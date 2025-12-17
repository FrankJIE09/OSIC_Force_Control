#!/usr/bin/env python3
"""
擦桌子仿真 - 基于混合力位控制（角速度在前版本）
使用 MuJoCo Franka Panda 模型实现表面擦拭任务

控制方法：
- 混合运动-力控制（基于约束）
- 运动子空间：XY位置控制（切向擦拭）
- 力子空间：Z方向力控制 + 旋转力矩控制（保持工具姿态）

**重要约定（书本第513行约定）**：
- Twist V = [ω_x, ω_y, ω_z, v_x, v_y, v_z]^T  （角速度在前，线速度在后）
- Wrench F = [m_x, m_y, m_z, f_x, f_y, f_z]^T （力矩在前，力在后）
- 所有6维向量：索引0-2为角速度/力矩，索引3-5为线速度/力
- 雅可比矩阵：J[:3, :]为角速度部分，J[3:, :]为线速度部分

注意：本代码使用书本第513行约定（角速度在前），与MuJoCo标准约定不同。
"""

import mujoco
import mujoco.viewer
import numpy as np
import re
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from ikpy.chain import Chain
from dynamics_calculator_wv import DynamicsCalculator


class WipeTableSimulation:
    """擦桌子仿真类"""

    def __init__(self, model_path: str = "surface_force_control.xml"):
        """
        初始化仿真

        参数:
            model_path: MuJoCo 模型文件路径
        """
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 获取关节信息
        self.joint_names = [f"panda_joint{i + 1}" for i in range(7)]
        self.n_joints = 7

        # 获取关节DOF索引
        self.joint_dof_indices = []
        for name in self.joint_names:
            try:
                joint_id = self.model.joint(name).id
                dof_adr = self.model.joint(joint_id).dofadr
                if len(dof_adr) > 0:
                    self.joint_dof_indices.append(dof_adr[0])
                else:
                    self.joint_dof_indices.append(-1)
            except:
                self.joint_dof_indices.append(-1)

        # 初始化 ikpy 链
        self.ikpy_chain, self.ikpy_active_indices = self._init_ikpy_chain()

        # 通过逆运动学求解末端Z轴朝下的初始配置
        q_init = self.compute_initial_configuration_ik()

        # 设置初始配置
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                self.data.qpos[dof_idx] = q_init[i]

        mujoco.mj_forward(self.model, self.data)

        # 初始化动力学计算器
        self.dynamics_calc = DynamicsCalculator(
            model_path=model_path,
            body_name="panda_hand",
            joint_names=self.joint_names
        )
        self.dynamics_calc.set_configuration(q_init)

        # 控制参数
        self.setup_control_parameters()

        # 状态变量
        self.contact_t = None
        self.is_contact_now = False  # 当前时刻是否检测到接触
        self.X_e_integral = np.zeros(6)  # 6维配置误差积分
        self.F_e_integral = np.zeros(6)  # 6维力误差积分（公式11.61）
        self.quat_ref = None  # 参考姿态（四元数）
        self.tau_velocity_feedforward = np.zeros(7)  # 速度前馈力矩（接近阶段使用）

        # 调试标志
        self.debug = False  # 设置为True启用调试输出

        print("✓ 擦桌子仿真初始化完成")

    def _init_ikpy_chain(self):
        """
        初始化 ikpy 链，根据 URDF 文件识别活动关节

        返回:
            chain: ikpy Chain 对象
            active_indices: 活动关节索引列表
        """
        urdf_path = "panda_arm.urdf"
        try:
            temp_chain = Chain.from_urdf_file(urdf_path, base_elements=['world'])

            # 精确匹配7个旋转关节：panda_joint1 到 panda_joint7
            # 排除：panda_joint_base, panda_link8_virtual_joint, panda_joint_wrist
            active_links_mask = [False] * len(temp_chain.links)
            active_indices = []

            for i, link in enumerate(temp_chain.links):
                link_name = getattr(link, 'name', '')
                # 精确匹配：panda_joint + 数字1-7，排除其他
                if re.match(r'.*panda_joint[1-7]', link_name):
                    # 进一步确认不是固定关节
                    if 'base' not in link_name.lower() and 'virtual' not in link_name.lower():
                        active_links_mask[i] = True
                        active_indices.append(i)

            if len(active_indices) == 7:
                chain = Chain.from_urdf_file(
                    urdf_path,
                    base_elements=['world'],
                    active_links_mask=active_links_mask
                )
                print(f"✓ 成功加载 URDF: {urdf_path}")
                print(f"  活动关节索引: {active_indices}")
                return chain, active_indices
            else:
                print(f"⚠ 未找到7个活动关节（找到{len(active_indices)}个）")
                return None, None
        except Exception as e:
            print(f"⚠ ikpy 初始化失败: {e}")
            return None, None

    def compute_initial_configuration_ik(self):
        """
        通过逆运动学求解末端Z轴朝下的初始配置（使用scipy优化）

        目标：
        - 位置：在表面上方（z = 0.3 m）
        - 姿态：Z轴朝下（末端坐标系Z轴指向 [0, 0, -1]）

        返回:
            q_init: 初始关节角度 (7,)
        """
        q_default = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

        if self.ikpy_chain is None or self.ikpy_active_indices is None:
            print("⚠ ikpy 链未加载，使用默认初始配置")
            return q_default

        try:
            # 目标位姿
            target_position = np.array([0.5, 0.0, 0.3])
            target_orientation = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
            ])

            # 关节限制（从URDF获取）
            joint_limits = np.array([
                [-2.8973, 2.8973],  # joint1
                [-1.7628, 1.7628],  # joint2
                [-2.8973, 2.8973],  # joint3
                [-3.0718, -0.0698],  # joint4
                [-2.8973, 2.8973],  # joint5
                [-0.0175, 3.7525],  # joint6
                [-2.8973, 2.8973]  # joint7
            ])

            # 使用ikpy求解（内部使用scipy优化）
            initial_guess = [0.0] * len(self.ikpy_chain.links)
            for i, active_idx in enumerate(self.ikpy_active_indices):
                if i < len(q_default):
                    initial_guess[active_idx] = q_default[i]

            ik_solution = self.ikpy_chain.inverse_kinematics(
                target_position=target_position,
                target_orientation=target_orientation,
                orientation_mode='all',
                initial_position=initial_guess,
                max_iter=100
            )

            # 提取活动关节角度
            q_init = np.array([ik_solution[idx] for idx in self.ikpy_active_indices])

            # 验证关节限制
            if np.all((q_init >= joint_limits[:, 0]) & (q_init <= joint_limits[:, 1])):
                # 验证正运动学
                joint_angles = [0.0] * len(self.ikpy_chain.links)
                for i, active_idx in enumerate(self.ikpy_active_indices):
                    joint_angles[active_idx] = q_init[i]

                T = self.ikpy_chain.forward_kinematics(joint_angles)
                pos_error = np.linalg.norm(T[:3, 3] - target_position)
                z_axis_error = np.linalg.norm(T[:3, 2] - target_orientation[:, 2])

                if pos_error < 0.05 and z_axis_error < 0.1:
                    print(f"✓ IK求解成功: 位置误差={pos_error:.4f}m, 姿态误差={z_axis_error:.4f}")
                    return q_init

            print("⚠ IK解验证失败，使用默认配置")
            return q_default
        except Exception as e:
            print(f"⚠ IK求解失败: {e}，使用默认配置")
            return q_default

    def setup_control_parameters(self):
        """设置控制参数"""
        # 运动控制增益（位置控制）
        self.K_p = np.diag([1000.0, 1000.0, 500.0])  # 位置增益
        self.K_i = np.diag([20.0, 20.0, 10.0])  # 积分增益
        self.K_d = np.diag([80.0, 80.0, 40.0])  # 微分增益

        # 姿态控制增益（旋转控制）
        self.K_p_rot = np.array([50.0, 50.0, 30.0])  # 旋转增益（向量形式）

        # 力控制增益（公式11.61：K_{fp}和K_{fi}是6×6矩阵）
        # 对于擦桌子任务，主要控制Z方向法向力，旋转力矩也使用PI控制
        self.K_fp = np.diag([0.5, 0.5, 1.0, 10.0, 10.0, 5.0])  # 力比例增益矩阵（6×6）
        self.K_fi = np.diag([0.1, 0.1, 0.2, 2.0, 2.0, 1.0])  # 力积分增益矩阵（6×6）

        # 目标力（法向，向下为负）
        self.F_desired = -15.0  # N

        # 约束类型：'full' 表示约束所有旋转
        self.constraint_type = 'full'

    def get_jacobian_6x7(self):
        """
        获取6DOF雅可比矩阵（位置+姿态）

        **Twist和Wrench的排序约定（书本第513行约定）**：
        - Twist V = [ω_x, ω_y, ω_z, v_x, v_y, v_z]^T  （角速度在前，线速度在后）
        - Wrench F = [m_x, m_y, m_z, f_x, f_y, f_z]^T （力矩在前，力在后）

        本代码使用书本第513行约定（角速度/力矩在前）。
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        try:
            hand_body_id = self.model.body("panda_hand").id
        except:
            return np.eye(6, 7)

        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, hand_body_id)

        J = np.zeros((6, self.n_joints))
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                J[:3, i] = jacr[:, dof_idx]  # 角速度部分 [ω_x, ω_y, ω_z]（索引0-2）
                J[3:, i] = jacp[:, dof_idx]  # 线速度部分 [v_x, v_y, v_z]（索引3-5）

        return J

    def get_contact_force(self):
        """
        获取机器人末端执行器与工作表面之间的接触力

        该方法使用两种方式检测接触：
        1. 距离判断：使用 ikpy 计算末端位置，通过 z 方向距离判断接触
        2. 力检测：从 MuJoCo 获取实际接触力用于力控制反馈

        工作表面定义：
        - 名称：'surface'（在 surface_force_control.xml 中定义）
        - 位置：z = 0.15 m（水平平面）
        - 尺寸：0.4 × 0.4 × 0.02 m

        返回:
            is_contact: bool
                是否检测到接触（基于距离判断）
                - True: 末端 z 位置接近表面（z ≤ 0.16 m）
                - False: 末端 z 位置远离表面（z > 0.18 m）
                - 中间状态：保持之前的接触状态（滞后机制，防止抖动）

            F_z: float
                Z方向法向力（单位：N，从 MuJoCo 接触力获取）
                - 负值：表示向下压力（机器人压向表面）
                - 正值：表示向上拉力（机器人被表面推开，通常不会发生）
                - 零值：无接触或力平衡
        """
        # ========== 获取关节角度 ==========
        q = np.zeros(self.n_joints)
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                q[i] = self.data.qpos[dof_idx]

        # ========== 使用 ikpy 计算末端位置（距离判断） ==========
        # 工作表面位置：z = 0.15 m
        # 使用距离判断接触，更稳定可靠
        surface_z = 0.15  # 工作表面 z 坐标
        contact_threshold_near = 0.24  # 接触阈值（接近表面）
        contact_threshold_far = 0.28  # 分离阈值（远离表面）

        try:
            # 使用 ikpy 计算末端位置
            pos_fk = self.compute_end_effector_position_from_fk(q)
            z_pos = pos_fk[2]  # Z 方向位置

            # ========== 基于距离的接触判断（滞后机制） ==========
            if z_pos <= contact_threshold_near:
                # 末端位置接近或低于表面 → 确定接触
                self.is_contact_now = True
                is_contact_by_distance = True
            elif z_pos >= contact_threshold_far:
                # 末端位置远离表面 → 确定未接触
                self.is_contact_now = False
                is_contact_by_distance = False
            else:
                # 中间状态：在 contact_threshold_near 和 contact_threshold_far 之间
                # 保持之前的状态，避免状态频繁切换
                is_contact_by_distance = (self.contact_t is not None)
        except:
            # 如果 ikpy 计算失败，回退到 MuJoCo 数据
            try:
                pos_curr = np.array(self.data.body("panda_hand").xpos)
                z_pos = pos_curr[2]
                if z_pos <= contact_threshold_near:
                    self.is_contact_now = True
                    is_contact_by_distance = True
                elif z_pos >= contact_threshold_far:
                    self.is_contact_now = False
                    is_contact_by_distance = False
                else:
                    is_contact_by_distance = (self.contact_t is not None)
            except:
                # 如果都失败，返回未接触
                return False, 0.0

        # ========== 获取接触力（用于力控制反馈） ==========
        force_z = 0.0  # Z方向接触力累加器

        try:
            # 获取工作表面几何体ID
            surf_geom_id = self.model.geom("surface").id

            # 遍历所有接触点，累加与表面相关的接触力
            for i in range(self.data.ncon):
                c = self.data.contact[i]

                # 只处理与工作表面相关的接触点
                if c.geom1 == surf_geom_id or c.geom2 == surf_geom_id:
                    f = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, f)
                    force_z += f[2]  # 累加 Z 方向力

                    # 如果检测到显著接触力，更新接触状态
                    if abs(f[2]) > 0.1:
                        self.is_contact_now = True
        except:
            # 如果无法获取接触力，使用距离判断的结果
            pass

        # ========== 返回结果 ==========
        # 接触判断：优先使用距离判断（更稳定）
        # 接触力：用于力控制反馈
        return is_contact_by_distance, force_z

    def compute_constraint_matrix(self, constraint_type='full'):
        """
        计算约束矩阵 A(θ) - 公式11.57

        对于表面接触：
        - 约束法向平移：v_z = 0
        - 约束所有旋转：ω_x = ω_y = ω_z = 0（保持工具与表面平行）

        **Twist排序约定（书本第513行约定）**：
        V = [ω_x, ω_y, ω_z, v_x, v_y, v_z]^T
        因此约束矩阵的列索引对应：
        - 列0,1,2: ω_x, ω_y, ω_z（角速度）
        - 列3,4,5: v_x, v_y, v_z（线速度）

        参数:
            constraint_type: 'full' (4约束) 或 'minimal' (3约束)

        返回:
            A: 约束矩阵 (k, 6)，k为约束数量
        """
        if constraint_type == 'full':
            # 完全约束：ω_x, ω_y, ω_z, v_z
            A = np.zeros((4, 6))
            A[0, 0] = 1.0  # 约束绕X轴旋转 ω_x（索引0对应ω_x）
            A[1, 1] = 1.0  # 约束绕Y轴旋转 ω_y（索引1对应ω_y）
            A[2, 2] = 1.0  # 约束绕Z轴旋转 ω_z（索引2对应ω_z）
            A[3, 5] = 1.0  # 约束Z方向平移 v_z（索引5对应v_z）
        else:
            # 最小约束：ω_x, ω_y, v_z（允许绕Z轴旋转）
            A = np.zeros((3, 6))
            A[0, 0] = 1.0  # ω_x
            A[1, 1] = 1.0  # ω_y
            A[2, 5] = 1.0  # v_z

        return A

    def compute_projection_matrix(self, Lambda, A):
        """
        计算投影矩阵 P(θ) - 公式11.63

        P = I - A^T(AΛ^{-1}A^T)^{-1}AΛ^{-1}

        参数:
            Lambda: 任务空间质量矩阵 (6, 6)
            A: 约束矩阵 (k, 6)

        返回:
            P: 投影矩阵 (6, 6)
        """
        try:
            Lambda_inv = np.linalg.inv(Lambda)
            A_Lambda_inv_AT = A @ Lambda_inv @ A.T

            if np.linalg.cond(A_Lambda_inv_AT) > 1e10:
                A_Lambda_inv_AT_inv = np.linalg.pinv(A_Lambda_inv_AT)
            else:
                A_Lambda_inv_AT_inv = np.linalg.inv(A_Lambda_inv_AT)

            P = np.eye(6) - A.T @ A_Lambda_inv_AT_inv @ A @ Lambda_inv
            return P
        except:
            return np.eye(6)

    def compute_end_effector_position_from_fk(self, q):
        """
        使用 ikpy 计算末端执行器的位置

        参数:
            q: 关节角度向量 (7,)

        返回:
            pos: 末端位置 (3,) [x, y, z]
        """
        if self.ikpy_chain is None:
            return np.array(self.data.body("panda_hand").xpos)

        try:
            joint_angles = [0.0] * len(self.ikpy_chain.links)
            for i, active_idx in enumerate(self.ikpy_active_indices):
                if i < len(q):
                    joint_angles[active_idx] = q[i]
            T = self.ikpy_chain.forward_kinematics(joint_angles)
            return T[:3, 3]
        except:
            return np.array(self.data.body("panda_hand").xpos)

    def compute_desired_orientation_from_fk(self, q, pos_ref):
        """
        通过正运动学计算期望的末端姿态四元数

        参数:
            q: 关节角度向量 (7,)
            pos_ref: 期望的末端位置 (3,)

        返回:
            quat: 期望姿态四元数 (4,) [w, x, y, z]
        """
        if self.ikpy_chain is None:
            return self.compute_desired_orientation(z_axis_direction=np.array([0.0, 0.0, -1.0]))

        try:
            joint_angles = [0.0] * len(self.ikpy_chain.links)
            for i, active_idx in enumerate(self.ikpy_active_indices):
                if i < len(q):
                    joint_angles[active_idx] = q[i]
            T = self.ikpy_chain.forward_kinematics(joint_angles)
            return self.rotation_matrix_to_quaternion(T[:3, :3])
        except:
            return self.compute_desired_orientation(z_axis_direction=np.array([0.0, 0.0, -1.0]))

    def compute_desired_orientation(self, z_axis_direction=np.array([0.0, 0.0, 1.0])):
        """
        计算期望的末端姿态四元数（使用scipy.spatial.transform.Rotation）

        对于擦桌子任务：
        - Z轴应该垂直于表面（向上或向下）
        - X和Y轴在表面平面内

        参数:
            z_axis_direction: 期望的Z轴方向（默认向上 [0, 0, 1]）

        返回:
            quat: 期望姿态四元数 (4,) [w, x, y, z] (MuJoCo格式)
        """
        try:
            # 归一化Z轴方向
            z_axis = z_axis_direction / np.linalg.norm(z_axis_direction)

            # 选择参考方向
            x_ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(z_axis, x_ref)) > 0.9:
                x_ref = np.array([0.0, 1.0, 0.0])

            # 构建正交坐标系
            y_axis = np.cross(z_axis, x_ref)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)

            # 构建旋转矩阵
            R = np.array([x_axis, y_axis, z_axis]).T

            # 使用scipy转换为四元数
            return self.rotation_matrix_to_quaternion(R)
        except:
            return np.array([1.0, 0.0, 0.0, 0.0])

    def rotation_matrix_to_quaternion(self, R):
        """
        将旋转矩阵转换为四元数（使用scipy.spatial.transform.Rotation）

        参数:
            R: 旋转矩阵 (3, 3)

        返回:
            quat: 四元数 (4,) [w, x, y, z]，已归一化（MuJoCo格式）
        """
        try:
            # scipy使用[x,y,z,w]格式，MuJoCo使用[w,x,y,z]格式
            rot = Rotation.from_matrix(R)
            quat_scipy = rot.as_quat()  # [x, y, z, w]
            # 转换为MuJoCo格式 [w, x, y, z]
            quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
            return quat
        except:
            return np.array([1.0, 0.0, 0.0, 0.0])

    def quaternion_error(self, q_ref, q_curr):
        """
        计算四元数误差（用于姿态控制，使用scipy.spatial.transform.Rotation）

        参数:
            q_ref: 参考四元数 (4,) [w, x, y, z] (MuJoCo格式)
            q_curr: 当前四元数 (4,) [w, x, y, z] (MuJoCo格式)

        返回:
            e_rot: 旋转误差向量 (3,) 旋转向量（轴角表示）
        """
        try:
            # 转换为scipy格式 [x, y, z, w]
            q_ref_scipy = np.array([q_ref[1], q_ref[2], q_ref[3], q_ref[0]])
            q_curr_scipy = np.array([q_curr[1], q_curr[2], q_curr[3], q_curr[0]])

            # 创建Rotation对象
            rot_ref = Rotation.from_quat(q_ref_scipy)
            rot_curr = Rotation.from_quat(q_curr_scipy)

            # 计算误差：R_err = R_ref * R_curr^{-1}
            rot_err = rot_ref * rot_curr.inv()

            # 转换为旋转向量（轴角表示）
            e_rot = rot_err.as_rotvec()
            return e_rot
        except:
            return np.zeros(3)

    def se3_log_map(self, R, p):
        """
        计算SE(3)的log映射：log(T)，其中T = [R, p; 0, 1]

        返回6维twist [ω, v]，其中：
        - ω: 旋转部分 (3,) 旋转向量
        - v: 平移部分 (3,)

        参数:
            R: 旋转矩阵 (3, 3)
            p: 位置向量 (3,)

        返回:
            twist: 6维twist [ω, v] (6,)（角速度在前，线速度在后）
        """
        try:
            # 计算旋转部分：ω = log(R)
            rot = Rotation.from_matrix(R)
            omega = rot.as_rotvec()  # 旋转向量

            # 计算平移部分：v = V^{-1} p，其中V是SE(3)的log映射的平移部分
            # 对于SE(3)的log映射，平移部分为：
            # v = V^{-1} p，其中V^{-1} = I - (1-cos(θ))/θ² [ω]× + (θ-sin(θ))/θ³ [ω]×²

            theta = np.linalg.norm(omega)
            if theta < 1e-6:
                # 小角度近似：V^{-1} ≈ I - 0.5 [ω]×
                omega_skew = np.array([
                    [0, -omega[2], omega[1]],
                    [omega[2], 0, -omega[0]],
                    [-omega[1], omega[0], 0]
                ])
                V_inv = np.eye(3) - 0.5 * omega_skew
                v = V_inv @ p
            else:
                # 精确计算
                omega_normalized = omega / theta
                omega_skew = np.array([
                    [0, -omega_normalized[2], omega_normalized[1]],
                    [omega_normalized[2], 0, -omega_normalized[0]],
                    [-omega_normalized[1], omega_normalized[0], 0]
                ])

                # V^{-1} = I - (1-cos(θ))/θ² [ω]× + (θ-sin(θ))/θ³ [ω]×²
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                V_inv = (np.eye(3) -
                         (1 - cos_theta) / (theta ** 2) * omega_skew +
                         (theta - sin_theta) / (theta ** 3) * omega_skew @ omega_skew)
                v = V_inv @ p

            return np.concatenate([omega, v])  # 角速度在前，线速度在后
        except:
            return np.zeros(6)

    def compute_configuration_error(self, pos_curr, quat_curr, pos_ref, quat_ref):
        """
        计算配置误差 X_e = log(X^{-1} X_d) - 公式11.61

        其中：
        - X = (R_curr, p_curr) 是当前位姿
        - X_d = (R_ref, p_ref) 是参考位姿
        - X_e 是在末端执行器坐标系中表示的6维twist

        参数:
            pos_curr: 当前位置 (3,)
            quat_curr: 当前姿态四元数 (4,) [w, x, y, z]
            pos_ref: 参考位置 (3,)
            quat_ref: 参考姿态四元数 (4,) [w, x, y, z]

        返回:
            X_e: 6维配置误差 [ω, v] (6,)（角速度误差在前，线速度误差在后）
        """
        try:
            # 转换为旋转矩阵
            q_curr_scipy = np.array([quat_curr[1], quat_curr[2], quat_curr[3], quat_curr[0]])
            q_ref_scipy = np.array([quat_ref[1], quat_ref[2], quat_ref[3], quat_ref[0]])

            rot_curr = Rotation.from_quat(q_curr_scipy)
            rot_ref = Rotation.from_quat(q_ref_scipy)

            R_curr = rot_curr.as_matrix()
            R_ref = rot_ref.as_matrix()

            # 计算 X^{-1} X_d = (R_curr^T R_ref, R_curr^T (p_ref - p_curr))
            R_err = R_curr.T @ R_ref
            p_err = R_curr.T @ (pos_ref - pos_curr)

            # 计算log映射：X_e = log(X^{-1} X_d)
            X_e = self.se3_log_map(R_err, p_err)

            return X_e
        except:
            # 如果计算失败，返回简化的位置和姿态误差
            e_pos = pos_ref - pos_curr
            e_rot = self.quaternion_error(quat_ref, quat_curr)
            return np.concatenate([e_rot, e_pos])  # 角速度误差在前，线速度误差在后

    def compute_velocity_error(self, pos_curr, quat_curr, pos_ref, quat_ref, vel_ref, V_curr):
        """
        计算速度误差 V_e = [Ad_{X^{-1}X_d}]V_d - V - 公式11.61

        其中：
        - V_d 是参考速度（在参考坐标系X_d中表示）
        - V 是当前速度（在当前坐标系X中表示）
        - [Ad_{X^{-1}X_d}] 将V_d从参考坐标系转换到当前坐标系

        参数:
            pos_curr: 当前位置 (3,)
            quat_curr: 当前姿态四元数 (4,)
            pos_ref: 参考位置 (3,)
            quat_ref: 参考姿态四元数 (4,)
            vel_ref: 参考速度 (6,) [ω_ref, v_ref]（在参考坐标系中，角速度在前）
            V_curr: 当前速度 (6,) [ω_curr, v_curr]（在当前坐标系中，角速度在前）

        返回:
            V_e: 6维速度误差 (6,)
        """
        try:
            # 转换为旋转矩阵
            q_curr_scipy = np.array([quat_curr[1], quat_curr[2], quat_curr[3], quat_curr[0]])
            q_ref_scipy = np.array([quat_ref[1], quat_ref[2], quat_ref[3], quat_ref[0]])

            rot_curr = Rotation.from_quat(q_curr_scipy)
            rot_ref = Rotation.from_quat(q_ref_scipy)

            R_curr = rot_curr.as_matrix()
            R_ref = rot_ref.as_matrix()

            # 计算 X^{-1} X_d 的旋转部分：R_curr^T R_ref
            R_err = R_curr.T @ R_ref

            # 计算 X^{-1} X_d 的平移部分：R_curr^T (p_ref - p_curr)
            p_err = R_curr.T @ (pos_ref - pos_curr)

            # 计算Adjoint变换 Ad_{X^{-1}X_d}
            Ad = self.compute_adjoint_transformation(R_err, p_err)

            # 如果vel_ref只有3维（只有平移速度），扩展为6维（角速度在前）
            if len(vel_ref) == 3:
                vel_ref_6d = np.concatenate([np.zeros(3), vel_ref])  # 角速度在前，线速度在后
            else:
                vel_ref_6d = vel_ref

            # 计算速度误差：V_e = [Ad_{X^{-1}X_d}]V_d - V
            V_e = Ad @ vel_ref_6d - V_curr

            return V_e
        except:
            # 如果计算失败，返回简化的速度误差（角速度在前）
            if len(vel_ref) == 3:
                V_e = np.concatenate([-V_curr[:3], vel_ref - V_curr[3:]])  # 角速度误差在前
            else:
                V_e = vel_ref - V_curr
            return V_e

    def generate_wipe_trajectory(self, t, t_contact):
        """
        生成擦拭轨迹

        参数:
            t: 当前时间
            t_contact: 接触后的时间

        返回:
            pos_ref: 参考位置 (3,)
            vel_ref: 参考速度 (3,)
            quat_ref: 参考姿态（四元数）(4,)
        """
        # 默认参考位置（表面中心）
        pos_ref = np.array([0.5, 0.0, 0.3])
        vel_ref = np.array([0.0, 0.0, 0.0])

        # 参考姿态：将在 control_step 中通过正运动学计算
        # 这里先使用默认值（会在调用时被正运动学结果覆盖）
        quat_ref = np.array([1.0, 0.0, 0.0, 0.0])  # 临时值

        if t_contact is None or t_contact < 2.0:
            # 接触前或刚接触：保持当前位置，准备下降
            return pos_ref, vel_ref, quat_ref

        # 接触后开始擦拭运动
        t_wipe = t_contact - 2.0

        if t_wipe < 10.0:
            # 前10秒：X轴前后擦拭
            cycle_t = t_wipe % 2.0  # 2秒一个周期
            amplitude = 0.02  # 20cm幅度

            if cycle_t < 1.0:
                # 向前
                progress = cycle_t / 1.0
                pos_ref[0] = 0.5 + amplitude * np.sin(progress * np.pi)
                vel_ref[0] = amplitude * np.pi / 1.0 * np.cos(progress * np.pi)
            else:
                # 向后
                progress = (cycle_t - 1.0) / 1.0
                pos_ref[0] = 0.5 + amplitude * np.sin((1.0 - progress) * np.pi)
                vel_ref[0] = -amplitude * np.pi / 1.0 * np.cos((1.0 - progress) * np.pi)

        elif t_wipe < 20.0:
            # 10-20秒：Y轴左右擦拭
            cycle_t = (t_wipe - 10.0) % 2.0
            amplitude = 0.02  # 15cm幅度

            if cycle_t < 1.0:
                # 向右
                progress = cycle_t / 1.0
                pos_ref[1] = amplitude * np.sin(progress * np.pi)
                vel_ref[1] = amplitude * np.pi / 1.0 * np.cos(progress * np.pi)
            else:
                # 向左
                progress = (cycle_t - 1.0) / 1.0
                pos_ref[1] = amplitude * np.sin((1.0 - progress) * np.pi)
                vel_ref[1] = -amplitude * np.pi / 1.0 * np.cos((1.0 - progress) * np.pi)

        else:
            # 20秒后：圆形擦拭
            cycle_t = (t_wipe - 20.0) % 4.0
            radius = 0.15
            angle = 2.0 * np.pi * cycle_t / 4.0

            pos_ref[0] = 0.5 + radius * np.cos(angle)
            pos_ref[1] = radius * np.sin(angle)
            vel_ref[0] = -radius * 2.0 * np.pi / 4.0 * np.sin(angle)
            vel_ref[1] = radius * 2.0 * np.pi / 4.0 * np.cos(angle)

        return pos_ref, vel_ref, quat_ref

    def compute_adjoint_transformation(self, R, p):
        """
        计算 Adjoint 变换矩阵 Ad_X（角速度在前版本）

        对于 SE(3) 中的变换 X = (R, p)，Adjoint 变换为：
        Ad_X = [R    0  ]
               [p×R  R ]

        其中 p× 是 p 的反对称矩阵

        对于角速度在前的约定，Adjoint矩阵结构为：
        Ad_X = [R    p×R]
               [0    R  ]

        参数:
            R: 旋转矩阵 (3×3)
            p: 位置向量 (3,)

        返回:
            Ad: Adjoint 变换矩阵 (6×6)（角速度在前）
        """
        # 计算 p 的反对称矩阵
        p_skew = np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ])

        # 构建 Adjoint 矩阵（角速度在前版本）
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = R  # 角速度到角速度
        Ad[:3, 3:] = p_skew @ R  # 线速度到角速度
        Ad[3:, :3] = np.zeros((3, 3))  # 角速度到线速度（为0）
        Ad[3:, 3:] = R  # 线速度到线速度

        return Ad

    def compute_desired_acceleration_feedforward(self, pos_ref, quat_ref, vel_ref, pos_curr, quat_curr, V, dt):
        """
        计算前馈加速度项：Λ̃(θ)d/dt([Ad_{X^{-1}X_d}]V_d)

        简化实现：计算期望加速度 V̇_d，并考虑坐标系变换

        参数:
            pos_ref, quat_ref: 参考位姿
            vel_ref: 参考速度（仅平移部分）
            pos_curr, quat_curr: 当前位姿
            V: 当前任务空间速度
            dt: 时间步长

        返回:
            V_dot_desired: 期望加速度 (6,)
        """
        # 简化实现：如果参考速度变化，计算加速度
        # 对于大多数情况，如果轨迹平滑，加速度项可以忽略或简化
        V_dot_desired = np.zeros(6)

        # 如果参考速度不为零，可以计算数值微分
        # 这里简化处理：对于平滑轨迹，加速度项较小，可以忽略
        # 或者使用期望速度的数值微分

        return V_dot_desired

    def control_step(self, t, dt=0.002):
        """
        执行一步混合力位控制 - 公式11.61（书本公式）

        控制律（公式11.61）：
        τ = J_b^T(θ)[P(θ)(Λ̃(θ)d/dt([Ad_{X^{-1}X_d}]V_d) + K_p X_e + K_i∫X_e + K_d V_e)
                  + (I-P(θ))(F_d + K_fp F_e + K_fi∫F_e) + η̃(θ, V_b)]

        其中：
        - J_b^T(θ): 末端执行器坐标系中的雅可比转置
        - P(θ): 投影矩阵（公式11.63）
        - Λ̃(θ): 任务空间质量矩阵
        - η̃(θ, V_b): 任务空间 Coriolis 项（末端执行器坐标系）
        - V_b: 末端执行器坐标系中的速度

        参数:
            t: 当前时间
            dt: 时间步长
        """
        mujoco.mj_forward(self.model, self.data)

        # 获取当前状态
        pos_curr = np.array(self.data.body("panda_hand").xpos)
        quat_curr = np.array(self.data.body("panda_hand").xquat)

        # 获取关节状态
        q = np.zeros(self.n_joints)
        qdot = np.zeros(self.n_joints)
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                q[i] = self.data.qpos[dof_idx]
                qdot[i] = self.data.qvel[dof_idx]

        # 计算雅可比矩阵
        J = self.get_jacobian_6x7()

        # 计算任务空间速度（末端执行器坐标系 {b}）
        # V_b = [ω_x, ω_y, ω_z, v_x, v_y, v_z]^T
        # **书本第513行约定：角速度在前，线速度在后**
        # 注意：需要在接近阶段之前计算，以便在接近阶段使用
        V_b = J @ qdot

        # 更新动力学计算器状态
        self.dynamics_calc.set_configuration(q, qdot)

        # 获取接触状态
        is_contact, F_curr = self.get_contact_force()

        # ========== DEBUG点1：接触状态检测 ==========
        if self.debug:
            pos_fk = self.compute_end_effector_position_from_fk(q)
            print(f"[DEBUG-接触] t={t:.3f}s | is_contact={is_contact} | "
                  f"F_curr={F_curr:.2f}N | z_pos={pos_fk[2]:.4f}m | "
                  f"contact_t={self.contact_t}")

        # 更新接触时间
        if is_contact and self.contact_t is None:
            self.contact_t = t
            self.F_e_integral = np.zeros(6)  # 6维力误差积分
            self.X_e_integral = np.zeros(6)  # 6维配置误差积分
            # 通过正运动学计算期望姿态（使用当前关节角度和期望位置）
            pos_ref = np.array([0.5, 0.0, 0.3])  # 期望位置
            self.quat_ref = self.compute_desired_orientation_from_fk(q, pos_ref)
            if self.debug:
                print(f"[DEBUG-接触建立] t={t:.3f}s | 接触建立！")
        elif not is_contact:
            if self.contact_t is not None and self.debug:
                print(f"[DEBUG-接触丢失] t={t:.3f}s | 接触丢失！之前接触时长={t - self.contact_t:.3f}s")
            self.contact_t = None

        # 生成参考轨迹
        if not is_contact:
            # ========== 接近阶段：使用速度控制保证Z轴朝下 ==========
            # 目标位置：在表面上方
            z_des = 0.4 - min(t / 2.0, 1.0) * 0.25  # 从0.4m降到0.15m
            pos_ref = np.array([0.5, 0.0, z_des])

            # 位置速度：Z方向下降速度
            vel_ref = np.array([0.0, 0.0, -0.25 / 2.0 if t < 2.0 else 0.0])

            # ========== 计算期望姿态（Z轴朝下） ==========
            # 期望姿态：Z轴朝下（末端坐标系Z轴指向 [0, 0, -1]）
            target_orientation = np.array([
                [1.0, 0.0, 0.0],  # X轴方向
                [0.0, -1.0, 0.0],  # Y轴方向
                [0.0, 0.0, -1.0]  # Z轴方向（向下）
            ])

            # 从旋转矩阵计算期望四元数（使用更稳健的方法）
            quat_ref = self.rotation_matrix_to_quaternion(target_orientation)
            self.quat_ref = quat_ref

            # ========== 使用速度控制：计算期望角速度 ==========
            # 计算姿态误差
            e_rot = self.quaternion_error(quat_ref, quat_curr)

            # 期望角速度：使得姿态误差减小（PD控制）
            # ω_des = K_p_rot * e_rot - K_d_rot * ω_curr
            # 使用末端执行器坐标系中的角速度（索引0-2）
            omega_des = self.K_p_rot * e_rot - 0.5 * self.K_p_rot * V_b[:3]

            # ========== 构建期望任务空间速度 ==========
            V_desired = np.zeros(6)
            V_desired[:3] = omega_des  # 角速度（索引0-2）
            V_desired[3:] = vel_ref  # 位置速度（索引3-5）

            # ========== 使用 J^(-1) @ V_desired 计算期望关节速度 ==========
            # 计算雅可比矩阵的伪逆
            try:
                J_pinv = np.linalg.pinv(J)
                qdot_desired = J_pinv @ V_desired
            except:
                # 如果伪逆计算失败，使用默认方法
                qdot_desired = np.zeros(7)

            # ========== 计算关节速度误差 ==========
            qdot_error = qdot_desired - qdot

            # ========== 使用速度误差生成控制力矩 ==========
            # 在接近阶段，使用速度控制来跟踪期望速度
            # 这里我们仍然使用位置控制，但加入速度前馈
            # 速度前馈项：K_v * qdot_error
            K_v = np.diag([50.0, 50.0, 50.0, 50.0, 30.0, 30.0, 30.0])  # 速度增益
            tau_velocity_feedforward = K_v @ qdot_error

            # 存储速度前馈项，将在后面添加到控制力矩中
            self.tau_velocity_feedforward = tau_velocity_feedforward
        else:
            # 接触后：使用擦拭轨迹
            t_contact = t - self.contact_t
            pos_ref, vel_ref, quat_ref = self.generate_wipe_trajectory(t, t_contact)

            quat_ref = self.compute_desired_orientation_from_fk(q, pos_ref)

            # 修改：接触后，Z方向参考位置应该基于实际接触位置
            # 避免位置误差过大导致位置控制与力控制冲突
            # 关键：参考位置应该等于或略低于当前位置，避免产生向上的位置控制力
            if t_contact < 0.5:  # 接触后0.5秒内，逐渐调整参考位置
                # 使用实际位置作为参考，等于当前位置（避免位置控制产生Z方向力）
                pos_ref[2] = pos_curr[2]  # 等于当前位置，避免位置误差
            else:
                # 稳定后，使用略低于当前位置的参考位置（允许轻微压缩）
                pos_ref[2] = pos_curr[2] - 0.005  # 略低于当前位置5mm，允许轻微压缩

            # 修改：接触后，姿态应该保持Z轴朝下，而不是通过正运动学计算
            # 因为正运动学计算的姿态可能不正确（特别是在移动时）
            # 使用固定的Z轴朝下姿态
            target_orientation = np.array([
                [1.0, 0.0, 0.0],  # X轴方向
                [0.0, -1.0, 0.0],  # Y轴方向
                [0.0, 0.0, -1.0]  # Z轴方向（向下）
            ])
            quat_ref = self.rotation_matrix_to_quaternion(target_orientation)
            self.quat_ref = quat_ref

            # 接触后不使用速度前馈
            self.tau_velocity_feedforward = np.zeros(7)

        # 计算动力学量
        Lambda = self.dynamics_calc.compute_task_space_mass_matrix(
            q, task_space_dim=6, use_pseudoinverse=True, damping=1e-6
        )

        # 计算任务空间 Coriolis 项 η̃(θ, V_b)（末端执行器坐标系）
        eta_tilde = self.dynamics_calc.compute_task_space_coriolis(
            q, V_b, task_space_dim=6, use_pseudoinverse=True, damping=1e-6
        )

        # 计算约束和投影矩阵
        if is_contact:
            A = self.compute_constraint_matrix(self.constraint_type)
            P = self.compute_projection_matrix(Lambda, A)
        else:
            P = np.eye(6)
            A = np.zeros((0, 6))

        # ========== 混合控制计算 ==========
        # 计算6维配置误差：X_e = log(X^{-1} X_d) - 公式11.61
        X_e = self.compute_configuration_error(
            pos_curr, quat_curr, pos_ref, self.quat_ref
        )

        # 计算6维速度误差：V_e = [Ad_{X^{-1}X_d}]V_d - V - 公式11.61
        # 如果vel_ref只有3维，需要扩展为6维（角速度部分为0，角速度在前）
        vel_ref_6d = vel_ref if len(vel_ref) == 6 else np.concatenate([np.zeros(3), vel_ref])
        V_e = self.compute_velocity_error(
            pos_curr, quat_curr, pos_ref, self.quat_ref, vel_ref_6d, V_b
        )

        # ========== DEBUG点2：位置和速度状态 ==========
        if self.debug and is_contact:
            print(f"[DEBUG-位置] t={t:.3f}s | pos_curr={pos_curr} | pos_ref={pos_ref} | "
                  f"X_e={X_e} | V_curr={V_b} | V_ref={vel_ref_6d}")
            print(f"[DEBUG-速度] t={t:.3f}s | V_z={V_b[5]:.4f}m/s | V_e_z={V_e[5]:.4f}m/s | "
                  f"omega_e={V_e[:3]}")

        # 更新积分项（6维）
        self.X_e_integral += X_e * dt
        self.X_e_integral = np.clip(self.X_e_integral, -0.02, 0.02)

        # ========== 运动控制部分（投影到运动子空间）==========
        # 公式11.61第一项：P(θ)(Λ̃(θ)d/dt([Ad_{X^{-1}X_d}]V_d) + K_p X_e + K_i∫X_e + K_d V_e)

        # 计算前馈加速度项（简化实现：对于平滑轨迹，此项可忽略或很小）
        V_dot_desired = self.compute_desired_acceleration_feedforward(
            pos_ref, self.quat_ref, vel_ref, pos_curr, quat_curr, V_b, dt
        )

        # 前馈项：Λ̃(θ) * V̇_d（简化：如果加速度项很小，可以忽略）
        # 对于大多数平滑轨迹，此项可以忽略
        feedforward_acceleration = V_dot_desired

        # 构建完整的6DOF控制向量
        # 公式11.61：F_motion = K_p X_e + K_i∫X_e + K_d V_e
        # 注意：X_e和V_e都是6维的

        # 扩展增益矩阵为6x6（如果当前是3x3）
        if self.K_p.shape == (3, 3):
            # 构建6x6增益矩阵（角速度在前）
            K_p_6d = np.zeros((6, 6))
            K_p_6d[:3, :3] = np.diag(self.K_p_rot)  # 旋转增益（索引0-2）
            K_p_6d[3:, 3:] = self.K_p  # 位置增益（索引3-5）

            K_i_6d = np.zeros((6, 6))
            K_i_6d[:3, :3] = np.diag(self.K_p_rot * 0.1)  # 旋转积分增益（较小，索引0-2）
            K_i_6d[3:, 3:] = self.K_i  # 位置积分增益（索引3-5）

            K_d_6d = np.zeros((6, 6))
            K_d_6d[:3, :3] = np.diag(self.K_p_rot * 0.5)  # 旋转微分增益（索引0-2）
            K_d_6d[3:, 3:] = self.K_d  # 位置微分增益（索引3-5）
        else:
            K_p_6d = self.K_p
            K_i_6d = self.K_i
            K_d_6d = self.K_d

        # 计算完整的6DOF运动控制力
        F_motion_6d = (
                K_p_6d @ X_e +
                K_i_6d @ self.X_e_integral +
                K_d_6d @ V_e
        )

        # 投影到运动子空间（完整的6DOF投影）
        # 公式11.61：F_motion = P @ (Λ̃(θ)d/dt([Ad_{X^{-1}X_d}]V_d) + K_p X_e + K_i∫X_e + K_d V_e)
        # 简化：F_motion = P @ (feedforward_acceleration + K_p X_e + K_i∫X_e + K_d V_e)
        F_motion = P @ Lambda @ (feedforward_acceleration + F_motion_6d)

        # 力控制部分（投影到力子空间）- 公式11.61
        # (I - P(θ))(F_d + K_{fp}F_e + K_{fi}∫F_e dt)
        F_force = np.zeros(6)
        if is_contact:
            # 计算6维力误差：F_e = F_d - F_curr
            # F_d是期望wrench（6维），F_curr是当前测量的wrench（6维）
            # 注意：wrench格式为[力矩, 力]，即[m_x, m_y, m_z, f_x, f_y, f_z]
            F_desired_6d = np.zeros(6)
            F_desired_6d[5] = self.F_desired  # Z方向期望法向力（索引5对应f_z）
            # 旋转力矩期望为0（保持姿态，索引0-2）
            # 注意：F_curr目前只有Z方向，需要扩展为6维
            F_curr_6d = np.zeros(6)
            F_curr_6d[5] = F_curr  # Z方向当前法向力（索引5对应f_z）
            # 可以从接触力中提取旋转力矩，这里简化处理

            # 计算6维力误差
            F_e_6d = F_desired_6d - F_curr_6d

            # 更新6维力误差积分（公式11.61）
            self.F_e_integral += F_e_6d * dt
            self.F_e_integral = np.clip(self.F_e_integral, -100.0, 100.0)

            # 力控制wrench：F_d + K_{fp}F_e + K_{fi}∫F_e dt（公式11.61）
            # 其中F_d是期望wrench，K_{fp}和K_{fi}是6×6增益矩阵
            F_force_cmd = F_desired_6d + self.K_fp @ F_e_6d + self.K_fi @ self.F_e_integral

            # 投影到力子空间：(I - P(θ)) @ F_force_cmd
            F_force = (np.eye(6) - P) @ F_force_cmd

            # ========== DEBUG点3：力控制 ==========
            if self.debug:
                print(f"[DEBUG-力控] t={t:.3f}s | F_curr={F_curr:.2f}N | F_desired={self.F_desired:.2f}N | "
                      f"F_e_6d={F_e_6d} | F_e_integral={self.F_e_integral} | "
                      f"F_force_cmd={F_force_cmd} | F_force={F_force}")

        # ========== 组合控制wrench ==========
        # 公式11.61：F_cmd = F_motion + F_force + η̃(θ, V_b)
        # 注意：虽然书本公式说Coriolis项不投影，但在约束方向（Z方向），
        # Coriolis项应该被投影掉，否则会在约束方向产生力，导致弹跳
        if is_contact:
            # 接触时：在约束方向，Coriolis项应该被投影到运动子空间
            # 这样可以避免在约束方向产生额外的力
            F_cmd = F_motion + F_force + eta_tilde
        else:
            # 无约束时：直接添加完整Coriolis项
            F_cmd = F_motion + F_force + eta_tilde

        # ========== 转换为关节力矩 ==========
        # 公式11.61：τ = J_b^T(θ) F_cmd
        # 其中 J_b 是末端执行器坐标系中的雅可比矩阵
        # 注意：如果需要禁用旋转力矩控制，可以设置 F_cmd[:3] = np.zeros(3)
        # F_cmd[:3] = np.zeros(3)  # 可选：禁用旋转力矩控制
        tau = J.T @ F_cmd

        # 在接近阶段，添加速度前馈项（使用 J^(-1) @ V 计算得到）
        if not is_contact:
            tau += self.tau_velocity_feedforward

        # 添加重力补偿
        # for i in range(self.n_joints):
        #     dof_idx = self.joint_dof_indices[i]
        #     if dof_idx >= 0:
        #         tau[i] += self.data.qfrc_bias[dof_idx]

        # 力矩限制
        tau_max = np.array([8700, 8700, 8700, 8700, 1200, 1200, 1200], dtype=float)
        tau = np.clip(tau, -tau_max, tau_max)

        # ========== DEBUG点4：控制力矩和wrench ==========
        if self.debug and is_contact:
            # Coriolis项（根据书本公式，不投影）
            print(f"[DEBUG-控制] t={t:.3f}s | F_motion[5]={F_motion[5]:.2f}N | "
                  f"F_force[5]={F_force[5]:.2f}N | eta_tilde[5]={eta_tilde[5]:.2f}N | "
                  f"F_cmd[5]={F_cmd[5]:.2f}N")
            print(f"[DEBUG-力矩] t={t:.3f}s | tau_max={tau_max} | "
                  f"tau_before_clip={J.T @ F_cmd} | tau_after_clip={tau}")
            print(f"[DEBUG-投影] t={t:.3f}s | P[5,5]={P[5, 5]:.4f} | "
                  f"(I-P)[5,5]={(np.eye(6) - P)[5, 5]:.4f} | "
                  f"F_motion_6d[5]={F_motion_6d[5]:.2f}N")
            # ========== DEBUG点5：姿态约束 ==========
            e_rot = X_e[:3]  # 从6维配置误差中提取旋转部分（索引0-2）
            omega_e = V_e[:3]  # 从6维速度误差中提取角速度部分（索引0-2）
            print(f"[DEBUG-姿态] t={t:.3f}s | e_rot={e_rot} | "
                  f"V_b[:3]={V_b[:3]} | omega_e={omega_e}")
            if is_contact:
                print(f"[DEBUG-旋转控制] t={t:.3f}s | F_d[:3]={F_force[:3]} | "
                      f"F_force[:3]={F_force[:3]} | F_motion[:3]={F_motion[:3]}")
                print(f"[DEBUG-投影旋转] t={t:.3f}s | P[:3,:3]对角线={np.diag(P[:3, :3])} | "
                      f"(I-P)[:3,:3]对角线={np.diag((np.eye(6) - P)[:3, :3])}")

        # 应用控制
        for i, name in enumerate(self.joint_names):
            try:
                ctrl_id = self.model.actuator(name).id
                self.data.actuator(ctrl_id).ctrl = float(tau[i])
            except:
                pass

    def run(self, duration=30.0):
        """
        运行仿真

        参数:
            duration: 仿真时长（秒）
        """
        print("\n" + "=" * 70)
        print("擦桌子仿真 - 混合力位控制")
        print("=" * 70)
        print("\n任务说明:")
        print("  1. 接近表面并建立接触")
        print("  2. 施加法向力（目标：15N）")
        print("  3. X轴前后擦拭（10秒）")
        print("  4. Y轴左右擦拭（10秒）")
        print("  5. 圆形擦拭（持续）")
        print("\n控制方法:")
        print("  - 约束：v_z, ω_x, ω_y, ω_z（保持工具与表面平行）")
        print("  - 运动控制：XY位置（切向擦拭）")
        print("  - 力控制：Z方向法向力 + 旋转力矩")
        print("=" * 70)
        print("\n[启动仿真...]")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 启用可视化选项
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

            dt = self.model.opt.timestep
            t = 0.0
            last_print = 0.0

            print("\n✓ Viewer 已启动")
            print("  关闭窗口停止仿真\n")

            while viewer.is_running() and t < duration:
                # 执行控制
                self.control_step(t, dt)

                # 物理仿真步进
                mujoco.mj_step(self.model, self.data)

                # 更新viewer
                viewer.sync()

                # 进度报告
                if t - last_print >= 0.01:
                    is_contact, F = self.get_contact_force()
                    contact_str = "✓接触" if is_contact else "✗未接触"

                    pos = np.array(self.data.body("panda_hand").xpos)

                    if t < 2.0:
                        phase = "接近"
                    elif t < 4.0:
                        phase = "建立接触"
                    elif t < 14.0:
                        phase = "X轴擦拭"
                    elif t < 24.0:
                        phase = "Y轴擦拭"
                    else:
                        phase = "圆形擦拭"

                    # print(f"[{phase:8}] t={t:5.1f}s | "
                    #       f"位置:[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                    #       f"力:{F:7.2f}N | {contact_str}")
                    last_print = t

                t += dt

            print(f"\n✓ 仿真完成！总时长: {t:.2f}s")

        print("\nViewer已关闭")


if __name__ == "__main__":
    sim = WipeTableSimulation(model_path="surface_force_control.xml")
    # ========== 调试开关 ==========
    # 设置为 True 启用详细调试输出（用于排查弹跳问题）
    sim.debug = True  # 改为 True 启用调试
    # ========== 调试开关 ==========
    sim.run(duration=30.0)
