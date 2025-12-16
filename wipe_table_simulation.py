#!/usr/bin/env python3
"""
擦桌子仿真 - 基于混合力位控制
使用 MuJoCo Franka Panda 模型实现表面擦拭任务

控制方法：
- 混合运动-力控制（基于约束）
- 运动子空间：XY位置控制（切向擦拭）
- 力子空间：Z方向力控制 + 旋转力矩控制（保持工具姿态）
"""

import mujoco
import mujoco.viewer
import numpy as np
import re
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from ikpy.chain import Chain
from dynamics_calculator import DynamicsCalculator


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
        self.X_e_integral = np.zeros(3)
        self.F_e_integral = 0.0
        self.quat_ref = None  # 参考姿态（四元数）
        self.tau_velocity_feedforward = np.zeros(7)  # 速度前馈力矩（接近阶段使用）

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

        # 力控制增益
        self.K_fp = 1.0  # 力比例增益
        self.K_fi = 0.2  # 力积分增益

        # 目标力（法向，向下为负）
        self.F_desired = -15.0  # N

        # 约束类型：'full' 表示约束所有旋转
        self.constraint_type = 'full'

    def get_jacobian_6x7(self):
        """获取6DOF雅可比矩阵（位置+姿态）"""
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
                J[:3, i] = jacp[:, dof_idx]
                J[3:, i] = jacr[:, dof_idx]

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
        
        参数:
            constraint_type: 'full' (4约束) 或 'minimal' (3约束)
        
        返回:
            A: 约束矩阵 (k, 6)，k为约束数量
        """
        if constraint_type == 'full':
            # 完全约束：v_z, ω_x, ω_y, ω_z
            A = np.zeros((4, 6))
            A[0, 2] = 1.0  # 约束Z方向平移 v_z
            A[1, 3] = 1.0  # 约束绕X轴旋转 ω_x
            A[2, 4] = 1.0  # 约束绕Y轴旋转 ω_y
            A[3, 5] = 1.0  # 约束绕Z轴旋转 ω_z
        else:
            # 最小约束：v_z, ω_x, ω_y（允许绕Z轴旋转）
            A = np.zeros((3, 6))
            A[0, 2] = 1.0  # v_z
            A[1, 3] = 1.0  # ω_x
            A[2, 4] = 1.0  # ω_y

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
            amplitude = 0.2  # 20cm幅度

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
            amplitude = 0.15  # 15cm幅度

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

    def control_step(self, t, dt=0.002):
        """
        执行一步混合力位控制 - 公式11.64
        
        控制律：
        τ = J^T[P(K_p X_e + K_i ∫X_e + K_d V_e) + (I-P)(F_d + K_fp F_e + K_fi ∫F_e) + η]
        
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

        # 计算任务空间速度
        V = J @ qdot  # [vx, vy, vz, wx, wy, wz]

        # 更新动力学计算器状态
        self.dynamics_calc.set_configuration(q, qdot)

        # 获取接触状态
        is_contact, F_curr = self.get_contact_force()

        # 更新接触时间
        if is_contact and self.contact_t is None:
            self.contact_t = t
            self.F_e_integral = 0.0
            self.X_e_integral = np.zeros(3)
            # 通过正运动学计算期望姿态（使用当前关节角度和期望位置）
            pos_ref = np.array([0.5, 0.0, 0.3])  # 期望位置
            self.quat_ref = self.compute_desired_orientation_from_fk(q, pos_ref)
        elif not is_contact:
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
            omega_des = self.K_p_rot * e_rot - 0.5 * self.K_p_rot * V[3:]

            # ========== 构建期望任务空间速度 ==========
            V_desired = np.zeros(6)
            V_desired[:3] = vel_ref  # 位置速度
            V_desired[3:] = omega_des  # 角速度

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

            # 使用正运动学重新计算姿态
            quat_ref = self.compute_desired_orientation_from_fk(q, pos_ref)
            self.quat_ref = quat_ref

            # 接触后不使用速度前馈
            self.tau_velocity_feedforward = np.zeros(7)

        # 计算动力学量
        Lambda = self.dynamics_calc.compute_task_space_mass_matrix(
            q, task_space_dim=6, use_pseudoinverse=True, damping=1e-6
        )

        eta = self.dynamics_calc.compute_task_space_coriolis(
            q, V, task_space_dim=6, use_pseudoinverse=True, damping=1e-6
        )

        # 计算约束和投影矩阵
        if is_contact:
            A = self.compute_constraint_matrix(self.constraint_type)
            P = self.compute_projection_matrix(Lambda, A)
        else:
            P = np.eye(6)
            A = np.zeros((0, 6))

        # ========== 混合控制计算 ==========
        # 位置误差和速度误差
        X_e = pos_ref - pos_curr
        V_e = vel_ref - V[:3]

        # 姿态误差
        e_rot = self.quaternion_error(self.quat_ref, quat_curr)
        omega_e = -V[3:]  # 角速度误差（目标角速度为0）

        # 更新积分项
        self.X_e_integral += X_e * dt
        self.X_e_integral = np.clip(self.X_e_integral, -0.2, 0.2)

        # 运动控制部分（投影到运动子空间）
        # 构建完整的6DOF控制向量
        F_motion_6d = np.zeros(6)

        # 平移控制（PID）
        F_motion_6d[:3] = (
                self.K_p @ X_e +
                self.K_i @ self.X_e_integral +
                self.K_d @ V_e
        )

        # 旋转控制（如果不在约束中）
        if not is_contact:
            F_motion_6d[3:] = (
                    self.K_p_rot * e_rot +
                    0.5 * self.K_p_rot * omega_e
            )
        # 接触时旋转部分保持为0（由约束处理）

        # 投影到运动子空间（完整的6DOF投影）
        # 公式11.64：F_motion = P @ (K_p X_e + K_i ∫X_e + K_d V_e)
        F_motion = P @ F_motion_6d

        # 力控制部分（投影到力子空间）
        F_force = np.zeros(6)
        if is_contact:
            # 力误差
            F_e = self.F_desired - F_curr
            self.F_e_integral += F_e * dt
            self.F_e_integral = np.clip(self.F_e_integral, -100.0, 100.0)

            # 力控制wrench
            F_d = np.zeros(6)
            F_d[2] = self.F_desired + self.K_fp * F_e + self.K_fi * self.F_e_integral

            # 旋转力矩控制（保持姿态）
            if self.constraint_type == 'full':
                # 完全约束时，通过力子空间控制旋转力矩
                F_d[3] = float(self.K_p_rot[0] * e_rot[0])  # 绕X轴
                F_d[4] = float(self.K_p_rot[1] * e_rot[1])  # 绕Y轴
                F_d[5] = float(self.K_p_rot[2] * e_rot[2])  # 绕Z轴

            # 投影到力子空间
            F_force = (np.eye(6) - P) @ F_d

        # 组合控制wrench
        F_cmd = F_motion + F_force + eta

        # 转换为关节力矩
        tau = J.T @ F_cmd

        # 在接近阶段，添加速度前馈项（使用 J^(-1) @ V 计算得到）
        if not is_contact:
            tau += self.tau_velocity_feedforward

        # 添加重力补偿
        for i in range(self.n_joints):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                tau[i] += self.data.qfrc_bias[dof_idx]

        # 力矩限制
        tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
        tau = np.clip(tau, -tau_max, tau_max)

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

                    print(f"[{phase:8}] t={t:5.1f}s | "
                          f"位置:[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                          f"力:{F:7.2f}N | {contact_str}")
                    last_print = t

                t += dt

            print(f"\n✓ 仿真完成！总时长: {t:.2f}s")

        print("\nViewer已关闭")


if __name__ == "__main__":
    sim = WipeTableSimulation(model_path="surface_force_control.xml")
    sim.run(duration=30.0)
