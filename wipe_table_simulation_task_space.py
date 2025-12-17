#!/usr/bin/env python3
"""
任务空间控制仿真 - 基于公式11.37（不使用投影矩阵版本）
使用 MuJoCo Franka Panda 模型实现任务空间控制

控制方法：
- 任务空间控制（公式11.37）
- 直接使用任务空间控制律，不使用投影矩阵

控制律（公式11.37）：
τ = J_b^T(θ)[Λ̃(θ)d/dt([Ad_{X^{-1}X_d}]V_d) + K_p X_e + K_i∫X_e + K_d V_e] + η̃(θ, V_b)

其中：
- X_e = log(X^{-1}X_d) 是配置误差
- V_e = [Ad_{X^{-1}X_d}]V_d - V 是速度误差
- Λ̃(θ) 是任务空间质量矩阵
- η̃(θ, V_b) 是任务空间 Coriolis 项

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


class TaskSpaceSimulation:
    """任务空间控制仿真类（公式11.37）"""

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
        self.X_e_integral = np.zeros(6)  # 6维配置误差积分
        self.quat_ref = None  # 参考姿态（四元数）

        # 调试标志
        self.debug = False  # 设置为True启用调试输出
        self.last_debug_time = -2.0  # 上次debug输出的时间

        print("✓ 任务空间控制仿真初始化完成")

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
            active_links_mask = [False] * len(temp_chain.links)
            active_indices = []

            for i, link in enumerate(temp_chain.links):
                link_name = getattr(link, 'name', '')
                if re.match(r'.*panda_joint[1-7]', link_name):
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
                return chain, active_indices
            else:
                print(f"⚠ 未找到7个活动关节（找到{len(active_indices)}个）")
                return None, None
        except Exception as e:
            print(f"⚠ ikpy 初始化失败: {e}")
            return None, None

    def compute_initial_configuration_ik(self):
        """
        通过逆运动学求解末端Z轴朝下的初始配置

        返回:
            q_init: 初始关节角度 (7,)
        """
        q_default = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

        if self.ikpy_chain is None or self.ikpy_active_indices is None:
            print("⚠ ikpy 链未加载，使用默认初始配置")
            return q_default

        try:
            target_position = np.array([0.5, 0.0, 0.3])
            target_orientation = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
            ])

            joint_limits = np.array([
                [-2.8973, 2.8973],
                [-1.7628, 1.7628],
                [-2.8973, 2.8973],
                [-3.0718, -0.0698],
                [-2.8973, 2.8973],
                [-0.0175, 3.7525],
                [-2.8973, 2.8973]
            ])

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

            q_init = np.array([ik_solution[idx] for idx in self.ikpy_active_indices])

            if np.all((q_init >= joint_limits[:, 0]) & (q_init <= joint_limits[:, 1])):
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
        """设置控制参数（公式11.37）"""
        # 6维任务空间控制增益（角速度在前）
        # K_p: 位置和姿态比例增益
        self.K_p = np.diag([50.0, 50.0, 50.0, 1000.0, 1000.0, 500.0])  # [旋转, 位置]
        
        # K_i: 积分增益
        self.K_i = np.diag([10.0, 10.0, 10.0, 20.0, 20.0, 10.0])
        
        # K_d: 微分增益
        self.K_d = np.diag([30.0, 30.0, 30.0, 80.0, 80.0, 40.0])

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

    def rotation_matrix_to_quaternion(self, R):
        """
        将旋转矩阵转换为四元数（使用scipy.spatial.transform.Rotation）

        参数:
            R: 旋转矩阵 (3, 3)

        返回:
            quat: 四元数 (4,) [w, x, y, z]，已归一化（MuJoCo格式）
        """
        try:
            rot = Rotation.from_matrix(R)
            quat_scipy = rot.as_quat()  # [x, y, z, w]
            quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
            return quat
        except:
            return np.array([1.0, 0.0, 0.0, 0.0])

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
            rot = Rotation.from_matrix(R)
            omega = rot.as_rotvec()

            theta = np.linalg.norm(omega)
            if theta < 1e-6:
                omega_skew = np.array([
                    [0, -omega[2], omega[1]],
                    [omega[2], 0, -omega[0]],
                    [-omega[1], omega[0], 0]
                ])
                V_inv = np.eye(3) - 0.5 * omega_skew
                v = V_inv @ p
            else:
                omega_normalized = omega / theta
                omega_skew = np.array([
                    [0, -omega_normalized[2], omega_normalized[1]],
                    [omega_normalized[2], 0, -omega_normalized[0]],
                    [-omega_normalized[1], omega_normalized[0], 0]
                ])

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
        计算配置误差 X_e = log(X^{-1} X_d) - 公式11.37

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
            e_pos = pos_ref - pos_curr
            e_rot = self.quaternion_error(quat_ref, quat_curr)
            return np.concatenate([e_rot, e_pos])

    def quaternion_error(self, q_ref, q_curr):
        """
        计算四元数误差（用于姿态控制）

        参数:
            q_ref: 参考四元数 (4,) [w, x, y, z]
            q_curr: 当前四元数 (4,) [w, x, y, z]

        返回:
            e_rot: 旋转误差向量 (3,)
        """
        try:
            q_ref_scipy = np.array([q_ref[1], q_ref[2], q_ref[3], q_ref[0]])
            q_curr_scipy = np.array([q_curr[1], q_curr[2], q_curr[3], q_curr[0]])

            rot_ref = Rotation.from_quat(q_ref_scipy)
            rot_curr = Rotation.from_quat(q_curr_scipy)

            rot_err = rot_ref * rot_curr.inv()
            e_rot = rot_err.as_rotvec()
            return e_rot
        except:
            return np.zeros(3)

    def compute_velocity_error(self, pos_curr, quat_curr, pos_ref, quat_ref, vel_ref, V_curr):
        """
        计算速度误差 V_e = [Ad_{X^{-1}X_d}]V_d - V - 公式11.37

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
                vel_ref_6d = np.concatenate([np.zeros(3), vel_ref])
            else:
                vel_ref_6d = vel_ref

            # 计算速度误差：V_e = [Ad_{X^{-1}X_d}]V_d - V
            V_e = Ad @ vel_ref_6d - V_curr

            return V_e
        except:
            if len(vel_ref) == 3:
                V_e = np.concatenate([-V_curr[:3], vel_ref - V_curr[3:]])
            else:
                V_e = vel_ref - V_curr
            return V_e

    def compute_adjoint_transformation(self, R, p):
        """
        计算 Adjoint 变换矩阵 Ad_X（角速度在前版本）

        对于 SE(3) 中的变换 X = (R, p)，Adjoint 变换为：
        Ad_X = [R    p×R]
               [0    R  ]

        参数:
            R: 旋转矩阵 (3×3)
            p: 位置向量 (3,)

        返回:
            Ad: Adjoint 变换矩阵 (6×6)（角速度在前）
        """
        p_skew = np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ])

        Ad = np.zeros((6, 6))
        Ad[:3, :3] = R  # 角速度到角速度
        Ad[:3, 3:] = p_skew @ R  # 线速度到角速度
        Ad[3:, :3] = np.zeros((3, 3))  # 角速度到线速度（为0）
        Ad[3:, 3:] = R  # 线速度到线速度

        return Ad

    def compute_desired_acceleration_feedforward(self, pos_ref, quat_ref, vel_ref, pos_curr, quat_curr, V, dt):
        """
        计算前馈加速度项：d/dt([Ad_{X^{-1}X_d}]V_d)

        简化实现：对于平滑轨迹，此项可以近似为 V̇_d 或忽略

        参数:
            pos_ref, quat_ref: 参考位姿
            vel_ref: 参考速度
            pos_curr, quat_curr: 当前位姿
            V: 当前任务空间速度
            dt: 时间步长

        返回:
            V_dot_desired: 期望加速度 (6,)
        """
        # 简化实现：对于大多数平滑轨迹，加速度项可以忽略或简化
        V_dot_desired = np.zeros(6)
        return V_dot_desired

    def control_step(self, t, dt=0.002):
        """
        执行一步任务空间控制 - 公式11.37

        控制律（公式11.37）：
        τ = J_b^T(θ)[Λ̃(θ)d/dt([Ad_{X^{-1}X_d}]V_d) + K_p X_e + K_i∫X_e + K_d V_e] + η̃(θ, V_b)

        其中：
        - J_b^T(θ): 末端执行器坐标系中的雅可比转置
        - Λ̃(θ): 任务空间质量矩阵
        - η̃(θ, V_b): 任务空间 Coriolis 项（末端执行器坐标系）
        - V_b: 末端执行器坐标系中的速度
        - X_e = log(X^{-1}X_d): 配置误差
        - V_e = [Ad_{X^{-1}X_d}]V_d - V: 速度误差

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
        V_b = J @ qdot

        # 更新动力学计算器状态
        self.dynamics_calc.set_configuration(q, qdot)

        # 生成参考轨迹
        # 目标位置：在表面上方
        z_des = 0.4 - min(t / 2.0, 1.0) * 0.25  # 从0.4m降到0.15m
        pos_ref = np.array([0.5, 0.0, z_des])

        # 参考速度：Z方向下降速度
        vel_ref = np.array([0.0, 0.0, -0.25 / 2.0 if t < 2.0 else 0.0])

        # 期望姿态：Z轴朝下
        target_orientation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        quat_ref = self.rotation_matrix_to_quaternion(target_orientation)
        self.quat_ref = quat_ref

        # 计算动力学量
        Lambda = self.dynamics_calc.compute_task_space_mass_matrix(
            q, task_space_dim=6, use_pseudoinverse=True, damping=1e-6
        )

        # 计算任务空间 Coriolis 项 η̃(θ, V_b)
        eta_tilde = self.dynamics_calc.compute_task_space_coriolis(
            q, V_b, task_space_dim=6, use_pseudoinverse=True, damping=1e-6
        )

        # ========== 计算配置误差和速度误差 ==========
        # 计算6维配置误差：X_e = log(X^{-1} X_d)
        X_e = self.compute_configuration_error(
            pos_curr, quat_curr, pos_ref, quat_ref
        )

        # 计算6维速度误差：V_e = [Ad_{X^{-1}X_d}]V_d - V
        vel_ref_6d = vel_ref if len(vel_ref) == 6 else np.concatenate([np.zeros(3), vel_ref])
        V_e = self.compute_velocity_error(
            pos_curr, quat_curr, pos_ref, quat_ref, vel_ref_6d, V_b
        )

        # 更新积分项（6维）
        self.X_e_integral += X_e * dt
        self.X_e_integral = np.clip(self.X_e_integral, -0.02, 0.02)

        # ========== 计算前馈加速度项 ==========
        V_dot_desired = self.compute_desired_acceleration_feedforward(
            pos_ref, quat_ref, vel_ref, pos_curr, quat_curr, V_b, dt
        )

        # ========== 任务空间控制律（公式11.37）==========
        # 公式11.37：τ = J_b^T(θ)[Λ̃(θ)d/dt([Ad_{X^{-1}X_d}]V_d) + K_p X_e + K_i∫X_e + K_d V_e] + η̃(θ, V_b)
        # 
        # 注意：η̃(θ, V_b) 是任务空间的 Coriolis 项（6维），在公式中位于括号外。
        # 但为了维度匹配，需要将 η̃ 也通过 J_b^T 转换到关节空间。
        # 实际上等价于：τ = J_b^T(θ)[... + η̃(θ, V_b)]
        # 
        # 任务空间控制力：F_cmd = Λ̃(θ)d/dt([Ad_{X^{-1}X_d}]V_d) + K_p X_e + K_i∫X_e + K_d V_e
        F_cmd = (
            Lambda @ V_dot_desired +
            self.K_p @ X_e +
            self.K_i @ self.X_e_integral +
            self.K_d @ V_e
        )

        # ========== 转换为关节力矩 ==========
        # 公式11.37：τ = J_b^T(θ) F_cmd + J_b^T(θ) η̃(θ, V_b)
        # 等价于：τ = J_b^T(θ)[F_cmd + η̃(θ, V_b)]
        # 为了维度匹配，将 η̃ 也通过 J^T 转换
        tau = J.T @ F_cmd + J.T @ eta_tilde

        # 力矩限制
        tau_max = np.array([8700, 8700, 8700, 8700, 1200, 1200, 1200], dtype=float)
        tau = np.clip(tau, -tau_max, tau_max)

        # ========== DEBUG输出 ==========
        should_debug = self.debug and (t - self.last_debug_time >= 2.0)
        if should_debug:
            self.last_debug_time = t
            print(f"[DEBUG] t={t:.3f}s | pos_curr={pos_curr} | pos_ref={pos_ref}")
            print(f"  X_e={X_e} | V_e={V_e}")
            print(f"  F_cmd={F_cmd} | tau={tau}")

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
        print("任务空间控制仿真 - 公式11.37")
        print("=" * 70)
        print("\n控制方法:")
        print("  - 任务空间控制（不使用投影矩阵）")
        print("  - 控制律：τ = J_b^T(θ)[Λ̃(θ)d/dt([Ad_{X^{-1}X_d}]V_d) + K_p X_e + K_i∫X_e + K_d V_e] + η̃(θ, V_b)")
        print("=" * 70)
        print("\n[启动仿真...]")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

            dt = self.model.opt.timestep
            t = 0.0

            print("\n✓ Viewer 已启动")
            print("  关闭窗口停止仿真\n")

            while viewer.is_running() and t < duration:
                # 执行控制
                self.control_step(t, dt)

                # 物理仿真步进
                mujoco.mj_step(self.model, self.data)

                # 更新viewer
                viewer.sync()

                t += dt

            print(f"\n✓ 仿真完成！总时长: {t:.2f}s")

        print("\nViewer已关闭")


if __name__ == "__main__":
    sim = TaskSpaceSimulation(model_path="surface_force_control.xml")
    sim.debug = False  # 设置为 True 启用调试输出
    sim.run(duration=30.0)

