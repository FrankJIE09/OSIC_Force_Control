#!/usr/bin/env python3
"""
任务空间位置控制仿真 - 基于公式236（位置控制版本）
使用 MuJoCo Franka Panda 模型实现任务空间位置控制

控制方法：
- 任务空间位置控制（公式236，扩展到6DOF）
- 通过速度积分得到位置命令：θ_cmd = θ_cmd + θ̇ * dt
- 不需要动力学模型

控制律（公式236，6DOF扩展）：
θ̇ = J^{-1}(θ)[Ẋ_d + K_p X_e]
θ_cmd = θ_cmd + θ̇ * dt

其中：
- X_e = log(X^{-1}X_d) 是配置误差（6维）
- Ẋ_d 是参考末端执行器速度（6维twist）
- J(θ) 是雅可比矩阵（6×7）
- K_p 是比例增益矩阵（6×6）
- dt 是时间步长

**重要约定（书本第513行约定）**：
- Twist V = [ω_x, ω_y, ω_z, v_x, v_y, v_z]^T  （角速度在前，线速度在后）
- 所有6维向量：索引0-2为角速度，索引3-5为线速度
- 雅可比矩阵：J[:3, :]为角速度部分，J[3:, :]为线速度部分

注意：本代码使用书本第513行约定（角速度在前），与MuJoCo标准约定不同。
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from tf2_ros import TransformBroadcaster
    from geometry_msgs.msg import TransformStamped
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class TaskSpaceVelocitySimulation:
    """任务空间位置控制仿真类（公式236，通过速度积分）"""

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

        # 获取位置执行器名称（position actuators）
        self.position_actuator_names = [f"pos_panda_joint{i + 1}" for i in range(7)]
        
        # 使用默认初始配置
        q_init = self.compute_initial_configuration()
        
        # 当前关节位置命令（用于积分）
        self.q_cmd = q_init.copy()

        # 设置初始配置
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                self.data.qpos[dof_idx] = q_init[i]

        mujoco.mj_forward(self.model, self.data)

        # 控制参数
        self.setup_control_parameters()

        # 状态变量
        self.quat_ref = None  # 参考姿态（四元数）

        # 调试标志
        self.debug = False  # 设置为True启用调试输出
        self.last_debug_time = -2.0  # 上次debug输出的时间

        # ROS2 TF 发布相关
        self.ros2_node = None
        self.tf_broadcaster = None
        self.tf_thread = None
        self.tf_running = False

        # TF 发布数据（线程安全）
        self.tf_lock = threading.Lock()
        self.pos_curr = np.zeros(3)
        self.quat_curr = np.array([1.0, 0.0, 0.0, 0.0])
        self.pos_ref = np.zeros(3)
        self.quat_ref_tf = np.array([1.0, 0.0, 0.0, 0.0])

        # 初始化 ROS2（如果可用）
        if ROS2_AVAILABLE:
            self.init_ros2()

        print("✓ 任务空间位置控制仿真初始化完成")

    def init_ros2(self):
        """初始化 ROS2 节点和 TF 广播器"""
        try:
            if not rclpy.ok():
                rclpy.init()

            self.ros2_node = Node('task_space_velocity_sim_tf_publisher')
            self.tf_broadcaster = TransformBroadcaster(self.ros2_node)
            self.tf_running = True

            # 启动 TF 发布线程
            self.tf_thread = threading.Thread(target=self._tf_publisher_thread, daemon=True)
            self.tf_thread.start()

            print("✓ ROS2 TF 发布器已启动")
        except Exception as e:
            print(f"⚠ ROS2 初始化失败: {e}")
            self.tf_running = False

    def _tf_publisher_thread(self):
        """TF 发布线程"""
        rate = 50  # 50 Hz
        period = 1.0 / rate

        while self.tf_running and rclpy.ok():
            try:
                with self.tf_lock:
                    pos_curr = self.pos_curr.copy()
                    quat_curr = self.quat_curr.copy()
                    pos_ref = self.pos_ref.copy()
                    quat_ref = self.quat_ref_tf.copy()

                # 发布当前位姿
                t_curr = TransformStamped()
                t_curr.header.stamp = self.ros2_node.get_clock().now().to_msg()
                t_curr.header.frame_id = 'world'
                t_curr.child_frame_id = 'panda_hand_current'
                t_curr.transform.translation.x = float(pos_curr[0])
                t_curr.transform.translation.y = float(pos_curr[1])
                t_curr.transform.translation.z = float(pos_curr[2])
                t_curr.transform.rotation.w = float(quat_curr[0])
                t_curr.transform.rotation.x = float(quat_curr[1])
                t_curr.transform.rotation.y = float(quat_curr[2])
                t_curr.transform.rotation.z = float(quat_curr[3])
                self.tf_broadcaster.sendTransform(t_curr)

                # 发布参考位姿
                t_ref = TransformStamped()
                t_ref.header.stamp = self.ros2_node.get_clock().now().to_msg()
                t_ref.header.frame_id = 'world'
                t_ref.child_frame_id = 'panda_hand_reference'
                t_ref.transform.translation.x = float(pos_ref[0])
                t_ref.transform.translation.y = float(pos_ref[1])
                t_ref.transform.translation.z = float(pos_ref[2])
                t_ref.transform.rotation.w = float(quat_ref[0])
                t_ref.transform.rotation.x = float(quat_ref[1])
                t_ref.transform.rotation.y = float(quat_ref[2])
                t_ref.transform.rotation.z = float(quat_ref[3])
                self.tf_broadcaster.sendTransform(t_ref)

                # 处理 ROS2 回调
                rclpy.spin_once(self.ros2_node, timeout_sec=0.001)

                time.sleep(period)
            except Exception as e:
                if self.tf_running:
                    print(f"⚠ TF 发布错误: {e}")
                break

    def compute_initial_configuration(self):
        """
        获取初始关节配置

        返回:
            q_init: 初始关节角度 (7,)
        """
        # 使用默认配置
        q_init = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        print("✓ 使用默认初始配置")
        return q_init

    def setup_control_parameters(self):
        """设置控制参数（公式236）"""
        # 6维任务空间控制增益（角速度在前）
        # K_p: 位置和姿态比例增益
        # 速度控制通常只需要比例项
        self.K_p = np.diag([5.0, 5.0, 5.0, 3.0, 3.0, 3.0])  # [旋转, 位置]

    def compute_adjoint_transformation(self, R, p):
        """
        计算 Adjoint 变换矩阵 Ad_{T_world_to_body}（从世界坐标系到 body 坐标系）
        
        对于 SE(3) 中的变换 T_world_to_body = (R, p_body)，Adjoint 变换为：
        Ad_{T_world_to_body} = [R           0        ]
                               [p_body×R     R       ]
        
        其中：
        - R: 从世界坐标系到 body 坐标系的旋转矩阵
        - p_body: body 坐标系中的位置（从世界原点看）
        - p_body = -R @ p_world，其中 p_world 是 body 在世界坐标系中的位置
        
        参数:
            R: 旋转矩阵 (3×3)，从世界坐标系到 body 坐标系
            p: 位置向量 (3,)，body 在世界坐标系中的位置 p_world
        
        返回:
            Ad: Adjoint 变换矩阵 (6×6)（角速度在前）
        """
        # body 坐标系中的位置（从世界原点看）
        p_body = -R @ p
        
        # 计算 p_body 的反对称矩阵
        p_skew = np.array([
            [0, -p_body[2], p_body[1]],
            [p_body[2], 0, -p_body[0]],
            [-p_body[1], p_body[0], 0]
        ])
        
        # 构建 Adjoint 变换矩阵（角速度在前版本）
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = R  # 角速度到角速度
        Ad[:3, 3:] = np.zeros((3, 3))  # 角速度到线速度（为0）
        Ad[3:, :3] = p_skew @ R  # 线速度到角速度
        Ad[3:, 3:] = R  # 线速度到线速度
        
        return Ad

    def get_jacobian_6x7(self):
        """
        获取6DOF物体雅可比矩阵（body frame，物体坐标系）
        
        计算步骤：
        1. 计算空间雅可比（世界坐标系）
        2. 获取 body 的位姿（旋转矩阵和位置）
        3. 计算 Adjoint 变换
        4. 将空间雅可比转换为物体雅可比

        **Twist和Wrench的排序约定（书本第513行约定）**：
        - Twist V = [ω_x, ω_y, ω_z, v_x, v_y, v_z]^T  （角速度在前，线速度在后）
        - Wrench F = [m_x, m_y, m_z, f_x, f_y, f_z]^T （力矩在前，力在后）

        本代码使用书本第513行约定（角速度/力矩在前）。
        
        返回:
            J_b: 物体雅可比矩阵 (6×n_joints)，在 body 坐标系中表示
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        try:
            hand_body_id = self.model.body("panda_hand").id
        except:
            return np.eye(6, 7)

        # 1. 计算空间雅可比（世界坐标系）
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, hand_body_id)

        # 构建空间雅可比 J_s（世界坐标系）
        J_s = np.zeros((6, self.n_joints))
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                J_s[:3, i] = jacr[:, dof_idx]  # 角速度部分 [ω_x, ω_y, ω_z]（索引0-2）
                J_s[3:, i] = jacp[:, dof_idx]  # 线速度部分 [v_x, v_y, v_z]（索引3-5）
        
        # 2. 获取 body 的位姿（世界坐标系）
        pos_world = np.array(self.data.body("panda_hand").xpos)  # 世界坐标系中的位置
        quat_world = np.array(self.data.body("panda_hand").xquat)  # 世界坐标系中的四元数 [w, x, y, z]
        
        # 将四元数转换为旋转矩阵
        # MuJoCo 四元数格式：[w, x, y, z] -> scipy 格式：[x, y, z, w]
        # xquat 表示从 body 到世界的旋转，所以需要转置得到从世界到 body 的旋转
        q_scipy = np.array([quat_world[1], quat_world[2], quat_world[3], quat_world[0]])
        rot = Rotation.from_quat(q_scipy)
        R_body_to_world = rot.as_matrix()  # 从 body 到世界的旋转矩阵
        R_world_to_body = R_body_to_world.T  # 从世界到 body 的旋转矩阵（转置）
        
        # 3. 计算 Adjoint 变换（从世界坐标系到 body 坐标系）
        Ad = self.compute_adjoint_transformation(R_world_to_body, pos_world)
        
        # 4. 转换为物体雅可比：J_b = Ad_{T^{-1}} J_s
        J_b = Ad @ J_s
        
        return J_b

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
        计算配置误差 X_e = log(X^{-1} X_d) - 公式236

        使用 SE(3) 的 4x4 变换矩阵直接计算

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
            # 将四元数转换为 scipy 格式 [x, y, z, w]
            q_curr_scipy = np.array([quat_curr[1], quat_curr[2], quat_curr[3], quat_curr[0]])
            q_ref_scipy = np.array([quat_ref[1], quat_ref[2], quat_ref[3], quat_ref[0]])

            rot_curr = Rotation.from_quat(q_curr_scipy)
            rot_ref = Rotation.from_quat(q_ref_scipy)

            R_curr = rot_curr.as_matrix()
            R_ref = rot_ref.as_matrix()

            # 构建 SE(3) 的 4x4 变换矩阵
            # T = [R  p]
            #     [0  1]
            T_curr = np.eye(4)
            T_curr[:3, :3] = R_curr
            T_curr[:3, 3] = pos_curr

            T_ref = np.eye(4)
            T_ref[:3, :3] = R_ref
            T_ref[:3, 3] = pos_ref

            # 计算 X^{-1} X_d = T_curr^{-1} @ T_ref
            # SE(3) 的逆：T^{-1} = [R^T  -R^T p]
            #                        [0    1     ]
            T_curr_inv = np.eye(4)
            T_curr_inv[:3, :3] = R_curr.T
            T_curr_inv[:3, 3] = -R_curr.T @ pos_curr

            # 计算相对变换：T_err = T_curr^{-1} @ T_ref
            T_err = T_curr_inv @ T_ref

            # 提取旋转和平移部分
            R_err = T_err[:3, :3]
            p_err = T_err[:3, 3]

            # 计算log映射：X_e = log(T_err)
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

    def control_step(self, t, dt=0.002):
        """
        执行一步任务空间位置控制 - 公式236（6DOF扩展）

        控制律（公式236，6DOF扩展）：
        θ̇ = J^{-1}(θ)[Ẋ_d + K_p X_e]
        θ_cmd = θ_cmd + θ̇ * dt

        其中：
        - J(θ): 雅可比矩阵（6×7）
        - Ẋ_d: 参考末端执行器速度（6维twist）
        - X_e = log(X^{-1}X_d): 配置误差（6维）
        - K_p: 比例增益矩阵（6×6）
        - dt: 时间步长

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

        # 生成参考轨迹
        # 目标位置：在表面上方
        z_des = 0.4 - min(t / 2.0, 1.0) * 0.25  # 从0.4m降到0.15m
        pos_ref = np.array([0.4, 0.0, 0.5])

        # 参考速度：Z方向下降速度（6维twist，角速度在前）
        vel_ref_linear = np.array([0.0, 0.0, -0.25 / 2.0 if t < 2.0 else 0.0])
        vel_ref_angular = np.zeros(3)  # 无角速度
        vel_ref_6d = np.concatenate([vel_ref_angular, vel_ref_linear])  # [ω, v]

        # 期望姿态：Z轴朝下
        target_orientation = Rotation.from_euler("xyz", [3.14, 0, 0]).as_matrix()
        quat_ref = self.rotation_matrix_to_quaternion(target_orientation)
        self.quat_ref = quat_ref

        # ========== 更新 TF 发布数据 ==========
        if ROS2_AVAILABLE and self.tf_running:
            with self.tf_lock:
                self.pos_curr = pos_curr.copy()
                self.quat_curr = quat_curr.copy()
                self.pos_ref = pos_ref.copy()
                self.quat_ref_tf = quat_ref.copy()

        # ========== 计算配置误差 ==========
        # 计算6维配置误差：X_e = log(X^{-1} X_d)
        X_e = self.compute_configuration_error(
            pos_curr, quat_curr, pos_ref, quat_ref
        )

        # ========== 任务空间速度控制律（公式236，6DOF扩展）==========
        # 公式236：θ̇ = J^{-1}(θ)[Ẋ_d + K_p X_e]
        # 
        # 计算期望任务空间速度
        X_dot_desired = vel_ref_6d + self.K_p @ X_e

        # 使用伪逆计算关节速度（处理冗余机器人）
        # J^+ = J^T (J J^T)^{-1} 或使用 SVD 伪逆
        J_pinv = np.linalg.pinv(J, rcond=1e-6)

        # 计算期望关节速度
        qdot_cmd = J_pinv @ X_dot_desired

        # 速度限制（rad/s）
        qdot_max = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5], dtype=float)
        qdot_cmd = np.clip(qdot_cmd, -qdot_max, qdot_max)

        # ========== 位置控制：速度积分得到位置命令 ==========
        # 使用速度*dt+theta的方式：q_cmd = q_cmd + qdot_cmd * dt
        self.q_cmd = self.q_cmd + qdot_cmd * dt
        
        # 关节位置限制（根据关节范围）
        q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.q_cmd = np.clip(self.q_cmd, q_min, q_max)

        # ========== DEBUG输出 ==========
        should_debug = self.debug and (t - self.last_debug_time >= 2.0)
        if should_debug:
            self.last_debug_time = t
            print(f"[DEBUG] t={t:.3f}s | pos_curr={pos_curr} | pos_ref={pos_ref}")
            print(f"  X_e={X_e} | X_dot_desired={X_dot_desired}")
            print(f"  qdot_cmd={qdot_cmd} | q_cmd={self.q_cmd}")

        # 应用控制（位置控制）
        for i, name in enumerate(self.position_actuator_names):
            try:
                ctrl_id = self.model.actuator(name).id
                self.data.actuator(ctrl_id).ctrl = float(self.q_cmd[i])
            except Exception as e:
                if self.debug:
                    print(f"⚠ 执行器 {name} 控制失败: {e}")

    def run(self, duration=30.0):
        """
        运行仿真

        参数:
            duration: 仿真时长（秒）
        """
        print("\n" + "=" * 70)
        print("任务空间位置控制仿真 - 公式236（6DOF扩展）")
        print("=" * 70)
        print("\n控制方法:")
        print("  - 任务空间位置控制（通过速度积分）")
        print("  - 控制律：θ̇ = J^{-1}(θ)[Ẋ_d + K_p X_e]")
        print("  - 位置命令：θ_cmd = θ_cmd + θ̇ * dt")
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

        # 停止 TF 发布
        if ROS2_AVAILABLE and self.tf_running:
            self.tf_running = False
            if self.tf_thread is not None:
                self.tf_thread.join(timeout=1.0)
            if self.ros2_node is not None:
                self.ros2_node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            print("✓ ROS2 TF 发布器已停止")

        print("\nViewer已关闭")


if __name__ == "__main__":
    # 注意：要使用 ROS2 TF 发布功能，需要先 source ROS2 环境：
    # source /opt/ros/humble/setup.bash  # 或对应版本的路径

    sim = TaskSpaceVelocitySimulation(model_path="surface_force_control.xml")
    sim.debug = False  # 设置为 True 启用调试输出
    sim.run(duration=30.0)

