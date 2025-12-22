#!/usr/bin/env python3
"""
任务空间控制仿真 - 空间坐标系版本 (Spatial Frame Control)
使用 MuJoCo Franka Panda 模型实现空间任务空间控制

控制方法：
- 空间任务空间控制 (Spatial Task Space Control)
- 所有计算（误差、雅可比、动力学）均在世界坐标系 (World Frame) 下进行

控制律 (Spatial Frame)：
F_s = Λ_s(θ) d/dt(V_d) + K_p X_e + K_d V_e + η_s(θ, V_s)
τ = J_s^T(θ) F_s

其中：
- V_s: 当前空间速度 (Spatial Velocity)
- X_e = log(X_d X^{-1}): 空间配置误差 (Spatial Twist)
- V_e = V_d - V_s: 空间速度误差
- Λ_s, η_s: 空间坐标系下的动力学矩阵

约定：
- Twist/Wrench: [角速度/力矩 (0-2), 线速度/力 (3-5)]
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation

from dynamics_calculator_wv import DynamicsCalculator

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from tf2_ros import TransformBroadcaster
    from geometry_msgs.msg import TransformStamped
except BaseException:
    pass

ROS2_AVAILABLE = False


class TaskSpaceSimulation:
    """空间任务空间控制仿真类"""

    def __init__(self, model_path: str = "surface_force_control.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = [f"panda_joint{i + 1}" for i in range(7)]
        self.n_joints = 7
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

        q_init = self.compute_initial_configuration()
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                self.data.qpos[dof_idx] = q_init[i]

        mujoco.mj_forward(self.model, self.data)

        self.dynamics_calc = DynamicsCalculator(
            model_path=model_path,
            body_name="panda_hand",
            joint_names=self.joint_names
        )
        self.dynamics_calc.set_configuration(q_init)

        self.setup_control_parameters()

        # 状态变量
        self.X_e_integral = np.zeros(6)

        # 前馈计算用的变量
        self.prev_V_ref = np.zeros(6)
        self.prev_t = 0.0

        # Debug
        self.debug = False
        self.last_debug_time = -2.0

        # ROS2
        self.ros2_node = None
        self.tf_broadcaster = None
        self.tf_thread = None
        self.tf_running = False
        self.tf_lock = threading.Lock()
        self.pos_curr = np.zeros(3)
        self.quat_curr = np.array([1.0, 0.0, 0.0, 0.0])
        self.pos_ref = np.zeros(3)
        self.quat_ref_tf = np.array([1.0, 0.0, 0.0, 0.0])

        if ROS2_AVAILABLE:
            self.init_ros2()

        print("✓ 空间任务空间控制仿真初始化完成")

    def init_ros2(self):
        try:
            if not rclpy.ok():
                rclpy.init()
            self.ros2_node = Node('spatial_control_sim_tf')
            self.tf_broadcaster = TransformBroadcaster(self.ros2_node)
            self.tf_running = True
            self.tf_thread = threading.Thread(target=self._tf_publisher_thread, daemon=True)
            self.tf_thread.start()
            print("✓ ROS2 TF 发布器已启动")
        except Exception as e:
            print(f"⚠ ROS2 初始化失败: {e}")
            self.tf_running = False

    def _tf_publisher_thread(self):
        rate = 50
        period = 1.0 / rate
        while self.tf_running and rclpy.ok():
            try:
                with self.tf_lock:
                    pos_curr = self.pos_curr.copy()
                    quat_curr = self.quat_curr.copy()
                    pos_ref = self.pos_ref.copy()
                    quat_ref = self.quat_ref_tf.copy()

                t_curr = TransformStamped()
                t_curr.header.stamp = self.ros2_node.get_clock().now().to_msg()
                t_curr.header.frame_id = 'world'
                t_curr.child_frame_id = 'panda_hand_spatial'
                t_curr.transform.translation.x = float(pos_curr[0])
                t_curr.transform.translation.y = float(pos_curr[1])
                t_curr.transform.translation.z = float(pos_curr[2])
                t_curr.transform.rotation.w = float(quat_curr[0])
                t_curr.transform.rotation.x = float(quat_curr[1])
                t_curr.transform.rotation.y = float(quat_curr[2])
                t_curr.transform.rotation.z = float(quat_curr[3])
                self.tf_broadcaster.sendTransform(t_curr)

                t_ref = TransformStamped()
                t_ref.header.stamp = self.ros2_node.get_clock().now().to_msg()
                t_ref.header.frame_id = 'world'
                t_ref.child_frame_id = 'ref_target_spatial'
                t_ref.transform.translation.x = float(pos_ref[0])
                t_ref.transform.translation.y = float(pos_ref[1])
                t_ref.transform.translation.z = float(pos_ref[2])
                t_ref.transform.rotation.w = float(quat_ref[0])
                t_ref.transform.rotation.x = float(quat_ref[1])
                t_ref.transform.rotation.y = float(quat_ref[2])
                t_ref.transform.rotation.z = float(quat_ref[3])
                self.tf_broadcaster.sendTransform(t_ref)

                rclpy.spin_once(self.ros2_node, timeout_sec=0.001)
                time.sleep(period)
            except Exception as e:
                break

    def compute_initial_configuration(self):
        return np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

    def setup_control_parameters(self):
        # 空间控制增益
        # K_p [旋转, 位置] - 空间刚度
        self.K_p = np.diag([100.0, 100.0, 100.0, 2000.0, 2000.0, 2000.0])
        # K_i
        self.K_i = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
        # K_d - 空间阻尼
        self.K_d = np.diag([10.0, 10.0, 10.0, 500.0, 500.0, 500.0])

    def get_spatial_jacobian(self):
        """
        获取空间雅可比 J_s (World Frame)
        MuJoCo mj_jacBody 返回的即为 J_s
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        hand_body_id = self.model.body("panda_hand").id

        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, hand_body_id)

        J_s = np.zeros((6, self.n_joints))
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                J_s[:3, i] = jacr[:, dof_idx]
                J_s[3:, i] = jacp[:, dof_idx]
        return J_s

    def rotation_matrix_to_quaternion(self, R):
        try:
            rot = Rotation.from_matrix(R)
            quat_scipy = rot.as_quat()  # [x, y, z, w]
            # 转换为 MuJoCo [w, x, y, z]
            return np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        except:
            return np.array([1.0, 0.0, 0.0, 0.0])

    def se3_log_map(self, R, p):
        """计算 SE(3) Log 映射，返回 Twist"""
        try:
            rot = Rotation.from_matrix(R)
            omega = rot.as_rotvec()
            theta = np.linalg.norm(omega)

            if theta < 1e-6:
                v = p
            else:
                omega_normalized = omega / theta

                omega_skew = np.array([
                    [0, -omega[2], omega[1]],
                    [omega[2], 0, -omega[0]],
                    [-omega[1], omega[0], 0]
                ])
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                V_inv = (np.eye(3) -
                         (1 - cos_theta) / (theta ** 2) * omega_skew +
                         (theta - sin_theta) / (theta ** 3) * omega_skew @ omega_skew)
                v = V_inv @ p
                # G_inv (Inverse of exp map linear part)
                term1 = np.eye(3) - 0.5 * omega_skew
                coef = (1 - (theta * np.cos(theta / 2) / (2 * np.sin(theta / 2)))) / (theta ** 2) if theta > 1e-4 else 0
                # 标准公式 V_inv = I - 0.5*w_x + ...
                # 这里使用简化或标准库计算更好，手动实现：
                # v = p  # 简化，近似认为 p 和 v 在小误差下相近，或者直接用 p
                # 更精确的求解 v 需要 G^{-1} p
                # 为保持代码简洁，在 theta 不大时 v approx p

            return np.concatenate([omega, p])  # 注意：这里近似 p 作为线速度部分
        except:
            return np.zeros(6)

    def compute_spatial_configuration_error(self, pos_curr, quat_curr, pos_ref, quat_ref):
        """
        计算空间配置误差 X_e = log(X_d X^{-1})
        这是在世界坐标系下表示的从当前位姿到期望位姿的 Twist。
        """
        # MuJoCo quat [w,x,y,z] -> Scipy [x,y,z,w]
        qc = np.roll(quat_curr, -1)
        qr = np.roll(quat_ref, -1)

        Rc = Rotation.from_quat(qc).as_matrix()
        Rr = Rotation.from_quat(qr).as_matrix()

        # 构建变换矩阵
        Tc = np.eye(4);
        Tc[:3, :3] = Rc;
        Tc[:3, 3] = pos_curr
        Tr = np.eye(4);
        Tr[:3, :3] = Rr;
        Tr[:3, 3] = pos_ref

        # 计算 X_d * X^{-1} (空间误差变换)
        Tc_inv = np.eye(4)
        Tc_inv[:3, :3] = Rc.T
        Tc_inv[:3, 3] = -Rc.T @ pos_curr

        T_err = Tr @ Tc_inv  # 注意顺序：X_d * X^{-1}

        # Log 映射
        R_err = T_err[:3, :3]
        p_err = T_err[:3, 3]
        # 使用旋转向量计算角误差
        rot_vec = Rotation.from_matrix(R_err).as_rotvec()

        # 空间误差向量 [omega, v]
        # 对于空间控制，线性误差通常直接取 p_ref - p_curr (当姿态误差小时)
        # 或者严格使用 Log(T_err) 的线性部分
        X_e = np.concatenate([rot_vec, pos_ref - pos_curr])
        X_e= self.se3_log_map(R_err, p_err)

        return X_e

    def control_step(self, t, dt=0.002):
        mujoco.mj_forward(self.model, self.data)

        # 1. 获取当前状态 (World Frame)
        pos_curr = np.array(self.data.body("panda_hand").xpos)
        quat_curr = np.array(self.data.body("panda_hand").xquat)

        q = np.zeros(self.n_joints)
        qdot = np.zeros(self.n_joints)
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                q[i] = self.data.qpos[dof_idx]
                qdot[i] = self.data.qvel[dof_idx]

        # 2. 计算空间雅可比 J_s 和 空间速度 V_s
        # J_s: [6 x 7], V_s: [6] (角速度在前)
        J_s = self.get_spatial_jacobian()
        V_s = J_s @ qdot  # V_s = J_s * q_dot

        # 3. 生成参考轨迹 (Spatial Frame)
        # 目标：画圈或下压
        z_des = 0.4 - min(t / 2.0, 1.0) * 0.25
        pos_ref = np.array([0.4, 0.0, 0.4])  # 保持固定或移动

        # 参考速度 (空间坐标系)
        # 简单起见，设定一个向下的速度
        vel_ref_linear = np.array([0.0, 0.0, -0.1 if t < 2.0 else 0.0])
        vel_ref_angular = np.array([0.0, 0.0, 0.0])
        vel_ref = np.concatenate([vel_ref_angular, vel_ref_linear])

        # 参考加速度 (数值微分)
        if t > 0:
            acc_ref = (vel_ref - self.prev_V_ref) / dt
        else:
            acc_ref = np.zeros(6)
        self.prev_V_ref = vel_ref.copy()

        # 参考姿态
        target_R = Rotation.from_euler("xyz", [3.14, 0, 0]).as_matrix()
        quat_ref = self.rotation_matrix_to_quaternion(target_R)
        self.quat_ref = quat_ref

        # 4. 计算动力学项 (Spatial Frame)
        # 注意：此处传入的是 V_s (空间速度)
        Lambda_s = self.dynamics_calc.compute_task_space_mass_matrix(
            q, task_space_dim=6, use_pseudoinverse=True, damping=1e-5
        )
        eta_s = self.dynamics_calc.compute_task_space_coriolis(
            q, V_s, task_space_dim=6, use_pseudoinverse=True, damping=1e-5
        )

        # 5. 计算误差 (Spatial Frame)
        # 空间配置误差 X_e = log(X_d X^{-1})
        X_e = self.compute_spatial_configuration_error(pos_curr, quat_curr, pos_ref, quat_ref)

        # 空间速度误差 V_e = V_d - V_s
        V_e = vel_ref - V_s

        # 积分项
        self.X_e_integral += X_e * dt
        self.X_e_integral = np.clip(self.X_e_integral, -0.1, 0.1)

        # 6. 计算空间控制力 F_s
        # F_s = Λ_s * acc_ref + η_s + K_p * X_e + K_d * V_e
        # 前馈项：Λ_s * acc_ref (期望空间加速度产生的惯性力)
        F_feedforward = Lambda_s @ acc_ref

        # 反馈项
        F_feedback = (self.K_p @ X_e + self.K_d @ V_e)  # + self.K_i @ self.X_e_integral)

        # 总空间力
        F_cmd = F_feedforward + eta_s + F_feedback

        # 7. 映射到关节力矩
        # τ = J_s^T * F_s
        tau = J_s.T @ F_cmd

        # 限制力矩
        tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
        tau = np.clip(tau, -tau_max, tau_max)

        # 应用控制
        for i, name in enumerate(self.joint_names):
            try:
                ctrl_id = self.model.actuator(name).id
                self.data.actuator(ctrl_id).ctrl = float(tau[i])
            except:
                pass

        # TF 发布
        if ROS2_AVAILABLE and self.tf_running:
            with self.tf_lock:
                self.pos_curr = pos_curr
                self.quat_curr = quat_curr
                self.pos_ref = pos_ref
                self.quat_ref_tf = quat_ref

        # Debug
        if self.debug and (t - self.last_debug_time >= 1.0):
            self.last_debug_time = t
            print(f"t={t:.2f} | ErrP={np.linalg.norm(X_e[3:]):.3f} | F_z={F_cmd[5]:.1f}")

    def run(self, duration=30.0):
        print("\n" + "=" * 60)
        print("空间任务空间控制 (Spatial Frame Task Space Control)")
        print("所有计算 (J, M, Λ, η) 均在空间坐标系下进行")
        print("=" * 60)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            dt = self.model.opt.timestep
            t = 0.0
            while viewer.is_running() and t < duration:
                self.control_step(t, dt)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                t += dt

        if ROS2_AVAILABLE and self.tf_running:
            self.tf_running = False
            if self.tf_thread: self.tf_thread.join()
            if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    sim = TaskSpaceSimulation(model_path="surface_force_control.xml")
    sim.run(duration=130.0)
