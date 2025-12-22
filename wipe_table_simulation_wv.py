#!/usr/bin/env python3
"""
表面力控制仿真 - 圆形擦拭任务
基于混合力位控制 (Hybrid Force/Position Control)

功能：
1. 接近阶段：机器人移动到工作表面上方。
2. 接触阶段：Z轴使用力控制保持恒定压力 (-15N)，XY平面使用位姿控制画圆。
3. 信号处理：对力传感器数据进行低通滤波。
"""

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from dynamics_calculator_wv import DynamicsCalculator


class WipeTableSimulation:
    def __init__(self, model_path: str = "surface_force_control_disk.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_names = [f"panda_joint{i + 1}" for i in range(7)]
        self.n_joints = 7
        self.joint_dof_indices = [self.model.joint(name).id for name in self.joint_names]
        self.dof_adr = [self.model.joint(idx).dofadr[0] for idx in self.joint_dof_indices]

        # 初始化动力学计算器
        self.dynamics_calc = DynamicsCalculator(model_path, "panda_hand", self.joint_names)

        # 初始关节配置
        q_init = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        for i, adr in enumerate(self.dof_adr):
            self.data.qpos[adr] = q_init[i]
        mujoco.mj_forward(self.model, self.data)

        self.setup_control_parameters()

        # 状态变量
        self.X_e_integral = np.zeros(6)
        self.F_e_integral = np.zeros(6)
        
        # 调试计数器
        self.debug_counter = 0

        # 力滤波变量（6维wrench）
        self.filtered_wrench = np.zeros(6)  # [fx, fy, fz, tx, ty, tz]
        self.force_alpha = 0.1  # 低通滤波系数 (0~1, 越小越平滑)

        # 任务状态机
        self.start_time = 0.0

        # 传感器初始化：6维力/力矩传感器
        try:
            self.sensor_id_force = self.model.sensor("force_sensor_force").id
            self.sensor_id_torque = self.model.sensor("force_sensor_torque").id
            print("✓ 6维力传感器初始化成功")
            print(f"  Force传感器ID: {self.sensor_id_force}")
            print(f"  Torque传感器ID: {self.sensor_id_torque}")
        except (AttributeError, KeyError) as e:
            print(f"⚠ 力传感器未找到: {e}")
            self.sensor_id_force = None
            self.sensor_id_torque = None

    def setup_control_parameters(self):
        """设置控制增益"""
        # 运动控制 PD 参数 [Rx, Ry, Rz, X, Y, Z]
        # 注意：Z轴增益在接触时会通过投影矩阵被屏蔽，但在非接触时需要保持悬停
        self.K_p = np.diag([80.0, 80.0, 80.0, 400.0, 400.0, 400.0])
        self.K_d = np.diag([5.0, 5.0, 5.0, 20.0, 20.0, 20.0])
        self.K_i = np.diag([0.1, 0.1, 0.1, 10.0, 10.0, 10.0])

        # 力控制 PI 参数 [Tx, Ty, Tz, Fx, Fy, Fz]
        self.K_fp = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # 仅 Z 轴有力控需求
        self.K_fi = np.diag([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

        self.F_desired_val = -15.0  # 期望压力 (N)

    def compute_spatial_error(self, p, q, p_d, q_d):
        """计算空间误差"""
        # MuJoCo quat is [w, x, y, z], Scipy needs [x, y, z, w]
        qc = np.roll(q, -1)
        qd = np.roll(q_d, -1)
        Rc = Rotation.from_quat(qc).as_matrix()
        Rd = Rotation.from_quat(qd).as_matrix()

        R_err = Rd @ Rc.T
        omega_e = Rotation.from_matrix(R_err).as_rotvec()
        pos_e = p_d - Rd @ Rc.T @ p
        return np.concatenate([omega_e, pos_e])

    def compute_projection_matrix(self, Lambda_s, A_s):
        """计算混合控制投影矩阵 P"""
        try:
            Lambda_inv = np.linalg.inv(Lambda_s)
            inner = A_s @ Lambda_inv @ A_s.T
            # 添加微小阻尼防止奇异
            inner_inv = np.linalg.inv(inner + 1e-6 * np.eye(A_s.shape[0]))
            P = np.eye(6) - A_s.T @ inner_inv @ A_s @ Lambda_inv
            return P
        except np.linalg.LinAlgError:
            return np.eye(6)

    def compute_adjoint_transformation(self, R, p):
        """
        计算伴随变换矩阵 Ad，用于wrench的坐标变换
        
        参数:
            R: 旋转矩阵 (3x3)，从源坐标系到目标坐标系的旋转
            p: 位置向量 (3,)，源坐标系原点在目标坐标系中的位置
        
        返回:
            Ad: 伴随变换矩阵 (6x6)
        
        Wrench变换公式: F_target = Ad^T @ F_source
        其中 F = [tx, ty, tz, fx, fy, fz]^T (力矩在前，力在后)
        """
        # 计算位置向量的反对称矩阵 [p]×
        p_skew = np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ])

        # 构建伴随变换矩阵
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = R  # 力矩到力矩
        Ad[3:, 3:] = R  # 力到力
        Ad[3:, :3] = p_skew @ R  # 力矩对力的影响

        return Ad

    def get_filtered_contact_wrench(self):
        """
        获取滤波后的6维接触wrench（力和力矩）
        
        返回:
            wrench: np.array([fx, fy, fz, tx, ty, tz])
            - force传感器返回3维力 [fx, fy, fz]（在site坐标系中）
            - torque传感器返回3维力矩 [tx, ty, tz]（在site坐标系中）
            - 注意：force传感器测量的是从子体(disk)指向父体(force_sensor_body)的力
            - 对于接触力，我们需要的是从环境作用到disk的力，所以需要取负号
        """
        if self.sensor_id_force is None or self.sensor_id_torque is None:
            return np.zeros(6)

        # 读取原始数据
        # force传感器：3维力 [fx, fy, fz]（在force_sensor_site坐标系中）
        force_raw = self.data.sensordata[self.sensor_id_force:self.sensor_id_force + 3]
        # torque传感器：3维力矩 [tx, ty, tz]（在force_sensor_site坐标系中）
        torque_raw = self.data.sensordata[self.sensor_id_torque * 3:self.sensor_id_torque * 3 + 3]

        # 组合成6维wrench
        wrench_raw = np.concatenate([torque_raw, force_raw])  # [ tx, ty, tz,fx, fy, fz,]

        # 注意：force传感器测量的是从子体(disk)指向父体(force_sensor_body)的力
        # 对于接触力控制，我们需要的是从环境作用到disk的力（即disk受到的力）
        # 根据牛顿第三定律，disk受到的力 = -force传感器测量的力
        # 所以这里取负号
        wrench = -wrench_raw

        return wrench

    def generate_wipe_trajectory(self, t):
        """生成擦拭轨迹"""
        # 阶段 1: 移动到中心上方 (0-2秒)
        # 阶段 2: 保持XY，降低Z试图接触 (2-4秒)
        # 阶段 3: XY画圆，Z由力控接管 (4秒+)

        center = np.array([0.55, 0.0])
        radius = 0.01
        freq = 1.0  # rad/s

        # 目标姿态：末端垂直向下
        quat_d = Rotation.from_euler('xyz', [3.14159, 0, 0]).as_quat()
        quat_d = np.array([quat_d[3], quat_d[0], quat_d[1], quat_d[2]])  # [w,x,y,z]

        if t < 2.0:
            pos_d = np.array([0.55, 0.0, 0.3])
            mode = "APPROACH"
        elif t < 4.0:
            # 缓慢下降到表面高度附近
            alpha = (t - 2.0) / 2.0
            z_target = 0.3 * (1 - alpha) + 0.14 * alpha  # 0.14 比表面0.15略低，确保能接触
            pos_d = np.array([0.5, 0.0, z_target])
            mode = "DESCEND"
        else:
            # 画圆
            time_circle = t - 4.0
            x = center[0] + radius * np.cos(freq * time_circle)
            y = center[1] + radius * np.sin(freq * time_circle)
            pos_d = np.array([x, y, 0.14])  # Z设定值此时不太重要，因为会被力控覆盖
            mode = "WIPE"

        return pos_d, quat_d, mode

    def control_step(self, t, dt):
        # 1. 状态获取
        q = np.array([self.data.qpos[adr] for adr in self.dof_adr])
        qdot = np.array([self.data.qvel[adr] for adr in self.dof_adr])
        pos = np.array(self.data.body("panda_hand").xpos)
        quat = np.array(self.data.body("panda_hand").xquat)

        # 2. 动力学计算
        J_s = self.dynamics_calc.compute_spatial_jacobian(q, 6)
        V_s = J_s @ qdot
        Lambda_s = self.dynamics_calc.compute_task_space_mass_matrix(q, 6)
        eta_s = self.dynamics_calc.compute_task_space_coriolis(q, V_s, 6)

        # 3. 接触力检测与模式判断
        # 获取6维wrench
        wrench_curr = self.get_filtered_contact_wrench()
        f_z_curr = wrench_curr[5]  # Z轴方向的力
        is_contact = abs(f_z_curr) > 0.5  # 接触阈值 0.5N

        # 4. 生成轨迹
        pos_d, quat_d, mode = self.generate_wipe_trajectory(t)

        # 5. 定义约束矩阵 A_s
        # 如果接触且处于 Descend 或 Wipe 阶段，开启力控
        enable_force_control = is_contact and (mode in ["DESCEND", "WIPE"])

        if enable_force_control:
            # 约束方向：[wx, wy, wz, fz] (索引0,1,2,5)
            # 在接触时，旋转方向(wx,wy,wz)和Z轴方向(fz)由力控接管
            # 此时 P_s 会将这些方向上的运动控制移除
            A_s = np.zeros((4, 6))
            A_s[0, 0] = 1.0  # wx
            A_s[1, 1] = 1.0  # wy
            A_s[2, 2] = 1.0  # wz
            A_s[3, 5] = 1.0  # fz
            P_s = self.compute_projection_matrix(Lambda_s, A_s)
        else:

            P_s = np.eye(6)

        # 6. 计算运动误差
        X_e = self.compute_spatial_error(pos, quat, pos_d, quat_d)
        
        # 调试：每循环5次打印空行
        self.debug_counter += 1
        if self.debug_counter % 30 == 0:
            print()
        
        V_e = -V_s

        # 积分限幅 (Anti-windup)
        self.X_e_integral += X_e * dt
        self.X_e_integral = np.clip(self.X_e_integral, -0.5, 0.5)

        # 7. 运动控制力 (Projected Motion Control)
        # F_motion = P * Lambda * (Kp*Xe + Kd*Ve)
        F_motion = P_s @ Lambda_s @ (self.K_p @ X_e + self.K_d @ V_e + self.K_i @ self.X_e_integral)

        # 8. 力控制力 (Force Control)
        F_force = np.zeros(6)
        if enable_force_control:
            # 获取ft_sensor_site和panda_link0的世界坐标位姿
            site_pos_world = np.array(self.data.site("ft_sensor_site").xpos)
            site_mat_world = np.array(self.data.site("ft_sensor_site").xmat).reshape(3, 3)

            # 构建SE(3)变换矩阵
            T_world_to_site = np.eye(4)
            T_world_to_site[:3, :3] = site_mat_world
            T_world_to_site[:3, 3] = site_pos_world

            # 计算从site到base的SE(3)变换矩阵：T_site_to_base = T_world_to_base^(-1) @ T_world_to_site
            T_world_to_site_inv = np.linalg.inv(T_world_to_site)
            T_site_to_world = T_world_to_site_inv

            # 从SE(3)变换矩阵提取旋转矩阵和位置向量
            R_site_to_world = T_site_to_world[:3, :3]
            p_site_in_world = T_site_to_world[:3, 3]

            # 计算伴随变换矩阵（从site到base）
            Ad_site_to_world = self.compute_adjoint_transformation(R_site_to_world, p_site_in_world)

            # 期望wrench：[wx, wy, wz, fx, fy, fz]（在base坐标系中）
            # 旋转方向(wx,wy,wz)期望力矩为0（保持当前姿态）
            # Z轴方向(fz)期望力为-15N（向下压力）
            F_d_base = np.zeros(6)
            F_d_base[0] = 0.0  # 期望力矩 wx = 0
            F_d_base[1] = 0.0  # 期望力矩 wy = 0
            F_d_base[2] = 0.0  # 期望力矩 wz = 0
            F_d_base[5] = self.F_desired_val  # 期望力 fz = -15N（Z轴方向）

            # 当前测量的wrench（从传感器获取，在site坐标系中，转换到base坐标系）
            F_curr_base = Ad_site_to_world.T @ wrench_curr

            # 力误差（在base坐标系中）
            F_e_base = F_d_base - F_curr_base
            self.F_e_integral += F_e_base * dt
            # 力积分限幅
            self.F_e_integral = np.clip(self.F_e_integral, -50.0, 50.0)

            # 计算力控指令（在base坐标系中）
            F_cmd_base = F_d_base + self.K_fp @ F_e_base + self.K_fi @ self.F_e_integral

            # (I - P) * F_cmd_base
            # 在约束方向上(wx,wy,wz,fz)，(I-P)会将力控指令施加到这些方向
            F_force = (np.eye(6) - P_s) @ F_cmd_base

        # 9. 合成力矩
        F_total = F_motion + F_force + eta_s
        tau = J_s.T @ F_total

        # 添加简单的零空间阻尼以稳定冗余自由度
        tau_null = -2.0 * qdot
        tau += (np.eye(7) - J_s.T @ np.linalg.pinv(J_s.T)) @ tau_null

        # 10. 发送控制
        tau = np.clip(tau, -87, 87)
        self.data.ctrl[:] = tau

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                step_start = self.data.time

                # 控制频率通常低于仿真频率，但这里每一帧都计算
                dt = self.model.opt.timestep
                self.control_step(self.data.time, dt)

                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    print("启动擦拭仿真...")
    sim = WipeTableSimulation(model_path="surface_force_control_disk.xml")
    sim.run()
