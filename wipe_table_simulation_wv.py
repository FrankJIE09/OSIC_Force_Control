#!/usr/bin/env python3
"""
表面力控制仿真 - 基于混合力位控制 (空间坐标系严格版)
基于公式 11.61 和 11.63 实现

控制律 (公式 11.61):
τ = J_s^T [ P_s (Λ_s d/dt(V_d) + K_p X_e + K_i∫X_e + K_d V_e)
          + (I - P_s)(F_d + K_fp F_e + K_fi∫F_e) + η_s ]

投影矩阵 (公式 11.63):
P_s = I - A_s^T (A_s Λ_s^-1 A_s^T)^-1 A_s Λ_s^-1
"""

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from dynamics_calculator_wv import DynamicsCalculator


class WipeTableSimulation:
    def __init__(self, model_path: str = "surface_force_control.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_names = [f"panda_joint{i + 1}" for i in range(7)]
        self.n_joints = 7
        self.joint_dof_indices = [self.model.joint(name).id for name in self.joint_names]
        self.dof_adr = [self.model.joint(idx).dofadr[0] for idx in self.joint_dof_indices]

        # 初始化动力学计算器 (使用空间坐标系接口)
        self.dynamics_calc = DynamicsCalculator(model_path, "panda_hand", self.joint_names)

        # 初始配置
        q_init = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        for i, adr in enumerate(self.dof_adr):
            self.data.qpos[adr] = q_init[i]
        mujoco.mj_forward(self.model, self.data)

        self.setup_control_parameters()

        # 积分器状态
        self.X_e_integral = np.zeros(6)
        self.F_e_integral = np.zeros(6)
        self.contact_t = None
        self.debug = True
        self.last_debug_time = 0.0

    def setup_control_parameters(self):
        """设置空间坐标系下的增益"""
        # 运动控制：[旋转, 平移]
        self.K_p = np.diag([80.0, 80.0, 80.0, 200.0, 200.0, 200.0])
        self.K_d = np.diag([10.0, 10.0, 10.0, 20.0, 20.0, 20.0])
        self.K_i = np.diag([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])

        # 力控制：[力矩, 力]
        self.K_fp = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 1.2])
        self.K_fi = np.diag([0.01, 0.01, 0.01, 0.05, 0.05, 0.2])
        self.F_desired_val = -15.0  # Z轴法向压力

    def compute_spatial_error(self, p, q, p_d, q_d):
        """计算空间配置误差 X_e = log(X_d * X^-1)"""
        # MuJoCo [w,x,y,z] -> Scipy [x,y,z,w]
        qc = np.roll(q, -1)
        qd = np.roll(q_d, -1)
        Rc = Rotation.from_quat(qc).as_matrix()
        Rd = Rotation.from_quat(qd).as_matrix()

        # 空间旋转误差
        R_err = Rd @ Rc.T
        omega_e = Rotation.from_matrix(R_err).as_rotvec()

        # 空间位置误差
        pos_e = p_d - p
        return np.concatenate([omega_e, pos_e])

    def compute_projection_matrix(self, Lambda_s, A_s):
        """实现公式 11.63: P = I - A^T (A Λ^-1 A^T)^-1 A Λ^-1"""
        try:
            Lambda_inv = np.linalg.inv(Lambda_s)
            # 中间项: (A * Λ^-1 * A^T)
            inner = A_s @ Lambda_inv @ A_s.T
            inner_inv = np.linalg.inv(inner + 1e-6 * np.eye(A_s.shape[0]))

            # P = I - A^T * inner_inv * A * Lambda_inv
            P = np.eye(6) - A_s.T @ inner_inv @ A_s @ Lambda_inv
            return P
        except np.linalg.LinAlgError:
            return np.eye(6)

    def get_contact_force_spatial(self):
        """在世界系下提取法向力（从圆盘与表面的接触）"""
        force_z = 0.0
        try:
            surf_id = self.model.geom("surface").id
            # 获取圆盘几何体ID
            disk_geom_id = self.model.geom("disk_geom").id
            disk_bottom_id = self.model.geom("disk_bottom").id
        except:
            # 如果没有surface或disk，返回0
            return 0.0
        
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            # 检查是否是圆盘与表面的接触
            if (c.geom1 == surf_id and (c.geom2 == disk_geom_id or c.geom2 == disk_bottom_id)) or \
               (c.geom2 == surf_id and (c.geom1 == disk_geom_id or c.geom1 == disk_bottom_id)):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                force_z += f6[0]  # MuJoCo 接触坐标系下 f[0] 为法向力
        return force_z

    def control_step(self, t, dt):
        # 1. 获取当前关节状态
        q = np.array([self.data.qpos[adr] for adr in self.dof_adr])
        qdot = np.array([self.data.qvel[adr] for adr in self.dof_adr])

        # 2. 获取空间几何状态
        pos = np.array(self.data.body("panda_hand").xpos)
        quat = np.array(self.data.body("panda_hand").xquat)

        # 3. 计算空间动力学量
        J_s = self.dynamics_calc.compute_spatial_jacobian(q, 6)
        V_s = J_s @ qdot
        Lambda_s = self.dynamics_calc.compute_task_space_mass_matrix(q, 6)
        eta_s = self.dynamics_calc.compute_task_space_coriolis(q, V_s, 6)

        # 4. 接触检测与约束矩阵 A_s 定义
        f_z = self.get_contact_force_spatial()
        is_contact = abs(f_z) > 0.1

        # 约束方向：在世界系下约束 ωx, ωy 和 vz
        # A_s 矩阵每一行代表一个约束向量
        A_s = np.zeros((3, 6))
        if is_contact:
            A_s[0, 0] = 1.0  # 约束绕世界 X 旋转
            A_s[1, 1] = 1.0  # 约束绕世界 Y 旋转
            A_s[2, 5] = 1.0  # 约束世界 Z 平移
            P_s = self.compute_projection_matrix(Lambda_s, A_s)
        else:
            P_s = np.eye(6)

        # 5. 参考轨迹定义 (空间坐标系)
        pos_d = np.array([0.5, 0.0, 0.15 if is_contact else 0.4 - 0.1 * t])
        quat_d = Rotation.from_euler('xyz', [3.14, 0, 0]).as_quat()
        quat_d = np.array([quat_d[3], quat_d[0], quat_d[1], quat_d[2]])  # [w,x,y,z]

        # 6. 计算误差项
        X_e = self.compute_spatial_error(pos, quat, pos_d, quat_d)
        V_e = -V_s  # 假设期望速度为0
        self.X_e_integral += X_e * dt

        # 7. 运动控制分支 (公式 11.61 第一项)
        # F_motion = P_s * (Λ_s * V_dot_d + Kp*Xe + Ki*∫Xe + Kd*Ve)
        F_motion = P_s @ Lambda_s @ (self.K_p @ X_e + self.K_d @ V_e + self.K_i @ self.X_e_integral)

        # 8. 力控制分支 (公式 11.61 第二项)
        F_force = np.zeros(6)
        if is_contact:
            F_d = np.zeros(6)
            F_d[5] = self.F_desired_val
            F_curr = np.zeros(6)
            F_curr[5] = -f_z  # 符号修正，对应 F_d 的压力方向

            F_e = F_d - F_curr
            self.F_e_integral += F_e * dt

            # (I - P_s) * (Fd + K_fp*Fe + K_fi*∫Fe)
            F_force_cmd = F_d + self.K_fp @ F_e + self.K_fi @ self.F_e_integral
            F_force = (np.eye(6) - P_s) @ F_force_cmd

        # 9. 总指令 Wrench 与 关节映射
        # F_total = F_motion + F_force + eta_s
        F_total = F_motion + F_force + eta_s
        tau = J_s.T @ F_total

        # 10. 安全限制与执行
        tau = np.clip(tau, -80, 80)
        for i in range(self.n_joints):
            self.data.ctrl[i] = tau[i]

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                dt = self.model.opt.timestep
                self.control_step(self.data.time, dt)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    sim = WipeTableSimulation(model_path="panda_with_disk.xml")
    sim.run()
