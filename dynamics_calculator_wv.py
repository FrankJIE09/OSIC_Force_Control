#!/usr/bin/env python3
"""
机器人动力学量计算器 - 空间坐标系版本 (Spatial Frame)
基于 MuJoCo 实现动力学量计算：
1. M(θ) - 关节空间质量矩阵 (不变)
2. h(θ, θ̇) - 关节空间 Coriolis/重力 (不变)
3. Λ_s(θ) - 空间任务空间质量矩阵 (Spatial Task Space Mass Matrix)
4. η_s(θ, V_s) - 空间任务空间科里奥利项 (Spatial Task Space Coriolis)

**重要约定**：
- 所有的任务空间量（雅可比、速度、力）均在**世界坐标系 (Spatial Frame)** 下表示。
- Twist V_s = [ω_x, ω_y, ω_z, v_x, v_y, v_z]^T （角速度在前，线速度在后）
- Wrench F_s = [m_x, m_y, m_z, f_x, f_y, f_z]^T （力矩在前，力在后）
"""

import mujoco
import mujoco.viewer
import numpy as np
from typing import Tuple, Optional
import time


class DynamicsCalculator:
    """机器人动力学量计算器 (Spatial/World Frame)"""

    def __init__(self, model_path: str = "surface_force_control.xml",
                 body_name: str = "panda_hand",
                 joint_names: Optional[list] = None):
        """
        初始化动力学计算器
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.body_name = body_name

        if joint_names is None:
            self.joint_names = []
            for i in range(self.model.njnt):
                jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if jnt_name:
                    self.joint_names.append(jnt_name)
        else:
            self.joint_names = joint_names

        self.n_joints = len(self.joint_names)
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

        try:
            self.body_id = self.model.body(body_name).id
        except:
            raise ValueError(f"找不到体 '{body_name}'")

        print(f"✓ 动力学计算器(Spatial)初始化完成")

    def set_configuration(self, q: np.ndarray, qdot: Optional[np.ndarray] = None):
        """设置机器人配置"""
        if len(q) != self.n_joints:
            raise ValueError(f"关节角度向量长度 {len(q)} 与关节数量 {self.n_joints} 不匹配")

        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                self.data.qpos[dof_idx] = q[i]

        if qdot is not None:
            for i, name in enumerate(self.joint_names):
                dof_idx = self.joint_dof_indices[i]
                if dof_idx >= 0:
                    self.data.qvel[dof_idx] = qdot[i]
        else:
            for i, name in enumerate(self.joint_names):
                dof_idx = self.joint_dof_indices[i]
                if dof_idx >= 0:
                    self.data.qvel[dof_idx] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """计算关节空间质量矩阵 M(θ)"""
        self.set_configuration(q, qdot=np.zeros(self.n_joints))
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_full, self.data.qM)

        M = np.zeros((self.n_joints, self.n_joints))
        for i in range(self.n_joints):
            dof_i = self.joint_dof_indices[i]
            if dof_i >= 0:
                for j in range(self.n_joints):
                    dof_j = self.joint_dof_indices[j]
                    if dof_j >= 0:
                        M[i, j] = M_full[dof_i, dof_j]
        return M

    def compute_coriolis_gravity(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """计算关节空间 Coriolis 和重力 h(θ, θ̇)"""
        self.set_configuration(q, qdot)
        h = np.zeros(self.n_joints)
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                h[i] = self.data.qfrc_bias[dof_idx]
        return h

    def compute_gravity_term(self, q: np.ndarray) -> np.ndarray:
        """计算关节空间重力项 g(θ)"""
        self.set_configuration(q, qdot=np.zeros(self.n_joints))
        g = np.zeros(self.n_joints)
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                g[i] = self.data.qfrc_bias[dof_idx]
        return g

    def compute_spatial_jacobian(self, q: np.ndarray, task_space_dim: int = 6) -> np.ndarray:
        """
        计算空间雅可比矩阵 J_s(θ) (Spatial Jacobian)

        MuJoCo 的 mj_jacBody 默认返回的就是空间雅可比（世界坐标系下的轴，作用于Body原点）。

        返回:
            J_s: 空间雅可比矩阵 (6×n) [角速度; 线速度]
        """
        self.set_configuration(q)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.body_id)

        if task_space_dim == 3:
            J = np.zeros((3, self.n_joints))
            for i in range(self.n_joints):
                dof_idx = self.joint_dof_indices[i]
                if dof_idx >= 0:
                    J[:, i] = jacp[:, dof_idx]
        elif task_space_dim == 6:
            J = np.zeros((6, self.n_joints))
            for i in range(self.n_joints):
                dof_idx = self.joint_dof_indices[i]
                if dof_idx >= 0:
                    # 空间雅可比：角速度在前，线速度在后
                    J[:3, i] = jacr[:, dof_idx]
                    J[3:, i] = jacp[:, dof_idx]
        else:
            raise ValueError("不支持的任务空间维度")
        return J

    def compute_jacobian_derivative(self, q: np.ndarray, qdot: np.ndarray,
                                    task_space_dim: int = 6,
                                    epsilon: float = 1e-6) -> np.ndarray:
        """计算空间雅可比矩阵的时间导数 d(J_s)/dt"""
        J_dot = np.zeros((task_space_dim, self.n_joints))
        for i in range(self.n_joints):
            q_plus = q.copy()
            q_plus[i] += epsilon
            J_plus = self.compute_spatial_jacobian(q_plus, task_space_dim)

            q_minus = q.copy()
            q_minus[i] -= epsilon
            J_minus = self.compute_spatial_jacobian(q_minus, task_space_dim)

            dJ_dqi = (J_plus - J_minus) / (2 * epsilon)
            J_dot += dJ_dqi * qdot[i]
        return J_dot

    def compute_task_space_mass_matrix(self, q: np.ndarray,
                                       task_space_dim: int = 6,
                                       use_pseudoinverse: bool = True,
                                       damping: float = 1e-6) -> np.ndarray:
        """
        计算空间任务空间质量矩阵 Λ_s(θ)
        Λ_s = (J_s M^{-1} J_s^T)^{-1}
        """
        M = self.compute_mass_matrix(q)
        J_s = self.compute_spatial_jacobian(q, task_space_dim)

        # 使用阻尼伪逆或直接求逆计算 Λ_s
        if use_pseudoinverse or self.n_joints > task_space_dim:
            # 计算惯性加权的伪逆相关项
            # Λ_s = (J_s * M^{-1} * J_s^T)^{-1}
            # 为了数值稳定性，通常计算 M^{-1} 然后组合
            try:
                M_inv = np.linalg.inv(M)
                Js_Minv_JsT = J_s @ M_inv @ J_s.T
                # 添加阻尼以防止奇异
                Js_Minv_JsT += damping * np.eye(task_space_dim)
                Lambda_s = np.linalg.inv(Js_Minv_JsT)
            except:
                # 备用方案：基于 Jacobian 伪逆
                J_pinv = J_s.T @ np.linalg.inv(J_s @ J_s.T + damping * np.eye(task_space_dim))
                Lambda_s = J_pinv.T @ M @ J_pinv
        else:
            M_inv = np.linalg.inv(M)
            Lambda_s = np.linalg.inv(J_s @ M_inv @ J_s.T)

        return Lambda_s

    def compute_task_space_coriolis(self, q: np.ndarray, V_s: np.ndarray,
                                    task_space_dim: int = 6,
                                    use_pseudoinverse: bool = True,
                                    damping: float = 1e-6) -> np.ndarray:
        """
        计算空间任务空间科里奥利项 η_s(θ, V_s)

        参数:
            q: 关节角度
            V_s: 当前的空间速度 (Spatial Velocity) [ω; v]
        """
        J_s = self.compute_spatial_jacobian(q, task_space_dim)

        # 计算关节速度 θ̇ = J_s^# V_s
        if use_pseudoinverse:
            J_inv = J_s.T @ np.linalg.inv(J_s @ J_s.T + damping * np.eye(task_space_dim))
        else:
            J_inv = np.linalg.pinv(J_s)

        qdot = J_inv @ V_s

        h = self.compute_coriolis_gravity(q, qdot)
        J_dot = self.compute_jacobian_derivative(q, qdot, task_space_dim)
        Lambda_s = self.compute_task_space_mass_matrix(q, task_space_dim, use_pseudoinverse, damping)

        # η_s = J_s^{-T} h - Λ_s J_dot J_s^{-1} V_s
        term1 = J_inv.T @ h
        term2 = Lambda_s @ J_dot @ J_inv @ V_s
        eta_s = term1 - term2

        return eta_s