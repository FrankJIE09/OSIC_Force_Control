#!/usr/bin/env python3
"""
机器人动力学量计算器
基于 MuJoCo 实现 LaTeX 文档中描述的动力学量计算：
1. M(θ) - 关节空间质量矩阵
2. h(θ, θ̇) - 科里奥利力、向心力和重力项
3. Λ(θ) - 任务空间质量矩阵
4. η(θ, V) - 任务空间科里奥利项

参考：chapter8dynamics_calculation_methods.tex
"""

import mujoco
import mujoco.viewer
import numpy as np
from typing import Tuple, Optional
import time


class DynamicsCalculator:
    """机器人动力学量计算器"""
    
    def __init__(self, model_path: str = "surface_force_control.xml", 
                 body_name: str = "panda_hand",
                 joint_names: Optional[list] = None):
        """
        初始化动力学计算器
        
        参数:
            model_path: MuJoCo 模型文件路径
            body_name: 任务空间参考体名称（用于计算雅可比矩阵）
            joint_names: 关节名称列表，如果为None则自动检测
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.body_name = body_name
        
        # 获取关节信息
        if joint_names is None:
            # 自动检测所有关节
            self.joint_names = []
            for i in range(self.model.njnt):
                jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if jnt_name:
                    self.joint_names.append(jnt_name)
        else:
            self.joint_names = joint_names
        
        self.n_joints = len(self.joint_names)
        
        # 获取关节的DOF索引
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
        
        # 获取任务空间参考体的ID
        try:
            self.body_id = self.model.body(body_name).id
        except:
            raise ValueError(f"找不到体 '{body_name}'")
        
        print(f"✓ 动力学计算器初始化完成")
        print(f"  - 关节数量: {self.n_joints}")
        print(f"  - 任务空间参考体: {body_name}")
    
    def set_configuration(self, q: np.ndarray, qdot: Optional[np.ndarray] = None):
        """
        设置机器人配置
        
        参数:
            q: 关节角度向量 (n_joints,)
            qdot: 关节角速度向量 (n_joints,)，可选
        """
        if len(q) != self.n_joints:
            raise ValueError(f"关节角度向量长度 {len(q)} 与关节数量 {self.n_joints} 不匹配")
        
        # 设置关节角度
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                self.data.qpos[dof_idx] = q[i]
        
        # 设置关节角速度（如果提供）
        if qdot is not None:
            if len(qdot) != self.n_joints:
                raise ValueError(f"关节角速度向量长度 {len(qdot)} 与关节数量 {self.n_joints} 不匹配")
            for i, name in enumerate(self.joint_names):
                dof_idx = self.joint_dof_indices[i]
                if dof_idx >= 0:
                    self.data.qvel[dof_idx] = qdot[i]
        else:
            # 设置为零
            for i, name in enumerate(self.joint_names):
                dof_idx = self.joint_dof_indices[i]
                if dof_idx >= 0:
                    self.data.qvel[dof_idx] = 0.0
        
        # 前向动力学计算
        mujoco.mj_forward(self.model, self.data)
    
    def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        计算关节空间质量矩阵 M(θ)
        
        方法：使用 MuJoCo 的 mj_fullM 函数（基于递归牛顿-欧拉方法）
        
        参数:
            q: 关节角度向量 (n_joints,)
        
        返回:
            M: 质量矩阵 (n_joints, n_joints)
        """
        # 设置配置（速度设为0）
        self.set_configuration(q, qdot=np.zeros(self.n_joints))
        
        # 计算完整质量矩阵（包括所有DOF）
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_full, self.data.qM)
        
        # 提取关节部分的质量矩阵
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
        """
        计算科里奥利力、向心力和重力项 h(θ, θ̇)
        
        方法：使用 MuJoCo 的 qfrc_bias（包含科里奥利、向心力和重力）
        
        参数:
            q: 关节角度向量 (n_joints,)
            qdot: 关节角速度向量 (n_joints,)
        
        返回:
            h: 科里奥利、向心力和重力项 (n_joints,)
        """
        # 设置配置
        self.set_configuration(q, qdot)
        
        # qfrc_bias 已经包含了科里奥利、向心力和重力项
        h = np.zeros(self.n_joints)
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                h[i] = self.data.qfrc_bias[dof_idx]
        
        return h
    
    def compute_gravity_term(self, q: np.ndarray) -> np.ndarray:
        """
        计算重力项 g(θ)
        
        方法：设置所有速度为0，运行逆动力学
        
        参数:
            q: 关节角度向量 (n_joints,)
        
        返回:
            g: 重力项 (n_joints,)
        """
        # 设置配置，速度设为0
        self.set_configuration(q, qdot=np.zeros(self.n_joints))
        
        # qfrc_bias 在零速度时只包含重力项
        g = np.zeros(self.n_joints)
        for i, name in enumerate(self.joint_names):
            dof_idx = self.joint_dof_indices[i]
            if dof_idx >= 0:
                g[i] = self.data.qfrc_bias[dof_idx]
        
        return g
    
    def compute_coriolis_term(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """
        计算科里奥利和向心力项 C(θ, θ̇)θ̇
        
        方法：h(θ, θ̇) - g(θ)
        
        参数:
            q: 关节角度向量 (n_joints,)
            qdot: 关节角速度向量 (n_joints,)
        
        返回:
            C_qdot: 科里奥利和向心力项 (n_joints,)
        """
        h = self.compute_coriolis_gravity(q, qdot)
        g = self.compute_gravity_term(q)
        return h - g
    
    def compute_jacobian(self, q: np.ndarray, task_space_dim: int = 6) -> np.ndarray:
        """
        计算雅可比矩阵 J(θ)
        
        参数:
            q: 关节角度向量 (n_joints,)
            task_space_dim: 任务空间维度（3=位置，6=位置+姿态）
        
        返回:
            J: 雅可比矩阵 (task_space_dim, n_joints)
        """
        # 设置配置
        self.set_configuration(q)
        
        # 计算雅可比矩阵
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.body_id)
        
        # 提取关节部分
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
                    J[:3, i] = jacp[:, dof_idx]
                    J[3:, i] = jacr[:, dof_idx]
        else:
            raise ValueError(f"不支持的任务空间维度 {task_space_dim}，应为3或6")
        
        return J
    
    def compute_jacobian_derivative(self, q: np.ndarray, qdot: np.ndarray, 
                                   task_space_dim: int = 6, 
                                   epsilon: float = 1e-6) -> np.ndarray:
        """
        计算雅可比矩阵的时间导数 Ḃ(θ)
        
        方法：数值微分
        Ḃ = Σ (∂J/∂θᵢ) θ̇ᵢ
        
        参数:
            q: 关节角度向量 (n_joints,)
            qdot: 关节角速度向量 (n_joints,)
            task_space_dim: 任务空间维度
            epsilon: 数值微分的步长
        
        返回:
            J_dot: 雅可比矩阵时间导数 (task_space_dim, n_joints)
        """
        J_dot = np.zeros((task_space_dim, self.n_joints))
        
        for i in range(self.n_joints):
            # 计算 ∂J/∂θᵢ
            q_plus = q.copy()
            q_plus[i] += epsilon
            J_plus = self.compute_jacobian(q_plus, task_space_dim)
            
            q_minus = q.copy()
            q_minus[i] -= epsilon
            J_minus = self.compute_jacobian(q_minus, task_space_dim)
            
            dJ_dqi = (J_plus - J_minus) / (2 * epsilon)
            
            # Ḃ += (∂J/∂θᵢ) θ̇ᵢ
            J_dot += dJ_dqi * qdot[i]
        
        return J_dot
    
    def compute_task_space_mass_matrix(self, q: np.ndarray, 
                                       task_space_dim: int = 6,
                                       use_pseudoinverse: bool = False,
                                       damping: float = 1e-6) -> np.ndarray:
        """
        计算任务空间质量矩阵 Λ(θ)
        
        公式：Λ(θ) = J⁻ᵀ(θ) M(θ) J⁻¹(θ)
        或：Λ(θ) = (J M⁻¹ Jᵀ)⁻¹（数值稳定）
        
        参数:
            q: 关节角度向量 (n_joints,)
            task_space_dim: 任务空间维度
            use_pseudoinverse: 是否使用伪逆（对于冗余机器人）
            damping: 阻尼系数（用于阻尼伪逆）
        
        返回:
            Lambda: 任务空间质量矩阵 (task_space_dim, task_space_dim)
        """
        # 计算关节空间质量矩阵
        M = self.compute_mass_matrix(q)
        
        # 计算雅可比矩阵
        J = self.compute_jacobian(q, task_space_dim)
        
        # 方法1：使用 (J M⁻¹ Jᵀ)⁻¹（数值稳定）
        try:
            M_inv = np.linalg.inv(M)
            Lambda = np.linalg.inv(J @ M_inv @ J.T)
        except np.linalg.LinAlgError:
            # 如果求逆失败，使用伪逆
            if use_pseudoinverse or self.n_joints > task_space_dim:
                # 冗余机器人或奇异情况
                if damping > 0:
                    # 阻尼伪逆
                    J_damped = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(task_space_dim))
                    Lambda = J_damped.T @ M @ J_damped
                else:
                    # 标准伪逆
                    J_pinv = np.linalg.pinv(J)
                    Lambda = J_pinv.T @ M @ J_pinv
            else:
                raise ValueError("无法计算任务空间质量矩阵：雅可比矩阵不可逆且未启用伪逆")
        
        return Lambda
    
    def compute_task_space_coriolis(self, q: np.ndarray, V: np.ndarray,
                                   task_space_dim: int = 6,
                                   use_pseudoinverse: bool = False,
                                   damping: float = 1e-6) -> np.ndarray:
        """
        计算任务空间科里奥利项 η(θ, V)
        
        公式：η(θ, V) = J⁻ᵀ(θ) h(θ, J⁻¹V) - Λ(θ) Ḃ(θ) J⁻¹(θ) V
        
        参数:
            q: 关节角度向量 (n_joints,)
            V: 任务空间速度 (task_space_dim,)
            task_space_dim: 任务空间维度
            use_pseudoinverse: 是否使用伪逆
            damping: 阻尼系数
        
        返回:
            eta: 任务空间科里奥利项 (task_space_dim,)
        """
        # 计算雅可比矩阵
        J = self.compute_jacobian(q, task_space_dim)
        
        # 计算关节速度：θ̇ = J⁻¹ V
        if use_pseudoinverse or self.n_joints > task_space_dim:
            if damping > 0:
                J_inv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(task_space_dim))
            else:
                J_inv = np.linalg.pinv(J)
        else:
            J_inv = np.linalg.inv(J)
        
        qdot = J_inv @ V
        
        # 计算 h(θ, θ̇)
        h = self.compute_coriolis_gravity(q, qdot)
        
        # 计算 Ḃ(θ)
        J_dot = self.compute_jacobian_derivative(q, qdot, task_space_dim)
        
        # 计算 Λ(θ)
        Lambda = self.compute_task_space_mass_matrix(q, task_space_dim, 
                                                    use_pseudoinverse, damping)
        
        # 计算第一项：J⁻ᵀ h(θ, θ̇)
        term1 = J_inv.T @ h
        
        # 计算第二项：Λ(θ) Ḃ(θ) J⁻¹(θ) V
        term2 = Lambda @ J_dot @ J_inv @ V
        
        # 组合
        eta = term1 - term2
        
        return eta
    
    def compute_all(self, q: np.ndarray, qdot: Optional[np.ndarray] = None,
                   V: Optional[np.ndarray] = None,
                   task_space_dim: int = 6) -> dict:
        """
        一次性计算所有动力学量
        
        参数:
            q: 关节角度向量
            qdot: 关节角速度向量（可选）
            V: 任务空间速度（可选，用于计算η）
            task_space_dim: 任务空间维度
        
        返回:
            dict: 包含所有动力学量的字典
        """
        if qdot is None:
            qdot = np.zeros(self.n_joints)
        
        results = {}
        
        # 1. 质量矩阵
        results['M'] = self.compute_mass_matrix(q)
        
        # 2. 科里奥利、向心力和重力项
        results['h'] = self.compute_coriolis_gravity(q, qdot)
        results['g'] = self.compute_gravity_term(q)
        results['C_qdot'] = self.compute_coriolis_term(q, qdot)
        
        # 3. 雅可比矩阵
        results['J'] = self.compute_jacobian(q, task_space_dim)
        
        # 4. 任务空间质量矩阵
        results['Lambda'] = self.compute_task_space_mass_matrix(q, task_space_dim)
        
        # 5. 雅可比矩阵时间导数（如果提供了速度）
        if np.any(qdot != 0):
            results['J_dot'] = self.compute_jacobian_derivative(q, qdot, task_space_dim)
        
        # 6. 任务空间科里奥利项（如果提供了任务空间速度）
        if V is not None:
            results['eta'] = self.compute_task_space_coriolis(q, V, task_space_dim)
        
        return results


def main():
    """示例用法"""
    print("=" * 70)
    print("机器人动力学量计算器 - 示例")
    print("=" * 70)
    
    # 创建计算器
    calc = DynamicsCalculator(
        model_path="surface_force_control.xml",
        body_name="panda_hand",
        joint_names=[f"panda_joint{i+1}" for i in range(7)]
    )
    
    # 设置初始配置
    q = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    qdot = np.array([0.1, -0.1, 0.05, -0.05, 0.02, 0.02, 0.01])
    
    print(f"\n配置:")
    print(f"  q = {q}")
    print(f"  qdot = {qdot}")
    
    # 计算所有动力学量
    print("\n计算动力学量...")
    results = calc.compute_all(q, qdot, task_space_dim=3)
    
    # 显示结果
    print("\n" + "=" * 70)
    print("计算结果:")
    print("=" * 70)
    
    print(f"\n1. 关节空间质量矩阵 M(θ) [{results['M'].shape[0]}×{results['M'].shape[1]}]")
    print(f"   特征值范围: [{np.min(np.linalg.eigvals(results['M'])):.4f}, "
          f"{np.max(np.linalg.eigvals(results['M'])):.4f}]")
    print(f"   条件数: {np.linalg.cond(results['M']):.4e}")
    
    print(f"\n2. 科里奥利、向心力和重力项 h(θ, θ̇) [{len(results['h'])}]")
    print(f"   h = {results['h']}")
    
    print(f"\n3. 重力项 g(θ) [{len(results['g'])}]")
    print(f"   g = {results['g']}")
    
    print(f"\n4. 科里奥利和向心力项 C(θ, θ̇)θ̇ [{len(results['C_qdot'])}]")
    print(f"   C_qdot = {results['C_qdot']}")
    
    print(f"\n5. 雅可比矩阵 J(θ) [{results['J'].shape[0]}×{results['J'].shape[1]}]")
    print(f"   条件数: {np.linalg.cond(results['J']):.4e}")
    
    print(f"\n6. 任务空间质量矩阵 Λ(θ) [{results['Lambda'].shape[0]}×{results['Lambda'].shape[1]}]")
    print(f"   特征值范围: [{np.min(np.linalg.eigvals(results['Lambda'])):.4f}, "
          f"{np.max(np.linalg.eigvals(results['Lambda'])):.4f}]")
    
    if 'J_dot' in results:
        print(f"\n7. 雅可比矩阵时间导数 Ḃ(θ) [{results['J_dot'].shape[0]}×{results['J_dot'].shape[1]}]")
        print(f"   ||Ḃ||_F = {np.linalg.norm(results['J_dot'], 'fro'):.6f}")
    
    # 验证动力学方程
    print("\n" + "=" * 70)
    print("验证动力学方程:")
    print("=" * 70)
    print("τ = M(θ)θ̈ + h(θ, θ̇)")
    print("\n对于给定的配置和零加速度:")
    qddot = np.zeros(7)
    tau = results['M'] @ qddot + results['h']
    print(f"τ = {tau}")
    print(f"  (这应该等于重力项 g(θ) = {results['g']})")
    print(f"  误差: {np.linalg.norm(tau - results['g']):.6e}")
    
    print("\n✓ 计算完成！")


if __name__ == "__main__":
    main()

