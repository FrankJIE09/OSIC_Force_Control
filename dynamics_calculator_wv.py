#!/usr/bin/env python3
"""
机器人动力学量计算器（角速度在前版本）
基于 MuJoCo 实现 LaTeX 文档中描述的动力学量计算：
1. M(θ) - 关节空间质量矩阵
2. h(θ, θ̇) - 科里奥利力、向心力和重力项
3. Λ(θ) - 任务空间质量矩阵
4. η(θ, V) - 任务空间科里奥利项

**重要约定（书本第513行约定）**：
- Twist V = [ω_x, ω_y, ω_z, v_x, v_y, v_z]^T  （角速度在前，线速度在后）
- Wrench F = [m_x, m_y, m_z, f_x, f_y, f_z]^T （力矩在前，力在后）
- 所有6维向量：索引0-2为角速度/力矩，索引3-5为线速度/力
- 雅可比矩阵：J[:3, :]为角速度部分，J[3:, :]为线速度部分

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

        动力学方程：
            τ = M(θ)θ̈ + h(θ, θ̇)

        拉格朗日方法（理论）：
            M(θ) = Σᵢ [mᵢ Jᵥ,ᵢᵀ(θ) Jᵥ,ᵢ(θ) + Jω,ᵢᵀ(θ) Iᵢ Jω,ᵢ(θ)]

        其中：
            - mᵢ: 连杆i的质量
            - Jᵥ,ᵢ: 连杆i质心的线速度雅可比矩阵
            - Jω,ᵢ: 连杆i的角速度雅可比矩阵
            - Iᵢ: 连杆i在质心处的惯性张量

        递归牛顿-欧拉方法（实现）：
            通过设置 θ̈ⱼ = 1, θ̈ₖ = 0 (k≠j), θ̇ = 0, g = 0
            运行递归算法得到 Mᵢⱼ(θ) = τᵢ

        参数:
            q: 关节角度向量 (n_joints,)

        返回:
            M: 质量矩阵 (n_joints, n_joints)，对称正定
        """
        # 设置配置（速度设为0）
        # 对于质量矩阵计算，需要 θ̇ = 0, θ̈ = 0, g = 0
        self.set_configuration(q, qdot=np.zeros(self.n_joints))

        # 计算完整质量矩阵（包括所有DOF）
        # 使用 MuJoCo 的 mj_fullM 函数（基于递归牛顿-欧拉方法）
        # M_full ∈ ℝ^(nv×nv)，包含所有自由度
        M_full = np.zeros((self.model.nv, self.model.nv))
        # mj_fullM: 计算完整质量矩阵（包括所有自由度）
        # 参数:
        #   - model: MuJoCo 模型
        #   - M_full: 输出矩阵 (nv×nv)，将被填充为完整质量矩阵
        #   - qM: 输入，从 data.qM 获取（MuJoCo 内部预计算的质量矩阵数据）
        # 功能:
        #   - 基于递归牛顿-欧拉方法计算质量矩阵
        #   - 将 data.qM 中的稀疏质量矩阵数据转换为密集矩阵 M_full
        #   - M_full[i,j] 表示第 i 个自由度对第 j 个自由度的惯性耦合
        # 注意:
        #   - 质量矩阵是对称正定矩阵：M_full = M_full^T
        #   - 包含所有自由度（关节、自由体等），不仅仅是关节
        mujoco.mj_fullM(self.model, M_full, self.data.qM)

        # 提取关节部分的质量矩阵
        # M(θ) ∈ ℝ^(n×n)，只包含关节自由度
        #
        # 目的：从完整质量矩阵 M_full（包含所有自由度）中提取关节空间质量矩阵 M
        #
        # 原因：
        #   - M_full 是 nv×nv 矩阵，包含所有自由度（关节、自由体、约束等）
        #   - M 是 n_joints×n_joints 矩阵，只包含关节自由度
        #   - 我们需要的是关节空间质量矩阵 M(θ)，用于动力学计算
        #
        # 方法：
        #   - 通过 joint_dof_indices 映射，找到每个关节对应的自由度索引
        #   - 从 M_full 中提取对应的元素：M[i,j] = M_full[dof_i, dof_j]
        #   - 其中 dof_i 是第 i 个关节的自由度索引，dof_j 是第 j 个关节的自由度索引
        #
        # 示例：
        #   - 如果关节1对应自由度3，关节2对应自由度5
        #   - 则 M[0,1] = M_full[3,5]（关节1和关节2之间的惯性耦合）
        M = np.zeros((self.n_joints, self.n_joints))
        for i in range(self.n_joints):
            dof_i = self.joint_dof_indices[i]  # 第 i 个关节对应的自由度索引
            if dof_i >= 0:  # 确保自由度索引有效
                for j in range(self.n_joints):
                    dof_j = self.joint_dof_indices[j]  # 第 j 个关节对应的自由度索引
                    if dof_j >= 0:  # 确保自由度索引有效
                        # 从完整质量矩阵中提取关节 i 和关节 j 之间的惯性耦合
                        # M[i,j] 表示：当关节 j 有单位加速度时，关节 i 需要的力矩
                        M[i, j] = M_full[dof_i, dof_j]

        return M

    def compute_coriolis_gravity(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """
        计算科里奥利力、向心力和重力项 h(θ, θ̇)

        方法：使用 MuJoCo 的 qfrc_bias（包含科里奥利、向心力和重力）

        公式：
            h(θ, θ̇) = C(θ, θ̇)θ̇ + g(θ) + b(θ̇)

        其中：
            - C(θ, θ̇)θ̇: 科里奥利力和向心力项
            - g(θ): 重力项
            - b(θ̇): 摩擦项（可选）

        从拉格朗日方程：
            C(θ, θ̇)θ̇ = Ṁ(θ)θ̇ - (1/2) ∂/∂θ [θ̇ᵀ M(θ)θ̇]
            g(θ) = ∂U/∂θ  (U为势能)

        递归牛顿-欧拉方法（实现）：
            使用给定的 θ 和 θ̇ 运行递归算法（包含重力）
            得到的 τ 就是 h(θ, θ̇)

        参数:
            q: 关节角度向量 (n_joints,)
            qdot: 关节角速度向量 (n_joints,)

        返回:
            h: 科里奥利、向心力和重力项 (n_joints,)
        """
        # 设置配置
        # 使用给定的 θ 和 θ̇ 运行递归牛顿-欧拉算法
        self.set_configuration(q, qdot)

        # qfrc_bias 已经包含了科里奥利、向心力和重力项
        # h(θ, θ̇) = C(θ, θ̇)θ̇ + g(θ)
        # 在 MuJoCo 中，qfrc_bias 就是 h(θ, θ̇)
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

        公式：
            g(θ) = ∂U/∂θ

        其中 U 是系统的势能。

        递归牛顿-欧拉方法（实现）：
            设置 θ̇ = 0, θ̈ = 0
            运行递归算法（包含重力 g）
            得到的 τ 就是 g(θ)

        注意：在零速度时，qfrc_bias 只包含重力项

        参数:
            q: 关节角度向量 (n_joints,)

        返回:
            g: 重力项 (n_joints,)
        """
        # 设置配置，速度设为0
        # 对于重力项计算：θ̇ = 0, θ̈ = 0，但包含重力 g
        self.set_configuration(q, qdot=np.zeros(self.n_joints))

        # qfrc_bias 在零速度时只包含重力项
        # g(θ) = ∂U/∂θ，其中 U 是势能
        # 当 θ̇ = 0 时，C(θ, θ̇)θ̇ = 0，所以 h(θ, 0) = g(θ)
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

        公式：
            C(θ, θ̇)θ̇ = h(θ, θ̇) - g(θ)

        从拉格朗日方程：
            C(θ, θ̇)θ̇ = Ṁ(θ)θ̇ - (1/2) ∂/∂θ [θ̇ᵀ M(θ)θ̇]

        递归牛顿-欧拉方法（实现）：
            1. 计算 h(θ, θ̇)（使用给定速度，包含重力）
            2. 计算 g(θ)（零速度，只有重力）
            3. C(θ, θ̇)θ̇ = h(θ, θ̇) - g(θ)

        参数:
            q: 关节角度向量 (n_joints,)
            qdot: 关节角速度向量 (n_joints,)

        返回:
            C_qdot: 科里奥利和向心力项 (n_joints,)
        """
        # 计算科里奥利和向心力项
        # C(θ, θ̇)θ̇ = h(θ, θ̇) - g(θ)
        h = self.compute_coriolis_gravity(q, qdot)
        g = self.compute_gravity_term(q)
        return h - g

    def compute_jacobian(self, q: np.ndarray, task_space_dim: int = 6) -> np.ndarray:
        """
        计算雅可比矩阵 J(θ)

        公式：
            V = J(θ)θ̇

        其中：
            - V ∈ ℝ⁶: 任务空间速度（twist）
              V = [ωₓ, ωᵧ, ω_z, vₓ, vᵧ, v_z]ᵀ
              **注意：使用书本第513行约定，角速度在前，线速度在后**
            - J(θ) ∈ ℝ⁶ˣⁿ: 雅可比矩阵
            - θ̇ ∈ ℝⁿ: 关节角速度

        雅可比矩阵结构：
            J(θ) = [Jω(θ)]
                  [Jᵥ(θ)]

        其中：
            - Jω(θ): 角速度雅可比矩阵 (3×n)
            - Jᵥ(θ): 线速度雅可比矩阵 (3×n)

        计算方法：
            使用 MuJoCo 的 mj_jacBody 函数计算
            Jᵥ 和 Jω，然后重新排列为角速度在前

        参数:
            q: 关节角度向量 (n_joints,)
            task_space_dim: 任务空间维度（3=位置，6=位置+姿态）

        返回:
            J: 雅可比矩阵 (task_space_dim, n_joints)
        """
        # 设置配置
        self.set_configuration(q)

        # 计算雅可比矩阵
        # 使用 MuJoCo 的 mj_jacBody 函数
        # jacp: 线速度雅可比 Jᵥ ∈ ℝ^(3×nv)
        # jacr: 角速度雅可比 Jω ∈ ℝ^(3×nv)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.body_id)

        # 提取关节部分
        # 雅可比矩阵：V = J(θ)θ̇
        # 其中 V = [ωₓ, ωᵧ, ω_z, vₓ, vᵧ, v_z]ᵀ
        # **书本第513行约定：角速度在前，线速度在后**
        if task_space_dim == 3:
            # 仅位置雅可比：J = Jᵥ（但注意：在3D情况下，我们仍然使用线速度部分）
            J = np.zeros((3, self.n_joints))
            for i in range(self.n_joints):
                dof_idx = self.joint_dof_indices[i]
                if dof_idx >= 0:
                    J[:, i] = jacp[:, dof_idx]
        elif task_space_dim == 6:
            # 完整雅可比：J = [Jω; Jᵥ]（角速度在前，线速度在后）
            J = np.zeros((6, self.n_joints))
            for i in range(self.n_joints):
                dof_idx = self.joint_dof_indices[i]
                if dof_idx >= 0:
                    J[:3, i] = jacr[:, dof_idx]  # 角速度部分（索引0-2）
                    J[3:, i] = jacp[:, dof_idx]  # 线速度部分（索引3-5）
        else:
            raise ValueError(f"不支持的任务空间维度 {task_space_dim}，应为3或6")

        return J

    def compute_jacobian_derivative(self, q: np.ndarray, qdot: np.ndarray,
                                   task_space_dim: int = 6,
                                   epsilon: float = 1e-6) -> np.ndarray:
        """
        计算雅可比矩阵的时间导数 Ḃ(θ)

        公式：
            Ḃ(θ) = d/dt J(θ) = Σᵢ (∂J/∂θᵢ) θ̇ᵢ

        链式法则：
            Ḃ(θ) = Σᵢ₌₁ⁿ (∂J/∂θᵢ) · (dθᵢ/dt)
                  = Σᵢ₌₁ⁿ (∂J/∂θᵢ) θ̇ᵢ

        数值微分方法（实现）：
            ∂J/∂θᵢ ≈ [J(θ + εeᵢ) - J(θ - εeᵢ)] / (2ε)

        其中：
            - ε: 数值微分步长（默认 1e-6）
            - eᵢ: 第i个单位向量

        然后：
            Ḃ(θ) = Σᵢ (∂J/∂θᵢ) θ̇ᵢ

        注意：解析方法更精确，但需要符号计算

        参数:
            q: 关节角度向量 (n_joints,)
            qdot: 关节角速度向量 (n_joints,)
            task_space_dim: 任务空间维度
            epsilon: 数值微分的步长

        返回:
            J_dot: 雅可比矩阵时间导数 (task_space_dim, n_joints)
        """
        # 初始化雅可比时间导数
        # Ḃ(θ) = Σᵢ (∂J/∂θᵢ) θ̇ᵢ
        J_dot = np.zeros((task_space_dim, self.n_joints))

        for i in range(self.n_joints):
            # 计算 ∂J/∂θᵢ 使用中心差分
            # ∂J/∂θᵢ ≈ [J(θ + εeᵢ) - J(θ - εeᵢ)] / (2ε)
            q_plus = q.copy()
            q_plus[i] += epsilon
            J_plus = self.compute_jacobian(q_plus, task_space_dim)

            q_minus = q.copy()
            q_minus[i] -= epsilon
            J_minus = self.compute_jacobian(q_minus, task_space_dim)

            dJ_dqi = (J_plus - J_minus) / (2 * epsilon)

            # 累加：Ḃ += (∂J/∂θᵢ) θ̇ᵢ
            # 链式法则：dJ/dt = Σᵢ (∂J/∂θᵢ) · (dθᵢ/dt)
            J_dot += dJ_dqi * qdot[i]

        return J_dot

    def compute_task_space_mass_matrix(self, q: np.ndarray,
                                       task_space_dim: int = 6,
                                       use_pseudoinverse: bool = False,
                                       damping: float = 1e-6) -> np.ndarray:
        """
        计算任务空间质量矩阵 Λ(θ)

        公式（标准形式）：
            Λ(θ) = J⁻ᵀ(θ) M(θ) J⁻¹(θ)

        公式（数值稳定形式）：
            Λ(θ) = (J M⁻¹ Jᵀ)⁻¹

        验证：
            (J M⁻¹ Jᵀ)⁻¹ = (Jᵀ)⁻¹ M (J⁻¹)
                         = J⁻ᵀ M J⁻¹ = Λ(θ)

        对于冗余机器人（n > task_space_dim）：
            使用伪逆：Λ(θ) = (J†)ᵀ M(θ) J†
            其中 J† = (JᵀJ)⁻¹Jᵀ 是伪逆

        阻尼伪逆（处理奇异点）：
            J† = Jᵀ (J Jᵀ + λI)⁻¹
            其中 λ 是小的正数（阻尼系数）

        物理意义：
            Λ(θ) 描述了任务空间加速度与任务空间力的关系：
            F = Λ(θ) V̇ + η(θ, V)

        参数:
            q: 关节角度向量 (n_joints,)
            task_space_dim: 任务空间维度
            use_pseudoinverse: 是否使用伪逆（对于冗余机器人）
            damping: 阻尼系数（用于阻尼伪逆）

        返回:
            Lambda: 任务空间质量矩阵 (task_space_dim, task_space_dim)，对称正定
        """
        # 计算关节空间质量矩阵 M(θ)
        M = self.compute_mass_matrix(q)

        # 计算雅可比矩阵 J(θ)
        J = self.compute_jacobian(q, task_space_dim)

        # 方法1：使用 (J M⁻¹ Jᵀ)⁻¹（数值稳定形式）
        # 公式：Λ(θ) = (J M⁻¹ Jᵀ)⁻¹
        # 等价于：Λ(θ) = J⁻ᵀ M J⁻¹，但数值上更稳定
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

        公式（公式8.91）：
            η(θ, V) = J⁻ᵀ(θ) h(θ, J⁻¹V) - Λ(θ) Ḃ(θ) J⁻¹(θ) V

        推导过程：
            从关节空间动力学：τ = M(θ)θ̈ + h(θ, θ̇)
            任务空间动力学：F = Λ(θ)V̇ + η(θ, V)

            通过坐标变换得到：
            η(θ, V) = J⁻ᵀ(θ) h(θ, J⁻¹V) - Λ(θ) Ḃ(θ) J⁻¹(θ) V

        计算步骤：
            1. 计算关节速度：θ̇ = J⁻¹(θ) V
            2. 计算 h(θ, θ̇) = h(θ, J⁻¹V)
            3. 计算 Ḃ(θ) = d/dt J(θ)
            4. 计算 Λ(θ)
            5. 第一项：J⁻ᵀ(θ) h(θ, θ̇)
            6. 第二项：Λ(θ) Ḃ(θ) J⁻¹(θ) V
            7. 组合：η = 第一项 - 第二项

        物理意义：
            η(θ, V) 包含任务空间的科里奥利力、向心力和重力项

        参数:
            q: 关节角度向量 (n_joints,)
            V: 任务空间速度 (task_space_dim,)
            task_space_dim: 任务空间维度
            use_pseudoinverse: 是否使用伪逆
            damping: 阻尼系数

        返回:
            eta: 任务空间科里奥利项 (task_space_dim,)
        """
        # 计算雅可比矩阵 J(θ)
        J = self.compute_jacobian(q, task_space_dim)

        # 计算关节速度：θ̇ = J⁻¹(θ) V
        # 从任务空间速度 V 反算关节速度
        if use_pseudoinverse or self.n_joints > task_space_dim:
            # 冗余机器人或接近奇异：使用伪逆
            if damping > 0:
                # 阻尼伪逆：J† = Jᵀ (J Jᵀ + λI)⁻¹
                J_inv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(task_space_dim))
            else:
                # 标准伪逆：J† = (JᵀJ)⁻¹Jᵀ
                J_inv = np.linalg.pinv(J)
        else:
            # 非冗余：直接求逆
            J_inv = np.linalg.inv(J)

        qdot = J_inv @ V  # θ̇ = J⁻¹ V

        # 计算 h(θ, θ̇) = h(θ, J⁻¹V)
        # 科里奥利、向心力和重力项
        h = self.compute_coriolis_gravity(q, qdot)

        # 计算 Ḃ(θ) = d/dt J(θ)
        # 雅可比矩阵的时间导数
        J_dot = self.compute_jacobian_derivative(q, qdot, task_space_dim)

        # 计算 Λ(θ) = J⁻ᵀ M J⁻¹ 或 (J M⁻¹ Jᵀ)⁻¹
        # 任务空间质量矩阵
        Lambda = self.compute_task_space_mass_matrix(q, task_space_dim,
                                                    use_pseudoinverse, damping)

        # 计算第一项：J⁻ᵀ(θ) h(θ, θ̇)
        # 将关节空间的科里奥利项转换到任务空间
        term1 = J_inv.T @ h

        # 计算第二项：Λ(θ) Ḃ(θ) J⁻¹(θ) V
        # 由于雅可比变化引起的额外项
        term2 = Lambda @ J_dot @ J_inv @ V

        # 组合：η(θ, V) = J⁻ᵀ h - Λ Ḃ J⁻¹ V
        # 公式8.91：η(θ, V) = J⁻ᵀ(θ) h(θ, J⁻¹V) - Λ(θ) Ḃ(θ) J⁻¹(θ) V
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

    def run_with_viewer(self, q_init: Optional[np.ndarray] = None,
                       update_rate: float = 10.0,
                       task_space_dim: int = 3,
                       show_detailed: bool = True):
        """
        使用 MuJoCo Viewer 实时显示机器人并计算动力学量

        参数:
            q_init: 初始关节角度，如果为None则使用模型默认值
            update_rate: 更新频率（Hz），控制计算和显示的频率
            task_space_dim: 任务空间维度
            show_detailed: 是否显示详细的计算结果
        """
        # 设置初始配置
        if q_init is not None:
            self.set_configuration(q_init)
        else:
            # 使用默认配置
            q_init = np.zeros(self.n_joints)
            for i, name in enumerate(self.joint_names):
                dof_idx = self.joint_dof_indices[i]
                if dof_idx >= 0:
                    q_init[i] = self.data.qpos[dof_idx]
            self.set_configuration(q_init)

        print("\n" + "=" * 70)
        print("机器人动力学量计算器 - 实时可视化")
        print("=" * 70)
        print("\n使用说明:")
        print("  - 在 viewer 窗口中拖动机器人关节来改变配置")
        print("  - 动力学量会实时计算并显示在终端")
        print("  - 按 ESC 或关闭窗口退出")
        print("=" * 70)
        print("\n[启动 viewer...]")

        last_update_time = time.time()
        update_interval = 1.0 / update_rate

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 启用接触点可视化
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

            print("\n✓ Viewer 已启动")
            print("  开始实时计算...\n")

            frame_count = 0

            while viewer.is_running():
                # 从 viewer 获取当前配置（用户可能已经修改）
                q_current = np.zeros(self.n_joints)
                qdot_current = np.zeros(self.n_joints)

                for i, name in enumerate(self.joint_names):
                    dof_idx = self.joint_dof_indices[i]
                    if dof_idx >= 0:
                        q_current[i] = self.data.qpos[dof_idx]
                        qdot_current[i] = self.data.qvel[dof_idx]

                # 按更新频率计算和显示
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    # 更新前向动力学
                    mujoco.mj_forward(self.model, self.data)

                    # 计算动力学量
                    try:
                        results = self.compute_all(q_current, qdot_current,
                                                   task_space_dim=task_space_dim)

                        # 显示结果
                        print("\n" + "-" * 70)
                        print(f"帧 #{frame_count} | 时间: {current_time:.2f}s")
                        print("-" * 70)

                        if show_detailed:
                            # 显示关节角度和速度
                            print(f"\n关节配置:")
                            print(f"  q = [{', '.join([f'{q:.3f}' for q in q_current])}]")
                            if np.any(qdot_current != 0):
                                print(f"  qdot = [{', '.join([f'{qd:.3f}' for qd in qdot_current])}]")

                            # 质量矩阵信息
                            M = results['M']
                            M_eigvals = np.linalg.eigvals(M)
                            print(f"\n1. 质量矩阵 M(θ) [{M.shape[0]}×{M.shape[1]}]")
                            print(f"   特征值: [{np.min(M_eigvals):.4f}, {np.max(M_eigvals):.4f}]")
                            print(f"   条件数: {np.linalg.cond(M):.4e}")
                            print(f"   行列式: {np.linalg.det(M):.6e}")

                            # 重力项
                            g = results['g']
                            print(f"\n2. 重力项 g(θ) [N·m]")
                            print(f"   g = [{', '.join([f'{gi:7.3f}' for gi in g])}]")
                            print(f"   ||g|| = {np.linalg.norm(g):.4f}")

                            # 科里奥利项（如果有速度）
                            if np.any(qdot_current != 0):
                                C_qdot = results['C_qdot']
                                print(f"\n3. 科里奥利项 C(θ,θ̇)θ̇ [N·m]")
                                print(f"   C_qdot = [{', '.join([f'{c:7.3f}' for c in C_qdot])}]")
                                print(f"   ||C_qdot|| = {np.linalg.norm(C_qdot):.4f}")

                            # 雅可比矩阵信息
                            J = results['J']
                            J_cond = np.linalg.cond(J)
                            print(f"\n4. 雅可比矩阵 J(θ) [{J.shape[0]}×{J.shape[1]}]")
                            print(f"   条件数: {J_cond:.4e}")
                            if J_cond > 1e6:
                                print(f"   ⚠ 警告: 接近奇异点！")

                            # 任务空间质量矩阵
                            Lambda = results['Lambda']
                            Lambda_eigvals = np.linalg.eigvals(Lambda)
                            print(f"\n5. 任务空间质量矩阵 Λ(θ) [{Lambda.shape[0]}×{Lambda.shape[1]}]")
                            print(f"   特征值: [{np.min(Lambda_eigvals):.4f}, {np.max(Lambda_eigvals):.4f}]")
                            print(f"   条件数: {np.linalg.cond(Lambda):.4e}")

                            # 雅可比时间导数（如果有速度）
                            if 'J_dot' in results:
                                J_dot = results['J_dot']
                                print(f"\n6. 雅可比时间导数 Ḃ(θ) [{J_dot.shape[0]}×{J_dot.shape[1]}]")
                                print(f"   ||Ḃ||_F = {np.linalg.norm(J_dot, 'fro'):.6f}")

                            # 末端执行器位置
                            try:
                                pos = np.array(self.data.body(self.body_name).xpos)
                                print(f"\n7. 末端执行器位置 [m]")
                                print(f"   p = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                            except:
                                pass
                        else:
                            # 简化显示
                            M = results['M']
                            g = results['g']
                            J = results['J']
                            Lambda = results['Lambda']

                            print(f"M(θ) 条件数: {np.linalg.cond(M):.2e} | "
                                  f"J(θ) 条件数: {np.linalg.cond(J):.2e} | "
                                  f"Λ(θ) 条件数: {np.linalg.cond(Lambda):.2e}")
                            print(f"||g|| = {np.linalg.norm(g):.4f} N·m")

                        last_update_time = current_time
                        frame_count += 1

                    except Exception as e:
                        print(f"\n⚠ 计算错误: {e}")
                        import traceback
                        traceback.print_exc()

                # 同步 viewer
                viewer.sync()
                time.sleep(0.001)  # 避免占用过多CPU

            print("\n" + "=" * 70)
            print("Viewer 已关闭")
            print(f"总共计算了 {frame_count} 帧")
            print("=" * 70)


def main():
    """示例用法"""
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--viewer':
        # 可视化模式
        print("=" * 70)
        print("机器人动力学量计算器 - 实时可视化模式")
        print("=" * 70)

        calc = DynamicsCalculator(
            model_path="surface_force_control.xml",
            body_name="panda_hand",
            joint_names=[f"panda_joint{i+1}" for i in range(7)]
        )

        # 设置初始配置
        q_init = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

        # 运行可视化
        calc.run_with_viewer(q_init=q_init, update_rate=5.0,
                           task_space_dim=3, show_detailed=True)
    else:
        # 静态计算模式
        print("=" * 70)
        print("机器人动力学量计算器 - 静态计算示例")
        print("=" * 70)
        print("\n提示: 使用 'python dynamics_calculator.py --viewer' 启动可视化模式")
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

