#!/usr/bin/env python3
"""
实时仿真显示 - 使用混合力位控制（基于约束）
使用公式11.57-11.64实现混合运动-力控制
"""

import mujoco
import mujoco.viewer
import numpy as np
from dynamics_calculator import DynamicsCalculator


class OSICViewer:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("surface_force_control.xml")
        self.data = mujoco.MjData(self.model)
        
        self.n_joints = 7
        self.joint_names = [f"panda_joint{i+1}" for i in range(self.n_joints)]
        
        # 初始配置（原始）
        qpos0 = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        for i, name in enumerate(self.joint_names):
            dof = self.model.joint(name).dofadr
            if len(dof) > 0:
                self.data.qpos[dof[0]] = qpos0[i]
        
        mujoco.mj_forward(self.model, self.data)
        
        # 初始化动力学计算器（用于计算Λ和η）
        # 注意：DynamicsCalculator 有自己的 model 和 data，需要同步状态
        self.dynamics_calc = DynamicsCalculator(
            model_path="surface_force_control.xml",
            body_name="panda_hand",
            joint_names=self.joint_names
        )
        
        # 同步初始状态
        q_init = np.zeros(self.n_joints)
        for i, name in enumerate(self.joint_names):
            dof = self.model.joint(name).dofadr
            if len(dof) > 0:
                q_init[i] = self.data.qpos[dof[0]]
        self.dynamics_calc.set_configuration(q_init)
        
        # 接触状态
        self.contact_t = None
        self.is_contact_stable = False
        self.is_contact_ever_established = False
        
        # 积分项（用于PI控制）
        self.X_e_integral = np.zeros(3)  # 位置误差积分
        self.F_e_integral = 0.0  # 力误差积分
        
        # 控制增益
        # 运动控制增益
        self.K_p = np.diag([800.0, 800.0, 300.0])  # 位置增益
        self.K_i = np.diag([10.0, 10.0, 5.0])  # 积分增益
        self.K_d = np.diag([50.0, 50.0, 20.0])  # 微分增益
        
        # 力控制增益
        self.K_fp = 0.5  # 力比例增益
        self.K_fi = 0.1  # 力积分增益
        
        # 目标力（法向，向下为负）
        self.F_desired = -10.0  # N
        
        # 约束方向（法向，Z轴向上）
        self.constraint_normal = np.array([0.0, 0.0, 1.0])  # 表面法向（向上）
        
        # 约束类型：'minimal' (3约束: v_z, ω_x, ω_y) 或 'full' (4约束: v_z, ω_x, ω_y, ω_z)
        self.constraint_type = 'full'  # 默认使用完全约束（包括所有旋转）
    
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
            dof = self.model.joint(name).dofadr
            if len(dof) > 0:
                J[:3, i] = jacp[:, dof[0]]
                J[3:, i] = jacr[:, dof[0]]
        
        return J
    
    def get_jacobian_3x7(self):
        """获取3DOF位置雅可比矩阵（向后兼容）"""
        J6 = self.get_jacobian_6x7()
        return J6[:3, :]
    
    def get_contact_force(self):
        force_z = 0.0
        is_contact_now = False
        
        try:
            surf_geom_id = self.model.geom("surface").id
        except:
            return False, 0.0
        
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == surf_geom_id or c.geom2 == surf_geom_id:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                force_z += f[2]
                # 使用绝对值判断接触
                if abs(f[2]) > 0.1:  # 任何小于0.1的力都忽略
                    is_contact_now = True
        
        # 改进的滞后机制：一旦有显著接触力，就保持接触状态
        # 只有当力完全消失时才切换为未接触
        if abs(force_z) > 1.0:  # 有显著力
            self.is_contact_stable = True
        elif abs(force_z) < 0.05:  # 力完全消失
            self.is_contact_stable = False
        # 其他情况（0.05-1.0之间）保持现有状态
        
        return self.is_contact_stable, force_z
    
    def compute_constraint_matrix(self, constraint_type='full'):
        """
        计算约束矩阵 A(θ) - 公式11.57
        对于表面接触，约束法向运动和旋转
        
        参数:
            constraint_type: 'minimal' (3约束: v_z, ω_x, ω_y) 
                           或 'full' (4约束: v_z, ω_x, ω_y, ω_z)
        """
        if constraint_type == 'minimal':
            # 方案1：最小约束（法向平移 + 俯仰 + 横滚）
            # 允许绕Z轴旋转（偏航角），增加灵活性
            A = np.zeros((3, 6))
            A[0, 2] = 1.0  # 约束Z方向平移 v_z（法向）
            A[1, 3] = 1.0  # 约束绕X轴旋转 ω_x（俯仰角）
            A[2, 4] = 1.0  # 约束绕Y轴旋转 ω_y（横滚角）
        elif constraint_type == 'full':
            # 方案2：完全约束（包括所有旋转）
            # 适用于需要固定工具方向的场景
            A = np.zeros((4, 6))
            A[0, 2] = 1.0  # 约束Z方向平移 v_z（法向）
            A[1, 3] = 1.0  # 约束绕X轴旋转 ω_x（俯仰角）
            A[2, 4] = 1.0  # 约束绕Y轴旋转 ω_y（横滚角）
            A[3, 5] = 1.0  # 约束绕Z轴旋转 ω_z（偏航角）
        else:
            # 默认：完全约束
            A = np.zeros((4, 6))
            A[0, 2] = 1.0  # 约束Z方向平移 v_z
            A[1, 3] = 1.0  # 约束绕X轴旋转 ω_x
            A[2, 4] = 1.0  # 约束绕Y轴旋转 ω_y
            A[3, 5] = 1.0  # 约束绕Z轴旋转 ω_z
        
        return A
    
    def compute_projection_matrix(self, Lambda, A):
        """
        计算投影矩阵 P - 公式11.63
        P = I - A^T(AΛ^{-1}A^T)^{-1}AΛ^{-1}
        """
        try:
            Lambda_inv = np.linalg.inv(Lambda)
            # 计算 (AΛ^{-1}A^T)^{-1}
            A_Lambda_inv_AT = A @ Lambda_inv @ A.T
            if np.linalg.cond(A_Lambda_inv_AT) > 1e10:
                # 接近奇异，使用伪逆
                A_Lambda_inv_AT_inv = np.linalg.pinv(A_Lambda_inv_AT)
            else:
                A_Lambda_inv_AT_inv = np.linalg.inv(A_Lambda_inv_AT)
            
            # 投影矩阵
            P = np.eye(6) - A.T @ A_Lambda_inv_AT_inv @ A @ Lambda_inv
            return P
        except:
            # 如果计算失败，返回单位矩阵（无约束）
            return np.eye(6)
    
    def control_step(self, t, dt=0.002):
        """
        执行一步混合力位控制 - 公式11.64
        使用约束和投影矩阵实现混合运动-力控制
        """
        mujoco.mj_forward(self.model, self.data)
        
        # 获取当前状态
        pos_curr = np.array(self.data.body("panda_hand").xpos)
        quat_curr = np.array(self.data.body("panda_hand").xquat)
        
        # 获取关节状态
        q = np.zeros(self.n_joints)
        qdot = np.zeros(self.n_joints)
        for i, name in enumerate(self.joint_names):
            dof = self.model.joint(name).dofadr
            if len(dof) > 0:
                q[i] = self.data.qpos[dof[0]]
                qdot[i] = self.data.qvel[dof[0]]
        
        # 获取雅可比矩阵（6DOF）
        J = self.get_jacobian_6x7()
        
        # 计算任务空间速度（twist）
        V = J @ qdot  # 6x1: [vx, vy, vz, wx, wy, wz]
        
        # 更新动力学计算器的状态（用于计算Λ和η）
        self.dynamics_calc.set_configuration(q, qdot)
        
        # 获取接触状态
        is_contact, F_curr = self.get_contact_force()
        
        # ========== 阶段识别和参考轨迹生成 ==========
        if abs(F_curr) < 1.0:  # 没有显著接触力
            phase = 0
            if self.contact_t is not None:
                self.contact_t = None
                self.F_e_integral = 0.0
        else:
            if self.contact_t is None:
                self.contact_t = t
                self.is_contact_ever_established = True
                self.F_e_integral = 0.0
                self.X_e_integral = np.zeros(3)
            phase = 1
        
        # 生成参考位置和速度
        pos_ref = np.array([0.5, 0.0, 0.3])  # 默认参考位置
        vel_ref = np.zeros(3)
        
        if phase == 0:
            # Phase 0: 快速下降到接近表面
            z_des = 0.4 - min(t / 3.0, 1.0) * 0.25
            pos_ref = np.array([0.5, 0.0, z_des])
            vel_ref = np.array([0.0, 0.0, -0.25/3.0 if t < 3.0 else 0.0])
        else:
            # Phase 1+: 接触后，使用混合控制
            t_contact = t - self.contact_t
            
            # XY位置参考（切向运动）
            pos_ref_xy = np.array([0.5, 0.0])
            vel_ref_xy = np.array([0.0, 0.0])
            
            if t_contact >= 3.0:
                t_wipe = t_contact - 3.0
                if t_wipe < 15.0:
                    # X轴前后擦拭
                    cycle_t = t_wipe % 1.0
                    if cycle_t < 0.5:
                        progress = cycle_t / 0.5
                        pos_ref_xy[0] = 0.5 + 0.15 * np.sin(progress * np.pi)
                        vel_ref_xy[0] = 0.15 * np.pi / 0.5 * np.cos(progress * np.pi)
                    else:
                        progress = (cycle_t - 0.5) / 0.5
                        pos_ref_xy[0] = 0.5 + 0.15 * np.sin((1.0 - progress) * np.pi)
                        vel_ref_xy[0] = -0.15 * np.pi / 0.5 * np.cos((1.0 - progress) * np.pi)
                else:
                    # Y轴左右擦拭
                    cycle_t = (t_wipe - 15.0) % 1.0
                    if cycle_t < 0.5:
                        progress = cycle_t / 0.5
                        pos_ref_xy[1] = -0.1 * np.sin(progress * np.pi)
                        vel_ref_xy[1] = -0.1 * np.pi / 0.5 * np.cos(progress * np.pi)
                    else:
                        progress = (cycle_t - 0.5) / 0.5
                        pos_ref_xy[1] = -0.1 * np.sin((1.0 - progress) * np.pi)
                        vel_ref_xy[1] = 0.1 * np.pi / 0.5 * np.cos((1.0 - progress) * np.pi)
            
            pos_ref = np.array([pos_ref_xy[0], pos_ref_xy[1], 0.3])  # Z在接触表面
            vel_ref = np.array([vel_ref_xy[0], vel_ref_xy[1], 0.0])
        
        # ========== 计算动力学量 ==========
        # 计算任务空间质量矩阵 Λ(θ)
        Lambda = self.dynamics_calc.compute_task_space_mass_matrix(
            q, task_space_dim=6, use_pseudoinverse=True, damping=1e-6
        )
        
        # 计算任务空间科里奥利项 η(θ, V)
        eta = self.dynamics_calc.compute_task_space_coriolis(
            q, V, task_space_dim=6, use_pseudoinverse=True, damping=1e-6
        )
        
        # ========== 约束和投影矩阵 ==========
        if is_contact:
            # 计算约束矩阵 A(θ) - 公式11.57
            # 约束：v_z, ω_x, ω_y, ω_z（法向平移 + 所有旋转）
            A = self.compute_constraint_matrix(constraint_type=self.constraint_type)
            
            # 计算投影矩阵 P(θ) - 公式11.63
            P = self.compute_projection_matrix(Lambda, A)
        else:
            # 无约束时，投影矩阵为单位矩阵
            P = np.eye(6)
            A = np.zeros((0, 6))
        
        # ========== 混合控制 - 公式11.64 ==========
        # 位置误差和速度误差（仅位置部分，3DOF）
        X_e = pos_ref - pos_curr  # 位置误差
        V_e = vel_ref - V[:3]  # 速度误差（仅平移部分）
        
        # 更新积分项
        self.X_e_integral += X_e * dt
        self.X_e_integral = np.clip(self.X_e_integral, -0.1, 0.1)  # 限制积分饱和
        
        # 运动控制部分（投影到运动子空间）
        # 简化：使用PD控制 + 积分项
        F_motion = P[:3, :3] @ (
            self.K_p @ X_e + 
            self.K_i @ self.X_e_integral + 
            self.K_d @ V_e
        )
        
        # 力控制部分（投影到力子空间）
        F_force = np.zeros(6)
        if is_contact:
            # 力误差（法向）
            F_e = self.F_desired - F_curr  # F_curr是负值（向下）
            self.F_e_integral += F_e * dt
            self.F_e_integral = np.clip(self.F_e_integral, -50.0, 50.0)
            
            # 力控制wrench（仅在法向）
            F_d = np.zeros(6)
            F_d[2] = self.F_desired + self.K_fp * F_e + self.K_fi * self.F_e_integral
            
            # 投影到力子空间
            F_force = (np.eye(6) - P) @ F_d
        
        # 组合控制wrench
        F_cmd = np.zeros(6)
        F_cmd[:3] = F_motion  # 运动控制（平移）
        F_cmd += F_force  # 力控制
        F_cmd += eta  # 科里奥利和重力补偿
        
        # ========== 转换为关节力矩 ==========
        # τ = J^T F_cmd - 公式11.64的最后部分
        tau = J.T @ F_cmd
        
        # 添加重力补偿（从关节空间）
        for i in range(self.n_joints):
            dof = self.model.joint(self.joint_names[i]).dofadr
            if len(dof) > 0:
                tau[i] += self.data.qfrc_bias[dof[0]]
        
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
    
    def run_with_viewer(self, duration=60.0):
        """用viewer运行仿真"""
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            
            dt = self.model.opt.timestep
            t = 0.0
            
            print("\n" + "="*70)
            print("表面力控仿真 - 实时显示")
            print("="*70)
            print("\n演示阶段:")
            print("  0-3s:   接近表面（Z轴快速下降）")
            print("  3s+:    混合力位控制 + 切向擦拭运动")
            print("\n控制方法: 混合运动-力控制（基于约束）")
            print("  - 约束: v_z, ω_x, ω_y, ω_z（法向平移 + 所有旋转）")
            print("  - 运动子空间: XY位置控制（切向运动）")
            print("  - 力子空间: Z方向力控制 + 旋转力矩控制")
            print("  - 使用投影矩阵分离运动和力控制")
            print("="*70)
            print("\n[运行中...]")
            print("关闭窗口停止仿真\n")
            
            last_print = 0.0
            
            while viewer.is_running() and t < duration:
                # 执行控制
                self.control_step(t)
                
                # 物理仿真步进
                mujoco.mj_step(self.model, self.data)
                
                # 更新viewer
                viewer.sync()
                
                # 进度报告
                if t - last_print >= 5.0:
                    is_contact, F = self.get_contact_force()
                    contact_str = "✓" if is_contact else "✗"
                    
                    if t < 10.0:
                        phase = "接近"
                    elif t < 13.0:
                        phase = "锁定/下降"
                    elif t < 30.0:
                        phase = "前后擦拭"
                    elif t < 50.0:
                        phase = "左右擦拭"
                    else:
                        phase = "收尾"
                    
                    print(f"[{phase:8}] t={t:5.1f}s | 力:{F:7.2f}N | 接触:{contact_str}")
                    last_print = t
                
                t += dt
            
            print(f"\n✓ 仿真完成！")
        
        print("\nViewer已关闭")


if __name__ == "__main__":
    sim = OSICViewer()
    sim.run_with_viewer(duration=60.0)
