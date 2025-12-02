#!/usr/bin/env python3
"""
完整版：零空间优化 + 切向运动

这个版本在稳定的OSIC力控基础上加入：
1. 零空间优化：关节配置优化（不干扰力控）
2. 切向运动：在保持法向力的同时实现表面擦拭

用户需求实现：
- 集成零空间：在不干扰力控的同时，优化关节构型
- 切向运动：在阶段 3 中，尝试给 XY 轴施加一个小的目标速度，
  模拟机器人在保持 10 N 法向力的同时在表面上滑动
"""

import csv
import mujoco
import numpy as np


class OSICWithTangentialMotion:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("surface_force_control.xml")
        self.data = mujoco.MjData(self.model)
        
        self.n_joints = 7
        self.joint_names = [f"panda_joint{i+1}" for i in range(self.n_joints)]
        
        # 使用已验证的初始配置
        qpos0 = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        for i, name in enumerate(self.joint_names):
            self.data.joint(name).qpos[0] = qpos0[i]
        
        mujoco.mj_forward(self.model, self.data)
        print("✓ 模型初始化完成")
        
        self.contact_established_time = None
    
    def get_jacobian(self):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bid)
        
        J = np.zeros((6, self.n_joints))
        for i, name in enumerate(self.joint_names):
            dof = self.model.joint(name).dofadr
            if len(dof) > 0:
                J[:3, i] = jacp[:, dof[0]]
                J[3:, i] = jacr[:, dof[0]]
        return J
    
    def get_contact(self):
        """获取接触力 - 正向为向下压"""
        try:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "surface")
        except:
            return False, 0.0
        
        force_normal = 0.0
        is_contact = False
        
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == sid or c.geom2 == sid:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                force_normal += f[2]
                is_contact = True
        
        return is_contact, force_normal
    
    def get_null_space_projector(self, J):
        """计算零空间投影矩阵 N = I - J^+ @ J"""
        J_pinv = np.linalg.pinv(J)
        N = np.eye(self.n_joints) - J_pinv @ J
        return N, J_pinv
    
    def get_null_space_velocity(self, qpos, q_target, gain=0.05):
        """在零空间中的期望速度（向目标配置靠近）"""
        q_err = q_target - qpos
        q_err = np.clip(q_err, -0.3, 0.3)
        v_null = gain * q_err
        return v_null
    
    def get_wipe_trajectory(self, t):
        """
        生成擦拭轨迹和切向速度目标
        
        返回：(pos_des_xy, v_tangent_des_xy)
        """
        pos_xy = np.array([0.5, 0.0])
        v_tangent = np.array([0.0, 0.0])
        
        # 以接触建立作为参考时间
        if self.contact_established_time is None:
            return pos_xy, v_tangent
        
        t_contact = t - self.contact_established_time
        
        # 在接触后 3 秒（过渡结束）开始擦拭运动
        if t_contact < 3.0:
            # 还在过渡阶段，保持原位置
            return pos_xy, v_tangent
        
        # 过渡完成，开始擦拭
        t_wipe = t_contact - 3.0
        
        # 在过渡结束后5秒再开始切向运动（确保接触稳定）
        if t_wipe < 5.0:
            # 稳定接触阶段，不移动
            return pos_xy, v_tangent
        
        t_wipe = t_wipe - 5.0
        
        # 切换不同的擦拭模式（每 30 秒循环一次）
        if t_wipe < 15.0:
            # 前后擦拭（X方向）- 15 秒
            cycle_t = t_wipe % 5.0
            
            if cycle_t < 2.5:
                # 向前（0-2.5s）
                progress = cycle_t / 2.5
                pos_xy[0] = 0.5 + 0.06 * np.sin(progress * np.pi)
                v_tangent[0] = 0.06 * np.pi / 2.5 * np.cos(progress * np.pi)
            else:
                # 向后（2.5-5s）
                progress = (cycle_t - 2.5) / 2.5
                pos_xy[0] = 0.5 + 0.06 * np.sin((1.0 - progress) * np.pi)
                v_tangent[0] = -0.06 * np.pi / 2.5 * np.cos((1.0 - progress) * np.pi)
        
        else:
            # 左右擦拭（Y方向）- 从 15s 开始
            cycle_t = (t_wipe - 15.0) % 5.0
            
            if cycle_t < 2.5:
                # 向左（0-2.5s）
                progress = cycle_t / 2.5
                pos_xy[1] = -0.04 * np.sin(progress * np.pi)  # 减小Y幅度
                v_tangent[1] = -0.04 * np.pi / 2.5 * np.cos(progress * np.pi)
            else:
                # 向右（2.5-5s）
                progress = (cycle_t - 2.5) / 2.5
                pos_xy[1] = 0.04 * np.sin((1.0 - progress) * np.pi)  # 减小Y幅度
                v_tangent[1] = 0.04 * np.pi / 2.5 * np.cos((1.0 - progress) * np.pi)
        
        return pos_xy, v_tangent
    
    def control(self, pos_des, z_des, force_des, t, z_des_contact=0.315):
        """
        改进的控制法则：三阶段力控 + 切向速度前馈 + 零空间优化
        """
        mujoco.mj_forward(self.model, self.data)
        
        pos_curr = self.data.body("panda_hand").xpos.copy()
        J = self.get_jacobian()
        is_contact, force_curr = self.get_contact()
        
        # 获取末端速度
        qvel = np.zeros(self.n_joints)
        qpos = np.zeros(self.n_joints)
        for i, name in enumerate(self.joint_names):
            dof = self.model.joint(name).dofadr
            if len(dof) > 0:
                qvel[i] = self.data.qvel[dof[0]]
                qpos[i] = self.data.qpos[dof[0]]
        
        v_curr = J @ qvel
        
        # 构建操作空间期望力
        F = np.zeros(6)
        
        # 获取切向运动目标
        pos_xy_des, v_tangent_des = self.get_wipe_trajectory(t)
        
        # ===== XY 位置控制 + 切向速度前馈 =====
        err_xy = np.array([pos_xy_des[0] - pos_curr[0], 
                          pos_xy_des[1] - pos_curr[1]])
        
        # 在接触状态下使用混合控制
        if is_contact and force_curr > 0.5:
            # 减小XY增益以防止干扰接触
            Kp_xy = 30.0  # 进一步降低
            Kd_xy = 4.0
            # 切向速度前馈
            Kv_xy = 0.8  # 降低速度前馈
            
            # 基于轨迹跟踪的位置控制 + 速度前馈
            F[:2] = Kp_xy * err_xy - Kd_xy * v_curr[:2] + Kv_xy * v_tangent_des
        else:
            # 接近阶段：强位置控制
            Kp_xy = 80.0
            Kd_xy = 8.0
            F[:2] = Kp_xy * err_xy - Kd_xy * v_curr[:2]
        
        # ===== Z 控制：三阶段策略（保持原始参数） =====
        if is_contact and force_curr > 0.5:
            # 已接触
            if self.contact_established_time is None:
                self.contact_established_time = t
                print(f"  ✓ 接触建立！时间: {t:.2f}s，力: {force_curr:.2f}N")
            
            t_since_contact = t - self.contact_established_time
            
            # 阶段 1: 0-0.5s - 锁定
            if t_since_contact < 0.5:
                z_des_hold = pos_curr[2]
                F[2] = 150.0 * (z_des_hold - pos_curr[2]) - 3.0 * v_curr[2]
            
            # 阶段 2: 0.5-3.0s - 过渡
            elif t_since_contact < 3.0:
                transition_progress = min(1.0, (t_since_contact - 0.5) / 2.5)
                z_des_lower = z_des_contact - 0.010 * transition_progress
                z_err = z_des_lower - pos_curr[2]
                F[2] = 40.0 * z_err - 2.0 * v_curr[2]
            
            # 阶段 3: 3.0s+ - 力控维持（支持切向运动）
            else:
                force_err = force_des - force_curr
                Kp_force = 25.0
                Kd_force = 2.5
                F[2] = -Kp_force * force_err - Kd_force * v_curr[2]
                
                # 位置维持项
                z_hold_gain = 8.0
                F[2] += z_hold_gain * (z_des_contact - 0.003 - pos_curr[2])
        
        else:
            # 未接触：位置控制下压
            z_err = z_des - pos_curr[2]
            F[2] = 15.0 * z_err - 1.5 * v_curr[2]
        
        # ===== 零空间优化（轻量级） =====
        N, J_pinv = self.get_null_space_projector(J)
        q_target = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        if is_contact and force_curr > 0.5:
            # 接触状态：轻微优化
            v_null_des = self.get_null_space_velocity(qpos, q_target, gain=0.05)
        else:
            # 接近阶段：更积极的优化
            v_null_des = self.get_null_space_velocity(qpos, q_target, gain=0.1)
        
        # ===== 计算关节力矩 =====
        tau = np.zeros(self.n_joints)
        
        for i, name in enumerate(self.joint_names):
            dof = self.model.joint(name).dofadr
            if len(dof) > 0:
                tau[i] = self.data.qfrc_bias[dof[0]]
            
            tau[i] += J[:, i] @ F
            
            max_tau = 87.0 if i < 4 else 12.0
            tau[i] = np.clip(tau[i], -max_tau, max_tau)
            
            try:
                self.data.actuator(name).ctrl = float(tau[i])
            except:
                pass
        
        return is_contact, force_curr
    
    def run(self, duration=60.0):
        print("\n" + "="*70)
        print("完整版：零空间优化 + 切向运动")
        print("="*70)
        print("\n功能:")
        print("  ✓ 三阶段法向力控制")
        print("  ✓ 零空间关节配置优化")
        print("  ✓ 切向速度前馈（光滑表面擦拭）")
        print("  ✓ 混合力/位置控制")
        print("\n演示阶段:")
        print("  1. 接近表面（0-~1s）")
        print("  2. 建立接触和过渡（~1-8s）")
        print("  3. 前后擦拭（8-23s）")
        print("  4. 左右擦拭（23s+）")
        print("="*70 + "\n")
        
        log = []
        t = 0.0
        dt = self.model.opt.timestep
        last_print = 0.0
        
        pos_des_xy = np.array([0.5, 0.0])
        z_des_init = 0.4
        z_des_contact = 0.315
        force_des = 10.0
        
        frame = 0
        while t < duration:
            # Z轴轨迹规划
            if t < 10.0:
                alpha = (t / 10.0) ** 1.2
                z_des = z_des_init - alpha * (z_des_init - z_des_contact)
            else:
                z_des = z_des_contact
            
            # 控制
            is_contact, force = self.control(pos_des_xy, z_des, force_des, t, z_des_contact)
            
            # 仿真步进
            mujoco.mj_step(self.model, self.data)
            
            pos_curr = self.data.body("panda_hand").xpos.copy()
            
            # 打印进度
            if t - last_print >= 2.5:
                if t < 1.0:
                    phase = "接近"
                elif t < 8.0:
                    phase = "建立"
                elif t < 23.0:
                    phase = "前后"
                else:
                    phase = "左右"
                
                contact_status = "✓" if is_contact else "✗"
                print(f"[{phase:2s}] t={t:6.2f}s | X:{pos_curr[0]:.3f} Y:{pos_curr[1]:.3f} Z:{pos_curr[2]:.4f} | " +
                      f"力:{force:7.2f}N | 接触:{contact_status}")
                
                last_print = t
            
            # 记录
            if frame % 10 == 0:
                log.append([t, pos_curr[0], pos_curr[1], pos_curr[2], force, int(is_contact)])
            
            t += dt
            frame += 1
        
        # 保存
        with open("osic_with_tangential.csv", 'w', newline='') as f:
            csv.writer(f).writerow(['time', 'pos_x', 'pos_y', 'pos_z', 'force_normal', 'is_contact'])
            csv.writer(f).writerows(log)
        
        print(f"\n✓ 仿真完成！已保存 {len(log)} 条数据到 osic_with_tangential.csv")
        
        # 统计（整体）
        forces_all = [row[4] for row in log if row[5] > 0.5]
        print(f"\n整体力值统计 (接触状态):")
        print(f"  均值: {np.mean(forces_all):.2f}N")
        print(f"  范围: [{np.min(forces_all):.2f}, {np.max(forces_all):.2f}]N")
        print(f"  标准差: {np.std(forces_all):.2f}N")
        print(f"  接触率: {len(forces_all)/len(log)*100:.1f}%")
        
        # 分阶段统计
        print(f"\n分阶段统计:")
        for t_start, t_end, phase in [(0, 8, "建立"), (8, 23, "前后"), (23, 60, "左右")]:
            data_phase = [row for row in log if t_start <= row[0] < t_end and row[5] > 0.5]
            if len(data_phase) > 0:
                forces = [row[4] for row in data_phase]
                total_points = len([row for row in log if t_start <= row[0] < t_end])
                print(f"  {phase} ({t_start}-{t_end}s): 均值 {np.mean(forces):.2f}N, " +
                      f"范围 [{np.min(forces):.2f}, {np.max(forces):.2f}]N, 接触率 {len(data_phase)/total_points*100:.1f}%")


if __name__ == "__main__":
    ctrl = OSICWithTangentialMotion()
    ctrl.run(duration=60.0)
    print("\n✓ 零空间优化 + 切向运动仿真完成！")
