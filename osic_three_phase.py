"""
表面摩擦力控仿真 - 改进的三阶段策略
1. 接近阶段（0-10s）：位置控制下压到接触高度附近
2. 过渡阶段（10-12s）：混合力位置控制，缓慢建立目标压力
3. 维持阶段（12s+）：纯力控制维持压力
"""

import csv
import mujoco
import numpy as np


class ImprovedOSICController:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("surface_force_control.xml")
        self.data = mujoco.MjData(self.model)
        
        self.n_joints = 7
        self.joint_names = [f"panda_joint{i+1}" for i in range(self.n_joints)]
        
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
    
    def control(self, pos_des, z_des, force_des, t, z_des_contact=0.315):
        """单步控制 - 三阶段策略"""
        mujoco.mj_forward(self.model, self.data)
        
        pos_curr = self.data.body("panda_hand").xpos.copy()
        J = self.get_jacobian()
        is_contact, force_curr = self.get_contact()
        
        # 获取末端速度
        qvel = np.zeros(self.n_joints)
        for i, name in enumerate(self.joint_names):
            dof = self.model.joint(name).dofadr
            if len(dof) > 0:
                qvel[i] = self.data.qvel[dof[0]]
        v_curr = J @ qvel
        
        # 构建操作空间期望力
        F = np.zeros(6)
        
        # XY 位置控制
        err_xy = pos_des[:2] - pos_curr[:2]
        F[:2] = 80.0 * err_xy - 8.0 * v_curr[:2]
        
        # Z 控制：三阶段
        if is_contact and force_curr > 0.5:
            # 已接触
            if self.contact_established_time is None:
                self.contact_established_time = t
                print(f"  ✓ 接触建立！时间: {t:.2f}s，力: {force_curr:.2f}N")
            
            t_since_contact = t - self.contact_established_time
            
            # 阶段 1: 0-0.5s - 快速建立接触，用高刚度位置控制锁定位置
            if t_since_contact < 0.5:
                z_des_hold = pos_curr[2]
                F[2] = 150.0 * (z_des_hold - pos_curr[2]) - 3.0 * v_curr[2]
            
            # 阶段 2: 0.5-3.0s - 扩展过渡阶段，持续下压以增加接触压力
            elif t_since_contact < 3.0:
                # 缓慢下压以增加压力，更长的下压时间
                transition_progress = min(1.0, (t_since_contact - 0.5) / 2.5)
                
                # 下压目标：从接触点逐渐深入表面
                z_des_lower = z_des_contact - 0.010 * transition_progress
                
                # 位置控制项 - 通过下压增加压力
                z_err = z_des_lower - pos_curr[2]
                pos_term = 40.0 * z_err
                
                F[2] = pos_term - 2.0 * v_curr[2]
            
            # 阶段 3: 3.0s+ - 纯力控制维持压力
            else:
                force_err = force_des - force_curr
                
                # 增加力控制增益
                Kp_force = 25.0
                Kd_force = 2.5
                
                F[2] = -Kp_force * force_err - Kd_force * v_curr[2]
                
                # 加入微弱的位置维持项
                z_hold_gain = 8.0
                F[2] += z_hold_gain * (z_des_contact - 0.003 - pos_curr[2])  # 目标稍微深入一些
        else:
            # 未接触：位置控制下压
            z_err = z_des - pos_curr[2]
            Kp_approach = 15.0
            Kd_approach = 1.5
            F[2] = Kp_approach * z_err - Kd_approach * v_curr[2]
        
        # 计算关节力矩
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
    
    def run(self, duration=20.0):
        print("\n" + "="*70)
        print("表面摩擦力控仿真 - 改进的三阶段策略")
        print("="*70 + "\n")
        
        log = []
        t = 0.0
        dt = self.model.opt.timestep
        last_print = 0.0
        
        pos_des = np.array([0.5, 0.0, 0.4])
        z_des_init = 0.4
        z_des_contact = 0.315
        force_des = 10.0
        
        frame = 0
        while t < duration:
            # 轨迹规划：缓慢下压到接触高度
            if t < 10.0:
                alpha = (t / 10.0) ** 1.2
                z_des = z_des_init - alpha * (z_des_init - z_des_contact)
            else:
                z_des = z_des_contact
            
            is_contact, force = self.control(pos_des, z_des, force_des, t, z_des_contact)
            mujoco.mj_step(self.model, self.data)
            
            pos_curr = self.data.body("panda_hand").xpos.copy()
            
            # 打印
            if t - last_print >= 0.5:
                phase = "接近" if t < 10.0 else "稳定"
                contact_status = "是" if is_contact else "否"
                print(f"[{phase}] t={t:6.2f}s | pos: [{pos_curr[0]:6.3f}, {pos_curr[1]:6.3f}, {pos_curr[2]:6.4f}] | "
                      f"力: {force:6.2f}N (目标{force_des:.1f}N) | 接触: {contact_status}")
                last_print = t
            
            # 记录
            if frame % 10 == 0:
                log.append([t, pos_curr[0], pos_curr[1], pos_curr[2], force, int(is_contact)])
            
            t += dt
            frame += 1
        
        # 保存
        with open("force_log.csv", 'w', newline='') as f:
            csv.writer(f).writerow(['time', 'pos_x', 'pos_y', 'pos_z', 'force_normal', 'is_contact'])
            csv.writer(f).writerows(log)
        
        print(f"\n✓ 仿真完成！已保存 {len(log)} 条数据到 force_log.csv")


if __name__ == "__main__":
    ctrl = ImprovedOSICController()
    ctrl.run(duration=20.0)
