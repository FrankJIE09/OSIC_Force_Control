#!/usr/bin/env python3
"""
实时仿真显示 - 使用 mujoco.viewer
直接显示60秒完整仿真动画
"""

import mujoco
import mujoco.viewer
import numpy as np


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
        
        self.contact_t = None
        self.F_integral = 0.0
        self.t_start = None
        self.is_contact_stable = False  # 稳定的接触状态（带滞后）
        self.is_contact_ever_established = False  # 一旦接触过，不再断开
    
    def get_jacobian_3x7(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        
        try:
            hand_body_id = self.model.body("panda_hand").id
        except:
            return np.eye(3, 7)
        
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, hand_body_id)
        
        J = np.zeros((3, self.n_joints))
        for i, name in enumerate(self.joint_names):
            dof = self.model.joint(name).dofadr
            if len(dof) > 0:
                J[:, i] = jacp[:, dof[0]]
        
        return J
    
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
    
    def control_step(self, t, F_target=-3.5, dt=0.002):  # 目标压力应该是负数（向下）
        """执行一步控制"""
        
        mujoco.mj_forward(self.model, self.data)
        
        pos_curr = np.array(self.data.body("panda_hand").xpos)
        J = self.get_jacobian_3x7()
        qvel = np.zeros(self.n_joints)
        for i, name in enumerate(self.joint_names):
            dof = self.model.joint(name).dofadr
            if len(dof) > 0:
                qvel[i] = self.data.qvel[dof[0]]
        v_curr = J @ qvel
        
        is_contact, F_curr = self.get_contact_force()
        
        # ========== 阶段识别 ==========
        # 基于接触力检测
        if abs(F_curr) < 1.0:  # 没有显著接触力
            phase = 0
        else:
            # 有显著接触力
            # 第一次有接触力时记录时间，之后永不改变
            if self.contact_t is None:
                self.contact_t = t
                self.is_contact_ever_established = True
            
            # Phase按照接触后的时间递进
            t_since_contact = t - self.contact_t
            if t_since_contact < 1.0:
                phase = 1
            elif t_since_contact < 3.0:
                phase = 2
            else:
                phase = 3
        
        # ========== 控制力生成 ==========
        F_cmd = np.zeros(3)
        
        if phase == 0:
            # Phase 0: 快速下降到接近表面
            # 每秒下降0.2m，需要3秒从0.4m到表面0.15m左右
            z_des = 0.4 - min(t / 3.0, 1.0) * 0.25  # 从0.4降到0.15
            # 中等增益（不要太高导致不稳定）
            F_cmd[2] = 300.0 * (z_des - pos_curr[2]) - 20.0 * v_curr[2]
        
        elif phase == 1:
            # Phase 1: 继续缓慢下降并产生压力
            z_des = max(0.15 - 0.01 * (t - self.contact_t), 0.1)  # 从0.15缓慢降到0.1
            F_cmd[2] = 250.0 * (z_des - pos_curr[2]) - 15.0 * v_curr[2] + 30.0  # 压力30N
        
        elif phase == 2:
            # Phase 2: 保持位置并产生稳定压力
            z_des = 0.12
            F_cmd[2] = 200.0 * (z_des - pos_curr[2]) - 10.0 * v_curr[2] + 40.0  # 压力40N
        
        elif phase == 3:
            # Phase 3: 维持接触，不再改变Z轴
            z_des = 0.24  # 锁定在当前接触深度
            F_cmd[2] = 150.0 * (z_des - pos_curr[2]) - 5.0 * v_curr[2]  # 弱力维持位置
        
        # XY轴跟踪
        pos_ref_xy = np.array([0.5, 0.0])
        vel_ref_xy = np.array([0.0, 0.0])  # 切向速度参考
        
        # 切向运动
        if is_contact and self.contact_t is not None:
            t_contact = t - self.contact_t
            if t_contact >= 3.0:
                # 稳定接触5秒后再开始擦拭
                t_wipe = t_contact - 3.0 - 5.0
                
                if t_wipe >= 0:  # 过渡完成后
                    # 前后擦拭（X轴）：15秒，增加幅度到0.15m
                    if t_wipe < 15.0:
                        cycle_t = t_wipe % 1.0  # 1秒一个周期
                        if cycle_t < 0.5:
                            progress = cycle_t / 0.5
                            pos_ref_xy[0] = 0.5 + 0.15 * np.sin(progress * np.pi)  # 幅度从0.12→0.15
                            vel_ref_xy[0] = 0.15 * np.pi / 0.5 * np.cos(progress * np.pi)  # 速度前馈
                        else:
                            progress = (cycle_t - 0.5) / 0.5
                            pos_ref_xy[0] = 0.5 + 0.15 * np.sin((1.0 - progress) * np.pi)
                            vel_ref_xy[0] = -0.15 * np.pi / 0.5 * np.cos((1.0 - progress) * np.pi)
                    
                    # 左右擦拭（Y轴）：从15秒后开始，幅度0.1m
                    else:
                        cycle_t = (t_wipe - 15.0) % 1.0
                        if cycle_t < 0.5:
                            progress = cycle_t / 0.5
                            pos_ref_xy[1] = 0.0 - 0.1 * np.sin(progress * np.pi)  # 幅度从0.08→0.1
                            vel_ref_xy[1] = -0.1 * np.pi / 0.5 * np.cos(progress * np.pi)
                        else:
                            progress = (cycle_t - 0.5) / 0.5
                            pos_ref_xy[1] = 0.0 - 0.1 * np.sin((1.0 - progress) * np.pi)
                            vel_ref_xy[1] = 0.1 * np.pi / 0.5 * np.cos((1.0 - progress) * np.pi)
        
        err_xy = pos_ref_xy - pos_curr[:2]
        if is_contact and abs(F_curr) > 2.0:  # 接触时使用速度前馈
            # 位置反馈 + 速度前馈
            F_cmd[:2] = 800.0 * err_xy + 2.0 * vel_ref_xy - 3.0 * v_curr[:2]
        else:
            F_cmd[:2] = 200.0 * err_xy - 15.0 * v_curr[:2]
        
        # ========== 转换为关节力矩 ==========
        tau = np.zeros(self.n_joints)
        
        for i in range(self.n_joints):
            dof = self.model.joint(self.joint_names[i]).dofadr
            if len(dof) > 0:
                tau[i] = self.data.qfrc_bias[dof[0]]
        
        tau_op = J.T @ F_cmd
        tau += tau_op
        
        tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
        tau = np.clip(tau, -tau_max, tau_max)
        
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
            print("  0-10s:  接近表面（Z轴快速下降）")
            print("  10-13s: 接触后位置锁定和下降")
            print("  13-30s: 力控 + 前后擦拭 (X方向)")
            print("  30-50s: 力控 + 左右擦拭 (Y方向)")
            print("  50-60s: 收尾（返回待命位置）")
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
