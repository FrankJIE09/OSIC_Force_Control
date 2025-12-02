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
        
        # 初始配置
        qpos0 = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        for i, name in enumerate(self.joint_names):
            dof = self.model.joint(name).dofadr
            if len(dof) > 0:
                self.data.qpos[dof[0]] = qpos0[i]
        
        mujoco.mj_forward(self.model, self.data)
        
        self.contact_t = None
        self.F_integral = 0.0
        self.t_start = None
    
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
        is_contact = False
        
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
                if f[2] > 0.5:
                    is_contact = True
        
        return is_contact, force_z
    
    def control_step(self, t, F_target=10.0, dt=0.002):
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
        if not is_contact:
            phase = 0
            if self.contact_t is None:
                pass
            else:
                self.contact_t = None
        else:
            if self.contact_t is None:
                self.contact_t = t
            
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
            z_des = 0.4 - min(t / 15.0, 1.0) * (0.4 - 0.315)
            F_cmd[2] = 100.0 * (z_des - pos_curr[2]) - 5.0 * v_curr[2]
        
        elif phase == 1:
            F_cmd[2] = 150.0 * (pos_curr[2] - pos_curr[2]) - 4.0 * v_curr[2] + 20.0
        
        elif phase == 2:
            z_des = 0.315 - 0.010 * ((t - self.contact_t - 1.0) / 2.0)
            F_cmd[2] = 40.0 * (z_des - pos_curr[2]) - 3.0 * v_curr[2]
        
        elif phase == 3:
            F_err = F_target - F_curr
            self.F_integral = np.clip(self.F_integral + F_err * dt, -1.0, 1.0)
            F_cmd[2] = -12.0 * F_err - 0.3 * self.F_integral - 2.0 * v_curr[2]
            F_cmd[2] += 2.0 * (0.313 - pos_curr[2])
        
        # XY轴跟踪
        pos_ref_xy = np.array([0.5, 0.0])
        
        # 切向运动
        if is_contact and self.contact_t is not None:
            t_contact = t - self.contact_t
            if t_contact >= 3.0:
                # 前后擦拭
                if t_contact < 20.0:
                    wipe_t = (t_contact - 3.0) % 10.0
                    if wipe_t < 5.0:
                        pos_ref_xy[0] += 0.08 * np.sin((wipe_t / 5.0) * np.pi)
                    else:
                        pos_ref_xy[0] += 0.08 * np.sin((1.0 - (wipe_t - 5.0) / 5.0) * np.pi)
                # 左右擦拭
                elif t_contact < 40.0:
                    wipe_t = (t_contact - 20.0) % 10.0
                    if wipe_t < 5.0:
                        pos_ref_xy[1] -= 0.08 * np.sin((wipe_t / 5.0) * np.pi)
                    else:
                        pos_ref_xy[1] -= 0.08 * np.sin((1.0 - (wipe_t - 5.0) / 5.0) * np.pi)
        
        err_xy = pos_ref_xy - pos_curr[:2]
        if is_contact and F_curr > 1.0:
            F_cmd[:2] = 25.0 * err_xy - 5.0 * v_curr[:2]
        else:
            F_cmd[:2] = 40.0 * err_xy - 8.0 * v_curr[:2]
        
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
