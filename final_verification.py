#!/usr/bin/env python3
"""
最终验证：运动摇晃幅度测量
"""

import numpy as np
import mujoco
from osic_viewer import OSICViewer

print("\n" + "="*100)
print("✅ 最终验证：末端摇晃运动")
print("="*100 + "\n")

sim = OSICViewer()

# 快速跑到t=10s（运动开始时间约）
for step in range(5000):
    t = step * 0.002
    sim.control_step(t)
    mujoco.mj_step(sim.model, sim.data)

# 记录运动数据
print(f"【运动摇晃周期数据】（t=10-15秒，高采样率0.02s）")
print(f"{'时间(s)':<10} {'末端x':<12} {'末端y':<12} {'末端z':<12} {'状态':<15}")
print("-" * 70)

x_positions = []
y_positions = []
z_positions = []
times = []

for step in range(5000, 7500):  # 额外运行5秒（到15s）
    t = step * 0.002
    
    if step % 10 == 0:  # 0.02s采样
        end_effector = sim.model.body("panda_hand")
        pos_curr = sim.data.xpos[end_effector.id]
        
        x_positions.append(pos_curr[0])
        y_positions.append(pos_curr[1])
        z_positions.append(pos_curr[2])
        times.append(t)
        
        if t >= 10.0 and t <= 15.0:
            status = "运动中" if sim.contact_t is not None else "准备中"
            print(f"{t:<10.2f} {pos_curr[0]:<12.6f} {pos_curr[1]:<12.6f} {pos_curr[2]:<12.6f} {status:<15}")
    
    sim.control_step(t)
    mujoco.mj_step(sim.model, sim.data)

# 计算统计
if len(x_positions) > 2500:
    x_array = np.array(x_positions[-2500:])  # 最后2500个点（t=10-15s）
    y_array = np.array(y_positions[-2500:])
    z_array = np.array(z_positions[-2500:])
else:
    x_array = np.array(x_positions)
    y_array = np.array(y_positions)
    z_array = np.array(z_positions)

print("-" * 70)
print(f"\n【运动幅度统计】（t=10-15秒）")
print(f"X轴：")
print(f"  最小值: {x_array.min():.6f}m")
print(f"  最大值: {x_array.max():.6f}m")
print(f"  摇晃幅度（p2p）: {x_array.max() - x_array.min():.6f}m")
print(f"  目标摇晃幅度: 0.30m (±0.15m)")

print(f"\nY轴：")
print(f"  最小值: {y_array.min():.6f}m")
print(f"  最大值: {y_array.max():.6f}m")
print(f"  摇晃幅度（p2p）: {y_array.max() - y_array.min():.6f}m")

print(f"\nZ轴（接触稳定性）：")
print(f"  最小值: {z_array.min():.6f}m")
print(f"  最大值: {z_array.max():.6f}m")
print(f"  变化范围: {z_array.max() - z_array.min():.6f}m（应该<0.01）")

if (x_array.max() - x_array.min()) > 0.05:
    print(f"\n✅ 成功！末端在摇晃，幅度 {(x_array.max() - x_array.min()):.3f}m")
else:
    print(f"\n⚠️  运动幅度较小，可能需要进一步调整参数")

print("\n" + "="*100 + "\n")
