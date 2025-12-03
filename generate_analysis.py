#!/usr/bin/env python3
"""
生成6面板轨迹分析图 - 中英文支持版本

使用方法:
  python3 generate_analysis.py        # 默认英文标签
  python3 generate_analysis.py --en   # 英文标签（推荐）
  python3 generate_analysis.py --cn   # 中文标签（需要系统安装中文字体）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

def setup_font(language='en'):
    """设置matplotlib字体支持"""
    if language == 'cn':
        try:
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except:
            print("⚠ 中文字体设置失败，使用英文标签")
            language = 'en'
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    return language

def create_tangential_analysis(csv_file='osic_with_tangential.csv', language='en'):
    """生成6面板轨迹分析图表"""
    
    if not os.path.exists(csv_file):
        print(f"❌ 找不到文件: {csv_file}")
        return False
    
    df = pd.read_csv(csv_file)
    print(f"✓ 已加载 {len(df)} 条数据点")
    
    # 定义中英文标签
    labels = {
        'en': {
            'time': 'Time (s)',
            'force': 'Normal Force (N)',
            'contact': 'Contact Status',
            'x_position': 'X Position (m)',
            'y_position': 'Y Position (m)',
            'z_position': 'Z Position (m)',
            'xy_trajectory': 'XY Plane Trajectory',
            'phase_0_1': 'Phase 0-1: Approach & Lock',
            'phase_2_3': 'Phase 2-3: Stabilize & Wipe',
            'force_vs_time': 'Force vs Time',
            'contact_vs_time': 'Contact Status vs Time',
            'x_vs_time': 'End-Effector X Position vs Time',
            'y_vs_time': 'End-Effector Y Position vs Time',
            'z_vs_time': 'End-Effector Z Position vs Time',
            'xy_plane': 'End-Effector XY Plane Trajectory',
            'force_distribution': 'Force Distribution',
            'force_target': 'Force Target (10N)',
            'contact_time': 'Contact Duration',
            'approach': 'Approach Phase',
            'contact_phase': 'Contact Phase',
            'start': 'Start',
            'end': 'End',
            'target': 'Target',
            'freq': 'Frequency',
            'mean': 'Mean',
            'std': 'Std',
            'title': 'OSIC Simulation - Trajectory Analysis\nNullspace Optimization + Tangential Motion',
            'axis_x': 'X (m)',
            'axis_y': 'Y (m)'
        },
        'cn': {
            'time': '时间 (s)',
            'force': '法向力 (N)',
            'contact': '接触状态',
            'x_position': 'X位置 (m)',
            'y_position': 'Y位置 (m)',
            'z_position': 'Z位置 (m)',
            'xy_trajectory': 'XY平面轨迹',
            'phase_0_1': '阶段0-1：接近和锁定',
            'phase_2_3': '阶段2-3：稳定和擦拭',
            'force_vs_time': '力vs时间',
            'contact_vs_time': '接触状态vs时间',
            'x_vs_time': '末端X位置vs时间',
            'y_vs_time': '末端Y位置vs时间',
            'z_vs_time': '末端Z位置vs时间',
            'xy_plane': '末端XY平面轨迹',
            'force_distribution': '力分布',
            'force_target': '力目标 (10N)',
            'contact_time': '接触时长',
            'approach': '接近阶段',
            'contact_phase': '接触阶段',
            'start': '起点',
            'end': '终点',
            'target': '目标',
            'freq': '频数',
            'mean': '均值',
            'std': '标准差',
            'title': 'OSIC仿真 - 轨迹分析\n零空间优化 + 切向运动',
            'axis_x': 'X (m)',
            'axis_y': 'Y (m)'
        }
    }
    
    L = labels[language]
    
    # 确定列名
    time_col = 'time' if 'time' in df.columns else 't'
    force_col = 'force_normal' if 'force_normal' in df.columns else 'force'
    
    time = df[time_col].values
    force = df[force_col].values
    pos_x = df['pos_x'].values
    pos_y = df['pos_y'].values
    pos_z = df['pos_z'].values
    is_contact = df['is_contact'].values
    
    # 创建6面板图表
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # ===== Panel 1: Force vs Time =====
    ax = axes[0, 0]
    contact_idx = is_contact > 0.5
    ax.plot(time, force, 'b-', linewidth=1, alpha=0.7)
    ax.axhline(y=10, color='r', linestyle='--', label=L['force_target'])
    ax.fill_between(time, force.min(), force.max(), 
                     where=contact_idx, alpha=0.15, color='green', label=L['contact'])
    ax.set_xlabel(L['time'])
    ax.set_ylabel(L['force'])
    ax.set_title(L['force_vs_time'], fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ===== Panel 2: Contact Status vs Time =====
    ax = axes[0, 1]
    colors = np.where(is_contact > 0.5, 'green', 'lightcoral')
    ax.bar(time, is_contact, width=0.02, color=colors, alpha=0.7)
    ax.set_xlabel(L['time'])
    ax.set_ylabel(L['contact'])
    ax.set_title(L['contact_vs_time'], fontweight='bold')
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ===== Panel 3: X Position vs Time =====
    ax = axes[1, 0]
    ax.plot(time, pos_x, 'r-', linewidth=1, alpha=0.8)
    ax.scatter(time[contact_idx], pos_x[contact_idx], c='red', s=1, alpha=0.3)
    ax.set_xlabel(L['time'])
    ax.set_ylabel(L['x_position'])
    ax.set_title(L['x_vs_time'], fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # ===== Panel 4: Y Position vs Time =====
    ax = axes[1, 1]
    ax.plot(time, pos_y, 'g-', linewidth=1, alpha=0.8)
    ax.scatter(time[contact_idx], pos_y[contact_idx], c='green', s=1, alpha=0.3)
    ax.set_xlabel(L['time'])
    ax.set_ylabel(L['y_position'])
    ax.set_title(L['y_vs_time'], fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # ===== Panel 5: Z Position vs Time =====
    ax = axes[2, 0]
    ax.plot(time, pos_z, 'b-', linewidth=1, alpha=0.8)
    ax.axhline(y=0.315, color='k', linestyle='--', alpha=0.5, label=L['contact_time'])
    ax.scatter(time[contact_idx], pos_z[contact_idx], c='blue', s=1, alpha=0.3)
    ax.fill_between(time, pos_z, alpha=0.2, color='blue')
    ax.set_xlabel(L['time'])
    ax.set_ylabel(L['z_position'])
    ax.set_title(L['z_vs_time'], fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ===== Panel 6: XY Plane Trajectory =====
    ax = axes[2, 1]
    # Plot different phases with different colors
    ax.plot(pos_x[~contact_idx], pos_y[~contact_idx], 'o-', color='orange', 
           alpha=0.5, markersize=2, label=L['approach'])
    ax.plot(pos_x[contact_idx], pos_y[contact_idx], 's-', color='green', 
           alpha=0.8, markersize=3, label=L['contact_phase'])
    ax.plot(0.5, 0.0, 'r*', markersize=15, label=L['target'], zorder=5)
    ax.set_xlabel(L['axis_x'])
    ax.set_ylabel(L['axis_y'])
    ax.set_title(L['xy_plane'], fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.suptitle(L['title'], fontsize=13, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    # 保存
    output_file = 'osic_tangential_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.4)
    print(f"✓ 轨迹分析图已保存: {output_file}")
    
    plt.close()
    
    # 生成力分布直方图
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    contact_forces = force[is_contact > 0.5]
    phases = [
        (is_contact == 0, '0-1: Approach' if language == 'en' else '0-1: 接近'),
        (is_contact > 0.5, '2-3: Contact' if language == 'en' else '2-3: 接触'),
    ]
    
    for idx, (mask, phase_name) in enumerate(phases[:2]):
        ax = axes[idx]
        forces_phase = force[mask]
        if len(forces_phase) > 0:
            ax.hist(forces_phase, bins=50, color=['orange', 'green'][idx], alpha=0.7, edgecolor='black')
            ax.set_xlabel(L['force'])
            ax.set_ylabel(L['freq'])
            ax.set_title(f'{phase_name}\n{L["mean"]}: {forces_phase.mean():.2f}N ± {forces_phase.std():.2f}N',
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    # Combined distribution
    ax = axes[2]
    if len(contact_forces) > 0:
        ax.hist(contact_forces, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        mean_f = np.mean(contact_forces)
        std_f = np.std(contact_forces)
        ax.axvline(mean_f, color='r', linestyle='--', linewidth=2, label=f'{L["mean"]}: {mean_f:.2f}N')
        ax.set_xlabel(L['force'])
        ax.set_ylabel(L['freq'])
        ax.set_title(f'{L["force_distribution"]}\n{L["mean"]}: {mean_f:.2f}N ± {std_f:.2f}N',
                    fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Force Distribution Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.subplots_adjust(wspace=0.3)
    
    dist_file = 'osic_force_distribution.png'
    plt.savefig(dist_file, dpi=150, bbox_inches='tight', pad_inches=0.4)
    print(f"✓ 力分布分析图已保存: {dist_file}")
    
    plt.close()
    return True

if __name__ == "__main__":
    # 解析命令行参数
    language = 'en'  # 默认英文
    
    if len(sys.argv) > 1:
        if '--cn' in sys.argv:
            language = 'cn'
        elif '--en' in sys.argv:
            language = 'en'
    
    # 设置字体
    language = setup_font(language)
    # 防止文字重影
    matplotlib.rcParams['text.antialiased'] = True
    matplotlib.rcParams['figure.max_open_warning'] = 0
    
    print("="*70)
    print("生成6面板轨迹分析图 - OSIC表面力控仿真")
    print("="*70)
    print(f"使用语言: {'中文' if language == 'cn' else '英文'}\n")
    
    # 检查数据文件
    csv_file = 'osic_with_tangential.csv'
    if os.path.exists(csv_file):
        print(f"✓ 发现数据文件: {csv_file}\n")
    else:
        print(f"❌ 找不到数据文件: {csv_file}")
        sys.exit(1)
    
    # 生成可视化
    if create_tangential_analysis(csv_file, language=language):
        print("\n✓ 轨迹分析完成！")
        print("  • osic_tangential_analysis.png - 6面板轨迹分析")
        print("  • osic_force_distribution.png - 力分布直方图")
    else:
        print("\n❌ 轨迹分析失败")
        sys.exit(1)
