#!/usr/bin/env python3
"""
生成9面板完整性能分析图 - 中英文支持版本

使用方法:
  python3 generate_plots.py          # 默认使用中文，英文标签
  python3 generate_plots.py --en     # 英文标签（推荐，避免乱码）
  python3 generate_plots.py --cn     # 中文标签（需要系统安装中文字体）
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
            # 尝试使用SimHei（黑体）
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except:
            print("⚠ 中文字体设置失败，使用英文标签")
            language = 'en'
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    return language

def create_full_visualization(csv_file='osic_with_tangential.csv', language='en', output_file=None):
    """生成9面板完整分析图表"""
    
    if not os.path.exists(csv_file):
        print(f"❌ 找不到文件: {csv_file}")
        return False
    
    df = pd.read_csv(csv_file)
    print(f"✓ 已加载 {len(df)} 条数据点")
    
    # 创建多面板图表
    fig = plt.figure(figsize=(16, 12))
    
    # 确定列名
    time_col = 'time' if 'time' in df.columns else 't'
    force_col = 'force_normal' if 'force_normal' in df.columns else 'force'
    
    # 转换为numpy以避免pandas多维索引问题
    time = df[time_col].values
    force = df[force_col].values
    pos_x = df['pos_x'].values
    pos_y = df['pos_y'].values
    pos_z = df['pos_z'].values
    is_contact = df['is_contact'].values
    
    # 定义中英文标签
    labels = {
        'en': {
            'time': 'Time (s)',
            'force': 'Normal Force (N)',
            'position': 'Position (m)',
            'contact': 'Contact',
            'force_timeseries': 'Force Time Series',
            'xyz_position': 'End-Effector Position',
            'xy_trajectory': 'XY Plane Trajectory (color=time)',
            'xy_changes': 'XY Position Changes',
            'z_position': 'Z-Axis Position (Descent Depth)',
            'contact_status': 'Contact Status Time Series',
            'force_distribution': 'Contact Force Distribution',
            'force_detailed': 'Normal Force Time Series (Detailed)',
            'force_error': 'Force Error (10N Target)',
            'title': 'OSIC Surface Force Control Simulation - Complete Performance Analysis\nNullspace Optimization + Tangential Motion (60s)',
            'start': 'Start',
            'end': 'End',
            'target': 'Target',
            'x_axis': 'X-axis (approach)',
            'y_axis': 'Y-axis (front-back/left-right)',
            'z_axis': 'Z-axis (descent)',
            'target_10n': 'Target: 10N',
            'mean': 'Mean',
            'freq': 'Frequency',
            'deviation': 'Deviation (N)',
            'n_points': 'n='
        },
        'cn': {
            'time': '时间 (s)',
            'force': '法向力 (N)',
            'position': '位置 (m)',
            'contact': '接触',
            'force_timeseries': '法向力时间序列',
            'xyz_position': '末端执行器位置',
            'xy_trajectory': 'XY平面轨迹（彩色=时间）',
            'xy_changes': 'XY轴位置变化',
            'z_position': '末端Z轴位置（下降深度）',
            'contact_status': '接触状态时间序列',
            'force_distribution': '接触力分布',
            'force_detailed': '法向力时间序列（详细）',
            'force_error': '力误差（10N目标）',
            'title': 'OSIC 表面力控仿真 - 完整性能分析\n零空间优化 + 切向运动（60秒）',
            'start': '起点',
            'end': '终点',
            'target': '目标',
            'x_axis': 'X轴（接近）',
            'y_axis': 'Y轴（前后/左右）',
            'z_axis': 'Z轴（下降）',
            'target_10n': '目标: 10N',
            'mean': '均值',
            'freq': '频数',
            'deviation': '偏差 (N)',
            'n_points': 'n='
        }
    }
    
    L = labels[language]
    
    # ===== 1. 力随时间 =====
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(time, force, 'b-', linewidth=0.8, alpha=0.7)
    contact_idx = is_contact > 0.5
    ax1.fill_between(time, force.min(), force.max(), 
                      where=contact_idx, alpha=0.15, color='green', label=L['contact'])
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_xlabel(L['time'])
    ax1.set_ylabel(L['force'])
    ax1.set_title(L['force_timeseries'], fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ===== 2. 末端XYZ位置 =====
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(time, pos_x, 'r-', label=L['x_axis'], linewidth=1.5, alpha=0.8)
    ax2.plot(time, pos_y, 'g-', label=L['y_axis'], linewidth=1.5, alpha=0.8)
    ax2.plot(time, pos_z, 'b-', label=L['z_axis'], linewidth=1.5, alpha=0.8)
    ax2.set_xlabel(L['time'])
    ax2.set_ylabel(L['position'])
    ax2.set_title(L['xyz_position'], fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ===== 3. XY平面轨迹 =====
    ax3 = plt.subplot(3, 3, 3)
    scatter = ax3.scatter(pos_x, pos_y, c=time, cmap='viridis', s=2, alpha=0.6)
    ax3.scatter([pos_x[0]], [pos_y[0]], c='green', s=100, marker='o', 
               edgecolors='black', linewidth=2, label=L['start'], zorder=5)
    ax3.scatter([pos_x[-1]], [pos_y[-1]], c='red', s=100, marker='s',
               edgecolors='black', linewidth=2, label=L['end'], zorder=5)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(L['xy_trajectory'], fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3, label=L['time'])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # ===== 4. XY轴随时间变化 =====
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(time, pos_x, 'r-', label='X', linewidth=1.5, alpha=0.8)
    ax4.plot(time, pos_y, 'g-', label='Y', linewidth=1.5, alpha=0.8)
    ax4.set_xlabel(L['time'])
    ax4.set_ylabel(L['position'])
    ax4.set_title(L['xy_changes'], fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ===== 5. Z轴位置 =====
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(time, pos_z, 'b-', linewidth=1.5, alpha=0.8)
    contact_idx = is_contact > 0.5
    ax5.scatter(time[contact_idx], pos_z[contact_idx], c='red', s=1, alpha=0.3, label=L['contact'])
    ax5.fill_between(time, pos_z, alpha=0.2, color='blue')
    ax5.set_xlabel(L['time'])
    ax5.set_ylabel('Z (m)')
    ax5.set_title(L['z_position'], fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # ===== 6. 接触状态指示 =====
    ax6 = plt.subplot(3, 3, 6)
    contact_data = is_contact
    colors = np.where(contact_data > 0.5, 'green', 'lightcoral')
    ax6.bar(time, contact_data, width=0.01, color=colors, alpha=0.7)
    ax6.set_xlabel(L['time'])
    ax6.set_ylabel(L['contact'])
    ax6.set_title(L['contact_status'], fontweight='bold')
    ax6.set_ylim(-0.1, 1.2)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # ===== 7. 力的直方图 =====
    ax7 = plt.subplot(3, 3, 7)
    contact_forces = force[is_contact > 0.5]
    if len(contact_forces) > 0:
        ax7.hist(contact_forces, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        mean_f = np.mean(contact_forces)
        std_f = np.std(contact_forces)
        ax7.axvline(mean_f, color='r', linestyle='--', linewidth=2.5, label=f'{L["mean"]}: {mean_f:.2f}N')
        ax7.axvline(mean_f + std_f, color='orange', linestyle=':', linewidth=2, label=f'±σ: ±{std_f:.2f}N')
        ax7.axvline(mean_f - std_f, color='orange', linestyle=':', linewidth=2)
        ax7.set_xlabel(L['force'])
        ax7.set_ylabel(L['freq'])
        ax7.set_title(f"{L['force_distribution']} ({L['n_points']}{len(contact_forces)})", fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3, axis='y')
    
    # ===== 8. 力与时间 =====
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(time, force, 'b-', linewidth=1, alpha=0.7)
    ax8.axhline(y=10.0, color='g', linestyle='--', linewidth=2, label=L['target_10n'])
    ax8.fill_between(time, force, alpha=0.3, color='blue')
    ax8.set_xlabel(L['time'])
    ax8.set_ylabel(L['force'])
    ax8.set_title(L['force_detailed'], fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # ===== 9. 力误差（与目标10N的偏差） =====
    ax9 = plt.subplot(3, 3, 9)
    force_error = 10.0 - np.abs(force)  # 到10N的距离
    ax9.plot(time, force_error, 'purple', linewidth=0.8, alpha=0.7)
    ax9.fill_between(time, force_error, alpha=0.3, color='purple')
    ax9.axhline(y=0, color='g', linestyle='--', linewidth=2, label=L['target'])
    ax9.set_xlabel(L['time'])
    ax9.set_ylabel(L['deviation'])
    ax9.set_title(L['force_error'], fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(L['title'], fontsize=13, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.subplots_adjust(hspace=0.4, wspace=0.35)
    
    # 保存
    if output_file is None:
        output_file = 'osic_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.4)
    print(f"✓ 完整分析图已保存: {output_file}")
    
    plt.close()
    return True

if __name__ == "__main__":
    # 解析命令行参数
    language = 'en'  # 默认英文，避免乱码
    
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
    print("生成9面板完整性能分析图 - OSIC表面力控仿真")
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
    if create_full_visualization(csv_file, language=language):
        print("\n✓ 图表生成完成！")
        print("  • osic_visualization.png - 9面板完整分析")
    else:
        print("\n❌ 图表生成失败")
        sys.exit(1)
