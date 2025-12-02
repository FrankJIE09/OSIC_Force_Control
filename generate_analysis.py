#!/usr/bin/env python3
"""
生成详细分析图 - 英文标签
6面板轨迹分析 + 力分布
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置英文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def create_detailed_analysis(csv_file='osic_with_tangential.csv'):
    """Generate detailed trajectory analysis"""
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return False
    
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} data points")
    
    # Determine column names
    time_col = 'time' if 'time' in df.columns else 't'
    force_col = 'force_normal' if 'force_normal' in df.columns else 'force'
    
    # Convert to numpy arrays
    time = df[time_col].values
    force = df[force_col].values
    pos_x = df['pos_x'].values
    pos_y = df['pos_y'].values
    pos_z = df['pos_z'].values
    is_contact = df['is_contact'].values
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 10))
    
    # ===== Top Row: 3 Trajectory Analysis =====
    
    # 1. Force over trajectory
    ax1 = plt.subplot(2, 3, 1)
    scatter1 = ax1.scatter(range(len(time)), force, c=time, cmap='plasma', s=1, alpha=0.6)
    contact_idx = is_contact > 0.5
    ax1.scatter(np.arange(len(time))[contact_idx], force[contact_idx], 
               c='green', s=0.5, alpha=0.4, label='Contact')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axhline(y=10, color='r', linestyle='--', linewidth=1.5, label='Target')
    ax1.set_xlabel('Sample Index', fontsize=10)
    ax1.set_ylabel('Force (N)', fontsize=10)
    ax1.set_title('Force Evolution During Motion', fontsize=11, fontweight='bold')
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='Time (s)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. 3D trajectory projection (XY + time)
    ax2 = plt.subplot(2, 3, 2)
    scatter2 = ax2.scatter(pos_x, pos_y, c=time, cmap='cool', s=3, alpha=0.7)
    # Add phase markers
    ax2.scatter([pos_x[0]], [pos_y[0]], c='green', s=200, marker='o', 
               edgecolors='black', linewidth=2, label='Start', zorder=5)
    
    # Mark phase transitions
    n_approach = int(10.0 / 0.01)  # 10s
    n_contact = int(13.0 / 0.01)   # 13s
    n_wipe_y = int(30.0 / 0.01)    # 30s
    
    if n_approach < len(pos_x):
        ax2.scatter([pos_x[n_approach]], [pos_y[n_approach]], c='yellow', s=100, 
                   marker='^', edgecolors='black', linewidth=1.5, zorder=4)
    if n_wipe_y < len(pos_x):
        ax2.scatter([pos_x[n_wipe_y]], [pos_y[n_wipe_y]], c='orange', s=100,
                   marker='s', edgecolors='black', linewidth=1.5, zorder=4)
    
    ax2.scatter([pos_x[-1]], [pos_y[-1]], c='red', s=200, marker='s',
               edgecolors='black', linewidth=2, label='End', zorder=5)
    
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_title('XY Trajectory with Phase Transitions', fontsize=11, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Time (s)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 3. Contact depth analysis
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time, pos_z, 'b-', linewidth=1, label='Z Position', alpha=0.8)
    ax3.scatter(time[contact_idx], pos_z[contact_idx], c='red', s=1, alpha=0.3, label='Contact')
    ax3.fill_between(time, pos_z, alpha=0.2, color='blue')
    ax3.axhline(y=0.315, color='g', linestyle='--', linewidth=1.5, label='Target depth')
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Z Position (m)', fontsize=10)
    ax3.set_title('Vertical Depth Over Time', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ===== Bottom Row: 3 Force Distribution Histograms =====
    
    # 4. Overall force histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(force, bins=60, color='steelblue', alpha=0.7, edgecolor='black', label='All data')
    contact_forces = force[is_contact > 0.5]
    if len(contact_forces) > 0:
        ax4.hist(contact_forces, bins=60, color='darkgreen', alpha=0.5, edgecolor='black', label='Contact only')
        mean_f = np.mean(contact_forces)
        ax4.axvline(mean_f, color='r', linestyle='--', linewidth=2.5, label=f'Mean: {mean_f:.2f}N')
    ax4.set_xlabel('Force (N)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Overall Force Distribution', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Lateral (X-Y) motion distribution
    ax5 = plt.subplot(2, 3, 5)
    # Calculate lateral motion amount
    lateral_motion = np.sqrt((pos_x - pos_x[0])**2 + (pos_y - pos_y[0])**2)
    scatter5 = ax5.scatter(time, lateral_motion, c=force, cmap='RdYlGn_r', s=2, alpha=0.7)
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('Lateral Displacement (m)', fontsize=10)
    ax5.set_title('Lateral Motion Trajectory (color=force)', fontsize=11, fontweight='bold')
    cbar5 = plt.colorbar(scatter5, ax=ax5, label='Force (N)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Contact rate over sliding window
    ax6 = plt.subplot(2, 3, 6)
    window_size = 200  # ~2 seconds
    contact_rate = np.convolve(is_contact, np.ones(window_size)/window_size, mode='valid')
    time_window = time[window_size-1:]
    ax6.plot(time_window, contact_rate * 100, 'g-', linewidth=2, alpha=0.8)
    ax6.axhline(y=95, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='95% threshold')
    ax6.fill_between(time_window, contact_rate * 100, alpha=0.3, color='green')
    ax6.set_xlabel('Time (s)', fontsize=10)
    ax6.set_ylabel('Contact Rate (%)', fontsize=10)
    ax6.set_title('Contact Rate (2s sliding window)', fontsize=11, fontweight='bold')
    ax6.set_ylim([0, 105])
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('OSIC Force Control - Detailed Motion Analysis',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_file = 'osic_tangential_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()
    return True

if __name__ == "__main__":
    print("="*70)
    print("OSIC Detailed Analysis Generator - English Version")
    print("="*70 + "\n")
    
    csv_file = 'osic_with_tangential.csv'
    
    if os.path.exists(csv_file):
        print(f"✓ Found data file: {csv_file}\n")
        if create_detailed_analysis(csv_file):
            print("\n✓ Analysis completed!")
    else:
        print("❌ Data file not found!")
        print("Please run osic_full_solution.py first")
