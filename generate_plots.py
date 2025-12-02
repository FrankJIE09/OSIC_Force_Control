#!/usr/bin/env python3
"""
生成可视化图表 - 英文标签（避免中文乱码）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置英文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def create_visualization_en(csv_file='osic_with_tangential.csv'):
    """Generate visualization with English labels"""
    
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
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # ===== 1. Force vs Time =====
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(time, force, 'b-', linewidth=0.8, alpha=0.7)
    contact_idx = is_contact > 0.5
    ax1.fill_between(time, force.min(), force.max(), 
                      where=contact_idx, alpha=0.15, color='green', label='Contact')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Force (N)', fontsize=10)
    ax1.set_title('Normal Force Time Series', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ===== 2. End-effector XYZ Position =====
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(time, pos_x, 'r-', label='X (front-back)', linewidth=1.5, alpha=0.8)
    ax2.plot(time, pos_y, 'g-', label='Y (left-right)', linewidth=1.5, alpha=0.8)
    ax2.plot(time, pos_z, 'b-', label='Z (up-down)', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Position (m)', fontsize=10)
    ax2.set_title('End-Effector Position', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ===== 3. XY Plane Trajectory =====
    ax3 = plt.subplot(3, 3, 3)
    scatter = ax3.scatter(pos_x, pos_y, c=time, cmap='viridis', s=2, alpha=0.6)
    ax3.scatter([pos_x[0]], [pos_y[0]], c='green', s=100, marker='o', 
               edgecolors='black', linewidth=2, label='Start', zorder=5)
    ax3.scatter([pos_x[-1]], [pos_y[-1]], c='red', s=100, marker='s',
               edgecolors='black', linewidth=2, label='End', zorder=5)
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    ax3.set_title('XY Plane Trajectory (color=time)', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3, label='Time (s)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # ===== 4. X and Y Axes =====
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(time, pos_x, 'r-', label='X axis', linewidth=1.5, alpha=0.8)
    ax4.plot(time, pos_y, 'g-', label='Y axis', linewidth=1.5, alpha=0.8)
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Position (m)', fontsize=10)
    ax4.set_title('Lateral Axis Motion', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ===== 5. Z Axis (Depth) =====
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(time, pos_z, 'b-', linewidth=1.5, alpha=0.8)
    contact_idx = is_contact > 0.5
    ax5.scatter(time[contact_idx], pos_z[contact_idx], c='red', s=1, alpha=0.3, label='Contact')
    ax5.fill_between(time, pos_z, alpha=0.2, color='blue')
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('Z Position (m)', fontsize=10)
    ax5.set_title('Depth Penetration', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # ===== 6. Contact State =====
    ax6 = plt.subplot(3, 3, 6)
    contact_data = is_contact
    colors = np.where(contact_data > 0.5, 'green', 'lightcoral')
    ax6.bar(time, contact_data, width=0.01, color=colors, alpha=0.7)
    ax6.set_xlabel('Time (s)', fontsize=10)
    ax6.set_ylabel('Contact State', fontsize=10)
    ax6.set_title('Contact Detection', fontsize=11, fontweight='bold')
    ax6.set_ylim(-0.1, 1.2)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # ===== 7. Force Distribution Histogram =====
    ax7 = plt.subplot(3, 3, 7)
    contact_forces = force[is_contact > 0.5]
    if len(contact_forces) > 0:
        ax7.hist(contact_forces, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        mean_f = np.mean(contact_forces)
        std_f = np.std(contact_forces)
        ax7.axvline(mean_f, color='r', linestyle='--', linewidth=2.5, label=f'Mean: {mean_f:.2f}N')
        ax7.axvline(mean_f + std_f, color='orange', linestyle=':', linewidth=2, label=f'Std: +/-{std_f:.2f}N')
        ax7.axvline(mean_f - std_f, color='orange', linestyle=':', linewidth=2)
        ax7.set_xlabel('Force (N)', fontsize=10)
        ax7.set_ylabel('Frequency', fontsize=10)
        ax7.set_title(f'Force Distribution (n={len(contact_forces)})', fontsize=11, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3, axis='y')
    
    # ===== 8. Force Detail =====
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(time, force, 'b-', linewidth=1, alpha=0.7)
    ax8.axhline(y=10.0, color='g', linestyle='--', linewidth=2, label='Target: 10N')
    ax8.fill_between(time, force, alpha=0.3, color='blue')
    ax8.set_xlabel('Time (s)', fontsize=10)
    ax8.set_ylabel('Force (N)', fontsize=10)
    ax8.set_title('Force Detail', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # ===== 9. Force Error =====
    ax9 = plt.subplot(3, 3, 9)
    force_error = 10.0 - np.abs(force)
    ax9.plot(time, force_error, 'purple', linewidth=0.8, alpha=0.7)
    ax9.fill_between(time, force_error, alpha=0.3, color='purple')
    ax9.axhline(y=0, color='g', linestyle='--', linewidth=2, label='Target: 10N')
    ax9.set_xlabel('Time (s)', fontsize=10)
    ax9.set_ylabel('Error (N)', fontsize=10)
    ax9.set_title('Force Error (vs 10N Target)', fontsize=11, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('OSIC Surface Force Control Simulation - 60 Second Complete Analysis',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_file = 'osic_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()
    return True

def print_statistics_en(csv_file='osic_with_tangential.csv'):
    """Print statistics with English labels"""
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    # Determine column names
    time_col = 'time' if 'time' in df.columns else 't'
    force_col = 'force_normal' if 'force_normal' in df.columns else 'force'
    
    time = df[time_col].values
    force = df[force_col].values
    is_contact = df['is_contact'].values
    
    print("\n" + "="*70)
    print("OSIC Force Control Simulation - Statistics Report")
    print("="*70)
    
    # Overall statistics
    print("\n[OVERALL STATISTICS]")
    print(f"  Total Duration: {time.max():.2f}s")
    print(f"  Data Points: {len(df)} samples")
    print(f"  Timestep: {time[1] - time[0]:.4f}s")
    
    # Contact statistics
    contact_points = df[is_contact > 0.5]
    contact_rate = len(contact_points) / len(df) * 100
    print(f"\n[CONTACT STATISTICS]")
    print(f"  Contact Duration: {contact_points[time_col].max() - contact_points[time_col].min():.2f}s")
    print(f"  Contact Points: {len(contact_points)} samples")
    print(f"  Contact Rate: {contact_rate:.1f}%")
    
    # Force statistics
    contact_forces = contact_points[force_col].values
    
    print(f"\n[ALL FORCE DATA]")
    print(f"  Maximum: {np.max(force):7.2f}N")
    print(f"  Minimum: {np.min(force):7.2f}N")
    print(f"  Mean: {np.mean(force):7.2f}N")
    print(f"  Std Dev: {np.std(force):7.2f}N")
    
    if len(contact_forces) > 0:
        print(f"\n[CONTACT FORCE DATA]")
        print(f"  Maximum: {np.max(contact_forces):7.2f}N")
        print(f"  Minimum: {np.min(contact_forces):7.2f}N")
        print(f"  Mean: {np.mean(contact_forces):7.2f}N")
        print(f"  Std Dev: {np.std(contact_forces):7.2f}N")
        print(f"  Target: 10.00N")
        print(f"  Error: {abs(np.mean(contact_forces) - 10.0):.2f}N")
    
    # Position statistics
    print(f"\n[END-EFFECTOR POSITION]")
    print(f"  X Range: [{df['pos_x'].min():.4f}, {df['pos_x'].max():.4f}]m (span: {df['pos_x'].max()-df['pos_x'].min():.4f}m)")
    print(f"  Y Range: [{df['pos_y'].min():.4f}, {df['pos_y'].max():.4f}]m (span: {df['pos_y'].max()-df['pos_y'].min():.4f}m)")
    print(f"  Z Range: [{df['pos_z'].min():.4f}, {df['pos_z'].max():.4f}]m (span: {df['pos_z'].max()-df['pos_z'].min():.4f}m)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print("="*70)
    print("OSIC Visualization Generator - English Version")
    print("="*70 + "\n")
    
    csv_file = 'osic_with_tangential.csv'
    
    if os.path.exists(csv_file):
        print(f"✓ Found data file: {csv_file}\n")
        if create_visualization_en(csv_file):
            print("\n✓ Visualization completed!\n")
            print_statistics_en(csv_file)
    else:
        print("❌ Data file not found!")
        print("Please run osic_full_solution.py first")
