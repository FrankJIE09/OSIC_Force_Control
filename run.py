#!/usr/bin/env python3
"""
快速启动脚本 - OSIC 表面力控仿真
选择要运行的版本
"""

import sys
import os
import subprocess

def print_menu():
    print("\n" + "="*70)
    print("OSIC 表面力控仿真 - 快速启动菜单")
    print("="*70)
    print("\n请选择要运行的版本：\n")
    print("  1. 🎬 实时3D可视化 (推荐)")
    print("     → osic_viewer.py")
    print("     → 打开MuJoCo窗口，实时显示60秒仿真")
    print()
    print("  2. 📊 完整数据版本")
    print("     → osic_full_solution.py")
    print("     → 生成CSV数据+统计信息，无可视化")
    print()
    print("  3. ✅ 基础验证版本")
    print("     → osic_three_phase.py")
    print("     → 快速20秒测试，验证基本功能")
    print()
    print("  4. ❌ 退出")
    print("\n" + "="*70)

def main():
    while True:
        print_menu()
        choice = input("请输入选择 (1-4): ").strip()
        
        scripts = {
            "1": "osic_viewer.py",
            "2": "osic_full_solution.py",
            "3": "osic_three_phase.py",
            "4": None
        }
        
        if choice not in scripts:
            print("\n❌ 无效选择，请重新输入")
            continue
        
        if choice == "4":
            print("\n👋 再见！")
            sys.exit(0)
        
        script = scripts[choice]
        
        print(f"\n⏳ 正在启动 {script}...\n")
        
        try:
            subprocess.run(["python3", script], check=False)
        except KeyboardInterrupt:
            print("\n\n⏸ 仿真已中断")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
        
        print("\n")
        again = input("要继续吗？(y/n): ").strip().lower()
        if again != "y":
            print("\n👋 退出成功！")
            sys.exit(0)

if __name__ == "__main__":
    main()
