#!/usr/bin/env python3
"""
中英文标签切换说明
================================================================================

【问题】
原始图表中的中文标签显示为乱码，因为matplotlib默认配置没有正确支持中文字体。

【解决方案】
提供两个生成脚本，支持中英文标签切换：
  • generate_plots.py    - 9面板完整分析图
  • generate_analysis.py - 6面板轨迹分析图 + 力分布直方图

【使用方法】

1️⃣  生成英文标签图表（推荐，避免乱码）
   cd OSIC_Force_Control
   python3 generate_plots.py      # 9面板完整分析
   python3 generate_analysis.py   # 6面板轨迹分析

   结果：
   ✓ osic_visualization.png - 305K（9面板）
   ✓ osic_tangential_analysis.png - 244K（6面板）  
   ✓ osic_force_distribution.png - 60K（力分布）
   
   特点：
   ✅ 所有系统都支持（无字体依赖）
   ✅ 显示效果清晰
   ✅ 无乱码风险
   ✅ 国际协作友好

2️⃣  生成中文标签图表（需要系统字体支持）
   python3 generate_plots.py --cn    # 9面板完整分析（中文）
   python3 generate_analysis.py --cn # 6面板轨迹分析（中文）
   
   前提条件：
   • Linux/Mac 需安装中文字体（如 SimHei、WenQuanYi 等）
   • Windows 通常已包含中文字体
   
   安装字体示例（Ubuntu/Debian）：
   sudo apt-get install fonts-wqy-microhei
   
   如果仍显示乱码，可使用 fc-cache 更新字体缓存：
   fc-cache -fv

【图表内容】

📊 osic_visualization.png（9面板完整分析）
├─ 1️⃣ 法向力时间序列 - 力随时间变化，绿色背景表示接触状态
├─ 2️⃣ 末端执行器位置 - X（红）Y（绿）Z（蓝）三轴位置
├─ 3️⃣ XY平面轨迹 - 彩色渐变表示时间进度
├─ 4️⃣ XY轴位置变化 - X和Y轴独立分量
├─ 5️⃣ Z轴位置（下降深度） - 机械臂下降过程
├─ 6️⃣ 接触状态时间序列 - 绿色=接触，浅红=未接触
├─ 7️⃣ 接触力分布直方图 - 均值和标准差
├─ 8️⃣ 法向力时间序列（详细） - 与10N目标对比
└─ 9️⃣ 力误差分析 - 到目标力的距离

📊 osic_tangential_analysis.png（6面板轨迹分析）
├─ 1️⃣ 法向力vs时间 - 显示10N目标线
├─ 2️⃣ 接触状态vs时间 - 条形图表示
├─ 3️⃣ X位置vs时间 - 包含接触时点标记
├─ 4️⃣ Y位置vs时间 - 包含接触时点标记
├─ 5️⃣ Z位置vs时间 - 显示接触深度
└─ 6️⃣ XY平面轨迹 - 彩色标记不同阶段

📊 osic_force_distribution.png（力分布直方图）
├─ 左图：接近阶段力分布（橙色）
├─ 中图：接触阶段力分布（绿色）
└─ 右图：总体力分布（蓝色）

【标签对应表】

英文                          中文
─────────────────────────────────────────────
Time (s)                      时间 (s)
Normal Force (N)              法向力 (N)
Position (m)                  位置 (m)
End-Effector Position         末端执行器位置
XY Plane Trajectory           XY平面轨迹
X-axis (approach)             X轴（接近）
Y-axis (front-back/left-right) Y轴（前后/左右）
Z-axis (descent)              Z轴（下降）
Force Time Series             法向力时间序列
Contact Status                接触状态
Force Distribution            力分布
Target: 10N                   目标: 10N
Start                         起点
End                           终点

【技术细节】

matplotlib 中文字体配置：
  • 英文：DejaVu Sans（系统默认）
  • 中文：SimHei（需系统安装）
  • 回退：自动降级到英文（无中文字体时）

字体查询命令：
  fc-list :lang=zh            # 列出所有中文字体
  fc-match SimHei             # 检查SimHei是否可用

【常见问题】

Q: 为什么生成的图表仍然显示乱码？
A: 
  1. 系统未安装中文字体
  2. matplotlib 缓存未更新
  3. 使用了 --cn 但字体不可用
  
  解决方案：使用英文标签（默认设置）
  python3 generate_plots.py    # 自动使用英文

Q: 如何在 SSH 远程连接中生成图表？
A: SSH 连接无需显示，直接运行即可（图表保存为PNG文件）
  ssh user@host
  cd OSIC_Force_Control
  python3 generate_plots.py
  # PNG 文件已保存到服务器

Q: 能否同时显示中英文标签？
A: 不建议。建议统一使用英文标签以保证兼容性和清晰度。

【推荐用法】

✅ 生产环境（无中文字体依赖）：
   python3 generate_plots.py
   python3 generate_analysis.py

✅ 学术报告（需要中文）：
   1. 确认系统已安装中文字体
   2. python3 generate_plots.py --cn
   3. 若显示乱码，改用英文版本

✅ 国际协作：
   python3 generate_plots.py
   python3 generate_analysis.py
   # 英文标签通用于所有地区

【版本信息】

生成脚本版本：1.0（2025-12-03）
支持格式：PNG （DPI=150，高清）
数据来源：osic_with_tangential.csv
matplotlib 依赖：2.1.0+
pandas 依赖：0.20.0+

================================================================================
"""

if __name__ == "__main__":
    print(__doc__)
