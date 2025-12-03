# 图表生成和中英文标签说明

## 概述

本项目包含两个**图表自动生成脚本**，支持中英文标签切换。解决了原始图表中文标签显示乱码的问题。

## 快速开始

### ✅ 推荐方式：英文标签（无需额外配置）

```bash
# 生成9面板完整分析图
python3 generate_plots.py

# 生成6面板轨迹分析图 + 力分布直方图  
python3 generate_analysis.py
```

输出文件：
- `osic_visualization.png` - 9面板完整分析（305K）
- `osic_tangential_analysis.png` - 6面板轨迹分析（244K）
- `osic_force_distribution.png` - 力分布直方图（60K）

### 🌐 可选方式：中文标签（需要系统字体）

```bash
python3 generate_plots.py --cn
python3 generate_analysis.py --cn
```

前置条件：系统已安装中文字体

## 文件说明

### `generate_plots.py` - 9面板完整分析

生成 `osic_visualization.png`，包含：

| 位置 | 内容 | 说明 |
|------|------|------|
| 左上 | 法向力时间序列 | 蓝线+绿色接触背景 |
| 中上 | 末端XYZ位置 | 红绿蓝三色 |
| 右上 | XY平面轨迹 | 彩色表示时间 |
| 左中 | XY轴独立分量 | X和Y分别显示 |
| 中中 | Z轴下降过程 | 显示接触过程 |
| 右中 | 接触状态 | 条形图 |
| 左下 | 力分布直方图 | 均值±标准差 |
| 中下 | 力详细曲线 | 与10N目标 |
| 右下 | 力误差分析 | 偏差曲线 |

### `generate_analysis.py` - 轨迹分析和力分布

生成两个文件：

1. **osic_tangential_analysis.png** - 6面板轨迹分析
   - 力vs时间、接触状态、X/Y/Z位置、XY平面轨迹

2. **osic_force_distribution.png** - 力分布直方图
   - 接近阶段、接触阶段、总体分布

### `LABELS_README.py` - 详细说明文档

查看中英文标签对应、技术细节、常见问题等。

```bash
python3 LABELS_README.py
```

## 标签对应表

| 英文 | 中文 |
|------|------|
| Time (s) | 时间 (s) |
| Normal Force (N) | 法向力 (N) |
| Position (m) | 位置 (m) |
| Contact | 接触 |
| Force Time Series | 法向力时间序列 |
| Contact Status | 接触状态 |
| End-Effector Position | 末端执行器位置 |
| XY Plane Trajectory | XY平面轨迹 |
| Start | 起点 |
| End | 终点 |
| Target: 10N | 目标: 10N |

## 为什么推荐英文标签？

✅ **兼容性好** - 所有系统都支持  
✅ **无字体依赖** - 开箱即用  
✅ **清晰无乱码** - 专业显示效果  
✅ **国际协作友好** - 全球通用  
❌ **中文标签需要** - 系统字体安装、版本匹配、缓存更新

## 中文字体配置（可选）

### Linux/Mac - 安装中文字体

```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei fonts-simhei

# Fedora/RHEL
sudo dnf install wqy-microhei-fonts

# macOS
brew install font-wqy-microhei
```

### 更新字体缓存

```bash
fc-cache -fv
```

### 验证字体安装

```bash
fc-list :lang=zh           # 列出所有中文字体
fc-match SimHei            # 检查SimHei是否可用
```

### Windows

通常已预装中文字体，可直接使用 `--cn` 选项

## 常见问题

**Q: 为什么我的中文标签还是显示乱码？**

A: 
1. 系统可能未安装中文字体
2. matplotlib 缓存未更新
3. 使用了 `--cn` 但字体路径不对

**解决**：回到英文标签（默认设置）

```bash
python3 generate_plots.py    # 自动使用英文
```

**Q: 在 SSH 远程连接中能生成图表吗？**

A: 可以！SSH 无需显示服务，直接保存PNG文件

```bash
ssh user@host
cd OSIC_Force_Control
python3 generate_plots.py
# osic_visualization.png 已保存到服务器
scp -r user@host:OSIC_Force_Control/*.png ./
```

**Q: 如何选择中文还是英文标签？**

A: 根据场景选择：
- **生产环境** → 英文（无依赖）
- **学术报告** → 中文（确认字体后）
- **国际协作** → 英文（通用）

## 脚本API

### 生成9面板图表

```python
from generate_plots import create_full_visualization

# 英文标签
create_full_visualization('osic_with_tangential.csv', language='en')

# 中文标签
create_full_visualization('osic_with_tangential.csv', language='cn')
```

### 生成轨迹分析

```python
from generate_analysis import create_tangential_analysis

# 英文标签
create_tangential_analysis('osic_with_tangential.csv', language='en')

# 中文标签
create_tangential_analysis('osic_with_tangential.csv', language='cn')
```

## 输出文件格式

所有PNG文件均为：
- **分辨率** - 150 DPI（高清）
- **格式** - PNG（无损压缩）
- **大小** - 60-305 KB
- **兼容性** - 所有主流图像查看器

## 更新日志

**v1.0** (2025-12-03)
- ✅ 支持中英文标签切换
- ✅ 9面板完整分析脚本
- ✅ 6面板轨迹分析脚本
- ✅ 自动字体降级（无中文字体时）
- ✅ 详细文档和API

---

**推荐用法**：`python3 generate_plots.py && python3 generate_analysis.py`

**快速查看说明**：`python3 LABELS_README.py`
