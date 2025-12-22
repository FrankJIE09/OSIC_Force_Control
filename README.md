# 机器人控制仿真 | Robot Control Simulation

本项目包含两种控制方法的实现：
- **阻抗控制 (Impedance Control)** - 空间任务空间控制
- **混合力位控制 (Hybrid Force/Position Control)** - 表面力控制与圆形擦拭任务

## 🎬 演示视频

### 阻抗控制演示

![阻抗控制演示](阻抗控制.gif)

> 上方 GIF 展示了阻抗控制（空间任务空间控制）的完整仿真演示过程。如需查看高清视频，请打开 `阻抗控制.mkv`

### 混合力位控制演示

![力位混合控制演示](力位混合控制.gif)

> 上方 GIF 展示了混合力位控制的完整仿真演示过程。如需查看高清视频，请打开 `力位混合控制.mkv`

---

## 📋 项目简介

基于 MuJoCo 物理引擎的机器人控制仿真系统，实现了两种主要的控制方法：

### 1. 阻抗控制 (Impedance Control)
- ✅ 空间任务空间控制 (Spatial Task Space Control)
- ✅ 所有计算（误差、雅可比、动力学）均在世界坐标系下进行
- ✅ 空间速度控制和配置误差计算
- ✅ 实时可视化与 ROS2 TF 发布支持

### 2. 混合力位控制 (Hybrid Force/Position Control)
- ✅ 在空间坐标系中实现力位混合控制
- ✅ 圆形擦拭任务 - XY 平面位置控制 + Z 轴力控制
- ✅ 6维力/力矩传感器 - 实时测量接触 wrench
- ✅ 投影矩阵控制 - 基于任务空间质量矩阵的投影控制
- ✅ 实时可视化 - MuJoCo Viewer 3D 显示

---

## 🚀 快速开始

### 环境要求

```bash
# 安装依赖
pip install -r requirements.txt
```

**依赖包：**
- `mujoco >= 2.3.0` - 物理仿真引擎
- `numpy >= 1.19` - 数值计算
- `scipy` - 科学计算（用于旋转变换）

### 运行仿真

#### 1. 阻抗控制仿真

```bash
# 运行阻抗控制程序（实时可视化）
python3 impedance_control_simulation.py
```

**功能说明：**
- 打开 MuJoCo 3D 显示窗口
- 实时渲染机械臂运动
- 空间任务空间控制
- 支持 ROS2 TF 发布（可选）

#### 2. 混合力位控制仿真

```bash
# 运行混合力位控制程序（实时可视化）
python3 hybrid_force_position_control_simulation.py
```

**功能说明：**
- 打开 MuJoCo 3D 显示窗口
- 实时渲染机械臂运动
- 显示接触力和接触状态
- 关闭窗口自动停止

**演示流程：**
```
0-2s:    接近阶段 - 移动到工作表面上方
2-4s:    下降阶段 - 缓慢下降到接触表面
4s+:     擦拭阶段 - XY 平面画圆，Z 轴力控维持恒定压力
```

---

## 📁 项目结构

```
OSIC_Force_Control/
├── 阻抗控制.gif                  # ⭐ 阻抗控制演示动画（GIF）
├── 阻抗控制.mkv                  # 阻抗控制演示视频（高清原版）
├── 力位混合控制.gif              # ⭐ 混合力位控制演示动画（GIF）
├── 力位混合控制.mkv              # 混合力位控制演示视频（高清原版）
│
├── impedance_control_simulation.py              # ⭐ 阻抗控制主程序
├── hybrid_force_position_control_simulation.py # ⭐ 混合力位控制主程序
├── dynamics_calculator_wv.py                   # 动力学计算器
│
├── surface_force_control_disk.xml # MuJoCo 模型配置（带圆盘）
├── surface_force_control.xml      # MuJoCo 模型配置
├── panda_with_disk.xml           # Franka Panda 机器人模型（带圆盘）
├── panda.xml                     # Franka Panda 机器人模型
│
├── hybrid_force_position_control_spatial_frame.tex  # 理论文档
├── wipe_table_simulation_wv_explanation.tex         # 代码说明文档
├── force_error_coordinate_transformation.tex        # 坐标变换理论
├── control_parameters_tuning_guide.tex              # 参数调优指南
│
├── mesh/                          # 3D 模型文件
├── texture/                       # 纹理文件
│
├── requirements.txt               # Python 依赖
├── environment.yml                # Conda 环境配置
└── README.md                      # 本文件
```

---

## 🔧 技术细节

### 1. 阻抗控制 (Impedance Control)

#### 控制架构

阻抗控制实现了基于**空间坐标系 (Spatial Frame)** 的任务空间控制：

**控制律：**
```
F_s = Λ_s(θ) d/dt(V_d) + K_p X_e + K_d V_e + η_s(θ, V_s)
τ = J_s^T(θ) F_s
```

其中：
- `V_s`: 当前空间速度 (Spatial Velocity)
- `X_e = log(X_d X^{-1})`: 空间配置误差 (Spatial Twist)
- `V_e = V_d - V_s`: 空间速度误差
- `Λ_s, η_s`: 空间坐标系下的动力学矩阵
- `J_s`: 空间雅可比矩阵

**控制参数：**
```python
K_p = diag([100, 100, 100, 2000, 2000, 2000])  # 空间刚度
K_d = diag([10, 10, 10, 500, 500, 500])        # 空间阻尼
K_i = diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])     # 积分增益
```

**特点：**
- 所有计算在世界坐标系下进行
- 使用空间雅可比矩阵 `J_s`
- 支持前馈加速度控制
- 可选的 ROS2 TF 发布功能

---

### 2. 混合力位控制 (Hybrid Force/Position Control)

#### 控制架构

#### 混合力位控制原理

本项目实现了基于**空间坐标系 (Spatial Frame)** 的混合力位控制：

1. **投影矩阵计算**
   ```
   P_s = I - A_s^T (A_s Λ_s^{-1} A_s^T)^{-1} A_s Λ_s^{-1}
   ```
   - `A_s`: 约束矩阵，定义力控方向
   - `Λ_s`: 任务空间质量矩阵
   - `P_s`: 投影矩阵，将运动控制投影到非约束方向

2. **运动控制力**
   ```
   F_motion = P_s Λ_s (K_p X_e + K_d V_e + K_i ∫X_e)
   ```
   - 在非约束方向上使用位置控制
   - XY 平面：位置跟踪（画圆）
   - 旋转方向：姿态保持

3. **力控制力**
   ```
   F_force = (I - P_s) (F_d + K_fp F_e + K_fi ∫F_e)
   ```
   - 在约束方向上使用力控制
   - Z 轴：维持恒定压力 (-30N)
   - 旋转方向：保持零力矩

4. **总控制力**
   ```
   F_total = F_motion + F_force + η_s
   τ = J_s^T F_total
   ```
   - `η_s`: 科里奥利和离心力项
   - `J_s`: 空间雅可比矩阵

#### 控制参数

**运动控制增益：**
```python
K_p = diag([20, 20, 20, 100, 100, 100]) * 5  # 位置/姿态比例增益
K_d = diag([2, 2, 2, 10, 10, 10]) * 5        # 速度/角速度微分增益
K_i = diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])   # 积分增益
```

**力控制增益：**
```python
K_fp = diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) * 2  # 力比例增益
K_fi = diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # 力积分增益
F_desired = -30.0  # 期望压力 (N)
```

#### 约束矩阵定义

在接触状态下，约束矩阵 `A_s` 定义为：
```python
A_s = [
    [1, 0, 0, 0, 0, 0],  # wx 方向（力矩控制）
    [0, 1, 0, 0, 0, 0],  # wy 方向（力矩控制）
    [0, 0, 0, 0, 0, 0],  # wz 方向（自由）
    [0, 0, 0, 0, 0, 1],  # fz 方向（力控制）
]
```

这意味着：
- **wx, wy**: 力矩控制（保持零力矩）
- **wz**: 自由旋转（位置控制）
- **fx, fy**: 位置控制（XY 平面画圆）
- **fz**: 力控制（维持恒定压力）

### 物理模型参数

#### 机械臂
- **型号:** Franka Panda (7-DOF)
- **最大力矩:** ±87 N·m (前4个关节), ±12 N·m (后3个关节)
- **初始配置:** `[0, -0.785, 0, -2.356, 0, 1.571, 0.785]` rad

#### 仿真环境
- **时间步长:** 0.002s (500Hz)
- **积分器:** Euler
- **求解器:** Newton
- **求解迭代:** 50次

#### 接触表面
- **几何:** 0.4m × 0.4m × 0.02m 盒子
- **位置:** z = 0.15m
- **摩擦系数:** 0.8
- **接触模型:** 软接触

---

## 📊 关键特性

### ✅ 已实现功能

#### 阻抗控制功能

1. **空间任务空间控制**
   - 空间坐标系下的完整动力学计算
   - 空间雅可比矩阵计算
   - 空间配置误差和速度误差计算
   - 前馈加速度控制

2. **ROS2 集成**
   - TF 变换发布（可选）
   - 实时位姿和参考位姿广播
   - 多线程安全设计

#### 混合力位控制功能

3. **混合力位控制**
   - 空间坐标系中的投影矩阵控制
   - 运动控制和力控制的解耦
   - 基于任务空间质量矩阵的动态补偿

4. **6维力/力矩传感器**
   - 实时测量接触 wrench
   - 传感器坐标系到世界坐标系的变换
   - 低通滤波处理

5. **圆形擦拭轨迹**
   - XY 平面圆形运动（半径 0.02m）
   - Z 轴恒定压力控制（-30N）
   - 平滑的轨迹生成

6. **坐标变换**
   - 伴随变换矩阵（Adjoint Transformation）
   - Wrench 在不同坐标系间的转换
   - 传感器数据到控制坐标系的映射

#### 通用功能

7. **力矩限制与安全**
   - 关节力矩限幅（±87 N·m）
   - 超限报警机制
   - 零空间阻尼稳定

8. **实时可视化**
   - MuJoCo Viewer 集成
   - 3D 机械臂模型显示
   - 实时状态监控

### 🔮 未来改进方向

- [ ] 自适应力控增益调整
- [ ] 更复杂的擦拭轨迹（螺旋、8字形等）
- [ ] 切向力控制（摩擦约束）
- [ ] 多接触点支持
- [ ] 参数自动调优
- [ ] 数据记录与分析工具

---

## 📚 理论文档

项目包含详细的理论文档（LaTeX 格式）：

1. **`hybrid_force_position_control_spatial_frame.tex`**
   - 混合力位控制理论基础
   - 投影矩阵推导
   - 空间坐标系中的控制公式

2. **`wipe_table_simulation_wv_explanation.tex`**
   - 代码实现详解
   - 算法流程说明
   - 关键参数解释

3. **`force_error_coordinate_transformation.tex`**
   - 力误差坐标变换
   - 伴随变换矩阵理论
   - Wrench 变换公式

4. **`control_parameters_tuning_guide.tex`**
   - 控制参数调优指南
   - 稳定性分析
   - 性能优化建议

---

## 🐛 常见问题

### Q1: 运行时看不到窗口
**A:** MuJoCo viewer 需要显示服务。如在 SSH 环境：
```bash
# 使用 X11 转发
ssh -X user@host
python3 impedance_control_simulation.py
# 或
python3 hybrid_force_position_control_simulation.py
```

### Q2: 接触力达不到期望值
**A:** 调整以下参数：
- 增加力控增益 `K_fp` 和 `K_fi`
- 调整期望力 `F_desired_val`
- 检查接触检测阈值

### Q3: 运行报错 "No module named 'mujoco'"
**A:** 安装 MuJoCo：
```bash
pip install mujoco
```

### Q4: 力矩超限警告
**A:** 这是正常的保护机制。如果频繁出现：
- 降低控制增益
- 减小运动速度
- 检查轨迹规划是否合理

---

## 📝 使用示例

### 修改阻抗控制参数

编辑 `impedance_control_simulation.py` 的 `setup_control_parameters` 方法：
```python
self.K_p = np.diag([100.0, 100.0, 100.0, 2000.0, 2000.0, 2000.0])  # 空间刚度
self.K_d = np.diag([10.0, 10.0, 10.0, 500.0, 500.0, 500.0])        # 空间阻尼
```

### 修改混合力位控制期望力

编辑 `hybrid_force_position_control_simulation.py` 第 83 行：
```python
self.F_desired_val = -30.0  # 改为其他值，如 -20.0
```

### 调整圆形轨迹

编辑 `hybrid_force_position_control_simulation.py` 第 170-201 行的 `generate_wipe_trajectory` 方法：
```python
center = np.array([0.55, 0.0])  # 圆心位置
radius = 0.02                    # 半径 (m)
freq = 1.0                       # 角频率 (rad/s)
```

### 修改控制增益

编辑对应程序的 `setup_control_parameters` 方法中的增益参数。

---

## 🛠 系统要求

### 依赖包
```bash
numpy >= 1.19
mujoco >= 2.3.0
scipy
```

### 版本要求
- Python 3.8+
- MuJoCo 2.3.0+
- NumPy 1.19+

### 硬件要求
- **CPU:** Intel/AMD 双核以上
- **内存:** 4GB+
- **显卡:** 可选（集成 GPU/独立显卡均支持）

---

## 📧 支持与反馈

如遇到问题或有改进建议，欢迎反馈！

**已知限制：**
- 当前实现针对圆形擦拭任务优化
- 力控参数需要根据具体任务调整
- 力矩限制为安全保护机制

---

## 📄 许可证

本项目遵循 MuJoCo 开源许可。

---

## 🙏 致谢

基于开源项目：
- **MuJoCo** - DeepMind Physics Engine
- **Franka Panda** - Open Source Manipulation Platform

---

**最后更新:** 2025年1月
