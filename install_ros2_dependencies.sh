#!/bin/bash
# 安装 ROS2 相关依赖到 conda 环境 osic_force_control

set -e

ENV_NAME="osic_force_control"

echo "=========================================="
echo "安装 ROS2 依赖到 conda 环境: $ENV_NAME"
echo "=========================================="

# 激活 conda 环境
echo "激活 conda 环境: $ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# 检查环境是否存在
if [ $? -ne 0 ]; then
    echo "错误: conda 环境 '$ENV_NAME' 不存在！"
    echo "请先创建环境: conda create -n $ENV_NAME python=3.10"
    exit 1
fi

echo "当前 Python 版本: $(python --version)"
echo "当前 conda 环境: $CONDA_DEFAULT_ENV"

# 检查 ROS2 是否已安装
echo ""
echo "检查 ROS2 安装..."
if [ -f "/opt/ros/humble/setup.bash" ]; then
    ROS2_DISTRO="humble"
    echo "检测到 ROS2 Humble"
elif [ -f "/opt/ros/iron/setup.bash" ]; then
    ROS2_DISTRO="iron"
    echo "检测到 ROS2 Iron"
elif [ -f "/opt/ros/jazzy/setup.bash" ]; then
    ROS2_DISTRO="jazzy"
    echo "检测到 ROS2 Jazzy"
else
    echo "⚠ 未检测到 ROS2 系统安装"
    echo "请先安装 ROS2:"
    echo "  Ubuntu 22.04: sudo apt install ros-humble-desktop"
    echo "  Ubuntu 24.04: sudo apt install ros-iron-desktop"
    ROS2_DISTRO=""
fi

# 如果 ROS2 已安装，source 环境并安装 Python 包
if [ -n "$ROS2_DISTRO" ]; then
    echo ""
    echo "Source ROS2 环境..."
    source /opt/ros/$ROS2_DISTRO/setup.bash
    
    echo ""
    echo "安装 ROS2 Python 包到 conda 环境..."
    # 使用 pip 安装可用的包
    pip install rclpy geometry-msgs || echo "⚠ 某些包可能已通过系统安装"
    
    # 验证安装
    echo ""
    echo "验证安装..."
    python -c "import rclpy; print('✓ rclpy 安装成功')" || echo "✗ rclpy 安装失败"
    python -c "from geometry_msgs.msg import TransformStamped; print('✓ geometry_msgs 安装成功')" || echo "✗ geometry_msgs 安装失败"
    python -c "from tf2_ros import TransformBroadcaster; print('✓ tf2_ros 安装成功')" || echo "✗ tf2_ros 安装失败（可能需要系统安装）"
    
    echo ""
    echo "如果 tf2_ros 安装失败，请运行:"
    echo "  sudo apt install ros-$ROS2_DISTRO-tf2-ros"
else
    echo ""
    echo "跳过 Python 包安装（需要先安装 ROS2 系统包）"
fi

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "注意: 要使用 ROS2 TF 发布功能，您还需要："
echo "1. 安装 ROS2 (Humble 或 Iron 版本)"
echo "2. 在运行脚本前 source ROS2 环境:"
echo "   source /opt/ros/humble/setup.bash  # 或对应版本的路径"
echo ""

