# OSIC_Force_Control Project Structure

## 📁 File Overview

### Core Simulation Programs
- **`osic_viewer.py`** ⭐ (8.1 KB)
  - Main program with real-time 3D visualization
  - Opens MuJoCo viewer window
  - Shows contact points and force vectors
  - **Recommended for visualization**

- **`osic_full_solution.py`** (14 KB)
  - Complete implementation with null-space optimization
  - Generates CSV data output
  - Outputs 6001 data points over 60 seconds
  - For post-processing analysis

- **`osic_three_phase.py`** (7.5 KB)
  - Baseline verification version
  - Quick 20-second test
  - Validates basic three-phase force control

### Configuration Files
- **`surface_force_control.xml`** (2.1 KB)
  - MuJoCo model configuration
  - Defines robot, surface, and contact properties

- **`panda.xml`** (7.0 KB)
  - Franka Panda robot model definition
  - 7-DOF robotic arm specification

### Visualization & Analysis
- **`osic_visualization.png`** (273 KB) ⭐
  - 9-panel complete performance analysis
  - All metrics in English (no character encoding issues)
  - Regenerated 2025-12-02

- **`osic_tangential_analysis.png`** (245 KB)
  - 6-panel detailed motion analysis
  - Trajectory visualization with phase transitions
  - Contact rate statistics

- **`osic_with_tangential.csv`** (593 KB)
  - Complete simulation data (6001 samples)
  - Columns: time, pos_x, pos_y, pos_z, force_normal, is_contact
  - Raw data for custom analysis

### Helper Scripts
- **`generate_plots.py`** (9.5 KB) ✨ New!
  - Regenerates 9-panel visualization with English labels
  - No character encoding issues
  - Use this to recreate plots if needed

- **`generate_analysis.py`** (6.7 KB) ✨ New!
  - Generates detailed 6-panel analysis
  - Trajectory and force distribution analysis
  - Use this for in-depth post-processing

- **`run.py`** (1.9 KB) ✨ New!
  - Interactive menu launcher
  - Select which version to run
  - Beginner-friendly interface

### Documentation
- **`README.md`** (9.3 KB)
  - Complete project documentation
  - Technical details and parameter explanations
  - Troubleshooting guide

---

## 🚀 Quick Start Guide

### Option 1: Interactive Menu (Easiest)
```bash
python3 run.py
```
Choose which version to run from the menu.

### Option 2: Direct Launch
```bash
# Real-time 3D visualization (recommended)
python3 osic_viewer.py

# Complete data version
python3 osic_full_solution.py

# Quick baseline test
python3 osic_three_phase.py
```

### Option 3: Regenerate Plots
```bash
# Regenerate with English labels (no encoding issues)
python3 generate_plots.py      # Main 9-panel analysis
python3 generate_analysis.py   # Detailed 6-panel analysis
```

---

## 📊 Key Performance Metrics

| Metric | Value |
|--------|-------|
| **Contact Rate** | 97.0% |
| **Duration** | 60 seconds |
| **Avg. Force** | -5.71 N |
| **Force Std Dev** | 2.54 N |
| **X-axis range** | 0.3070-0.5093 m |
| **Y-axis range** | -0.0013-0.0026 m |
| **Z-axis range** | 0.3865-0.6175 m |

---

## 🔧 System Requirements

- **Python** 3.8+
- **MuJoCo** 2.3.0+
- **NumPy** 1.19+
- **Pandas** (for CSV analysis)
- **Matplotlib** (for visualization)

### Installation
```bash
pip install mujoco numpy pandas matplotlib
```

---

## 📝 File Generation History

### Latest Changes (2025-12-02)
- ✅ Regenerated PNG files with English labels
- ✅ Fixed Chinese character encoding issues
- ✅ Created `generate_plots.py` for reproducibility
- ✅ Created `generate_analysis.py` for detailed analysis
- ✅ Updated README with new commands

### Previous Versions
- Original PNG files had Chinese labels (乱码 issues)
- CSV data unchanged (already properly formatted)
- Core simulation code unchanged

---

## 💡 Usage Tips

1. **For Visualization**: Run `osic_viewer.py` (requires display/X11)
2. **For Data Analysis**: Run `osic_full_solution.py` then analyze CSV
3. **For Plots**: Use `generate_plots.py` to recreate visualizations
4. **For SSH Sessions**: Use data version, not viewer
5. **For Batch Processing**: Use CSV output and custom scripts

---

## 🐛 Troubleshooting

**Q: No display/cannot run viewer?**
```bash
# Use SSH with X11 forwarding
ssh -X user@host
python3 osic_viewer.py

# Or use data version
python3 osic_full_solution.py
```

**Q: Plots look corrupted or have encoding errors?**
```bash
# Regenerate with English labels
python3 generate_plots.py
python3 generate_analysis.py
```

**Q: ModuleNotFoundError: mujoco**
```bash
pip install mujoco
```

---

## 📚 Related Files

Located in parent directory `/home/zhq/OrangeSpace/panda_mujoco-master/`:
- Various experimental versions
- Additional analysis scripts
- Physics simulation logs

Main project is self-contained in this folder.

---

## 🎯 Next Steps

1. **Run the simulation**: `python3 run.py`
2. **View results**: Open PNG files or watch 3D animation
3. **Analyze data**: Use CSV file with Pandas/Matlab
4. **Customize**: Modify parameters in Python source code
5. **Extend**: Add your own control logic

---

**Project Last Updated**: 2025-12-02  
**Status**: ✅ Complete and Working
