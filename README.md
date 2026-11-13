# HOFA-UAV-Control

**基于高阶全驱系统方法（HOFA）、中间估计器（IE）与模型预测控制（MPC）的四旋翼无人机轨迹跟踪研究。**

本项目是本科毕业论文《无人机系统建模与控制方法研究》的配套代码。项目使用 Python 与 ROS 2 实现控制算法，通过 PX4 Offboard 接口下发控制指令，提供正方形、螺旋上升及扰动注入实验，并配套 PX4 原生控制对照节点、CSV 数据记录与可视化脚本。

## 目录

- [项目功能](#项目功能)
- [控制架构](#控制架构)
- [仓库结构](#仓库结构)
- [运行环境](#运行环境)
- [构建与仿真启动](#构建与仿真启动)
- [实验设置](#实验设置)
- [数据记录与绘图](#数据记录与绘图)
- [实验结果与统计说明](#实验结果与统计说明)

## 项目功能

- **HOFA 位置控制**：通过参考轨迹前馈和极点配置反馈构造期望加速度。
- **扰动估计**：使用 Standard IE 估计复合扰动，并将估计值引入 MPC 预测模型。
- **约束优化**：使用 CVXPY 建模、OSQP 求解有限时域二次规划问题。
- **PX4 接口**：将期望加速度映射为姿态目标，再生成机体角速度和归一化推力指令。
- **对照实验**：通过位置和速度设定点调用 PX4 原生控制链路，与 HOFA 控制节点分别开展实验。
- **实验分析**：记录参考位置和实际位置，绘制轨迹、单轴响应及误差统计图。
- **RViz 显示**：提供位姿、路径与 TF 桥接节点。

## 控制架构

```text
参考轨迹 ──→ 跟踪误差 ──→ IE 扰动估计 ──→ MPC 滚动优化
                ↑                              │
           PX4 状态反馈                         ↓
                ↑                     HOFA 前馈与反馈合成
                │                              ↓
         PX4 角速率内环 ←─ 角速度/推力指令 ←─ 姿态映射
```

HOFA 节点发布 `VehicleRatesSetpoint`，由 PX4 执行底层角速率控制。对照节点通过 `TrajectorySetpoint` 发布位置与速度参考，由 PX4 原生控制链路执行轨迹跟踪。

| 话题 | 消息类型 | 用途 |
| --- | --- | --- |
| `/fmu/out/vehicle_odometry` | `VehicleOdometry` | 位置、速度及姿态反馈 |
| `/fmu/in/offboard_control_mode` | `OffboardControlMode` | Offboard 控制模式 |
| `/fmu/in/vehicle_rates_setpoint` | `VehicleRatesSetpoint` | HOFA 角速度及推力指令 |
| `/fmu/in/trajectory_setpoint` | `TrajectorySetpoint` | PX4 对照组轨迹指令 |
| `/fmu/in/vehicle_command` | `VehicleCommand` | 解锁和模式切换 |

控制节点采用 PX4 的 NED 坐标约定，Z 轴向下为正，因此 `z = -5` 对应相对高度约 5 m。RViz 桥接节点对位置进行 NED 到 ENU 转换。

## 仓库结构

```text
HOFA-UAV-Control/
├── README.md
├── .gitignore
└── src/
    ├── hofa_design/
    │   ├── hofa_design/
    │   │   ├── __init__.py
    │   │   ├── hofa_node.py             # HOFA 正方形轨迹
    │   │   ├── baseline_node.py         # PX4 正方形对照
    │   │   ├── hofa_round.py            # HOFA 螺旋上升轨迹
    │   │   ├── baseline_round.py        # PX4 螺旋上升对照
    │   │   ├── hofa_wind_node.py        # HOFA 扰动注入实验
    │   │   ├── baseline_wind_node.py    # PX4 扰动注入对照
    │   │   ├── plot_result.py           # 正方形实验绘图
    │   │   ├── plot_round.py            # 螺旋实验绘图
    │   │   ├── plot_wind_test.py        # 扰动实验绘图
    │   │   └── px4_rviz_bridge.py       # RViz 位姿与路径桥接
    │   ├── launch/rviz.py
    │   ├── rviz/rviz.rviz
    │   ├── resource/hofa_design
    │   ├── test/
    │   ├── package.xml
    │   ├── setup.cfg
    │   └── setup.py
    └── px4_msgs/                        # Git 子仓库引用
```

## 运行环境

### 仿真平台

| 组件 | 环境配置 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 |
| ROS 2 | Humble |
| 飞控 | PX4 Autopilot 17 / SITL（作者确认） |
| 仿真器 | Gazebo，具体版本待补充 |
| 仿真机型 | `gz_x500` |
| 通信桥接 | Micro XRCE-DDS Agent，UDP 8888 |
| 地面站 | QGroundControl |
| 算法依赖 | Python、NumPy、SciPy、CVXPY、OSQP |
| 数据分析 | pandas、Matplotlib |
| 可视化 | RViz2 |

Python、科学计算库和 Agent 的实际测试版本尚未锁定。`package.xml` 与 `setup.py` 当前仅声明了部分运行依赖。

## 构建与仿真启动

以下步骤面向已安装 ROS 2 Humble、colcon、PX4 仿真环境及 Micro XRCE-DDS Agent 的 Ubuntu 系统。示例沿用作者的 `~/grad_design_ws` 工作空间名称。

### 1. 获取代码并补齐依赖

```bash
git clone https://github.com/Ma0nesir/HOFA-UAV-Control.git ~/grad_design_ws
cd ~/grad_design_ws
source /opt/ros/humble/setup.bash
```

如果该目录已存在，请使用其他空目录或现有检出目录。继续构建前，需要补齐 `src/px4_msgs` 并确认其与所用 PX4 固件匹配。

在 rosdep 已初始化的环境中安装已声明的依赖：

```bash
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install numpy scipy cvxpy osqp pandas matplotlib
```

上述 Python 安装命令未锁定版本，仅说明所需依赖。可视化还依赖 `rviz2`、`tf2_ros`、`geometry_msgs`、`nav_msgs`、`launch_ros` 和 `ament_index_python`，应确认这些组件已安装。

### 2. 首次构建

```bash
cd ~/grad_design_ws
colcon build --symlink-install
source install/setup.bash
```

依赖构建完成后，如需更新控制包，可使用命令：

```bash
colcon build --packages-select hofa_design
source install/setup.bash
```

### 3. 启动 PX4 与 Gazebo

在终端 A 中执行：

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500
```

该启动目标来自作者实验记录，使用其他 PX4 版本时需核对其是否支持。

### 4. 启动 DDS Agent 和地面站

在终端 B 中执行并保持运行：

```bash
MicroXRCEAgent udp4 -p 8888
```

打开 QGroundControl，确认已连接仿真飞控。在已加载 ROS 2 环境的终端中检查状态反馈：

```bash
ros2 topic list
ros2 topic info /fmu/out/vehicle_odometry
```

### 5. 启动 RViz（可选）

为显示起飞开始的路径，可在控制节点启动前，在独立终端中执行：

```bash
source /opt/ros/humble/setup.bash
cd ~/grad_design_ws
source install/setup.bash
ros2 launch hofa_design rviz.py
```

桥接节点发布 `/rviz/uav_pose` 和 `/rviz/uav_path`。当前 RViz 配置的 Path 显示订阅 `/drone_path`，需在界面中改为 `/rviz/uav_path`；固定坐标系为 `map`。

### 6. 启动控制节点

在终端 C 中加载环境：

```bash
source /opt/ros/humble/setup.bash
cd ~/grad_design_ws
source install/setup.bash
```

从下表选择一个节点运行：

| 实验 | HOFA + IE + MPC | PX4 对照组 |
| --- | --- | --- |
| 正方形 | `ros2 run hofa_design hofa_node` | `ros2 run hofa_design baseline_node` |
| 螺旋上升 | `ros2 run hofa_design hofa_round` | `ros2 run hofa_design baseline_round_node` |
| 扰动注入 | `ros2 run hofa_design hofa_wind_node` | `ros2 run hofa_design baseline_wind_node` |

每次只运行一个控制节点。对照实验应分别进行，并重置仿真状态。

> **运行注意：**当前节点会自动发送解锁和 Offboard 切换指令，默认轨迹涉及 5–15 m 高度。已检查的 HOFA 节点未实现完整的自动降落流程；停止节点不等于执行降落。上述流程仅用于仿真。

## 实验设置

| 场景 | 当前代码中的参考轨迹 |
| --- | --- |
| 正方形 | 4 s 起飞至 5 m，悬停至 6 s；沿边长 5 m 的正方形运动，每边 4 s；22 s 后悬停 |
| 螺旋上升 | 起飞至 5 m 后盘旋；半径参数 3 m、角速度 0.3 rad/s、爬升速度 0.15 m/s、高度上限 15 m；包含半径渐入过程 |
| 扰动注入 | 采用螺旋参考轨迹，轨迹计时超过 8 s 后注入确定性的时变指令扰动 |

主要共同参数：

| 参数 | 默认值 |
| --- | --- |
| 控制定时周期 | 0.02 s，名义频率 50 Hz |
| 模型质量 | 2.0 kg |
| 重力加速度 | 9.81 m/s² |
| 悬停推力基准 | 0.57 |
| 极点配置参数 | 0.95 |
| MPC 预测步数 | 5 |
| MPC 求解器 | OSQP，启用 warm start |

50 Hz 是定时器设定频率，实际求解耗时和控制周期仍需运行日志验证。各场景的 MPC 权重与滤波系数存在差异，参数目前直接定义在 Python 文件中。

### 扰动实验的适用范围

当前 HOFA 节点在期望加速度进入姿态映射之前叠加扰动；PX4 对照节点将扰动写入 `TrajectorySetpoint.acceleration`。两组使用的扰动幅值也不同。

因此，当前实现应描述为“指令侧扰动注入实验”，不能直接等同于两组飞行器受到相同 Gazebo 物理风场。严格的抗扰性能对比需要统一扰动幅值、作用位置及实验条件。

## 数据记录与绘图

### CSV 日志

控制节点将日志写入启动进程的当前工作目录，同名文件会被覆盖。按上述流程运行时，日志位于 `~/grad_design_ws`。

| 场景 | HOFA 日志 | PX4 日志 |
| --- | --- | --- |
| 正方形 | `hofa_square.csv` | `baseline_square.csv` |
| 螺旋上升 | `hofa_round.csv` | `baseline_round.csv` |
| 扰动注入 | `hofa_round_windy.csv` | `baseline_round_wind.csv` |

CSV 字段为：

```text
timestamp,ref_x,act_x,ref_y,act_y,ref_z,act_z
```

`timestamp` 是根据控制步数计算的轨迹时间，单位为秒；位置单位为米。日志不包含求解耗时、扰动估计或全部控制输出。

### 绘图脚本

绘图脚本未注册为 ROS 2 命令行入口，需从工作空间根目录直接运行：

```bash
python3 src/hofa_design/hofa_design/plot_result.py
python3 src/hofa_design/hofa_design/plot_round.py
python3 src/hofa_design/hofa_design/plot_wind_test.py
```

运行前必须核对以下路径设置：

| 脚本 | 当前数据及输出目录 | 图像文件 |
| --- | --- | --- |
| `plot_result.py` | 硬编码 `/home/xu/grad_design_ws` | `square_*.png` |
| `plot_round.py` | `~/grad_design_ws` | `round_*.png` |
| `plot_wind_test.py` | 当前工作目录 | `wind_*.png` |

如果工作空间位置不同，请先修改脚本中的目录变量。中文显示异常时，需要安装脚本指定的中文字体或调整字体配置。

### PX4 飞行日志

启用日志记录后，可通过 QGroundControl 获取 PX4 ULog，并使用 [PX4 Flight Review](https://logs.px4.io/upload) 分析。CSV 与 ULog 的记录内容不同，建议保存原始日志及对应的代码提交和飞控参数。

## 实验结果与统计说明

### 论文报告的仿真结果

下表来自论文初稿及作者提供的对比图，尚未依据原始 CSV 独立复算。

| 场景 | HOFA + MPC 平均 RMSE / m | PX4 Baseline 平均 RMSE / m |
| --- | ---: | ---: |
| 正方形轨迹 | 0.057 | 0.206 |
| 螺旋上升轨迹 | 0.067 | 0.107 |
| 风扰场景 | 0.071 | 0.236 |

这里的 Average RMSE 为三个单轴 RMSE 的算术平均：

```text
Average RMSE = (RMSE_x + RMSE_y + RMSE_z) / 3
```
