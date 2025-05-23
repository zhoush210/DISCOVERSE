# MMK2 导航示例

此目录包含用于控制 MMK2 机器人导航的示例。

## 1. `move_to_point_mujoco.py`

此脚本演示了如何在 MuJoCo 仿真环境中将 MMK2 机器人移动到指定的点 (x, y, theta)。

**功能：**

* 初始化 MuJoCo 模型和数据。
* 使用 `MMK2_Receiver` 获取机器人状态信息，并使用 `MMK2_Controller` 发送命令。
* `MoveToPoint` 类计算到达目标点所需的转向和移动。
  * 它首先将机器人转向目标点。
  * 然后，它将机器人向前移动到目标点。
  * 最后，它将机器人转到所需的最终朝向。

**如何运行：**

确保已安装 MuJoCo 和 `discoverse` 库。在终端中导航到 `discoverse/examples/mmk2/navigation/` 目录并运行：

```bash
python move_to_point_mujoco.py
```

该脚本将执行其 `main()` 函数中定义的移动序列。

## 2. `move_to_point_ros2.py`

此脚本演示了如何使用 ROS 2 将 MMK2 机器人移动到指定的点 (x, y, theta)。

**功能：**

* 初始化 ROS 2 节点。
* 使用 `MMK2_Receiver` 订阅机器人里程计信息，并使用 `MMK2_Controller` 发布控制命令。
* `MoveToPoint` 类：
  * 获取机器人当前的位姿（位置和方向）。
  * 计算朝向目标位置所需的旋转角度。
  * 将机器人移动到目标位置。
  * 计算最终的旋转角度以达到目标朝向。
  * 它会处理机器人已经接近目标位置的情况，此时仅调整朝向。

**先决条件：**

* 正在运行的 ROS 2 环境。
* MMK2 机器人仿真或物理机器人正在运行，并且正在发布/订阅必要的 ROS 2 主题。
* 已安装 `discoverse` 库。

**如何运行：**

1. 配置您的 ROS 2 环境。
2. 导航到 `discoverse/examples/mmk2/navigation/` 目录。
3. 运行脚本：

```bash
python move_to_point_ros2.py
```

该脚本将尝试连接到 ROS 2 网络，获取机器人的当前位姿，然后执行移动到其 `main()` 函数中定义的目标点的操作。目标点可以作为字典或列表提供。
