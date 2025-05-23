# MMK2 机器人通信 API 说明

本文档介绍了 `discoverse/examples/mmk2/communication/` 目录下各 Python 脚本中主要类和函数的功能及用法。这些脚本为 MMK2 机器人在 MuJoCo 仿真环境和 ROS2 环境下提供了传感器数据接收和机器人控制的接口。

## 目录结构

- `mujoco_mmk2_receiver_api.py`: MuJoCo 环境下的传感器数据接收器。
- `mujoco_mmk2_controller_api.py`: MuJoCo 环境下的机器人控制器。
- `ros2_mmk2_api.py`: ROS2 环境下的传感器数据接收器和机器人控制器封装。
- `ros2_communication.py`: ROS2 环境下通用的 Topic 订阅和发布节点实现。

## 1. MuJoCo 环境 API

### 1.1 `mujoco_mmk2_receiver_api.py`

#### 1.1.1 类 `MMK2_Receiver`

此类用于在 MuJoCo 仿真环境中接收 MMK2 机器人的传感器数据。

**主要方法:**

- `__init__(self, model: mujoco.MjModel, data: mujoco.MjData, save_dir: Optional[str] = None, debug: bool = False)`
  - 功能: 初始化接收器。
  - 参数:
    - `model`: MuJoCo 模型对象。
    - `data`: MuJoCo 数据对象。
    - `save_dir`: (可选) 数据保存目录，调试模式下使用。
    - `debug`: (可选) 是否启用调试模式，启用后会保存接收到的数据。
- `get_initial_joint_states(self) -> Optional[Dict]`
  - 功能: 获取机器人初始时刻记录的关节状态。
  - 返回: 包含各关节初始位置的字典。
- `get_base_position_orientation(self) -> Dict`
  - 功能: 获取机器人基座的当前位置和朝向（四元数）。
  - 返回: 包含 `position` 和 `orientation` 的字典。
- `get_head_rgb_camera_info(self) -> Optional[Dict]`
  - 功能: 获取头部 RGB 相机的内参等信息。
  - 返回: 包含相机宽度、高度和内参矩阵 `k` 的字典。
- `get_head_depth_camera_info(self) -> Optional[Dict]`
  - 功能: 获取头部深度相机的内参等信息。
  - 返回: 类似 `get_head_rgb_camera_info`。
- `get_left_arm_camera_info(self) -> Optional[Dict]`
  - 功能: 获取左臂相机的内参等信息。
- `get_right_arm_camera_info(self) -> Optional[Dict]`
  - 功能: 获取右臂相机的内参等信息。
- `get_head_depth(self) -> Optional[np.ndarray]`
  - 功能: 获取头部深度图像。
  - 返回: 深度图像 (uint16, 单位 mm)。
- `get_head_depth_rgb(self, min_depth=0.0, max_depth=2.0) -> Optional[np.ndarray]`
  - 功能: 获取头部深度图像，并转换为伪彩色图以便可视化。
  - 返回: BGR 格式的伪彩色深度图。
- `get_head_rgb(self) -> Optional[np.ndarray]`
  - 功能: 获取头部 RGB 图像。
  - 返回: BGR 格式的图像。
- `get_free_camera_rgb(self) -> Optional[np.ndarray]`
  - 功能: 获取 MuJoCo 自由视角的 RGB 图像。
- `get_left_arm_rgb(self) -> Optional[np.ndarray]`
  - 功能: 获取左臂 RGB 图像。
- `get_right_arm_rgb(self) -> Optional[np.ndarray]`
  - 功能: 获取右臂 RGB 图像。
- `get_joint_states(self) -> Optional[Dict]`
  - 功能: 获取机器人所有控制关节的当前状态（位置、速度、力矩）以及基座和末端传感器的信息。
  - 返回: 包含 `positions`, `velocities`, `effort` 等键的字典。
- `get_end_effector_pose(self, arm: str) -> Optional[Dict]`
  - 功能: 获取指定手臂末端执行器（夹爪中心）的位姿。
  - 参数: `arm`: `'l'` (左臂) 或 `'r'` (右臂)。
  - 返回: 包含 `position`, `rotation_matrix`, `euler` 的字典。
- `get_object_pose(self, name: str) -> tuple`
  - 功能: 获取仿真环境中指定名称物体（body 或 geom）的位姿。
  - 参数: `name`: 物体名称。
  - 返回: `(position, quaternion)` 元组。
- `render(self, use_mujoco_viewer: bool = True, use_cv_viewer: bool = True, enable_depth: bool = False) -> None`
  - 功能: 渲染仿真视图。
  - 参数:
    - `use_mujoco_viewer`: 是否使用 MuJoCo 内置查看器。
    - `use_cv_viewer`: 是否使用 OpenCV 窗口显示头部图像。
    - `enable_depth`: 当 `use_cv_viewer` 为 True 时，是否额外显示深度图像。

### 1.2 `mujoco_mmk2_controller_api.py`

#### 1.2.1 类 `MMK2_Controller`

此类用于在 MuJoCo 仿真环境中控制 MMK2 机器人。

**主要方法:**

- `__init__(self, model: mujoco.MjModel, data: mujoco.MjData, receiver: MMK2_Receiver) -> None`
  - 功能: 初始化控制器。
  - 参数:
    - `model`: MuJoCo 模型对象。
    - `data`: MuJoCo 数据对象。
    - `receiver`: `MMK2_Receiver` 对象，用于获取机器人当前状态。
- `set_move_forward_pos(self, relative_position: float) -> None`
  - 功能: 控制机器人底盘前进指定的相对距离。
  - 参数: `relative_position`: 前进距离 (米)。
- `set_move_backward_pos(self, relative_position: float) -> None`
  - 功能: 控制机器人底盘后退指定的相对距离。
  - 参数: `relative_position`: 后退距离 (米)。
- `set_turn_left_angle(self, angle: float) -> None`
  - 功能: 控制机器人底盘向左转动指定的角度。
  - 参数: `angle`: 转动角度 (度)。
- `set_turn_right_angle(self, angle: float) -> None`
  - 功能: 控制机器人底盘向右转动指定的角度。
  - 参数: `angle`: 转动角度 (度)。
- `set_head_slide_pos(self, position: float) -> None`
  - 功能: 设置升降台（躯干）的高度。
  - 参数: `position`: 目标高度。
- `set_head_yaw_angle(self, position: float) -> None`
  - 功能: 设置头部偏航（左右转动）角度。
  - 参数: `position`: 目标偏航角 (弧度)。
- `set_head_pitch_angle(self, position: float) -> None`
  - 功能: 设置头部俯仰（上下点头）角度。
  - 参数: `position`: 目标俯仰角 (弧度)。
- `set_head_position(self, yaw: float, pitch: float) -> None`
  - 功能: 同时设置头部的偏航和俯仰角度。
- `set_left_arm_pos(self, positions: List[float]) -> None`
  - 功能: 设置左臂各关节的目标位置。
  - 参数: `positions`: 包含 6 个关节角度 (弧度) 或 7 个值 (最后为夹爪) 的列表。
- `set_left_arm_gripper_open(self, position: float = GRIPPER_OPEN) -> None`
  - 功能: 打开左手夹爪。
  - 参数: `position`: 夹爪开度，`GRIPPER_OPEN` (0.0) 为完全打开。
- `set_left_arm_gripper_close(self, position: float = GRIPPER_CLOSE) -> None`
  - 功能: 关闭左手夹爪。
  - 参数: `position`: 夹爪开度，`GRIPPER_CLOSE` (1.0) 为完全关闭。
- `set_right_arm_pos(self, positions: List[float]) -> None`
  - 功能: 设置右臂各关节的目标位置。
- `set_right_arm_gripper_open(self, position: float = GRIPPER_OPEN) -> None`
  - 功能: 打开右手夹爪。
- `set_right_arm_gripper_close(self, position: float = GRIPPER_CLOSE) -> None`
  - 功能: 关闭右手夹爪。
- `stop_all(self) -> None`
  - 功能: 停止机器人所有正在进行的运动。
- `reset(self) -> bool`
  - 功能: 将机器人重置到预定义的初始姿态。

## 2. ROS2 环境 API

### 2.1 `ros2_communication.py`

此文件定义了通用的 ROS2 Topic 订阅和发布节点。

#### 类 `TopicSubscriber(Node)`

- 功能: 创建一个 ROS2 节点，用于订阅指定的话题，并将接收到的数据存储或（在调试模式下）保存到文件。
- `__init__(self, save_dir: str = None, topics: List = None, debug: bool = False)`: 初始化订阅者。
  - `topics`: 需要订阅的话题名称列表 (例如 `['head_rgb', 'joint_states']`)。
- 各话题回调函数 (如 `head_rgb_callback`, `joint_states_callback`): 内部使用，处理对应话题的消息。
- `record_data()`: 调试模式下，将收集到的图像序列保存为视频。

#### 类 `TopicPublisher(Node)`

- 功能: 创建一个 ROS2 节点，用于向指定的话题发布消息。
- `__init__(self, save_dir=None, topics=None, debug=False)`: 初始化发布者。
- `publish_cmd_vel(self, linear: float, angular: float) -> None`: 发布底盘运动指令。
- `publish_head_position(self, positions: List[float]) -> None`: 发布头部关节位置指令。
- `publish_left_arm_position(self, positions: List[float]) -> None`: 发布左臂关节位置指令。
- `publish_right_arm_position(self, positions: List[float]) -> None`: 发布右臂关节位置指令。
- `publish_spine_position(self, positions: List[float]) -> None`: 发布升降台（躯干）位置指令。

### 2.2 `ros2_mmk2_api.py`

此文件基于 `ros2_communication.py` 封装了针对 MMK2 机器人的 ROS2 接收器和控制器。

#### 2.2.1 类 `MMK2_Receiver`

- 功能: 通过 ROS2 Topic 接收 MMK2 机器人的传感器数据。内部使用 `TopicSubscriber`。
- `__init__(self, save_dir: Optional[str] = None, topics: List[str] = None, debug: bool = False)`: 初始化。
- `get_initial_joint_states(self) -> Optional[Dict]`: 获取初始关节状态。
- `get_clock(self) -> Optional[str]`: 获取 `/clock` 时间。
- `get_head_rgb_camera_info(self) -> Optional[Dict]`: 获取头部 RGB 相机信息。
- `get_head_depth_camera_info(self) -> Optional[Dict]`: 获取头部深度相机信息。
- `get_left_arm_camera_info(self) -> Optional[Dict]`: 获取左臂相机信息。
- `get_right_arm_camera_info(self) -> Optional[Dict]`: 获取右臂相机信息。
- `get_odom(self) -> Optional[Dict]`: 获取里程计数据。
- `get_taskinfo(self) -> Optional[str]`: 获取任务信息。
- `get_gameinfo(self) -> Optional[str]`: 获取比赛信息。
- `get_head_depth(self) -> Optional[np.ndarray]`: 获取头部深度图。
- `get_head_rgb(self) -> Optional[np.ndarray]`: 获取头部 RGB 图。
- `get_left_arm_rgb(self) -> Optional[np.ndarray]`: 获取左臂 RGB 图。
- `get_right_arm_rgb(self) -> Optional[np.ndarray]`: 获取右臂 RGB 图。
- `get_joint_states(self) -> Optional[Dict]`: 获取关节状态。
- `get_base_position(self) -> Optional[Dict]`: 获取底盘位姿。
- `get_arm_position(self) -> Optional[Dict]`: 获取双臂关节位置。
- `get_head_position(self) -> Optional[Dict]`: 获取头部（升降、俯仰、偏航）位置。
- `stop(self)`: 停止订阅线程并（在调试模式下）保存数据。

#### 2.2.2 类 `MMK2_Controller`

- 功能: 通过 ROS2 Topic 控制 MMK2 机器人。内部使用 `TopicPublisher`。
- `__init__(self, receiver: MMK2_Receiver = None, save_dir: Optional[str] = None, topics: List[str] = None, debug: bool = False)`: 初始化。
  - `receiver`: `MMK2_Receiver` (ROS2版) 对象，用于获取当前状态以实现闭环或平滑控制。
- `reset_all(self) -> bool`: 重置机器人到初始状态。
- `reset_head(self) -> bool`: 重置头部到初始状态。
- `reset_slide(self) -> bool`: 重置升降台到初始状态。
- `reset_arms(self) -> bool`: 重置双臂到初始状态。
- `set_move_forward_position(self, relative_position: float, pos_tolerance: float = 0.01, flush_print: bool = False) -> None`: 控制底盘前进（带反馈）。
- `set_move_backward_position(self, relative_position: float, pos_tolerance: float = 0.01, flush_print: bool = False) -> None`: 控制底盘后退（带反馈）。
- `set_turn_left_angle(self, angle: float, angle_tolerance: float = 0.5, flush_print: bool = False) -> None`: 控制底盘左转（带反馈）。
- `set_turn_right_angle(self, angle: float, angle_tolerance: float = 0.5, flush_print: bool = False) -> None`: 控制底盘右转（带反馈）。
- `set_head_yaw_angle(self, position: float, execute_time_threshold: float = 5.0) -> None`: 平滑设置头部偏航角。
- `set_head_pitch_angle(self, position: float, execute_time_threshold: float = 5.0) -> None`: 平滑设置头部俯仰角。
- `set_head_position(self, pitch: float, yaw: float, execute_time_threshold: float = 5.0) -> None`: 平滑设置头部姿态。
- `set_head_slide_position(self, position: float, execute_time_threshold: float = 5.0) -> None`: 平滑设置升降台高度。
- `set_left_arm_position(self, positions: List[float], execute_time_threshold: float = 10.0) -> None`: 平滑设置左臂关节位置。
- `set_right_arm_position(self, positions: List[float], execute_time_threshold: float = 10.0) -> None`: 平滑设置右臂关节位置。
- `set_arm_position(self, lft_positions: List[float], rgt_positions: List[float], execute_time_threshold: float = 10.0) -> None`: 平滑设置双臂关节位置。
- `set_left_arm_gripper(self, position: float = GRIPPER_OPEN) -> None`: 设置左夹爪开合。
- `set_right_arm_gripper(self, position: float = GRIPPER_OPEN) -> None`: 设置右夹爪开合。
- `stop_all(self) -> None`: 停止底盘运动。

## 3. 使用示例

具体使用方法请参考各脚本中的 `main()` 函数（如果提供）或相关示例代码。

**MuJoCo 环境基本流程:**

1. 加载 MuJoCo 模型 (`model`) 和数据 (`data`)。
2. 实例化 `MMK2_Receiver(model, data)`。
3. 实例化 `MMK2_Controller(model, data, receiver)`。
4. 调用 `receiver` 的 `get_*` 方法获取数据。
5. 调用 `controller` 的 `set_*` 方法控制机器人。
6. 在仿真循环中调用 `mujoco.mj_step(model, data)` 和 `receiver.render()`。

**ROS2 环境基本流程:**

1. 初始化 ROS2 (`rclpy.init()`)。
2. 实例化 `MMK2_Receiver()` (ROS2版)。可能需要一些时间来订阅和接收初始消息。
3. 实例化 `MMK2_Controller(receiver)` (ROS2版)。
4. 调用 `receiver` 的 `get_*` 方法获取数据。
5. 调用 `controller` 的 `set_*` 方法控制机器人。
6. 确保 ROS2 节点在运行 (例如，通过 `rclpy.spin_once` 或在独立线程中 `rclpy.spin`)。
7. 使用完毕后 `rclpy.shutdown()`。

**注意:**

- ROS2 API 中的控制函数（尤其是运动相关的）通常包含平滑处理或闭环逻辑，执行可能需要一定时间。
- 常量（如 `DEFAULT_LINEAR_VEL`, `HEAD_YAW_MIN` 等）定义在各文件开头，用于限制范围或提供默认值。
- 调试模式 (`debug=True`) 会在特定目录下生成日志文件或图像/视频数据。
  