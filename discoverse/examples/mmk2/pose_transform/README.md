# Pose Transform 功能说明

该 `pose_transform` 模块提供了一系列用于 MMK2 机器人位姿变换和运动学计算的工具。主要包含以下几个部分：

## 核心组件

1. **`utils/airbot_play/airbot_play_ik_nopin.py`**:
    * 实现了 Airbot Play 机械臂的逆运动学 (IK) 解析解。
    * `AirbotPlayIK_nopin` 类：
        * `properIK(pos, ori, ref_q=None)`: 计算给定末端位姿（位置 `pos` 和旋转矩阵 `ori`）的关节角度。`ref_q` 可用于在多个解中选择最接近的解。
        * `inverseKin(pos, ori, ref_q=None)`: 内部调用的逆运动学核心算法。
        * `add_bias(angle)`: 为关节角度添加偏置。
        * `move_joint6_2_joint5(pos, ori)`: 将第六关节的坐标转换到第五关节。

2. **`utils/mmk2/mmk2_fk.py`**:
    * 实现了 MMK2 机器人的正向运动学 (FK)，依赖 MuJoCo 进行仿真计算。
    * `MMK2FK` 类：
        * 初始化时加载 MMK2 的 MJCF 模型。
        * `set_base_pose(position, orientation)`: 设置机器人基座的位姿。
        * `set_slide_joint(joint_angle)`: 设置滑动关节的角度。
        * `set_head_joints(joint_angles)`: 设置头部关节的角度。
        * `set_left_arm_joints(joint_angles)`: 设置左臂关节的角度。
        * `set_right_arm_joints(joint_angles)`: 设置右臂关节的角度。
        * `get_head_camera_world_pose()`: 获取头部相机在世界坐标系下的位姿。
        * `get_left_arm_camera_world_pose()`: 获取左臂相机在世界坐标系下的位姿。
        * `get_right_arm_camera_world_pose()`: 获取右臂相机在世界坐标系下的位姿。
        * `get_left_endeffector_world_pose()`: 获取左臂末端执行器在世界坐标系下的位姿。
        * `get_right_endeffector_world_pose()`: 获取右臂末端执行器在世界坐标系下的位姿。
        * 内部方法如 `_forward_kinematics`, `_get_site_transform` 等用于 MuJoCo 的数据更新和信息提取。

3. **`utils/mmk2/mmk2_fik.py`**:
    * 实现了 MMK2 机器人的正向和逆向运动学集成。
    * `MMK2FIK` 类：
        * 包含机器人各部件之间的固定变换矩阵 (如 `TMat_footprint2chest`, `TMat_endpoint2camera` 等)。
        * 包含预定义的抓取动作旋转矩阵 (`action_rot`)。
        * `__init__`: 初始化时创建 `AirbotPlayIK_nopin` 实例用于手臂IK计算。
        * `get_arm_joint_pose_wrt_arm_base(point3d, action, arm, q_ref=None, action_rot=np.eye(3))`: 计算在**机械臂基座坐标系**下，给定目标点 `point3d` 和动作 `action` (字符串或旋转矩阵) 的机械臂关节角度。
        * `get_arm_joint_pose_wrt_base(point3d, action, arm, slide, q_ref=np.zeros(6), action_rot=np.eye(3))`: 计算在**机器人整体基座坐标系 (footprint)** 下，给定目标点 `point3d`、动作 `action`、手臂 `arm` 和滑动关节值 `slide` 的机械臂关节角度。

## 主要脚本

1. **`pose_transform.py`**:
    * 提供了一个通用的函数 `get_relative_pose(model, data, body1_name, body2_name)`。
    * 该函数用于在 MuJoCo 环境中计算两个物体 (body) 之间的相对位姿变换矩阵。
    * `main` 函数演示了如何加载模型并计算相机到末端执行器的相对位姿。

2. **`pose_transform_mujoco.py`**:
    * 定义了 `PoseTransform` 类，封装了在 MuJoCo 环境中进行位姿变换和运动学计算的功能。
    * `PoseTransform` 类：
        * `__init__(model, data)`: 初始化时接收 MuJoCo 的模型和数据。
        * `get_relative_pose(...)`: 与 `pose_transform.py` 中的功能类似，用于计算相对位姿。
        * `get_T_OC(object_name, camera_name)`: 计算物体相对于相机的位姿变换 T_OC。
        * `get_T_CB(camera_name, base_name)`: 计算相机相对于机械臂基座的位姿变换 T_CB。
        * `get_T_OB(object_name, base_name, camera_name=None)`: 计算物体相对于机械臂基座的位姿变换 T_OB。如果提供了相机名称，则通过 T_OB = T_CB @ T_OC 计算；否则直接计算。
        * `get_arm_joint_pose(pose3d, action, arm, q_ref=None, action_rot=np.eye(3))`: 根据物体在机械臂基座坐标系下的3D坐标 `pose3d` 和动作，计算关节角度。调用 `MMK2FIK` 的功能。
        * `get_arm_joint_pose_full(pose6d, arm, q_ref=None, action_rot=np.eye(3))`: 根据物体在机械臂基座坐标系下的完整6D位姿 (4x4矩阵) 计算关节角度。**慎用，可能有奇异解**。
        * `get_arm_joint_pose_from_object(...)`: 直接根据物体名称、基座名称等信息计算关节角度。
    * `main` 函数提供了使用示例，包括加载 `GraspAppleTask` 环境，进行各种位姿变换计算和逆运动学求解，并保存渲染图像。

3. **`pose_transform_ros2.py`**:
    * 定义了 `PoseTransform` 类，用于在 ROS2 环境下进行 MMK2 机器人的位姿变换和运动学计算。
    * `PoseTransform` 类：
        * `__init__(receiver: MMK2_Receiver)`: 初始化时接收一个 `MMK2_Receiver` 对象，用于从 ROS2 获取机器人状态。
        * `_update_pose_fk()`: 从 `MMK2_Receiver` 更新机器人当前的关节状态和基座姿态，并设置到内部的 `MMK2FK` 实例中。
        * `get_arm_joint_pose(target_position, arm_action, arm, q_ref=None, action_rot=np.eye(3))`: 根据目标在**机器人基座坐标系**下的位置 `target_position`、动作和手臂选择，计算关节角度。它会从 `receiver` 获取当前的滑轨位置，并调用 `MMK2FIK` 的 `get_arm_joint_pose_wrt_base`。
        * `get_world_position_from_head_camera(position_in_camera)`: 将头部相机坐标系下的点转换到世界坐标系。
        * `transform_position_wrt_camera_to_base(position_in_camera)`: 将头部相机坐标系下的点转换到机器人基座坐标系。
        * `transform_position_wrt_base_to_world(position_in_base)`: 将机器人基座坐标系下的点转换到世界坐标系。
        * `transform_position_wrt_world_to_base(position_in_world)`: 将世界坐标系下的点转换到机器人基座坐标系。
    * `main` 函数演示了如何初始化 ROS2节点、`MMK2_Receiver` 和 `PoseTransform`，并进行坐标变换测试。

## 仿真环境示例

* **`envs/grasp_apple.py`**:
  * 定义了 `GraspAppleTask` 类，这是一个基于 MuJoCo 的仿真环境，用于模拟 MMK2 抓取苹果的任务。
  * `GraspAppleTask` 类：
    * `__init__(obj_name, base_name, cam_name, arm, action)`: 初始化环境，加载 `grasp_apple.xml` 模型。
    * `reset()`: 重置环境，随机放置苹果的位置。
    * `render()`: 启动 MuJoCo 的被动查看器进行渲染。
    * `save_render_image(save_path)`: 将当前场景渲染为图像并保存。
    * `Base_Cam()`: 计算相机在机械臂基座坐标系下的位姿 T_BC (等价于 `pose_transform_mujoco.py` 中的 `get_T_CB`)。
    * `Cam_Obj()`: 计算物体在相机坐标系下的位姿 T_CO (等价于 `pose_transform_mujoco.py` 中的 `get_T_OC`)。
    * `Base_Obj()`: 计算物体在机械臂基座坐标系下的位姿 T_BO (T_BC @ T_CO)。
    * `inverse_kinematics(T_BO)`: (未使用 `MMK2FIK`) 内部调用 `ik_solver.properIK` (似乎 `ik_solver` 未在此类中完全初始化或使用，可能依赖外部设置或是一个占位符)。
    * `get_joint_states()`: 获取基于 `Base_Obj()` 计算的关节状态。
    * `get_joint_state_from_cam(action_rot=None)`: 调用 `MMK2FIK` 的 `get_armjoint_pose_wrt_armbase` 来根据物体位姿计算关节角度。

## 使用流程建议

1. **仿真 (MuJoCo)**:
    * 使用 `MMK2FK` 设置机器人各关节状态。
    * 使用 `pose_transform_mujoco.py` 中的 `PoseTransform` 类：
        * 通过 `get_T_OB` 获取目标物体相对于机械臂基座的位姿。
        * 使用 `get_arm_joint_pose` 或 `get_arm_joint_pose_from_object` 计算所需的关节角度。
    * `grasp_apple.py` 提供了一个具体的任务场景示例。

2. **ROS2 (实物或ROS2仿真)**:
    * 初始化 `MMK2_Receiver` 以接收机器人实时状态。
    * 使用 `pose_transform_ros2.py` 中的 `PoseTransform` 类：
        * 通过 `transform_position_wrt_camera_to_base` 或 `transform_position_wrt_world_to_base` 将感知到的物体位置转换到机器人基座坐标系。
        * 使用 `get_arm_joint_pose` 计算目标关节角度。

这个模块为MMK2机器人在不同环境（纯MuJoCo仿真、ROS2控制）下的位姿感知、坐标变换和运动规划提供了基础。
