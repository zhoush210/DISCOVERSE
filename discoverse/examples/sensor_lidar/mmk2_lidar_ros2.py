import threading
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray

from discoverse.robots_env.mmk2_base import MMK2Cfg
from discoverse.examples.ros2.mmk2_ros2_joy import MMK2ROS2JoyCtl

from mujoco_lidar.lidar_wrapper import MjLidarWrapper
from mujoco_lidar.scan_gen import create_lidar_single_line
from mujoco_lidar.examples.lidar_vis_ros2 import publish_scene, broadcast_tf, publish_point_cloud

if __name__ == "__main__":
    rclpy.init()

    # 设置NumPy打印选项：精度为3位小数，禁用科学计数法，行宽为500字符
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    
    cfg = MMK2Cfg()
    cfg.render_set["width"] = 1280
    cfg.render_set["height"] = 720
    cfg.init_key = "pick"
    cfg.mjcf_file_path = "mjcf/mmk2_lidar.xml"
    cfg.use_gaussian_renderer = False

    # 初始化仿真环境
    exec_node = MMK2ROS2JoyCtl(cfg)
    exec_node.reset()

    # 创建TF广播者
    tf_broadcaster = TransformBroadcaster(exec_node)

    # 创建激光雷达射线配置 - 参数360表示水平分辨率，2π表示完整360度扫描范围
    # 返回的rays_phi和rays_theta分别表示射线的俯仰角和方位角
    # 设置激光雷达数据发布频率为12Hz
    lidar_pub_rate = 12
    rays_theta, rays_phi = create_lidar_single_line(360, np.pi*2.)
    exec_node.get_logger().info("rays_phi, rays_theta: {}, {}".format(rays_phi.shape, rays_theta.shape))

    # 定义激光雷达的坐标系ID，用于TF发布
    lidar_frame_id = "mmk2_lidar_s2"

    # 创建MuJoCo激光雷达传感器对象，关联到当前渲染场景
    # enable_profiling=False表示不启用性能分析，verbose=False表示不输出详细日志
    lidar_s2 = MjLidarWrapper(exec_node.mj_model, exec_node.mj_data, site_name="laser")
    
    # Warm Start
    # 使用Taichi库进行光线投射计算，获取激光雷达点云数据
    lidar_s2.get_lidar_points(rays_phi, rays_theta, exec_node.mj_data)
    
    # 创建ROS发布者，用于将激光雷达数据发布为PointCloud2类型消息
    pub_lidar_s2 = exec_node.create_publisher(PointCloud2, '/mmk2/lidar_s2', 1)

    def publish_scene_thread():
        # 创建ROS发布者，用于将MuJoCo场景发布为MarkerArray类型消息
        rate = exec_node.create_rate(1)
        pub_scene = exec_node.create_publisher(MarkerArray, '/mujoco_scene', 1)
        while exec_node.running and rclpy.ok():
            stamp = exec_node.get_clock().now().to_msg()
            publish_scene(pub_scene, exec_node.renderer.scene, "world", stamp)
            rate.sleep()

    # 创建一个线程，用于发布场景可视化标记
    scene_pub_thread = threading.Thread(target=publish_scene_thread)
    scene_pub_thread.start()

    sim_step_cnt = 0
    lidar_pub_cnt = 0

    print("打开rviz2并在其中设置以下显示：")
    print("1. 添加TF显示，用于查看坐标系")
    print("2. 添加PointCloud2显示，话题为/mmk2/lidar_s2")
    print("3. 设置Fixed Frame为'world'")
    print("4. 添加MarkerArray显示，话题为/mujoco_scene")

    while exec_node.running and rclpy.ok():
        # 处理ROS消息
        rclpy.spin_once(exec_node, timeout_sec=0)
        
        # 处理手柄操作输入
        exec_node.teleopProcess()
        obs, _, _, _, _ = exec_node.step(exec_node.target_control)

        # 当累计的仿真时间（步数×时间步长×期望频率）超过已发布次数时，执行发布
        if sim_step_cnt * exec_node.delta_t * lidar_pub_rate > lidar_pub_cnt:
            lidar_pub_cnt += 1

            points = lidar_s2.get_lidar_points(rays_phi, rays_theta, exec_node.mj_data)

            stamp = exec_node.get_clock().now().to_msg()
            publish_point_cloud(pub_lidar_s2, points, lidar_frame_id, stamp)
           
            lidar_orientation = Rotation.from_matrix(lidar_s2.sensor_rotation).as_quat()
            broadcast_tf(tf_broadcaster, "world", lidar_frame_id, lidar_s2.sensor_position, lidar_orientation, stamp)

        sim_step_cnt += 1

    # 清理资源
    exec_node.destroy_node()
    rclpy.shutdown()
    scene_pub_thread.join()
