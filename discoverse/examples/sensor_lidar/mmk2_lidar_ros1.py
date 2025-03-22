import mujoco
import threading
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from discoverse.utils import get_site_tmat
from discoverse.envs.mmk2_base import MMK2Cfg
from discoverse.examples.ros1.mmk2_ros1_joy import MMK2ROS1JoyCtl

from discoverse.envs.mj_lidar import MjLidarSensor, create_lidar_single_line
from discoverse.examples.sensor_lidar.lidar_vis_ros1 import broadcast_tf, publish_point_cloud, publish_scene

if __name__ == "__main__":
    rospy.init_node('mmk2_lidar_node', anonymous=True)

    # 设置NumPy打印选项：精度为3位小数，禁用科学计数法，行宽为500字符
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    
    cfg = MMK2Cfg()
    cfg.render_set["width"] = 1280
    cfg.render_set["height"] = 720
    cfg.init_key = "pick"
    cfg.mjcf_file_path = "mjcf/mmk2_lidar.xml"
    cfg.use_gaussian_renderer = False

    # 初始化仿真环境
    exec_node = MMK2ROS1JoyCtl(cfg)
    exec_node.reset()

    # 创建TF广播者
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # 创建激光雷达射线配置 - 参数10表示角度分辨率，2π表示完整360度扫描范围
    # 返回的rays_phi和rays_theta分别表示射线的俯仰角和方位角
    # 设置激光雷达数据发布频率为12Hz
    lidar_pub_rate = 12
    rays_phi, rays_theta = create_lidar_single_line(360, np.pi*2.)
    rospy.loginfo("rays_phi, rays_theta: {}, {}".format(rays_phi.shape, rays_theta.shape))

    # 定义激光雷达的坐标系ID，用于TF发布
    lidar_frame_id = "mmk2_lidar_s2"

    # 创建MuJoCo激光雷达传感器对象，关联到当前渲染场景
    # enable_profiling=False表示不启用性能分析，verbose=False表示不输出详细日志
    lidar_s2 = MjLidarSensor(exec_node.renderer.scene, enable_profiling=False, verbose=False)

    # 更新MuJoCo场景，准备进行光线追踪 mjCAT_ALL表示更新所有类别的对象，包括几何体、关节、约束等
    mujoco.mjv_updateScene(
        exec_node.mj_model, exec_node.mj_data, mujoco.MjvOption(), 
        None, mujoco.MjvCamera(), 
        mujoco.mjtCatBit.mjCAT_ALL.value, exec_node.renderer.scene
    )
    
    # Warm Start
    # 使用Taichi库进行光线投射计算，获取激光雷达点云数据
    points = lidar_s2.ray_cast_taichi(rays_phi, rays_theta, np.eye(4), exec_node.renderer.scene)
    # 同步Taichi并行计算操作，确保计算完成
    ti.sync()

    # 创建ROS发布者，用于将激光雷达数据发布为PointCloud2类型消息
    pub_lidar_s2 = rospy.Publisher('/mmk2/lidar_s2', PointCloud2, queue_size=1)

    def publish_scene_thread():
        # 创建场景可视化标记发布者
        rate = rospy.Rate(1)
        pub_scene = rospy.Publisher('/mujoco_scene', MarkerArray, queue_size=1)
        while exec_node.running and not rospy.is_shutdown():
            # 发布场景可视化标记
            publish_scene(pub_scene, exec_node.renderer.scene)
            rate.sleep()
    # 创建一个线程，用于发布场景可视化标记

    scene_pub_thread = threading.Thread(target=publish_scene_thread)
    scene_pub_thread.start()

    sim_step_cnt = 0
    lidar_pub_cnt = 0

    print("打开rviz并在其中设置以下显示：")
    print("1. 添加TF显示，用于查看坐标系")
    print("2. 添加PointCloud2显示，话题为/mmk2/lidar_s2")
    print("3. 设置Fixed Frame为'world'")

    while exec_node.running and not rospy.is_shutdown():
        # 处理手柄操作输入
        exec_node.teleopProcess()
        obs, _, _, _, _ = exec_node.step(exec_node.target_control)

        # 当累计的仿真时间（步数×时间步长×期望频率）超过已发布次数时，执行发布
        if sim_step_cnt * exec_node.delta_t * lidar_pub_rate > lidar_pub_cnt:
            lidar_pub_cnt += 1

            lidar_pose = get_site_tmat(exec_node.mj_data, lidar_frame_id)
            points = lidar_s2.ray_cast_taichi(rays_phi, rays_theta, lidar_pose, exec_node.renderer.scene)
            ti.sync()
            publish_point_cloud(pub_lidar_s2, points, lidar_frame_id)
           
            lidar_position = lidar_pose[:3, 3]
            lidar_orientation = Rotation.from_matrix(lidar_pose[:3, :3]).as_quat()
            broadcast_tf(tf_broadcaster, "world", lidar_frame_id, lidar_position, lidar_orientation)

        sim_step_cnt += 1

    scene_pub_thread.join()