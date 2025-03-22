import mujoco
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from discoverse.utils import get_site_tmat
from discoverse.envs.mmk2_base import MMK2Cfg
from discoverse.examples.ros2.mmk2_joy_ros2 import MMK2ROS2JoyCtl

from discoverse.envs.mj_lidar import MjLidarSensor, create_lidar_single_line

def broadcast_tf_ros2(broadcaster, parent_frame, child_frame, translation, rotation):
    """
    广播TF变换 - ROS2版本
    
    参数:
    broadcaster: TransformBroadcaster对象
    parent_frame: 父坐标系名称
    child_frame: 子坐标系名称
    translation: 平移向量[x, y, z]
    rotation: 旋转四元数[x, y, z, w]
    """
    t = TransformStamped()
    t.header.stamp = Node.get_clock(broadcaster).now().to_msg()
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    
    t.transform.translation.x = float(translation[0])
    t.transform.translation.y = float(translation[1])
    t.transform.translation.z = float(translation[2])
    
    t.transform.rotation.x = float(rotation[0])
    t.transform.rotation.y = float(rotation[1])
    t.transform.rotation.z = float(rotation[2])
    t.transform.rotation.w = float(rotation[3])
    
    broadcaster.sendTransform(t)

def publish_point_cloud_ros2(publisher, points, frame_id):
    """
    将点云数据发布为ROS2 PointCloud2消息
    
    参数:
    publisher: PointCloud2发布者
    points: 点云数据，形状为(N, 3)或(3, N)
    frame_id: 坐标系ID
    """
    # 创建消息头
    header = Header()
    header.frame_id = frame_id
    header.stamp = Node.get_clock(publisher).now().to_msg()
    
    # 定义点云字段
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
    ]
    
    # 添加强度值
    if len(points.shape) == 2:
        # 如果是(N, 3)形状，转换为(3, N)以便处理
        points_transposed = points.T if points.shape[1] == 3 else points
        
        if points_transposed.shape[0] == 3:
            # 添加强度通道
            points_with_intensity = np.vstack([
                points_transposed, 
                np.ones(points_transposed.shape[1], dtype=np.float32)
            ])
        else:
            points_with_intensity = points_transposed
    else:
        # 如果点云已经是(3, N)形状
        if points.shape[0] == 3:
            points_with_intensity = np.vstack([
                points, 
                np.ones(points.shape[1], dtype=np.float32)
            ])
        else:
            points_with_intensity = points
    
    # 创建PointCloud2消息
    pc_msg = PointCloud2()
    pc_msg.header = header
    pc_msg.height = 1
    pc_msg.width = points_with_intensity.shape[1]
    pc_msg.fields = fields
    pc_msg.is_bigendian = False
    pc_msg.point_step = 16  # 4 * 4字节 (x, y, z, intensity)
    pc_msg.row_step = pc_msg.point_step * points_with_intensity.shape[1]
    pc_msg.is_dense = True
    
    # 转换数据为字节流
    pc_msg.data = np.transpose(points_with_intensity).astype(np.float32).tobytes()
    
    publisher.publish(pc_msg)

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
    rays_phi, rays_theta = create_lidar_single_line(360, np.pi*2.)
    exec_node.get_logger().info("rays_phi, rays_theta: {}, {}".format(rays_phi.shape, rays_theta.shape))

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
    pub_lidar_s2 = exec_node.create_publisher(PointCloud2, '/mmk2/lidar_s2', 1)

    sim_step_cnt = 0
    lidar_pub_cnt = 0

    print("打开rviz2并在其中设置以下显示：")
    print("1. 添加TF显示，用于查看坐标系")
    print("2. 添加PointCloud2显示，话题为/mmk2/lidar_s2")
    print("3. 设置Fixed Frame为'world'")

    while exec_node.running and rclpy.ok():
        # 处理ROS消息
        rclpy.spin_once(exec_node, timeout_sec=0)
        
        # 处理手柄操作输入
        exec_node.teleopProcess()
        obs, _, _, _, _ = exec_node.step(exec_node.target_control)

        # 当累计的仿真时间（步数×时间步长×期望频率）超过已发布次数时，执行发布
        if sim_step_cnt * exec_node.delta_t * lidar_pub_rate > lidar_pub_cnt:
            lidar_pub_cnt += 1

            lidar_pose = get_site_tmat(exec_node.mj_data, lidar_frame_id)
            points = lidar_s2.ray_cast_taichi(rays_phi, rays_theta, lidar_pose, exec_node.renderer.scene)
            ti.sync()
            publish_point_cloud_ros2(pub_lidar_s2, points, lidar_frame_id)
           
            lidar_position = lidar_pose[:3, 3]
            lidar_orientation = Rotation.from_matrix(lidar_pose[:3, :3]).as_quat()
            broadcast_tf_ros2(tf_broadcaster, "world", lidar_frame_id, lidar_position, lidar_orientation)

        sim_step_cnt += 1

    # 清理资源
    exec_node.destroy_node()
    rclpy.shutdown()
