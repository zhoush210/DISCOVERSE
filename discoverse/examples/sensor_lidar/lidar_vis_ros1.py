import time
import mujoco
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation

from pynput import keyboard

import rospy
import tf2_ros
import geometry_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray

from discoverse.envs.mj_lidar import MjLidarSensor, create_lidar_rays, create_demo_scene

def create_marker_from_geom(geom, marker_id, frame_id="world"):
    """从MuJoCo几何体创建ROS可视化标记"""
    # 胶囊体需要特殊处理，返回标记列表
    if geom.type == 3:  # CAPSULE
        return create_capsule_markers(geom, marker_id, frame_id)
    
    # 其他几何体正常处理，返回单个标记
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "mujoco_geoms"
    marker.id = marker_id
    marker.action = Marker.ADD
    
    # 设置位置
    marker.pose.position.x = geom.pos[0]
    marker.pose.position.y = geom.pos[1]
    marker.pose.position.z = geom.pos[2]
    
    # 设置旋转（从四元数）
    quat = Rotation.from_matrix(geom.mat.reshape(3, 3)).as_quat()
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    
    # 设置颜色
    marker.color.r = geom.rgba[0]
    marker.color.g = geom.rgba[1]
    marker.color.b = geom.rgba[2]
    marker.color.a = geom.rgba[3]
    
    # 根据几何体类型设置标记类型和大小
    if geom.type == 0:  # PLANE
        marker.type = Marker.CUBE
        marker.scale.x = geom.size[0] * 2  # 平面的宽度
        marker.scale.y = geom.size[1] * 2  # 平面的高度
        marker.scale.z = 0.01  # 平面厚度很小
    elif geom.type == 2:  # SPHERE
        marker.type = Marker.SPHERE
        marker.scale.x = geom.size[0] * 2  # 直径 = 半径 * 2
        marker.scale.y = geom.size[0] * 2
        marker.scale.z = geom.size[0] * 2
    elif geom.type == 4:  # ELLIPSOID
        marker.type = Marker.SPHERE
        # 椭球体通过缩放球体来显示
        marker.scale.x = geom.size[0] * 2
        marker.scale.y = geom.size[1] * 2
        marker.scale.z = geom.size[2] * 2
    elif geom.type == 5:  # CYLINDER
        marker.type = Marker.CYLINDER
        marker.scale.x = geom.size[0] * 2  # 直径
        marker.scale.y = geom.size[0] * 2
        marker.scale.z = geom.size[1] * 2  # 高度
    elif geom.type == 6:  # BOX
        marker.type = Marker.CUBE
        marker.scale.x = geom.size[0] * 2
        marker.scale.y = geom.size[1] * 2
        marker.scale.z = geom.size[2] * 2
    else:
        # 不支持的几何体类型
        marker.type = Marker.CUBE
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
    
    return [marker]  # 返回单个标记的列表，保持接口一致

def create_capsule_markers(geom, marker_id, frame_id="world"):
    """创建胶囊体的可视化标记（一个圆柱体和两个半球）"""
    markers = []
    radius = geom.size[0]         # 半径
    half_height = geom.size[1]    # 圆柱部分的半高
    
    # 获取旋转矩阵和四元数
    rot_matrix = geom.mat.reshape(3, 3)
    quat = Rotation.from_matrix(rot_matrix).as_quat()
    
    # 计算胶囊体中心点位置
    center = np.array([geom.pos[0], geom.pos[1], geom.pos[2]])
    
    # 计算圆柱体的高度（直接使用2*半高）
    cylinder_height = 2 * half_height
    
    # 计算胶囊体方向的单位向量 (假设沿z轴)
    z_dir = np.array([0, 0, 1])
    # 应用旋转矩阵得到实际方向
    capsule_dir = rot_matrix.dot(z_dir)
    capsule_dir = capsule_dir / np.linalg.norm(capsule_dir)  # 确保是单位向量
    
    # 1. 创建中间的圆柱体
    cylinder = Marker()
    cylinder.header.frame_id = frame_id
    cylinder.header.stamp = rospy.Time.now()
    cylinder.ns = "mujoco_geoms"
    cylinder.id = marker_id * 3     # 使用基础ID的3倍，确保唯一性
    cylinder.type = Marker.CYLINDER
    cylinder.action = Marker.ADD
    
    cylinder.pose.position.x = center[0]
    cylinder.pose.position.y = center[1]
    cylinder.pose.position.z = center[2]
    
    cylinder.pose.orientation.x = quat[0]
    cylinder.pose.orientation.y = quat[1]
    cylinder.pose.orientation.z = quat[2]
    cylinder.pose.orientation.w = quat[3]
    
    cylinder.scale.x = radius * 2  # 直径
    cylinder.scale.y = radius * 2
    cylinder.scale.z = cylinder_height  # 高度
    
    cylinder.color.r = geom.rgba[0]
    cylinder.color.g = geom.rgba[1]
    cylinder.color.b = geom.rgba[2]
    cylinder.color.a = geom.rgba[3]
    
    markers.append(cylinder)
    
    # 计算两个半球的中心位置
    # 半球中心位于距离主中心点半高距离的位置
    sphere1_center = center + capsule_dir * half_height
    sphere2_center = center - capsule_dir * half_height
    
    # 2. 创建上半球
    top_sphere = Marker()
    top_sphere.header.frame_id = frame_id
    top_sphere.header.stamp = rospy.Time.now()
    top_sphere.ns = "mujoco_geoms"
    top_sphere.id = marker_id * 3 + 1  # 使用基础ID的3倍+1
    top_sphere.type = Marker.SPHERE
    top_sphere.action = Marker.ADD
    
    top_sphere.pose.position.x = sphere1_center[0]
    top_sphere.pose.position.y = sphere1_center[1]
    top_sphere.pose.position.z = sphere1_center[2]
    
    top_sphere.pose.orientation.x = quat[0]
    top_sphere.pose.orientation.y = quat[1]
    top_sphere.pose.orientation.z = quat[2]
    top_sphere.pose.orientation.w = quat[3]
    
    top_sphere.scale.x = radius * 2
    top_sphere.scale.y = radius * 2
    top_sphere.scale.z = radius * 2
    
    top_sphere.color.r = geom.rgba[0]
    top_sphere.color.g = geom.rgba[1]
    top_sphere.color.b = geom.rgba[2]
    top_sphere.color.a = geom.rgba[3]
    
    markers.append(top_sphere)
    
    # 3. 创建下半球
    bottom_sphere = Marker()
    bottom_sphere.header.frame_id = frame_id
    bottom_sphere.header.stamp = rospy.Time.now()
    bottom_sphere.ns = "mujoco_geoms"
    bottom_sphere.id = marker_id * 3 + 2  # 使用基础ID的3倍+2
    bottom_sphere.type = Marker.SPHERE
    bottom_sphere.action = Marker.ADD
    
    bottom_sphere.pose.position.x = sphere2_center[0]
    bottom_sphere.pose.position.y = sphere2_center[1]
    bottom_sphere.pose.position.z = sphere2_center[2]
    
    bottom_sphere.pose.orientation.x = quat[0]
    bottom_sphere.pose.orientation.y = quat[1]
    bottom_sphere.pose.orientation.z = quat[2]
    bottom_sphere.pose.orientation.w = quat[3]
    
    bottom_sphere.scale.x = radius * 2
    bottom_sphere.scale.y = radius * 2
    bottom_sphere.scale.z = radius * 2
    
    bottom_sphere.color.r = geom.rgba[0]
    bottom_sphere.color.g = geom.rgba[1]
    bottom_sphere.color.b = geom.rgba[2]
    bottom_sphere.color.a = geom.rgba[3]
    
    markers.append(bottom_sphere)
    
    return markers

def publish_scene(publisher, mj_scene, frame_id="world"):
    """将MuJoCo场景发布为ROS可视化标记数组"""
    marker_array = MarkerArray()
    
    # 记录当前使用的标记ID
    current_id = 0
    
    # 创建每个几何体的标记
    for i in range(mj_scene.ngeom):
        geom = mj_scene.geoms[i]
        # 现在 create_marker_from_geom 返回一个标记列表
        markers = create_marker_from_geom(geom, current_id, frame_id)
        
        # 添加所有返回的标记到标记数组
        for marker in markers:
            marker_array.markers.append(marker)
            current_id += 1
    
    # 发布标记数组
    publisher.publish(marker_array)


def publish_point_cloud(publisher, points, frame_id):
    """将点云数据发布为ROS PointCloud2消息"""
    stamp = rospy.Time.now()
        
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
        
    # 转换为ROS消息格式的点云
    pc_msg = pc2.create_cloud(
        header=rospy.Header(frame_id=frame_id, stamp=stamp),
        fields=fields,
        points=np.transpose(points_with_intensity)  # 转置回(N, 4)格式
    )
    
    publisher.publish(pc_msg)

def broadcast_tf(broadcaster, parent_frame, child_frame, translation, rotation, stamp=None):
    """广播TF变换"""
    if stamp is None:
        stamp = rospy.Time.now()
        
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]
    
    t.transform.rotation.x = rotation[0]
    t.transform.rotation.y = rotation[1]
    t.transform.rotation.z = rotation[2]
    t.transform.rotation.w = rotation[3]
    
    broadcaster.sendTransform(t)

class KeyboardListener:
    def __init__(self, lidar_position, lidar_orientation):
        # 存储激光雷达的位置和方向
        self.lidar_position = lidar_position
        self.lidar_orientation = lidar_orientation
        
        # 存储当前的欧拉角（俯仰、偏航）
        self.euler_angles = Rotation.from_quat(lidar_orientation).as_euler('xyz')
        
        # 移动和旋转的速度
        self.move_speed = 1.  # 平移速度 (米/秒)
        self.rotate_speed = 1.  # 旋转速度 (弧度/秒)
        self.height_speed = 0.5  # 高度调整速度 (米/秒)
        
        # 当前按下的键
        self.pressed_keys = set()
        
        # 启动键盘监听器
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        
        print("键盘控制已启动:")
        print("  WASD: 控制水平移动")
        print("  Q/E: 控制高度上升/下降")
        print("  方向键上/下: 控制俯仰角")
        print("  方向键左/右: 控制偏航角")
        print("  ESC: 退出程序")

    def on_press(self, key):
        """处理按键按下事件"""
        try:
            # 将按键添加到已按下的键集合中
            if hasattr(key, 'char'):
                self.pressed_keys.add(key.char.lower())
            else:
                self.pressed_keys.add(key)
            
            # 如果按下ESC键，则停止监听
            if key == keyboard.Key.esc:
                print('ESC 键按下，退出程序')
                return False
        except AttributeError:
            # 忽略特殊键的AttributeError
            pass

    def on_release(self, key):
        """处理按键释放事件"""
        try:
            # 从已按下的键集合中移除释放的键
            if hasattr(key, 'char'):
                self.pressed_keys.discard(key.char.lower())
            else:
                self.pressed_keys.discard(key)
        except AttributeError:
            # 忽略特殊键的AttributeError
            pass

    def update_lidar_pose(self, dt):
        """根据当前按下的键更新激光雷达的位置和姿态"""
        # 获取当前的欧拉角
        pitch = self.euler_angles[1]
        yaw = self.euler_angles[2]
        
        # 计算前进方向（基于当前的yaw角）
        forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0])
        right_dir = np.array([np.sin(yaw), -np.cos(yaw), 0])
        
        # 处理平移（WASD）
        move_delta = np.zeros(3)
        
        if 'w' in self.pressed_keys:
            move_delta += forward_dir * self.move_speed * dt
        if 's' in self.pressed_keys:
            move_delta -= forward_dir * self.move_speed * dt
        if 'd' in self.pressed_keys:
            move_delta += right_dir * self.move_speed * dt
        if 'a' in self.pressed_keys:
            move_delta -= right_dir * self.move_speed * dt
            
        # 处理高度调整（QE）
        if 'q' in self.pressed_keys:
            move_delta[2] += self.height_speed * dt
        if 'e' in self.pressed_keys:
            move_delta[2] -= self.height_speed * dt
            
        # 应用平移
        self.lidar_position += move_delta
        
        # 确保激光雷达不会低于地面
        self.lidar_position[2] = max(0.1, self.lidar_position[2])
        
        # 处理旋转（方向键）
        if keyboard.Key.up in self.pressed_keys:
            pitch += self.rotate_speed * dt
        if keyboard.Key.down in self.pressed_keys:
            pitch -= self.rotate_speed * dt
        if keyboard.Key.left in self.pressed_keys:
            yaw += self.rotate_speed * dt
        if keyboard.Key.right in self.pressed_keys:
            yaw -= self.rotate_speed * dt
            
        # 限制俯仰角范围
        pitch = np.clip(pitch, -np.pi/2 + 0.1, np.pi/2 - 0.1)
        
        # 更新欧拉角
        self.euler_angles[1] = pitch
        self.euler_angles[2] = yaw
        
        # 将欧拉角转换为四元数
        self.lidar_orientation = Rotation.from_euler('xyz', self.euler_angles).as_quat()
        
        # 返回更新后的位置和姿态
        return self.lidar_position.copy(), self.lidar_orientation.copy()

if __name__ == "__main__":

    # 初始化ROS节点
    rospy.init_node('mujoco_lidar_test', anonymous=True)

    # 创建点云发布者
    pub_taichi = rospy.Publisher('/lidar_points_taichi', PointCloud2, queue_size=1)

    # 创建场景可视化标记发布者
    pub_scene = rospy.Publisher('/mujoco_scene', MarkerArray, queue_size=1)

    # 创建TF广播者
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # 创建MuJoCo场景
    mj_model, mj_data = create_demo_scene()

    # 创建场景渲染对象
    scene = mujoco.MjvScene(mj_model, maxgeom=100)
    # 更新模拟
    mujoco.mj_forward(mj_model, mj_data)

    # 更新场景
    mujoco.mjv_updateScene(
        mj_model, mj_data, mujoco.MjvOption(), 
        None, mujoco.MjvCamera(), 
        mujoco.mjtCatBit.mjCAT_ALL.value, scene
    )

    # 创建激光雷达传感器
    lidar = MjLidarSensor(scene)

    # 创建激光雷达射线 - 使用更高的分辨率进行性能测试
    rays_phi, rays_theta = create_lidar_rays(horizontal_resolution=3600, vertical_resolution=256)
    print(f"射线数量: {len(rays_phi)}")

    # 设置激光雷达位置
    lidar_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    lidar_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # 四元数(x,y,z,w)

    # 创建键盘监听器
    kb_listener = KeyboardListener(lidar_position, lidar_orientation)

    # 主循环
    rate = rospy.Rate(30)  # 提高帧率以获得更流畅的控制体验
    last_time = time.time()

    print("在RViz中设置以下显示：")
    print("1. 添加TF显示，用于查看坐标系")
    print("2. 添加PointCloud2显示，话题为/lidar_points_taichi")
    print("3. 添加MarkerArray显示，话题为/mujoco_scene")
    print("4. 设置Fixed Frame为'world'")
        
    try:
        while not rospy.is_shutdown():
            # 计算时间增量
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # 更新激光雷达位置和方向
            lidar_position, lidar_orientation = kb_listener.update_lidar_pose(dt)
            
            # 构建激光雷达位姿矩阵
            lidar_pose = np.eye(4, dtype=np.float32)
            
            # 旋转部分（从四元数构建旋转矩阵）
            lidar_pose[:3, :3] = Rotation.from_quat(lidar_orientation).as_matrix()

            # 平移部分
            lidar_pose[:3, 3] = lidar_position
            
            # 更新模拟
            mujoco.mj_step(mj_model, mj_data)
            
            # 更新场景
            mujoco.mjv_updateScene(
                mj_model, mj_data, mujoco.MjvOption(), 
                None, mujoco.MjvCamera(), 
                mujoco.mjtCatBit.mjCAT_ALL.value, scene
            )
            
            # 发布场景可视化标记
            publish_scene(pub_scene, scene)
            
            # 执行光线追踪
            start_time = time.time()
            points = lidar.ray_cast_taichi(rays_phi, rays_theta, lidar_pose, scene)
            ti.sync()
            end_time = time.time()
            
            # 打印性能信息和当前位置
            print(f"位置: [{lidar_position[0]:.2f}, {lidar_position[1]:.2f}, {lidar_position[2]:.2f}], "
                    f"欧拉角: [{kb_listener.euler_angles[0]:.2f}, {kb_listener.euler_angles[1]:.2f}, {kb_listener.euler_angles[2]:.2f}], "
                    f"耗时: {(end_time - start_time)*1000:.2f} ms, 射线数量: {len(rays_phi)}")
            
            # 发布点云
            publish_point_cloud(pub_taichi, points, "lidar")
            
            # 广播激光雷达的TF
            broadcast_tf(tf_broadcaster, "world", "lidar", lidar_position, lidar_orientation)
            
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        print("程序结束")