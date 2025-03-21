import time
import mujoco
import numpy as np
import taichi as ti

ti.init(
    arch=ti.gpu, 
    kernel_profiler=True,
    advanced_optimization=True,  # 启用高级优化
    offline_cache=True,          # 启用离线缓存
    default_fp=ti.f32,           # 设置默认浮点类型
    default_ip=ti.i32,           # 设置默认整数类型
    device_memory_GB=4.0,        # 限制设备内存使用
)

# mj_geom_type = {
#     0 : "mjGEOM_PLANE",
#     1 : "mjGEOM_HFIELD",
#     2 : "mjGEOM_SPHERE",
#     3 : "mjGEOM_CAPSULE",
#     4 : "mjGEOM_ELLIPSOID",
#     5 : "mjGEOM_CYLINDER",
#     6 : "mjGEOM_BOX"
# }
@ti.data_oriented
class MjLidarSensor:

    def __init__(self, mj_scene):
        self.n_geoms = mj_scene.ngeom
        print(f"n_geoms: {self.n_geoms}")

        # 预分配所有Taichi字段，避免重复创建
        self.geom_types = ti.field(dtype=ti.i32, shape=(self.n_geoms))
        self.geom_sizes = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_geoms))
        self.geom_positions = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_geoms))
        self.geom_rotations = ti.Vector.field(9, dtype=ti.f32, shape=(self.n_geoms))
        
        # 初始化几何体静态数据
        for i in range(self.n_geoms):
            geom = mj_scene.geoms[i]
            self.geom_types[i] = geom.type
            self.geom_sizes[i] = ti.math.vec3(geom.size[0], geom.size[1], geom.size[2])
            self.geom_positions[i] = ti.math.vec3(geom.pos[0], geom.pos[1], geom.pos[2])
            # TODO: 如果需要旋转，将geom.mat转为矩阵并保存
            # 暂时未完全实现旋转的处理
        
        # 预先分配传感器位姿数组
        self.sensor_pose_ti = ti.ndarray(dtype=ti.f32, shape=(4, 4))
        
        # 缓存射线数据
        self.cached_n_rays = 0
        self.rays_phi_ti = None
        self.rays_theta_ti = None
        self.hit_points = None
        
        # 预分配临时数组（用于内核计算）
        self.hit_points_world = None  # 世界坐标系下的命中点
        self.hit_mask = None  # 射线命中标志
        
        # 性能统计
        self.kernel_time = 0
        self.prepare_time = 0
        self.total_time = 0

    def set_sensor_pose(self, sensor_pose):
        assert sensor_pose.shape == (4, 4) and sensor_pose.dtype == np.float32, f"sensor_pose must be a 4x4 numpy array, but got {sensor_pose.shape} and {sensor_pose.dtype}"
        self.sensor_pose_ti.from_numpy(sensor_pose)

    def update_geom_positions(self, mj_scene):
        """更新几何体位置数据"""
        for i in range(self.n_geoms):
            geom = mj_scene.geoms[i]
            self.geom_positions[i] = ti.math.vec3(geom.pos[0], geom.pos[1], geom.pos[2])
            # self.geom_rotations[i].from_numpy(geom.mat.flatten())
    
    @ti.func
    def ray_plane_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3) -> ti.math.vec4:
        """计算射线与平面的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        normal = ti.math.vec3(0.0, 0.0, 1.0)  # 假设法向量为z轴
        half_width = size[0]
        half_height = size[1]
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        denom = ray_direction.dot(normal)
        
        # 避免除以零，检查光线是否与平面平行
        if ti.abs(denom) >= 1e-6:
            t = (center - ray_start).dot(normal) / denom
            
            # 如果t为正，表示有有效交点
            if t >= 0:
                hit_pos = ray_start + t * ray_direction
                local_pos = hit_pos - center
                
                # 检查交点是否在平面范围内
                if ti.abs(local_pos.x) <= half_width and ti.abs(local_pos.y) <= half_height:
                    hit_result = ti.math.vec4(hit_pos.x, hit_pos.y, hit_pos.z, t)
        
        return hit_result
    
    @ti.func
    def ray_sphere_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3) -> ti.math.vec4:
        """计算射线与球体的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        radius = size[0]
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        oc = ray_start - center
        a = ray_direction.dot(ray_direction)
        b = 2.0 * oc.dot(ray_direction)
        c = oc.dot(oc) - radius * radius
        
        discriminant = b * b - 4 * a * c
        
        # 计算交点
        if discriminant >= 0:
            t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
            
            # 如果t为负，则使用较大的t值
            if t < 0:
                t = (-b + ti.sqrt(discriminant)) / (2.0 * a)
            
            # 如果t为正，表示有有效交点
            if t >= 0:
                hit_pos = ray_start + t * ray_direction
                hit_result = ti.math.vec4(hit_pos.x, hit_pos.y, hit_pos.z, t)
        
        return hit_result
    
    @ti.func
    def ray_box_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3) -> ti.math.vec4:
        """计算射线与盒子的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        half_extents = size
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        inv_dir = ti.math.vec3(1.0/ray_direction.x, 1.0/ray_direction.y, 1.0/ray_direction.z)
        
        t_min = -1e10  # 使用大数而不是无穷
        t_max = 1e10
        
        # 检查x轴
        t1 = (center.x - half_extents.x - ray_start.x) * inv_dir.x
        t2 = (center.x + half_extents.x - ray_start.x) * inv_dir.x
        t_min = ti.max(t_min, ti.min(t1, t2))
        t_max = ti.min(t_max, ti.max(t1, t2))
        
        # 检查y轴
        t1 = (center.y - half_extents.y - ray_start.y) * inv_dir.y
        t2 = (center.y + half_extents.y - ray_start.y) * inv_dir.y
        t_min = ti.max(t_min, ti.min(t1, t2))
        t_max = ti.min(t_max, ti.max(t1, t2))
        
        # 检查z轴
        t1 = (center.z - half_extents.z - ray_start.z) * inv_dir.z
        t2 = (center.z + half_extents.z - ray_start.z) * inv_dir.z
        t_min = ti.max(t_min, ti.min(t1, t2))
        t_max = ti.min(t_max, ti.max(t1, t2))
        
        # 如果有有效的交点
        if t_max >= t_min and t_max >= 0:
            t = t_min if t_min >= 0 else t_max
            if t >= 0:
                hit_pos = ray_start + t * ray_direction
                hit_result = ti.math.vec4(hit_pos.x, hit_pos.y, hit_pos.z, t)
        
        return hit_result
    
    @ti.func
    def ray_cylinder_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3) -> ti.math.vec4:
        """计算射线与圆柱体的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        # size[0]是半径，size[1]是半高
        radius = size[0]
        half_height = size[1]
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        # 假设圆柱体的中心轴与z轴平行
        # 平移射线起点，使圆柱体中心在原点
        oc = ray_start - center
        
        # 仅考虑xy平面上的方向分量
        ray_dir_xy = ti.math.vec2(ray_direction.x, ray_direction.y)
        oc_xy = ti.math.vec2(oc.x, oc.y)
        
        # 解二次方程 at² + bt + c = 0
        a = ray_dir_xy.dot(ray_dir_xy)
        
        # 如果a很小，射线几乎与z轴平行
        if a < 1e-6:
            # 检查射线是否在圆柱体内部
            if oc_xy.norm() <= radius:
                # 计算与顶部或底部平面的交点
                t1 = (half_height - oc.z) / ray_direction.z
                t2 = (-half_height - oc.z) / ray_direction.z
                
                # 选择最小的正t值
                t = -1.0  # 默认为无效值
                if t1 >= 0 and (t2 < 0 or t1 < t2):
                    t = t1
                elif t2 >= 0:
                    t = t2
                
                if t >= 0:
                    hit_pos = ray_start + t * ray_direction
                    hit_result = ti.math.vec4(hit_pos.x, hit_pos.y, hit_pos.z, t)
        else:
            # 标准的圆柱体-射线相交测试
            b = 2.0 * oc_xy.dot(ray_dir_xy)
            c = oc_xy.dot(oc_xy) - radius * radius
            
            discriminant = b * b - 4 * a * c
            
            if discriminant >= 0:
                # 计算圆柱侧面的两个可能交点
                sqrt_disc = ti.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2.0 * a)
                t2 = (-b + sqrt_disc) / (2.0 * a)
                
                # 选择最小的正t值
                t = -1.0  # 默认为无效值
                if t1 >= 0:
                    t = t1
                elif t2 >= 0:
                    t = t2
                
                # 检查交点是否在圆柱体高度范围内
                if t >= 0:
                    hit_pos = ray_start + t * ray_direction
                    local_hit = hit_pos - center
                    
                    if ti.abs(local_hit.z) <= half_height:
                        # 交点在圆柱体侧面上
                        hit_result = ti.math.vec4(hit_pos.x, hit_pos.y, hit_pos.z, t)
                    else:
                        # 侧面交点不在圆柱体高度范围内，检查与顶部或底部平面的交点
                        cap_t = -1.0
                        
                        # 射线从上方射向底平面
                        if ray_direction.z < 0 and oc.z > half_height:
                            cap_t = (half_height - oc.z) / ray_direction.z
                        # 射线从下方射向顶平面
                        elif ray_direction.z > 0 and oc.z < -half_height:
                            cap_t = (-half_height - oc.z) / ray_direction.z
                        
                        if cap_t >= 0:
                            cap_hit = ray_start + cap_t * ray_direction
                            cap_local = cap_hit - center
                            cap_xy = ti.math.vec2(cap_local.x, cap_local.y)
                            
                            # 检查交点是否在圆盘内
                            if cap_xy.norm() <= radius:
                                hit_result = ti.math.vec4(cap_hit.x, cap_hit.y, cap_hit.z, cap_t)
        
        return hit_result
    
    @ti.func
    def ray_ellipsoid_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3) -> ti.math.vec4:
        """计算射线与椭球体的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        # 将问题转换为单位球相交，通过缩放空间
        inv_size = ti.math.vec3(1.0/size.x, 1.0/size.y, 1.0/size.z)
        
        # 转换射线原点和方向
        scaled_start = (ray_start - center) * inv_size
        scaled_dir = ray_direction * inv_size
        scaled_dir = scaled_dir.normalized()  # 重新归一化缩放后的方向
        
        # 现在执行标准球体相交测试
        a = scaled_dir.dot(scaled_dir)
        b = 2.0 * scaled_start.dot(scaled_dir)
        c = scaled_start.dot(scaled_start) - 1.0  # 单位球半径为1
        
        discriminant = b * b - 4 * a * c
        
        if discriminant >= 0:
            t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
            
            # 如果t为负，则使用较大的t值
            if t < 0:
                t = (-b + ti.sqrt(discriminant)) / (2.0 * a)
            
            # 如果t为正，表示有有效交点
            if t >= 0:
                # 因为我们使用了归一化的方向向量，需要调整t值以匹配原始未缩放空间中的距离
                # 这里做一个近似，使用原始方向的长度进行缩放
                scaled_hit = scaled_start + t * scaled_dir
                
                # 将缩放的命中点变换回原始空间
                hit_pos = center + scaled_hit * size
                
                # 计算实际距离作为t值
                actual_t = (hit_pos - ray_start).norm() / ray_direction.norm()
                hit_result = ti.math.vec4(hit_pos.x, hit_pos.y, hit_pos.z, actual_t)
        
        return hit_result
    
    @ti.func
    def ray_capsule_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3) -> ti.math.vec4:
        """计算射线与胶囊体的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        # 在MuJoCo中: size[0]是半径，size[1]是圆柱部分的半高
        radius = size[0]
        half_height = size[1]
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        # 计算胶囊体两个半球的中心
        # 半球中心点在距离胶囊体中心half_height处
        sphere1_center = center + ti.math.vec3(0.0, 0.0, half_height)
        sphere2_center = center - ti.math.vec3(0.0, 0.0, half_height)
        
        # 圆柱部分的半高就是half_height
        cylinder_half_height = half_height
        
        # 为圆柱部分创建新的size
        cylinder_size = ti.math.vec3(radius, cylinder_half_height, 0.0)
        
        # 首先检查与圆柱体部分的交点
        cylinder_hit = self.ray_cylinder_intersection(ray_start, ray_direction, center, cylinder_size)
        
        # 初始化最小距离为无穷大
        min_t = 1e10
        has_hit = False
        
        # 如果有圆柱体交点
        if cylinder_hit.w > 0 and cylinder_hit.w < min_t:
            min_t = cylinder_hit.w
            hit_result = cylinder_hit
            has_hit = True
        
        # 然后检查与两个半球的交点
        sphere_size = ti.math.vec3(radius, radius, radius)
        
        # 上半球
        sphere1_hit = self.ray_sphere_intersection(ray_start, ray_direction, sphere1_center, sphere_size)
        if sphere1_hit.w > 0 and sphere1_hit.w < min_t:
            # 确保交点在半球内，而不是在完整球体的下半部分
            hit_pos = ti.math.vec3(sphere1_hit.x, sphere1_hit.y, sphere1_hit.z)
            local_z = hit_pos.z - sphere1_center.z
            if local_z >= 0:  # 只取上半部分
                min_t = sphere1_hit.w
                hit_result = sphere1_hit
                has_hit = True
        
        # 下半球
        sphere2_hit = self.ray_sphere_intersection(ray_start, ray_direction, sphere2_center, sphere_size)
        if sphere2_hit.w > 0 and sphere2_hit.w < min_t:
            # 确保交点在半球内，而不是在完整球体的上半部分
            hit_pos = ti.math.vec3(sphere2_hit.x, sphere2_hit.y, sphere2_hit.z)
            local_z = hit_pos.z - sphere2_center.z
            if local_z <= 0:  # 只取下半部分
                min_t = sphere2_hit.w
                hit_result = sphere2_hit
                has_hit = True
        
        # 如果没有任何交点，返回无效结果
        if not has_hit:
            hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        return hit_result
    
    # 优化的Taichi核函数
    @ti.kernel
    def trace_rays(self, 
        sensor_origins: ti.types.ndarray(dtype=ti.f32, ndim=2), 
        rays_phi: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rays_theta: ti.types.ndarray(dtype=ti.f32, ndim=1),
        n_rays: ti.i32,
        hit_points: ti.template(),
    ):
        # 设置LiDAR传感器位置和姿态（只做一次）
        sensor_pose = ti.Matrix.identity(ti.f32, 4)
        for i in range(4):
            for j in range(4):
                sensor_pose[i, j] = sensor_origins[i, j]
        ray_start = sensor_pose[0:3, 3]  # 射线起点

        # 计算传感器位姿的逆矩阵（只计算一次）
        sensor_pose_inv = ti.Matrix.identity(ti.f32, 4)
        # 旋转部分的转置（因为正交矩阵的逆等于其转置）
        for i in range(3):
            for j in range(3):
                sensor_pose_inv[i, j] = sensor_pose[j, i]
        # 平移部分的变换
        for i in range(3):
            sensor_pose_inv[i, 3] = 0.0
            for j in range(3):
                sensor_pose_inv[i, 3] -= sensor_pose_inv[i, j] * sensor_pose[j, 3]

        # 使用预分配的字段，而不是在内核中创建新字段
        self.hit_points_world.fill(ti.Vector([0.0, 0.0, 0.0]))
        hit_points.fill(ti.Vector([0.0, 0.0, 0.0]))
        self.hit_mask.fill(0)
                
        # 为每条射线并行计算
        ti.loop_config(block_dim=512)
        for i in range(n_rays):
            min_distance = 1e10

            # 计算射线方向（球坐标系转笛卡尔坐标系）
            theta = rays_theta[i]  # 垂直角度
            phi = rays_phi[i]      # 水平角度
            
            # 预计算三角函数值
            cos_theta = ti.cos(theta)
            sin_theta = ti.sin(theta)
            cos_phi = ti.cos(phi)
            sin_phi = ti.sin(phi)

            dir_local = ti.Vector([
                cos_theta * cos_phi,  # x分量
                cos_theta * sin_phi,  # y分量
                sin_theta             # z分量
            ]).normalized()  # 单位化方向向量

            ray_direction = (sensor_pose @ ti.Vector([dir_local.x, dir_local.y, dir_local.z, 0.0])).xyz.normalized()

            # 检查与每个几何体的交点
            for j in range(self.n_geoms):
                hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
                
                # 获取几何体数据
                geom_type = self.geom_types[j]
                center = self.geom_positions[j]
                size = self.geom_sizes[j]
                
                # 根据几何体类型调用相应的交点计算函数
                if geom_type == 0:  # PLANE
                    hit_result = self.ray_plane_intersection(ray_start, ray_direction, center, size)
                elif geom_type == 2:  # SPHERE
                    hit_result = self.ray_sphere_intersection(ray_start, ray_direction, center, size)
                elif geom_type == 3:  # CAPSULE
                    hit_result = self.ray_capsule_intersection(ray_start, ray_direction, center, size)
                elif geom_type == 4:  # ELLIPSOID
                    hit_result = self.ray_ellipsoid_intersection(ray_start, ray_direction, center, size)
                elif geom_type == 5:  # CYLINDER
                    hit_result = self.ray_cylinder_intersection(ray_start, ray_direction, center, size)
                elif geom_type == 6:  # BOX
                    hit_result = self.ray_box_intersection(ray_start, ray_direction, center, size)
                # 暂不支持HFIELD(1)

                # 检查是否有有效交点，并且是否是最近的
                if hit_result.w > 0 and hit_result.w < min_distance:
                    # 记录世界坐标系中的最近交点
                    self.hit_points_world[i] = ti.math.vec3(hit_result.x, hit_result.y, hit_result.z)
                    min_distance = hit_result.w
                    self.hit_mask[i] = 1  # 标记此射线有命中

            # 在遍历完当前射线的所有几何体后，进行坐标转换
            if self.hit_mask[i] == 1:  # 如果有命中
                world_hit = self.hit_points_world[i]
                # 转换为齐次坐标并应用传感器位姿的逆矩阵
                local_hit = sensor_pose_inv @ ti.Vector([world_hit.x, world_hit.y, world_hit.z, 1.0])
                hit_points[i] = local_hit.xyz

    def ray_cast_taichi(self, rays_phi, rays_theta, sensor_pose, mj_scene):
        """
        使用Taichi进行真正的并行光线追踪
        Params:
            rays_phi: 水平角度数组
            rays_theta: 垂直角度数组
            sensor_pose: 4x4 matrix, the pose of the sensor in the world frame
            mj_scene: mujoco.MjvScene object, the scene to cast rays into
        Return:
            Nx3 matrix, each row is the intersection point of the corresponding ray in the sensor frame
        """
        assert rays_phi.shape == rays_theta.shape, "rays_phi和rays_theta的形状必须相同"
        n_rays = rays_phi.shape[0]
        
        # 性能计时
        start_total = time.time()
        
        # 确保sensor_pose是float32类型
        sensor_pose = sensor_pose.astype(np.float32)
        
        # 创建Taichi ndarray并从NumPy数组填充
        self.sensor_pose_ti.from_numpy(sensor_pose)
        
        # 如果光线数量变化，重新分配内存
        if self.cached_n_rays != n_rays:
            self.rays_phi_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
            self.rays_theta_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
            self.hit_points = ti.Vector.field(3, dtype=ti.f32, shape=n_rays)
            # 同时创建临时字段
            self.hit_points_world = ti.Vector.field(3, dtype=ti.f32, shape=n_rays)
            self.hit_mask = ti.field(dtype=ti.i32, shape=n_rays)
            self.cached_n_rays = n_rays
            
        # 更新光线数据
        self.rays_phi_ti.from_numpy(rays_phi.astype(np.float32))
        self.rays_theta_ti.from_numpy(rays_theta.astype(np.float32))
        
        # 更新几何体位置
        self.update_geom_positions(mj_scene)
        
        # 准备阶段结束，记录时间
        prepare_end = time.time()
        self.prepare_time = (prepare_end - start_total) * 1000
        
        # 开始Taichi内核计算
        ti.sync()  # 确保之前的操作完成
        kernel_start = time.time()
        
        # 调用Taichi内核
        self.trace_rays(
            self.sensor_pose_ti,
            self.rays_phi_ti,
            self.rays_theta_ti,
            n_rays,
            self.hit_points
        )
        
        # 等待内核完成
        ti.sync()
        kernel_end = time.time()
        self.kernel_time = (kernel_end - kernel_start) * 1000
        
        # 结果已经在内核中转换为局部坐标系
        result = self.hit_points.to_numpy()
        
        # 计算总时间
        end_total = time.time()
        self.total_time = (end_total - start_total) * 1000
        
        # 打印详细性能信息
        print(f"准备时间: {self.prepare_time:.2f}ms, 内核时间: {self.kernel_time:.2f}ms, 总时间: {self.total_time:.2f}ms")
        
        return result


def create_lidar_rays(horizontal_resolution=360, vertical_resolution=32, horizontal_fov=2*np.pi, vertical_fov=np.pi/3):
    """创建激光雷达扫描线的角度数组"""
    h_angles = np.linspace(-horizontal_fov/2, horizontal_fov/2, horizontal_resolution)
    v_angles = np.linspace(-vertical_fov/2, vertical_fov/2, vertical_resolution)

    phi_grid, theta_grid = np.meshgrid(h_angles, v_angles)
    
    # 展平网格为一维数组
    rays_phi = phi_grid.flatten()
    rays_theta = theta_grid.flatten()
    return rays_phi, rays_theta


def create_demo_scene():
    """创建一个用于测试的mujoco场景，包含所有支持的几何体类型"""
    xml = """
    <mujoco>
        <worldbody>
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <!-- 平面 -->
        <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.9 0.9 0.9 1"/>
        
        <!-- 盒子 -->
        <geom name="box1" type="box" size="0.5 0.5 0.5" pos="2 0 0.5" rgba="1 0 0 1"/>
        <geom name="box2" type="box" size="0.3 0.8 0.2" pos="-2 -1 0.2" rgba="1 0 0 0.7"/>
        
        <!-- 球体 -->
        <geom name="sphere1" type="sphere" size="0.5" pos="0 2 0.5" rgba="0 1 0 1"/>
        <geom name="sphere2" type="sphere" size="0.3" pos="-1 2 0.3" rgba="0 1 0 0.7"/>
        
        <!-- 圆柱体 -->
        <geom name="cylinder1" type="cylinder" size="0.4 0.6" pos="0 -2 0.6" rgba="0 0 1 1"/>
        <geom name="cylinder2" type="cylinder" size="0.2 0.3" pos="2 -2 0.3" rgba="0 0 1 0.7"/>
        
        <!-- 椭球体 -->
        <geom name="ellipsoid1" type="ellipsoid" size="0.4 0.3 0.5" pos="3 2 0.5" rgba="1 1 0 1"/>
        <geom name="ellipsoid2" type="ellipsoid" size="0.2 0.4 0.3" pos="3 -1 0.3" rgba="1 1 0 0.7"/>
        
        <!-- 胶囊体 -->
        <geom name="capsule1" type="capsule" size="0.3 0.5" pos="-3 1 0.8" rgba="1 0 1 1"/>
        <geom name="capsule2" type="capsule" size="0.2 0.4" pos="-3 -2 0.6" rgba="1 0 1 0.7"/>
        
        <!-- 角落放置一组排列的几何体 -->
        <body pos="-3 3 0">
            <geom name="corner_box" type="box" size="0.2 0.2 0.2" pos="0 0 0.2" rgba="0.5 0.5 0.5 1"/>
            <geom name="corner_sphere" type="sphere" size="0.2" pos="0.5 0 0.2" rgba="0.7 0.7 0.7 1"/>
            <geom name="corner_cylinder" type="cylinder" size="0.2 0.2" pos="0 0.5 0.2" rgba="0.6 0.6 0.6 1"/>
            <geom name="corner_capsule" type="capsule" size="0.1 0.3" pos="0.5 0.5 0.3" rgba="0.8 0.8 0.8 1" euler="0 1.57 0"/>
        </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # mujoco.mj_saveLastXML("test.xml", model)
    return model, data

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

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
    rays_phi, rays_theta = create_lidar_rays(horizontal_resolution=360, vertical_resolution=32)
    print(f"射线数量: {len(rays_phi)}")

    # 设置激光雷达传感器位姿
    lidar_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # 构建激光雷达位姿矩阵
    lidar_pose = np.eye(4, dtype=np.float32)
    lidar_pose[:3, 3] = lidar_position
    
    # 更新模拟
    mujoco.mj_step(mj_model, mj_data)
    
    # 更新场景
    mujoco.mjv_updateScene(
        mj_model, mj_data, mujoco.MjvOption(), 
        None, mujoco.MjvCamera(), 
        mujoco.mjtCatBit.mjCAT_ALL.value, scene
    )
                
    # 执行光线追踪
    for _ in range(3):
        start_time = time.time()
        points = lidar.ray_cast_taichi(rays_phi, rays_theta, lidar_pose, scene)
        ti.sync()
        end_time = time.time()
    
    # 打印性能信息和当前位置
    print(f"耗时: {(end_time - start_time)*1000:.2f} ms, 射线数量: {len(rays_phi)}")
    
    # 三维点云可视化
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis')

    # 设置轴标签
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D LiDAR Point Cloud Visualization')

    # 设置三轴等比例
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()