import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Slerp, Rotation

def interpolate_camera_poses(keypoints, num_points=100):
    """
    对相机位姿进行样条插值，生成平滑的相机轨迹
    
    参数:
    keypoints (list): 关键点列表，每个元素是一个元组 (position, quaternion)
                      position: 相机位置，形状为 (3,) 的numpy数组 [x, y, z]
                      quaternion: 相机旋转四元数，形状为 (4,) 的numpy数组 [qw, qx, qy, qz]
    num_points (int): 插值后轨迹的总点数
    
    返回:
    list: 插值后的相机位姿列表，格式与输入相同
    """
    # 检查输入是否为空
    if not keypoints:
        return []
    
    # 提取位置和四元数
    positions = np.array([kp[0] for kp in keypoints])
    quaternions = np.array([kp[1] for kp in keypoints])
    
    # 归一化四元数以避免插值问题
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    
    # 创建插值时间点（假设关键点均匀分布）
    n_keypoints = len(keypoints)
    key_times = np.linspace(0, 1, n_keypoints)
    interp_times = np.linspace(0, 1, num_points)
    
    # 位置插值（使用三次样条）
    cs_x = CubicSpline(key_times, positions[:, 0])
    cs_y = CubicSpline(key_times, positions[:, 1])
    cs_z = CubicSpline(key_times, positions[:, 2])
    
    # 计算插值后的位置
    interp_positions = np.column_stack([
        cs_x(interp_times),
        cs_y(interp_times),
        cs_z(interp_times)
    ])
    
    # 旋转插值（使用SLERP - 球面线性插值）
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])  # 转换为scipy的格式 [qx, qy, qz, qw]
    slerp = Slerp(key_times, rotations)
    interp_rotations = slerp(interp_times)
    
    # 转换回四元数格式 [qw, qx, qy, qz]
    interp_quaternions = interp_rotations.as_quat()[:, [3, 0, 1, 2]]
    
    # 组合位置和四元数
    return [(pos, quat) for pos, quat in zip(interp_positions, interp_quaternions)]

# 使用示例
if __name__ == "__main__":
    import os
    import shutil
    from discoverse import DISCOVERSE_ASSETS_DIR

    # 定义一些关键点
    keypoints = [
        # (位置, 四元数)
        (np.array([0, 0, 0]), np.array([1, 0, 0, 0])),  # 初始位姿
        (np.array([1, 1, 0.5]), np.array([0.9239, 0, 0, 0.3827])),  # 旋转45度
        (np.array([2, 2, 1]), np.array([0.7071, 0, 0, 0.7071])),  # 旋转90度
        (np.array([3, 1, 0.5]), np.array([0.3827, 0, 0, 0.9239])),  # 旋转135度
        (np.array([4, 0, 0]), np.array([0, 0, 0, 1]))  # 旋转180度
    ]
    
    body_str = ""
    for i in range(len(keypoints)):
        pos, quat = keypoints[i]
        body_str += f"""<body name="keypoint_{i}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"/>\n"""

    # 生成插值轨迹
    trajectory = interpolate_camera_poses(keypoints, num_points=50)
    
    # 打印前几个插值点
    print(f"生成了 {len(trajectory)} 个插值点")
    for i in range(min(5, len(trajectory))):
        pos, quat = trajectory[i]
        print(f"点 {i}: 位置={pos.round(3)}, 四元数={quat.round(3)}")

    site_str = ""
    for i in range(len(trajectory)):
        pos, quat = trajectory[i]
        site_str += f"""<site name="trajectory_{i}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"/>\n"""

    mj_xml_str = f"""
    <mujoco>
        <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".9 .8 .7" rgb2=".4 .4 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
            <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance=".2"/>
        </asset>

        <statistic meansize="0.1"/>

        <worldbody>
            <geom name="ground" type="plane" pos="0 0 0" size="3 3 0.1" material="grid" solimp=".9 .95 .001" solref='-10000 -1000'/>
            <light pos="0 0 3" dir="0 0 -1"/>
            {body_str}
            {site_str}
        </worldbody>
    </mujoco>
    """

    py_dir = shutil.which('python3')
    if not py_dir:
        py_dir = shutil.which('python')
    if not py_dir:
        print("错误：找不到 python 或 python3 可执行文件。无法启动查看器。")
        exit(1)

    tmp_world_mjcf = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "_tmp_preview.xml")

    with open(tmp_world_mjcf, "w") as f:
        f.write(mj_xml_str)

    cmd_line = f"{py_dir} -m mujoco.viewer --mjcf={tmp_world_mjcf}"
    print(f"执行命令: {cmd_line}")
    os.system(cmd_line)

    # print(f"删除临时预览文件: {tmp_world_mjcf}")
    # os.remove(tmp_world_mjcf)
