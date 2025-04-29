import os
import argparse

import glfw
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

class CamEnv(SimulatorBase):
    """相机环境类，用于控制和显示相机视角
    
    继承自SimulatorBase类，实现了一个可交互的相机环境，允许用户通过键盘和鼠标控制相机的移动和旋转。
    主要功能包括：
    - 支持双目相机设置
    - 实时渲染RGB和深度图像
    - 通过WASD键控制相机移动
    - 通过鼠标控制相机旋转
    """
    
    # 相机的偏航角和俯仰角
    camera_yaw = 0.0      # 偏航角(绕垂直轴旋转)
    camera_pitch = 0.0    # 俯仰角(抬头/低头)

    # 是否触发空格键
    key_space_triger = False

    def __init__(self, config: BaseConfig):
        """初始化相机环境
        
        Args:
            config: 包含环境配置参数的BaseConfig对象
        """
        super().__init__(config)

    def updateControl(self, action):
        pass

    def post_load_mjcf(self):
        pass

    def getObservation(self):
        """获取环境观测信息
        
        Returns:
            dict: 包含以下键值对的字典：
                - rgb_cam_posi: RGB相机位姿列表 (position:(x,y,z), quaternion:(w,x,y,z))
                - depth_cam_posi: 深度相机位姿列表 (position:(x,y,z), quaternion:(w,x,y,z))
                - rgb_img: RGB图像字典，键为相机ID
                - depth_img: 深度图像字典，键为相机ID
        """
        rgb_cam_pose_lst = [self.getCameraPose(id) for id in self.config.obs_rgb_cam_id]
        depth_cam_pose_lst = [self.getCameraPose(id) for id in self.config.obs_depth_cam_id]
        self.obs = {
            "rgb_cam_posi"   : rgb_cam_pose_lst,
            "depth_cam_posi" : depth_cam_pose_lst,
            "rgb_img"        : self.img_rgb_obs_s,
            "depth_img"      : self.img_depth_obs_s,
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs    

    def checkTerminated(self):
        return False
    
    def getReward(self):
         return None
    
    def on_mouse_move(self, window, xpos, ypos):
        """处理鼠标移动事件
        
        当在固定相机模式下（非自由视角），通过鼠标左键拖动来控制相机的偏航角和俯仰角。
        在自由视角模式下，使用父类的默认鼠标控制逻辑。
        
        Args:
            window: GLFW窗口对象
            xpos: 鼠标X坐标
            ypos: 鼠标Y坐标
        """
        if self.cam_id == -1:
            super().on_mouse_move(window, xpos, ypos)
        else:
            if self.mouse_pressed['left']:
                self.camera_pose_changed = True
                height = self.config.render_set["height"]
                dx = float(xpos) - self.mouse_pos["x"]
                dy = float(ypos) - self.mouse_pos["y"]
                # 根据鼠标移动更新相机角度
                self.camera_yaw -= 2. * dx / height    # 左右旋转
                self.camera_pitch += 2. * dy / height  # 上下俯仰

                # 将欧拉角转换为四元数并更新相机姿态
                quat_wxyz = Rotation.from_euler("xyz", [0.0, self.camera_pitch, self.camera_yaw]).as_quat()[[3,0,1,2]]
                self.mj_model.body("stereo").quat[:] = quat_wxyz

            self.mouse_pos['x'] = xpos
            self.mouse_pos['y'] = ypos

    def on_key(self, window, key, scancode, action, mods):
        """处理键盘事件
        
        实现了以下键盘控制：
        - WASD: 在水平面上移动相机
        - QE: 垂直方向移动相机
        - Shift: 按住可以增加移动速度
        
        Args:
            window: GLFW窗口对象
            key: 按键代码
            scancode: 扫描码
            action: 按键动作（按下/释放）
            mods: 修饰键状态
        """
        super().on_key(window, key, scancode, action, mods)

        # 检查Shift键是否按下，用于调整移动速度
        is_shift_pressed = (mods & glfw.MOD_SHIFT)
        move_step_ratio = 10.0 if is_shift_pressed else 3.0  # Shift按下时移动速度更快
        step = move_step_ratio / float(self.config.render_set["fps"])

        if action == glfw.PRESS or action == glfw.REPEAT:
            # 初始化移动方向
            dxlocal, dylocal, dz = 0.0, 0.0, 0.0
            
            # 检测WASDQE按键状态并更新移动方向
            if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS or glfw.get_key(window, glfw.KEY_W) == glfw.REPEAT:
                dxlocal += step  # 前进
            if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS or glfw.get_key(window, glfw.KEY_S) == glfw.REPEAT:
                dxlocal -= step  # 后退
            if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS or glfw.get_key(window, glfw.KEY_A) == glfw.REPEAT:
                dylocal += step  # 左移
            if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS or glfw.get_key(window, glfw.KEY_D) == glfw.REPEAT:
                dylocal -= step  # 右移
            if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS or glfw.get_key(window, glfw.KEY_Q) == glfw.REPEAT:
                dz += step      # 上升
            if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS or glfw.get_key(window, glfw.KEY_E) == glfw.REPEAT:
                dz -= step      # 下降

            # 根据当前相机朝向计算实际移动方向并更新相机位置
            self.mj_model.body("stereo").pos[0] += dxlocal * np.cos(self.camera_yaw) - dylocal * np.sin(self.camera_yaw)
            self.mj_model.body("stereo").pos[1] += dxlocal * np.sin(self.camera_yaw) + dylocal * np.cos(self.camera_yaw)
            self.mj_model.body("stereo").pos[2] += dz

            if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
                self.key_space_triger = True

    def printHelp(self):
        """打印帮助信息，显示控制说明"""
        print("Camera View")
        print("W/S/A/D: 移动相机")
        print("Q/E: 升降相机")
        print("Shift: 加快移动速度")
        print("左键按住拖动: 旋转相机视角")
        print("ESC: 切换到自由视角")
        print("]/[: 切换相机")
        print("Space: 保存当前视角和图像")
        print("Ctrl+G: 切换高斯渲染")
        print("Ctrl+D: 切换深度渲染")

if __name__ == "__main__":
    # 设置numpy打印格式
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='相机环境程序')
    parser.add_argument('--gsply', type=str, required=True, help='高斯渲染模型文件路径(.ply)')
    parser.add_argument('--mesh', type=str, default=None, help='场景网格文件路径(.obj)')
    parser.add_argument('--max-depth', type=float, default=5.0, help='最大渲染深度')
    parser.add_argument('--camera-distance', type=float, default=0.1, help='双目相机基线距离')
    parser.add_argument('--fovy', type=float, default=75.0, help='相机视场角(度)')
    parser.add_argument('--width', type=int, default=1920, help='渲染图像宽度')
    parser.add_argument('--height', type=int, default=1080, help='渲染图像高度')
    args = parser.parse_args()

    # 检查高斯渲染模型文件是否存在
    if not os.path.exists(args.gsply):
        raise FileNotFoundError(f"gsply文件不存在: {args.gsply}")

    # 准备场景网格的XML描述
    asset_xml = ''
    geom_xml = ''
    if args.mesh is None:
        # 如果未指定mesh文件，尝试使用默认的scene.obj
        obj_path = os.path.join(os.path.dirname(args.gsply), "scene.obj")
        if os.path.exists(obj_path):
            asset_xml = f'  <asset>\n    <mesh name="scene" file="{obj_path}"/>\n  </asset>'
            geom_xml = f'    <geom type="mesh" rgba="0.5 0.5 0.5 1" mesh="scene"/>'
    elif os.path.exists(args.mesh):
        # 使用指定的mesh文件
        asset_xml = f'  <asset>\n    <mesh name="scene" file="{args.mesh}"/>\n  </asset>'
        geom_xml = f'    <geom type="mesh" rgba="0.5 0.5 0.5 1" mesh="scene"/>'

    # 构建MuJoCo场景XML
    camera_env_xml = f"""
    <mujoco model="camera_env">
      <option integrator="RK4" solver="Newton" gravity="0 0 0"/>
      {asset_xml}
      <worldbody>
        {geom_xml}
        <body name="stereo" pos="0 0 1" quat="1 0 0 0">
          <camera name="camera_left" fovy="{args.fovy}" pos="0 {args.camera_distance/2.} 0" quat="0.5 0.5 -0.5 -0.5"/>
          <site pos="0 {args.camera_distance/2.} 0" quat="0.5 -0.5 0.5 -0.5"/>
          <camera name="camera_right" fovy="{args.fovy}" pos="0 {-args.camera_distance/2.} 0" quat="0.5 0.5 -0.5 -0.5"/>
          <site pos="0 {-args.camera_distance/2.} 0" quat="0.5 -0.5 0.5 -0.5"/>
          <geom type="box" size="0.05 0.2 0.05" rgba="1 1 1 0."/>
        </body>
      </worldbody>
    </mujoco>"""

    # 临时保存场景XML文件
    xml_save_path = os.path.join(os.path.dirname(args.gsply), "_camera_env.xml")

    # 配置环境参数
    cfg = BaseConfig()
    cfg.render_set["fps"] = 30                    # 渲染帧率
    cfg.render_set["width"] = args.width          # 渲染宽度
    cfg.render_set["height"] = args.height        # 渲染高度
    cfg.timestep = 1./cfg.render_set["fps"]       # 时间步长
    cfg.decimation = 1                            # 仿真步数
    cfg.mjcf_file_path = xml_save_path           # MuJoCo场景文件路径
    cfg.max_render_depth = args.max_depth        # 最大渲染深度
    cfg.obs_rgb_cam_id = [0, 1]                  # 启用的RGB相机ID（左右相机）
    cfg.obs_depth_cam_id = [0, 1]                # 启用的深度相机ID（左右相机）

    # 配置高斯渲染器
    cfg.use_gaussian_renderer = True
    cfg.gs_model_dict["background"] = args.gsply

    # 检查并加载环境点云
    env_ply_path = os.path.join(os.path.dirname(args.gsply), "environment.ply")
    if os.path.exists(env_ply_path):
        cfg.gs_model_dict["background_env"] = env_ply_path

    # 创建并配置相机环境
    with open(xml_save_path, "w") as f:
        f.write(camera_env_xml)
    robot = CamEnv(cfg)
    robot.options.label = mujoco.mjtLabel.mjLABEL_CAMERA.value
    robot.options.frame = mujoco.mjtFrame.mjFRAME_SITE.value
    robot.options.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
    os.remove(xml_save_path)

    # 重置环境并获取初始观测
    obs = robot.reset()
    rgb_cam_posi = obs["rgb_cam_posi"]
    depth_cam_posi = obs["depth_cam_posi"]
    rgb_img_0 = obs["rgb_img"][0]
    rgb_img_1 = obs["rgb_img"][1]
    depth_img_0 = obs["depth_img"][0]
    depth_img_1 = obs["depth_img"][1]

    # 打印左相机信息
    print('>>>>>> camera_left:')
    print("rgb_cam_posi    = ", rgb_cam_posi[0])
    print("depth_cam_posi  = ", depth_cam_posi[0])
    print("rgb_img_0.shape = ", rgb_img_0.shape, "rgb_img_0.dtype = ", rgb_img_0.dtype)
    print("depth_img_0.shape = ", depth_img_0.shape, "depth_img_0.dtype = ", depth_img_0.dtype)

    # 打印右相机信息
    print('>>>>>> camera_right:')
    print("rgb_cam_posi    = ", rgb_cam_posi[1])
    print("depth_cam_posi  = ", depth_cam_posi[1])
    print("rgb_img_1.shape = ", rgb_img_1.shape, "rgb_img_1.dtype = ", rgb_img_1.dtype)
    print("depth_img_1.shape = ", depth_img_1.shape, "depth_img_1.dtype = ", depth_img_1.dtype)

    # 打印控制说明
    print("-" * 50)
    robot.printHelp()
    print("-" * 50)

    # 设置初始相机和视角参数
    robot.cam_id = 0                  # 默认使用左相机视角
    robot.free_camera.distance = 1.   # 设置自由视角相机距离

    # 主循环
    while robot.running:
        obs, _, _, _, _ = robot.step()

        if robot.key_space_triger:
            robot.key_space_triger = False
            # 按下空格键会被触发，可以用来记录当前视角和图像，例如
            # cv2.imwrite(f"rgb_img_{robot.render_cnt}.png", rgb_img_0)
            # np.save(f"depth_img_{robot.render_cnt}.npy", depth_img_0)
            # 保存当前相机位姿
            # print(rgb_cam_posi[0], rgb_cam_posi[1])
            # print(depth_cam_posi[0], depth_cam_posi[1])
