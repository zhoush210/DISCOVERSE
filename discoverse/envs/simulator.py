import os
import time
import traceback
from abc import abstractmethod

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import glfw
import OpenGL.GL as gl
import ctypes
import cv2
import sys

from discoverse import DISCOVERSE_ASSERT_DIR
from discoverse.utils import BaseConfig

try:
    from discoverse.gaussian_renderer import GSRenderer
    from discoverse.gaussian_renderer.util_gau import multiple_quaternion_vector3d, multiple_quaternions
    DISCOVERSE_GAUSSIAN_RENDERER = True

except ImportError:
    traceback.print_exc()
    print("Warning: gaussian_splatting renderer not found. Please install the required packages to use it.")
    DISCOVERSE_GAUSSIAN_RENDERER = False


def setRenderOptions(options):
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
    options.frame = mujoco.mjtFrame.mjFRAME_BODY.value
    pass

#主进程
class SimulatorBase:
    # 核心属性
    running = True
    obs = None

    # 相机相关
    cam_id = -1  # -1表示自由视角
    last_cam_id = -1
    render_cnt = 0
    camera_names = []
    camera_pose_changed = False
    camera_rmat = np.array([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0],
    ])

    # 鼠标状态
    mouse_pressed = {
        'left': False,
        'right': False,
        'middle': False
    }
    mouse_pos = {
        'x': 0,
        'y': 0
    }

    # Mujoco选项
    options = mujoco.MjvOption()

    # 初始化
    def __init__(self, config:BaseConfig):
        self.config = config

        if self.config.mjcf_file_path.startswith("/"):
            self.mjcf_file = self.config.mjcf_file_path
        else:
            self.mjcf_file = os.path.join(DISCOVERSE_ASSERT_DIR, self.config.mjcf_file_path)
        if os.path.exists(self.mjcf_file):
            print("mjcf found: {}".format(self.mjcf_file))
        else:
            print("\033[0;31;40mFailed to load mjcf: {}\033[0m".format(self.mjcf_file))
            raise FileNotFoundError("Failed to load mjcf: {}".format(self.mjcf_file))
        self.load_mjcf()
        self.decimation = self.config.decimation
        self.delta_t = self.mj_model.opt.timestep * self.decimation
        self.render_fps = self.config.render_set["fps"]

        if self.config.enable_render:
            self.free_camera = mujoco.MjvCamera()
            self.free_camera.fixedcamid = -1
            self.free_camera.type = mujoco._enums.mjtCamera.mjCAMERA_FREE
            mujoco.mjv_defaultFreeCamera(self.mj_model, self.free_camera)

            self.config.use_gaussian_renderer = self.config.use_gaussian_renderer and DISCOVERSE_GAUSSIAN_RENDERER
            if self.config.use_gaussian_renderer:
                self.gs_renderer = GSRenderer(self.config.gs_model_dict, self.config.render_set["width"], self.config.render_set["height"])
                self.last_cam_id = self.cam_id
                self.show_gaussian_img = True
                if self.cam_id == -1:
                    self.gs_renderer.set_camera_fovy(self.mj_model.vis.global_.fovy * np.pi / 180.)
                else:
                    self.gs_renderer.set_camera_fovy(self.mj_model.cam_fovy[self.cam_id] * np.pi / 180.0)

        self.window = None  # 明确初始化window为None
        self.glfw_initialized = False  # 添加GLFW初始化标志
        
        # 确保render_set中包含所需的所有配置
        if not hasattr(self.config.render_set, "window_title"):
            self.config.render_set["window_title"] = "DISCOVERSE Simulator"
        
        if not self.config.headless:
            try:
                if not glfw.init():
                    raise RuntimeError("无法初始化GLFW")
                self.glfw_initialized = True
                
                # 设置OpenGL版本和窗口属性
                glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
                glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
                glfw.window_hint(glfw.VISIBLE, True)
                
                # 创建窗口
                self.window = glfw.create_window(
                    self.config.render_set["width"],
                    self.config.render_set["height"],
                    self.config.render_set.get("window_title", "DISCOVERSE Simulator"),
                    None, None
                )
                
                if not self.window:
                    glfw.terminate()
                    raise RuntimeError("无法创建GLFW窗口")
                
                glfw.make_context_current(self.window)
                
                # 初始化OpenGL设置
                gl.glClearColor(0.0, 0.0, 0.0, 1.0)
                gl.glShadeModel(gl.GL_SMOOTH)
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                
                # 设置回调
                glfw.set_key_callback(self.window, self.on_key)
                glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)
                glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
                
            except Exception as e:
                print(f"GLFW初始化失败: {e}")
                if self.glfw_initialized:
                    glfw.terminate()
                self.config.headless = True
                self.window = None

        # 1. 记录最后一次渲染的时间，用于控制帧率
        self.last_render_time = time.time()
        # 2. 重置物理引擎的数据到初始状态
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        # 3. 计算物理状态（位置、速度、加速度等）
        mujoco.mj_forward(self.mj_model, self.mj_data)

    # 加载MJCF模型文件
    def load_mjcf(self):
        if self.mjcf_file.endswith(".xml"):
            self.mj_model = mujoco.MjModel.from_xml_path(self.mjcf_file)
        elif self.mjcf_file.endswith(".mjb"):
            self.mj_model = mujoco.MjModel.from_binary_path(self.mjcf_file)
        self.mj_model.opt.timestep = self.config.timestep
        self.mj_data = mujoco.MjData(self.mj_model)
        if self.config.enable_render:
            for i in range(self.mj_model.ncam):
                self.camera_names.append(self.mj_model.camera(i).name)

            if type(self.config.obs_rgb_cam_id) is int:
                assert -2 < self.config.obs_rgb_cam_id < len(self.camera_names), "Invalid obs_rgb_cam_id {}".format(self.config.obs_rgb_cam_id)
                tmp_id = self.config.obs_rgb_cam_id
                self.config.obs_rgb_cam_id = [tmp_id]
            elif type(self.config.obs_rgb_cam_id) is list:
                for cam_id in self.config.obs_rgb_cam_id:
                    assert -2 < cam_id < len(self.camera_names), "Invalid obs_rgb_cam_id {}".format(cam_id)
            elif self.config.obs_rgb_cam_id is None:
                self.config.obs_rgb_cam_id = []
            
            if type(self.config.obs_depth_cam_id) is int:
                assert -2 < self.config.obs_depth_cam_id < len(self.camera_names), "Invalid obs_depth_cam_id {}".format(self.config.obs_depth_cam_id)
            elif type(self.config.obs_depth_cam_id) is list:
                for cam_id in self.config.obs_depth_cam_id:
                    assert -2 < cam_id < len(self.camera_names), "Invalid obs_depth_cam_id {}".format(cam_id)
            elif self.config.obs_depth_cam_id is None:
                self.config.obs_depth_cam_id = []

            self.renderer = mujoco.Renderer(self.mj_model, self.config.render_set["height"], self.config.render_set["width"])

        self.post_load_mjcf()

    # 加载MJCF模型文件后，设置窗口标题
    def post_load_mjcf(self):
        self.config.render_set["window_title"] = "DISCOVERSE Simulator"  # 添加默认标题

    # 3. ★在类中，只要调用render，就会把图像渲染到GLFW窗口
    def render(self):
        # 1. 更新高斯场景
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            self.update_gs_scene()
        
        # 2. 获取RGB图像
        depth_rendering = self.renderer._depth_rendering
        self.renderer.disable_depth_rendering()
        self.img_rgb_obs_s = {}
        for id in self.config.obs_rgb_cam_id:
            img = self.getRgbImg(id)
            self.img_rgb_obs_s[id] = img
        
        # 3. 获取深度图像
        self.renderer.enable_depth_rendering()
        self.img_depth_obs_s = {}
        for id in self.config.obs_depth_cam_id:
            img = self.getDepthImg(id)
            self.img_depth_obs_s[id] = img
        self.renderer._depth_rendering = depth_rendering
        
        # 4. 准备渲染图像
        if not self.renderer._depth_rendering:
            if self.cam_id in self.config.obs_rgb_cam_id:
                img_vis = self.img_rgb_obs_s[self.cam_id]
            else:
                img_rgb = self.getRgbImg(self.cam_id)
                img_vis = img_rgb
        else:
            if self.cam_id in self.config.obs_depth_cam_id:
                img_depth = self.img_depth_obs_s[self.cam_id]
            else:
                img_depth = self.getDepthImg(self.cam_id)
            
            if img_depth is not None:
                img_vis = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=25.5), cv2.COLORMAP_JET)
            else:
                img_vis = None

        # 5. GLFW渲染
        if not self.config.headless and self.window is not None:
            try:
                if glfw.window_should_close(self.window):
                    self.running = False
                    return
                    
                glfw.make_context_current(self.window)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                
                # 设置正确的视口
                gl.glViewport(0, 0, self.config.render_set["width"], 
                             self.config.render_set["height"])
                
                # 设置正确的光栅化位置
                gl.glRasterPos2i(-1, -1)
                
                # 在 macOS 上进行像素缩放
                if sys.platform == "darwin":  # 检查是否为 macOS 系统
                    gl.glPixelZoom(2.0, 2.0)
                
                if img_vis is not None:
                    # 将图像转换为opengl坐标
                    img_vis = img_vis[::-1]
                    # 确保数据连续性
                    img_vis = np.ascontiguousarray(img_vis)
                    # 直接绘制像素
                    gl.glDrawPixels(
                        img_vis.shape[1],
                        img_vis.shape[0],
                        gl.GL_RGB,
                        gl.GL_UNSIGNED_BYTE,
                        img_vis.tobytes()
                    )
                
                # 交换缓冲区并处理事件
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                
                # 帧率控制
                if self.config.sync:
                    current_time = time.time()
                    wait_time = max(1.0/self.render_fps - (current_time - self.last_render_time), 0)
                    if wait_time > 0:
                        time.sleep(wait_time)
                    self.last_render_time = time.time()
                    
            except Exception as e:
                print(f"渲染错误: {e}")

    # 基于mujoco与高斯渲染器获取RGB图像，按G切换
    def getRgbImg(self, cam_id):
        # 获取RGB图像, 3D场景渲染成2D图像
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            # 1.使用高斯渲染器
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
                self.gs_renderer.set_camera_fovy(self.mj_model.vis.global_.fovy * np.pi / 180.0)
            if self.last_cam_id != cam_id and cam_id > -1:
                self.gs_renderer.set_camera_fovy(self.mj_model.cam_fovy[cam_id] * np.pi / 180.0)
            self.last_cam_id = cam_id
            trans, quat_wxyz = self.getCameraPose(cam_id)
            self.gs_renderer.set_camera_pose(trans, quat_wxyz[[1,2,3,0]])
            return self.gs_renderer.render()
        else:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            elif cam_id > -1:
                self.renderer.update_scene(self.mj_data, self.camera_names[cam_id], self.options)
            else:
                return None
            # 2. 使用mujoco渲染器
            rgb_img = self.renderer.render()
            return rgb_img

    # 基于mujoco与高斯渲染器获取深度图像，按D切换
    def getDepthImg(self, cam_id):
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            if self.last_cam_id != cam_id:
                if cam_id == -1:
                    self.gs_renderer.set_camera_fovy(np.pi * 0.25)
                elif cam_id > -1:
                    self.gs_renderer.set_camera_fovy(self.mj_model.cam_fovy[cam_id] * np.pi / 180.0)
                else:
                    return None
            self.last_cam_id = cam_id
            trans, quat_wxyz = self.getCameraPose(cam_id)
            self.gs_renderer.set_camera_pose(trans, quat_wxyz[[1,2,3,0]])
            return self.gs_renderer.render(render_depth=True)
        else:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            elif cam_id > -1:
                self.renderer.update_scene(self.mj_data, self.camera_names[cam_id], self.options)
            else:
                return None
            depth_img = self.renderer.render()
            return depth_img
        
    def on_mouse_move(self, window, xpos, ypos):
        """鼠标移动事件处理"""
        if self.cam_id == -1:  # 只在自由视角模式下处理
            dx = xpos - self.mouse_pos['x']
            dy = ypos - self.mouse_pos['y']
            height = self.config.render_set["height"]
            
            action = None
            # 左键拖动：旋转相机
            if self.mouse_pressed['left']:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
            # 右键拖动：移动相机
            elif self.mouse_pressed['right']:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V
            # 中键拖动：缩放相机
            elif self.mouse_pressed['middle']:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM

            # 更新相机位置
            if action is not None:
                self.camera_pose_changed = True
                mujoco.mjv_moveCamera(
                    self.mj_model, 
                    action, 
                    dx/height, 
                    dy/height,
                    self.renderer.scene,
                    self.free_camera
                )

        self.mouse_pos['x'] = xpos
        self.mouse_pos['y'] = ypos

    def on_mouse_button(self, window, button, action, mods):
        """鼠标按键事件处理"""
        is_pressed = action == glfw.PRESS
        
        # 更新按键状态
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_pressed['left'] = is_pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_pressed['right'] = is_pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.mouse_pressed['middle'] = is_pressed

    def on_key(self, window, key, scancode, action, mods):
        """GLFW键盘回调函数"""

        if action == glfw.PRESS:  # 只在按下时触发，避免持续触发
            # 检查是否按下Ctrl键
            is_ctrl_pressed = (mods & glfw.MOD_CONTROL)
            
            # 处理组合键
            if is_ctrl_pressed:
                if key == glfw.KEY_G:  # Ctrl + G
                    if self.config.use_gaussian_renderer:
                        self.show_gaussian_img = not self.show_gaussian_img
                        self.gs_renderer.renderer.need_rerender = True
                        
                elif key == glfw.KEY_D:  # Ctrl + D
                    if self.config.use_gaussian_renderer:
                        self.gs_renderer.renderer.need_rerender = True
                    if self.renderer._depth_rendering:
                        self.renderer.disable_depth_rendering()
                    else:
                        self.renderer.enable_depth_rendering()

            # 处理单个按键
            else:
                if key == glfw.KEY_H:  # 'h': 显示帮助
                    self.printHelp()

                elif key == glfw.KEY_P:  # 'p': 打印信息
                    self.printMessage()

                elif key == glfw.KEY_R:  # 'r': 重置状态
                    self.reset()

                elif key == glfw.KEY_ESCAPE:  # ESC: 切换到自由视角
                    self.cam_id = -1
                    self.camera_pose_changed = True
                    
                elif key == glfw.KEY_RIGHT_BRACKET:  # ']': 下一个相机
                    if self.mj_model.ncam:
                        self.cam_id += 1
                        self.cam_id = self.cam_id % self.mj_model.ncam
                        
                elif key == glfw.KEY_LEFT_BRACKET:  # '[': 上一个相机
                    if self.mj_model.ncam:
                        self.cam_id += self.mj_model.ncam - 1
                        self.cam_id = self.cam_id % self.mj_model.ncam

    def printHelp(self):
        """打印帮助信息"""
        print("\n=== 键盘控制说明 ===")
        print("H: 显示此帮助信息")
        print("P: 打印当前状态信息")
        print("R: 重置模拟器状态")
        print("G: 切换高斯渲染（如果可用）")
        print("D: 切换深度渲染")
        print("Ctrl+G: 组合键切换高斯模式")
        print("Ctrl+D: 组合键切换深度图模式")
        print("ESC: 切换到自由视角")
        print("[: 切换到上一个相机")
        print("]: 切换到下一个相机")
        print("\n=== 鼠标控制说明 ===")
        print("左键拖动: 旋转视角")
        print("右键拖动: 平移视角")
        print("中键拖动: 缩放视角")
        print("================\n")

    def printMessage(self):
        """打印当前状态信息"""
        print("\n=== 当前状态 ===")
        print(f"当前相机ID: {self.cam_id}")
        if self.cam_id >= 0:
            print(f"相机名称: {self.camera_names[self.cam_id]}")
        print(f"高斯渲染: {'开启' if self.show_gaussian_img else '关闭'}")
        print(f"深度渲染: {'开启' if self.renderer._depth_rendering else '关闭'}")
        print("==============\n")

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.camera_pose_changed = True

    def update_gs_scene(self):
        # 更新场景状态
        for name in self.config.obj_list + self.config.rb_link_list:
            trans, quat_wxyz = self.getObjPose(name)
            self.gs_renderer.set_obj_pose(name, trans, quat_wxyz)

        if self.gs_renderer.update_gauss_data:
            self.gs_renderer.update_gauss_data = False
            self.gs_renderer.renderer.need_rerender = True
            self.gs_renderer.renderer.gaussians.xyz[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternion_vector3d(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]) + self.gs_renderer.renderer.gau_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]
            self.gs_renderer.renderer.gaussians.rot[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternions(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:])

    def getObjPose(self, name):
        try:
            position = self.mj_data.body(name).xpos
            quat = self.mj_data.body(name).xquat
            return position, quat
        except KeyError:
            try:
                position = self.mj_data.geom(name).xpos
                quat = Rotation.from_matrix(self.mj_data.geom(name).xmat.reshape((3,3))).as_quat()[[3,0,1,2]]
                return position, quat
            except KeyError:
                print("Invalid object name: {}".format(name))
                return None, None
    
    def getCameraPose(self, cam_id):
        if cam_id == -1:
            rotation_matrix = self.camera_rmat @ Rotation.from_euler('xyz', [self.free_camera.elevation * np.pi / 180.0, self.free_camera.azimuth * np.pi / 180.0, 0.0]).as_matrix()
            camera_position = self.free_camera.lookat + self.free_camera.distance * rotation_matrix[:3,2]
        else:
            rotation_matrix = np.array(self.mj_data.camera(self.camera_names[cam_id]).xmat).reshape((3,3))
            camera_position = self.mj_data.camera(self.camera_names[cam_id]).xpos

        return camera_position, Rotation.from_matrix(rotation_matrix).as_quat()[[3,0,1,2]]

    def __del__(self):
        """清理资源"""
        try:
            # 清理GLFW资源
            if hasattr(self, 'window') and self.window is not None:
                if glfw.get_current_context() is not None:
                    glfw.destroy_window(self.window)
                    self.window = None
            
            # 终止GLFW
            if hasattr(self, 'glfw_initialized') and self.glfw_initialized:
                try:
                    if glfw.get_current_context() is not None:
                        glfw.terminate()
                except Exception:
                    pass
            
        except Exception as e:
            print(f"清理资源时出错: {str(e)}")
        
        finally:
            try:
                super().__del__()
            except Exception:
                pass

    # ------------------------------------------------------------------------------
    # ---------------------------------- Override ----------------------------------
    def reset(self):
        self.resetState()
        self.render()
        self.render_cnt = 0
        return self.getObservation()

    def updateControl(self, action):
        pass

    # 包含了一些需要子类实现的抽象方法
    @abstractmethod
    def post_physics_step(self):
        pass

    @abstractmethod
    def getChangedObjectPose(self):
        raise NotImplementedError("pubObjectPose is not implemented")

    @abstractmethod
    def checkTerminated(self):
        raise NotImplementedError("checkTerminated is not implemented")    

    @abstractmethod
    def getObservation(self):
        raise NotImplementedError("getObservation is not implemented")

    @abstractmethod
    def getPrivilegedObservation(self):
        raise NotImplementedError("getPrivilegedObservation is not implemented")

    @abstractmethod
    def getReward(self):
        raise NotImplementedError("getReward is not implemented")
    
    # ---------------------------------- Override ----------------------------------
    # ------------------------------------------------------------------------------

    def step(self, action=None): # 主要的仿真步进函数
        # 1. 执行多步物理仿真
        for _ in range(self.decimation): # decimation是每次step执行的物理仿真次数
            self.updateControl(action) # 更新控制输入,如机器人关节力矩
            mujoco.mj_step(self.mj_model, self.mj_data) #★①物理引擎计算出场景状态

        # 2. 检查是否终止
        if self.checkTerminated(): # 检查是否终止
            self.resetState()
        
        self.post_physics_step()
        if self.config.enable_render and self.render_cnt-1 < self.mj_data.time * self.render_fps:
            self.render()

        # 4. 返回观测、私有观测、奖励、终止状态、其他信息
        return self.getObservation(), self.getPrivilegedObservation(), self.getReward(), self.checkTerminated(), {}

    def view(self):
        # 1. 更新时间
        self.mj_data.time += self.delta_t
        # 2. 设置速度为0
        self.mj_data.qvel[:] = 0
        # 3. 执行物理仿真
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # 4. 如果需要渲染，渲染图像
        if self.render_cnt-1 < self.mj_data.time * self.render_fps:
            self.render()