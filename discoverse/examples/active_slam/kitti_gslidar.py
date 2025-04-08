import os
import argparse

import glfw
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

class CamEnvConfig(BaseConfig):
    robot_name = "camera"

class CamEnv(SimulatorBase):
    def __init__(self, config: CamEnvConfig):
        super().__init__(config)

        # 加载LiDAR位姿数组
        lidar_poses = None
        if self.config.lidar_pose_npy is not None and os.path.exists(self.config.lidar_pose_npy):
            try:
                lidar_poses = np.load(self.config.lidar_pose_npy)
                print(f"已加载{len(lidar_poses)}个LiDAR位姿从文件: {self.config.lidar_pose_npy}")
                print(f"位姿数组形状: {lidar_poses.shape}")
                if len(lidar_poses.shape) != 3 or lidar_poses.shape[1:] != (4, 4):
                    print(f"警告: 位姿数组形状不是(N, 4, 4)，而是{lidar_poses.shape}，将不使用预设位姿")
                    lidar_poses = None
            except Exception as e:
                print(f"加载位姿文件失败: {e}")
                lidar_poses = None
        self.lidar_poses = lidar_poses
        self.current_pose_index = 0

    def load_pose_from_array(self, index):
        if self.lidar_poses is None:
            print("no lidar poses loaded")
            return

        if self.current_pose_index == index:
            return

        print(f"新相机索引: {index}")

        # 确保索引在有效范围内
        index = max(0, min(index, len(self.lidar_poses) - 1))
        self.current_pose_index = index
    
        # 加载位姿矩阵
        tmat_tmp = self.lidar_poses[index].copy()
        print(tmat_tmp)        

        self.mj_model.body("base_body").pos[:] = tmat_tmp[:3, 3]
        self.mj_model.body("base_body").quat[:] = Rotation.from_matrix(tmat_tmp[:3, :3]).as_quat()[[3,0,1,2]]
        self.cam_id = 0

    def updateControl(self, action):
        pass

    def get_base_pose(self):
        return self.mj_model.body(self.config.robot_name).pos.copy(), self.mj_model.body(self.config.robot_name).quat.copy()

    def post_load_mjcf(self):
        pass

    def getObservation(self):
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
        super().on_mouse_move(window, xpos, ypos)

    def on_key(self, window, key, scancode, action, mods):
        super().on_key(window, key, scancode, action, mods)

        is_shift_pressed = (mods & glfw.MOD_SHIFT)

        if action == glfw.PRESS or action == glfw.REPEAT:
            # 同时监控多个按键
            if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
                pass
            
            increment = 0
            if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
                increment = -1
            elif glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
                increment = 1

            elif glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
                pass

            new_index = (self.current_pose_index + increment) % len(self.lidar_poses)
            self.load_pose_from_array(new_index)

    def printHelp(self):
        super().printHelp()
        print("-------------------------------------")
        print("dummy robot control:")
        print("arrow left/right : switch camera")

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser(description='Camera Environment')
    parser.add_argument('--lidar-pose-npy', type=str, default=None, help='LiDAR位姿文件路径，包含多个4x4位姿矩阵')
    args = parser.parse_args()

    cfg = CamEnvConfig()
    cfg.lidar_pose_npy = args.lidar_pose_npy

    cfg.render_set["fps"] = 60
    cfg.render_set["width"] = 640
    cfg.render_set["height"] = 480
    cfg.timestep = 1./cfg.render_set["fps"]
    cfg.decimation = 1
    cfg.mjcf_file_path = "mjcf/camera_env.xml"

    cfg.use_gaussian_renderer = True
    # cfg.gs_model_dict["background"] = "scene/kitti/room_2dgs_dense.ply"
    # cfg.gs_model_dict["background"] = "scene/kitti/kitti.ply"
    cfg.gs_model_dict["background"] = "scene/kitti/qz_table_2dg.ply"

    robot = CamEnv(cfg)
    robot.cam_id = -1

    action = np.zeros(4)

    obs = robot.reset()
    robot.printHelp()

    while robot.running:
        obs, _, _, _, _ = robot.step(action)