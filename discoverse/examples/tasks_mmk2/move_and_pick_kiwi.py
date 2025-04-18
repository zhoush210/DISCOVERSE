import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import shutil
import argparse
import multiprocessing as mp

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.envs.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase, recoder_mmk2, copypy2
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine

class SimNode(MMK2TaskBase):
    gripper_done_limit = 1.0
    def domain_randomization(self):
        random_x = (np.random.random() - 0.5) * 0.1
        # 随机 kiwi位置
        self.mj_data.qpos[self.njq+7*0+0] += random_x
        # 随机 box 位置
        random_x = (np.random.random() - 0.5) * 0.1
        random_y = (np.random.random() - 0.5) * 0.1
        self.mj_data.qpos[self.njq+7*2+0] += random_x
        self.mj_data.qpos[self.njq+7*2+1] += random_y
        # 盒子里物品旋转
        # angle = (np.random.random() - 0.5) * np.radians(90) 
        # random_rotation = Rotation.from_euler('x', angle).as_quat()
        # current_quat = self.mj_data.qpos[self.njq+7*4+3 : self.njq+7*4+7]
        # new_quat = (Rotation.from_quat(random_rotation) * Rotation.from_quat(current_quat)).as_quat()
        # self.mj_data.qpos[self.njq+7*4+3 : self.njq+7*4+7] = new_quat

        # 盒子高度
        # if np.random.random() < 0.33:
        #     self.mj_data.qpos[self.njq+7*1+2] += -0.3
        # elif np.random.random() < 0.667:
        #     self.mj_data.qpos[self.njq+7*1+2] += 0.0
        # else:
        #     self.mj_data.qpos[self.njq+7*1+2] += 0.3

    def check_success(self):
        # box和disk都在桌面上
        if abs(self.mj_data.qpos[self.njq+7*1+2] - 0.75) < 0.1 and abs(self.mj_data.qpos[self.njq+7*14+2] - 0.75) < 0.1 :
            return True

        return False

cfg = MMK2Cfg()
cfg.use_gaussian_renderer = True
cfg.init_key = "pick"
cfg.gs_model_dict["box_disk"]       = "s2r2025/box.ply"
cfg.gs_model_dict["kiwi"]        = "object/kiwi.ply"
cfg.gs_model_dict["background"]     = "scene/s2r2025/point_cloud.ply"

cfg.mjcf_file_path = "mjcf/s2r2025_env_pick_kiwi.xml"
cfg.obj_list    = ["box_disk", "kiwi"]
cfg.sync     = False
cfg.headless = False
cfg.render_set  = {
    "fps"    : 25,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id = [0,1,2]
cfg.save_mjb_and_task_config = True