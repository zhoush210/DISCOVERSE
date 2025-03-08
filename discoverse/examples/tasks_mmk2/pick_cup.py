import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import shutil
import argparse
import multiprocessing as mp
from scipy.spatial.transform import Rotation

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.envs.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase, recoder_mmk2, copypy2
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine

class SimNode(MMK2TaskBase):
    # gripper_done_limit = 1.0

    def domain_randomization(self):
        # 随机box位置
        box_x_bios = (np.random.random() - 0.5) * 0.1
        box_y_bios = (np.random.random() - 0.5) * 0.1
        self.mj_data.qpos[self.njq+7*0+0] += box_x_bios
        self.mj_data.qpos[self.njq+7*0+1] += box_y_bios


        # 随机cup位置
        cup_x_bios = (np.random.random() - 0.5) * 0.1
        cup_y_bios = (np.random.random() - 0.5) * 0.1
        self.mj_data.qpos[self.njq+7*1+0] += cup_x_bios
        self.mj_data.qpos[self.njq+7*1+1] += cup_y_bios

    def check_success(self):
        v1 = np.array([self.mj_data.qpos[self.njq+7*0+0], self.mj_data.qpos[self.njq+7*0+1]])
        v2 = np.array([self.mj_data.qpos[self.njq+7*1+0], self.mj_data.qpos[self.njq+7*1+1]])
        
        # 计算差异
        diff = v1 - v2
        
        # 计算平方和
        squared_diff = np.sum(diff**2)
        
        # 取平方根
        distance = np.sqrt(squared_diff) - 0.0195
        print(distance)
        if distance < 0.1:
            return True

        return False
    
cfg = MMK2Cfg()
cfg.use_gaussian_renderer = True
cfg.init_key = "pick"
cfg.gs_model_dict["box"]     = "object/box.ply"
cfg.gs_model_dict["cup"]          = "object/cup_blue.ply"
cfg.gs_model_dict["background"]      = "scene/lab3/environment_aligned.ply"

cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_cup.xml"
cfg.obj_list    = ["box", "cup"]
cfg.sync     = False
cfg.headless = False
cfg.render_set  = {
    "fps"    : 25,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id = [0,1,2]
cfg.save_mjb_and_task_config = True