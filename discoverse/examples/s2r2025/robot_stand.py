import os
import numpy as np

from discoverse import DISCOVERSE_ASSERT_DIR
from discoverse.envs.mmk2_base import MMK2Base, MMK2Cfg

cfg = MMK2Cfg()
cfg.mjcf_file_path = "mjcf/s2r2025_env.xml"
cfg.init_key = "pick"
cfg.sync     = True
cfg.headless = False
cfg.render_set = {
    "fps"    : 30,
    "width"  : 1920,
    "height" : 1080 
}
cfg.obj_list = []

cfg.gs_model_dict["background"] = "scene/s2r2025/point_cloud.ply"
# cfg.gs_model_dict[""]        = ""

cfg.obs_rgb_cam_id   = None
cfg.obs_depth_cam_id = None
cfg.use_gaussian_renderer = False

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    sim_node = MMK2Base(cfg)
    obs = sim_node.reset()
    action = sim_node.init_joint_ctrl.copy()

    while sim_node.running:
        obs, _, _, _, _ = sim_node.step(action)
