import os
import numpy as np

from discoverse import DISCOVERSE_ASSERT_DIR
from discoverse.envs.mmk2_base import MMK2Base, MMK2Cfg

cfg = MMK2Cfg()
cfg.mjcf_file_path = "mjcf/s2r2025_env.xml"
cfg.timestep       = 0.002
cfg.decimation     = 5
cfg.init_key = "pick"
cfg.sync     = True
cfg.headless = False
cfg.render_set = {
    "fps"    : 30,
    "width"  : 1920,
    "height" : 1080 
}
cfg.obj_list = [
    "box_carton" , "box_disk"    , "box_sheet"     ,
    "carton_01"  , "disk_01"     , "sheet_01"      ,
    "apple"      , "book"        , "cup"           ,
    "kettle"     , "scissors"    , "timeclock"     ,
    "wood"       , "xbox"        , "yellow_bowl"   ,
    "toy_cabinet", "cabinet_door", "cabinet_drawer",
]

cfg.gs_model_dict["background"] = "scene/s2r2025/point_cloud_trans.ply"
cfg.gs_model_dict["box_carton"] = "s2r2025/box.ply"
cfg.gs_model_dict["box_disk"]   = "s2r2025/box.ply"
cfg.gs_model_dict["box_sheet"]  = "s2r2025/box.ply"
cfg.gs_model_dict["carton_01"]  = "s2r2025/carton_01.ply"
cfg.gs_model_dict["disk_01"]    = "s2r2025/disk_01.ply"
cfg.gs_model_dict["sheet_01"]   = "s2r2025/sheet_01.ply"
cfg.gs_model_dict["apple"]      = "s2r2025/apple.ply"
cfg.gs_model_dict["book"]       = "s2r2025/book.ply"
cfg.gs_model_dict["cup"]        = "s2r2025/cup.ply"
cfg.gs_model_dict["kettle"]     = "s2r2025/kettle.ply"
cfg.gs_model_dict["scissors"]   = "s2r2025/scissors.ply"
cfg.gs_model_dict["timeclock"]  = "s2r2025/timeclock.ply"
cfg.gs_model_dict["wood"]       = "s2r2025/wood.ply"
cfg.gs_model_dict["xbox"]       = "s2r2025/xbox.ply"
cfg.gs_model_dict["yellow_bowl"]= "s2r2025/yellow_bowl.ply"
cfg.gs_model_dict["toy_cabinet"] = "s2r2025/toy_cabinet.ply"
cfg.gs_model_dict["cabinet_door"] = "s2r2025/cabinet_door.ply"
cfg.gs_model_dict["cabinet_drawer"] = "s2r2025/cabinet_drawer.ply"

cfg.obs_rgb_cam_id   = None
cfg.obs_depth_cam_id = None
cfg.use_gaussian_renderer = True

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    sim_node = MMK2Base(cfg)
    obs = sim_node.reset()
    action = sim_node.init_joint_ctrl.copy()

    while sim_node.running:
        obs, _, _, _, _ = sim_node.step(action)
