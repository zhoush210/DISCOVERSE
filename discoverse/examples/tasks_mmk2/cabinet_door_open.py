import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import shutil
import argparse
import multiprocessing as mp

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.envs.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase, recoder_mmk2
from discoverse.utils import get_site_tmat, get_body_tmat, step_func, SimpleStateMachine

class MMK2TASK(MMK2TaskBase):

    def domain_randomization(self):
        # 随机 柜门位置
        self.mj_data.qpos[self.njq+0] += 2.*(np.random.random()-0.5) * 0.05
        self.mj_data.qpos[self.njq+1] += 2.*(np.random.random()-0.5) * 0.025
        self.origin_pos=self.mj_data.qpos.copy()

    def check_success(self):
        diff=np.sum(np.square(self.mj_data.qpos-self.origin_pos))
        return diff > 20.0

cfg = MMK2Cfg()
cfg.use_gaussian_renderer = False
cfg.init_key = "pick"
#cfg.gs_model_dict["coffeecup_white"] = "object/teacup.ply"
#cfg.gs_model_dict["plate_white"]     = "object/plate_white.ply"
#cfg.gs_model_dict["cup_lid"]         = "object/teacup_lid.ply"
#cfg.gs_model_dict["wood"]            = "object/wood.ply"
#cfg.gs_model_dict["background"]      = "scene/tsimf_library_0/point_cloud_for_mmk2.ply"

cfg.mjcf_file_path = "mjcf/tasks_mmk2/cabinet_door_open.xml"
cfg.obj_list    = ["cabinet_door"]

cfg.sync     = True
cfg.headless = False
cfg.render_set  = {
    "fps"    : 25,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id = [0,1,2]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/mmk2_cabinet_door_open")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sim_node = MMK2TASK(cfg)
    sim_node.teleop = None
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        shutil.copyfile(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 16
    max_time = 20.0 #s

    action = np.zeros_like(sim_node.target_control)
    process_list = []

    pick_lip_arm = "r"
    move_speed = 1
    obs = sim_node.reset()
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
        
        """
        tmat_door 是一个 4x4 的仿射变换矩阵，表示 "cabinet_door_handle" 的变换关系：
        tmat_door =
            [[ R11  R12  R13  x ]
                [ R21  R22  R23  y ]
                [ R31  R32  R33  z ]
                [  0    0    0   1 ]]
        tmat_door[:3, :3]：是一个 3x3 的旋转矩阵，表示门把手的朝向。
        tmat_door[:3, 3]：是门把手在世界坐标系中的位置 [x, y, z]。
        """
        
        try:
            if stm.trigger():
                if stm.state_idx == 0: #后退并降高度
                    action[0] = -0.2
                    sim_node.tctr_head[1] = 0.6
                    sim_node.tctr_slide[0] = 0.05
                elif stm.state_idx == 1: # 伸到柜门前
                    action[0] = 0
                    tmat_door = get_site_tmat(sim_node.mj_data, "cabinet_door_handle")
                    target_posi = tmat_door[:3, 3] + 0.1 * tmat_door[:3, 0]
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [0, -1.5807, -1]).as_matrix())
                    sim_node.tctr_lft_gripper[:] = 1
                elif stm.state_idx == 2: # 伸到柜门把手
                    tmat_door = get_site_tmat(sim_node.mj_data, "cabinet_door_handle")
                    target_posi = tmat_door[:3, 3] + np.array([-0.1,0.1,0])
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [0, -1.5807, -0.2]).as_matrix())
                elif stm.state_idx == 3: # 伸到柜门把手
                    tmat_door = get_site_tmat(sim_node.mj_data, "cabinet_door_handle")
                    target_posi = tmat_door[:3, 3] + np.array([-0.05,0.1,0])
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [0, -1.5807, -0.5]).as_matrix())
                elif stm.state_idx == 4: # 伸到柜门把手
                    tmat_door = get_site_tmat(sim_node.mj_data, "cabinet_door_handle")
                    target_posi = tmat_door[:3, 3] + np.array([-0.05,0.05,0])
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [0, -1.5807, -0.5]).as_matrix())
                elif stm.state_idx == 5: # 伸到柜门把手
                    tmat_door = get_site_tmat(sim_node.mj_data, "cabinet_door_handle")
                    target_posi = tmat_door[:3, 3] + np.array([-0.04,0,0])
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [0, -1.5807, -0.5]).as_matrix())
                elif stm.state_idx == 6: # 伸到柜门把手
                    tmat_door = get_site_tmat(sim_node.mj_data, "cabinet_door_handle")
                    target_posi = tmat_door[:3, 3] + np.array([-0.04,-0.05,0])
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [0, -1.5807, -0.5]).as_matrix())
                elif stm.state_idx == 7: # 抓住把手
                    sim_node.tctr_lft_gripper[:] = 0.0
                    sim_node.delay_cnt = int(0.5/sim_node.delta_t)
                elif stm.state_idx == 8: # 打开柜门 第一阶段（半开）# 门把手到铰链的距离=0.195
                    tmat_door = get_site_tmat(sim_node.mj_data, "cabinet_door_handle")
                    target_posi = tmat_door[:3, 3] + np.array([-0.16, 0.085, 0.])
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [0, -1.5807, 0.16]).as_matrix()) #0.18
                elif stm.state_idx == 9: # 打开柜门 第二阶段 （调整位姿）
                    tmat_door = get_site_tmat(sim_node.mj_data, "cabinet_door_handle")
                    target_posi = tmat_door[:3, 3] + np.array([-0.04, 0.05, 0])
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [0, -1.5807, 0.18]).as_matrix()) #0.2
                elif stm.state_idx == 10: # 打开柜门 第二阶段 （调整位姿）
                    tmat_door = get_site_tmat(sim_node.mj_data, "cabinet_door_handle")
                    target_posi = tmat_door[:3, 3] + np.array([-0.04, 0.05, 0])
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [0, -1.5807, 0.18]).as_matrix()) #0.2
                elif stm.state_idx == 11: # 打开柜门 第三阶段（全开）
                    tmat_door = get_site_tmat(sim_node.mj_data, "cabinet_door_handle")
                    target_posi = tmat_door[:3, 3] + np.array([0, 0.05, 0])
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [0, -1.5807, 0]).as_matrix())

                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)
                sim_node.joint_move_ratio[2] *= 0.5

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")

            else:
                stm.update()

            if sim_node.checkActionDone():
                stm.next()

        except ValueError as ve:
            # traceback.print_exc()
            sim_node.reset()

        for i in range(2, sim_node.njctrl):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
        # 固定角度
        # yaw = Rotation.from_quat(np.array(obs["base_orientation"])[[1,2,3,0]]).as_euler("xyz")[2]
        # action[1] = -10 * yaw

        obs, _, _, _, _ = sim_node.step(action)
        
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)

        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
                process = mp.Process(target=recoder_mmk2, args=(save_path, act_lst, obs_lst, cfg))
                process.start()
                process_list.append(process)

                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")

            obs = sim_node.reset()

    for p in process_list:
        p.join()
