import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import shutil
import argparse
import multiprocessing as mp

import json

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.envs.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase, recoder_mmk2
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine

class SimNode(MMK2TaskBase):

    def domain_randomization(self):
        # 随机 杯子位置
        self.mj_data.qpos[self.njq+0] += 2.*(np.random.random()-0.5) * 0.05
        self.mj_data.qpos[self.njq+1] += 2.*(np.random.random()-0.5) * 0.025

        # 随机 盘子位置
        wood_y_bios = (np.random.random()-0.75) * 0.05
        self.mj_data.qpos[self.njq+7+0] += 2.*(np.random.random()-0.5) * 0.05
        self.mj_data.qpos[self.njq+7+1] += wood_y_bios
        self.mj_data.qpos[self.njq+7+2] += 0.01

        # 随机 木盘位置
        self.mj_data.qpos[self.njq+7*2+0] += 2.*(np.random.random()-0.5) * 0.05
        self.mj_data.qpos[self.njq+7*2+1] += wood_y_bios

        # 随机 杯盖位置
        self.mj_data.qpos[self.njq+7*3+0] += 2.*(np.random.random()-0.5) * 0.1
        self.mj_data.qpos[self.njq+7*3+1] += 2.*(np.random.random()-0.5) * 0.02

    def check_success(self):
        tmat_coffeecup = get_body_tmat(self.mj_data, "coffeecup_white")
        tmat_cup_lid = get_body_tmat(self.mj_data, "cup_lid")
        return np.hypot(tmat_coffeecup[0, 3] - tmat_cup_lid[0, 3], tmat_coffeecup[1, 3] - tmat_cup_lid[1, 3]) < 0.02

cfg = MMK2Cfg()
cfg.use_gaussian_renderer = True
cfg.init_key = "pick"
cfg.gs_model_dict["coffeecup_white"] = "object/teacup.ply"
cfg.gs_model_dict["plate_white"]     = "object/plate_white.ply"
cfg.gs_model_dict["cup_lid"]         = "object/teacup_lid.ply"
cfg.gs_model_dict["wood"]            = "object/wood.ply"
cfg.gs_model_dict["background"]      = "scene/tsimf_library_0/point_cloud_for_mmk2.ply"

cfg.mjcf_file_path = "mjcf/tasks_mmk2/plate_coffeecup.xml"
cfg.obj_list    = ["plate_white", "coffeecup_white", "wood", "cup_lid"]
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

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/mmk2_plate_coffecup")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        shutil.copyfile(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 16
    max_time = 20.0 #s

    action = np.zeros_like(sim_node.target_control)
    process_list = []

    pick_lip_arm = "r"
    move_speed = 1.
    obs = sim_node.reset()

    with open('/home/andy/DISCOVERSE/discoverse/examples/ros1/robot_state.json', 'r') as f:
        robot_state = json.load(f)

    print("len(robot_state):", len(robot_state))

    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []

        try:
            if stm.trigger():
                if stm.state_idx < len(robot_state): # 准备抓coffeecup
                    
                    print("stm.state_idx:",stm.state_idx)

                    # 从 JSON 中读取数据
                    state_data = robot_state[stm.state_idx]

                    # 操作对象
                    object_name = state_data["object_name"]
                    tmat_coffee = get_body_tmat(sim_node.mj_data, object_name)

                    left_arm = state_data["left_arm"]
                    right_arm = state_data["right_arm"]

                    if left_arm["movement"] == "move":
                        # 左臂
                        
                        left_arm_pos = np.array(left_arm["position_object_local"])
                        left_arm_rot = Rotation.from_euler('zyx', left_arm["rotation_robot_local"]).as_matrix()
                        left_gripper = left_arm["gripper"]
                        
                    
                        target_posi1 = tmat_coffee[:3, 3] + left_arm_pos
                        sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi1)
                        sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, left_arm_rot)
                        sim_node.tctr_lft_gripper[:] = left_gripper

                    if right_arm["movement"] == "move":
                        # 右臂
                        
                        right_arm_pos = np.array(right_arm["position_object_local"])
                        right_arm_rot = Rotation.from_euler('zyx', right_arm["rotation_robot_local"]).as_matrix()
                        right_gripper = right_arm["gripper"]
                        
                    
                        target_posi2 = tmat_coffee[:3, 3] + right_arm_pos
                        sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi2)
                        sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, right_arm_rot)
                        sim_node.tctr_rgt_gripper[:] = right_gripper

                    # 头部和滑轨位置
                    slide_position = state_data["slide"]
                    head_position = state_data["head"]
                    delay_s = state_data["delay_s"]

                    sim_node.tctr_head[1] = head_position[1]
                    sim_node.tctr_head[0] = head_position[0]
                    sim_node.tctr_slide[0] = slide_position[0]

                    # 延迟处理
                    sim_node.delay_cnt = int(delay_s / sim_node.delta_t)

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
        yaw = Rotation.from_quat(np.array(obs["base_orientation"])[[1,2,3,0]]).as_euler("xyz")[2]
        action[1] = -10 * yaw

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
