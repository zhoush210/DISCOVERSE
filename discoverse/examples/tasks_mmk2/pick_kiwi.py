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

    def domain_randomization(self):
        # 随机 木盘位置
        # wood_x_bios = (np.random.random()) * 0.02
        # wood_y_bios = (np.random.random() - 1) * 0.05
        # self.mj_data.qpos[self.njq+7*1+0] += wood_x_bios
        # self.mj_data.qpos[self.njq+7*1+1] += wood_y_bios
        # self.mj_data.qpos[self.njq+7*1+2] += 0.01

        # 随机 枣位置
        # kiwi_x_bios = (np.random.random()- 0.5) * 0.04
        # kiwi_y_bios = (np.random.random()) * 0.12
        # self.mj_data.qpos[self.njq+7*2+0] += kiwi_x_bios
        # self.mj_data.qpos[self.njq+7*2+1] += kiwi_y_bios

        # # 随机 碗位置
        # bowl_x_bios = (np.random.random()) * 0.02
        # bowl_y_bios = (np.random.random()) * 0.04
        # self.mj_data.qpos[self.njq+7*0+0] += bowl_x_bios
        # self.mj_data.qpos[self.njq+7*0+1] += bowl_y_bios
        pass

    def check_success(self):
        v1 = np.array([self.mj_data.qpos[self.njq+7*2+0], self.mj_data.qpos[self.njq+7*2+1]])
        v2 = np.array([self.mj_data.qpos[self.njq+7*0+0], self.mj_data.qpos[self.njq+7*0+1]])
        
        # 计算差异
        diff = v1 - v2
        
        # 计算平方和
        squared_diff = np.sum(diff**2)
        
        # 取平方根
        distance = np.sqrt(squared_diff) - 0.0195
        print(distance)
        if distance < 1e-2:
            return True

        return False
    
cfg = MMK2Cfg()
cfg.use_gaussian_renderer = True
cfg.init_key = "pick"
cfg.gs_model_dict["bowl_yellow"]     = "object/red_bowl.ply"
cfg.gs_model_dict["kiwi"]          = "object/kiwi.ply"
cfg.gs_model_dict["background"]      = "scene/lab3/point_cloud.ply"

cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"
cfg.obj_list    = ["bowl_yellow", "kiwi"]
cfg.sync     = False
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

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/mmk2_pick_kiwi")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    sim_node.teleop = None
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 10
    max_time = 30.0 #s

    action = np.zeros_like(sim_node.target_control)
    process_list = []

    pick_lip_arm = "l"
    move_speed = 1.
    obs = sim_node.reset()
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []

        try:
            if stm.trigger():
                if stm.state_idx == 0: # 降高度
                    sim_node.tctr_head[1] = 0.8
                    sim_node.tctr_slide[0] = 0.2
                # elif stm.state_idx == 1: # 伸到碗前
                #     tmat_bowl = get_body_tmat(sim_node.mj_data, "bowl_yellow")
                #     target_posi = tmat_bowl[:3, 3] + 0.1 * tmat_bowl[:3, 1] + 0.1 * tmat_bowl[:3, 2]
                #     sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                #     sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [ 0., -0.0551, 0.]).as_matrix())
                #     sim_node.tctr_lft_gripper[:] = 1
                # elif stm.state_idx == 2: # 伸到碗壁
                #     tmat_bowl = get_body_tmat(sim_node.mj_data, "bowl_yellow")
                #     target_posi = tmat_bowl[:3, 3] + 0.046 * tmat_bowl[:3, 1] + 0.05 * tmat_bowl[:3, 2]
                #     sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                #     sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [ 0., -0.0551, 0.]).as_matrix())
                # elif stm.state_idx == 3: # 抓住碗壁
                #     sim_node.tctr_lft_gripper[:] = 0.0
                # elif stm.state_idx == 4: # 提起来碗
                #     sim_node.tctr_slide[0] = 0.1
                # elif stm.state_idx == 5: # 把碗放到托盘上空
                #     tmat_plate = get_body_tmat(sim_node.mj_data, "wood")
                #     target_posi = tmat_plate[:3, 3] + np.array([0.0, 0.045, 0.16])
                #     sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                #     sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [ 0., -0.0551, 0.]).as_matrix())                    
                # elif stm.state_idx == 6: # 下降高度
                #     sim_node.tctr_slide[0] = 0.2
                # elif stm.state_idx == 7: # 松开碗壁 放下碗
                #     sim_node.tctr_lft_gripper[:] = 1
                # elif stm.state_idx == 8: # 上升高度
                #     sim_node.tctr_slide[0] = 0.1
                # elif stm.state_idx == 9: # 移开手臂
                #     tmat_plate = get_body_tmat(sim_node.mj_data, "wood")
                #     target_posi = tmat_plate[:3, 3] + np.array([0.2, 0.03, 0.16])
                #     sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                #     sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [ 0., -0.0551, 0.]).as_matrix())
                
                elif stm.state_idx == 1: # 伸到kiwi上
                    tmat_kiwi = get_body_tmat(sim_node.mj_data, "kiwi")
                    target_posi = tmat_kiwi[:3, 3] + 0.1 * tmat_kiwi[:3, 1] + 0.1 * tmat_kiwi[:3, 2]
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [ 0., -0.0551, 0.]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 1
                elif stm.state_idx == 2: # 伸到枣前
                    tmat_kiwi = get_body_tmat(sim_node.mj_data, "kiwi")
                    target_posi = tmat_kiwi[:3, 3] + 0.132 * tmat_kiwi[:3, 2]
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [ 0., -0.0551, 0.]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 1 
                elif stm.state_idx == 3: # 降高度
                    sim_node.tctr_head[1] = 0.8
                    sim_node.tctr_slide[0] = 0.2
                elif stm.state_idx == 4: # 抓住枣
                    sim_node.tctr_rgt_gripper[:] = 0.0
                elif stm.state_idx == 5: # 提起枣
                    sim_node.tctr_slide[0] = 0.1
                elif stm.state_idx == 6: # 移动到碗上空
                    tmat_bowl = get_body_tmat(sim_node.mj_data, "bowl_yellow")
                    target_posi = tmat_bowl[:3, 3] + 0.15 * tmat_bowl[:3, 2]
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [ 0., -0.0551, 0.]).as_matrix())
                elif stm.state_idx == 7: # 降高度
                    sim_node.tctr_head[1] = 0.8
                    sim_node.tctr_slide[0] = 0.15
                elif stm.state_idx == 8: # 放开枣
                    sim_node.tctr_rgt_gripper[:] = 1.0
                    sim_node.delay_cnt = int(0.2/sim_node.delta_t)
                elif stm.state_idx == 9: # 升高度
                    sim_node.tctr_head[1] = 0.8
                    sim_node.tctr_slide[0] = 0.1
                
                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)
                sim_node.joint_move_ratio[2] *= 0.25

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
        # yaw = Rotation.from_quat(np.array(obs["base_orientation"])[[1,2,3,0]]).as_euler("xyz")[2] + np.pi / 2
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
