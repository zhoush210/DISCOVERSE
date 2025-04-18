import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import json
import shutil
import argparse
import importlib
import multiprocessing as mp
import traceback

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.task_base import recoder_mmk2
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, help='Name of the task, e.g. plate_coffeecup')
    parser.add_argument("--scheme_json", type=str, required=True, help='json-file name of the strategy, e.g. plate_coffeecup')
    parser.add_argument("--max_time_s", type=int, default=300, help='run task max time in seconds')
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--fps", type=int, default=20, help="data collection fps")
    parser.add_argument("--image_width", type=int, default=640, help="image width")
    parser.add_argument("--image_height", type=int, default=480, help="image height")
    parser.add_argument("--auto", action="store_true", help="auto run")
    parser.add_argument("--vis", action="store_true", help="visualize, if --auto is set, this will be ignored")
    parser.add_argument("--dim17", action="store_true", help="genegrate 17 joint num mmk2 data")

    args = parser.parse_args()

    try:
        task_module = importlib.import_module(f"discoverse.examples.tasks_mmk2.{args.task_name}")
        SimNode = task_module.SimNode
        cfg = task_module.cfg
    except ModuleNotFoundError:
        print("Task {} not found".format(args.task_name))
        quit()

    if os.path.exists(args.scheme_json):
        with open(args.scheme_json, 'r') as f:
            robot_state = json.load(f)
    else:
        raise FileNotFoundError("File {} not found".format(args.scheme_json))

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    cfg.headless = not args.vis
    if args.auto:
        cfg.headless = True
        cfg.sync = False
    
    if args.dim17:
        cfg.io_dim = 17

    cfg.render_set  = {
        "fps"    : args.fps,
        "width"  : args.image_width,
        "height" : args.image_height
    }

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", args.task_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        # :TODO: save task config
        # shutil.copyfile(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    stm = SimpleStateMachine()
    stm.max_state_cnt = len(robot_state)
    max_time = args.max_time_s

    action = np.zeros_like(sim_node.target_control)
    process_list = []

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
                if stm.state_idx < stm.max_state_cnt:
                    
                    # print("stm.state_idx:",stm.state_idx)
                    state_data = robot_state[stm.state_idx]

                    object_name = state_data["object_name"]
                    tmat_coffee = get_body_tmat(sim_node.mj_data, object_name)

                    base = state_data["base"]
                    left_arm = state_data["left_arm"]
                    right_arm = state_data["right_arm"]

                    action[0] = 0.0
                    action[1] = 0.0
                    sim_node.check_forward_done = False
                    sim_node.check_yaw_done = False

                    if base["movement"] == "x" or base["movement"] == "y":
                        target_global_pos = np.array(base["position"])
                        target_quat = np.array(base["rotation"])
                        target_yaw = Rotation.from_quat([target_quat[1], target_quat[2],
                                               target_quat[3], target_quat[0]]).as_euler('zyx')[0]
                        target_pos_local = sim_node.get_tmat_wrt_mmk2base(target_global_pos)  # 转为机器人本体坐标

                        # 提取目标相对坐标
                        rel_x = target_pos_local[0]  # 前后方向（前是正）
                        rel_y = target_pos_local[1]  # 左右方向（左是正）

                        # 设置目标完成标志和记录目标
                        if base["movement"] == "x":
                            # 机器人朝向上的位置控制（前进后退）
                            max_speed_linear = 0.1
                            kp_linear = 1.0
                            action[0] = max(min(kp_linear * rel_x, max_speed_linear), -max_speed_linear)
                            sim_node.check_forward_done = True
                            sim_node.forward_start_pose = sim_node.obs["base_position"].copy()
                            random_x = (np.random.random() - 0.5) * 0.1 # 随机位置噪声
                            random_y = (np.random.random() - 0.5) * 0.1
                            # print("random_x=",random_x,", random_y=",random_y)
                            target_global_pos[0] += random_x
                            target_global_pos[1] += random_y
                            sim_node.forward_target_pose = target_global_pos
                        if base["movement"] == "y":
                            # 旋转控制（向左转为正）
                            max_speed_angular = 0.1
                            kp_angular = 5.0
                            action[1] = max(min(kp_angular * rel_y, max_speed_angular), -max_speed_angular)
                            sim_node.check_yaw_done = True
                            random_yaw = (np.random.random() - 0.5) * 0.02 # 随机角度噪声
                            target_yaw += random_yaw
                            # print("random_yaw=",random_yaw,",target_yaw=",target_yaw)
                            sim_node.yaw_target_pose = target_yaw

                    if left_arm["movement"] == "move":
                        left_arm_pos = np.array(left_arm["position_object_local"])
                        left_arm_rot = Rotation.from_euler('zyx', left_arm["rotation_robot_local"]).as_matrix()
                        left_gripper = left_arm["gripper"]
                        
                        target_posi1 = tmat_coffee[:3, 3] + left_arm_pos
                        sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi1)
                        sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, left_arm_rot)
                        sim_node.tctr_lft_gripper[:] = left_gripper

                    if right_arm["movement"] == "move":
                        right_arm_pos = np.array(right_arm["position_object_local"])
                        right_arm_rot = Rotation.from_euler('zyx', right_arm["rotation_robot_local"]).as_matrix()
                        right_gripper = right_arm["gripper"]
                        
                        target_posi2 = tmat_coffee[:3, 3] + right_arm_pos
                        sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi2)
                        sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, right_arm_rot)
                        sim_node.tctr_rgt_gripper[:] = right_gripper

                    # 头部和升降位置
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
            # 调试时 Uncomment 下面这行
            traceback.print_exc()
            sim_node.reset()


        assert len(action) == sim_node.njctrl
        # action[:2] is x-linear vel and z-angular vel

        if sim_node.io_dim == sim_node.njctrl:
            for i in range(2, sim_node.njctrl):
                action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
            yaw = Rotation.from_quat(np.array(obs["base_orientation"])[[1,2,3,0]]).as_euler("xyz")[2]
            obs, _, _, _, _ = sim_node.step(action)
        elif sim_node.io_dim == 17:
            for i in range(2, sim_node.njctrl):
                action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
            obs, _, _, _, _ = sim_node.step(action[2:])
        else:
            raise ValueError(f"Wrong io dim: {sim_node.io_dim}")
        
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            if sim_node.io_dim == sim_node.njctrl:
                act_lst.append(action.tolist().copy())
            elif sim_node.io_dim == 17:
                act_lst.append(action[2:].tolist().copy())
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
