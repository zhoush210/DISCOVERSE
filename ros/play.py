import os
import numpy as np
import threading
import time
from discoverse.examples.hardware_in_loop.airbot_arm import (
    AirbotArm,
    cfg,
    ControlMode,
)

import argparse

import rclpy
from rclpy.node import Node

from airbot_msgs.msg import MotorFeedback, MotorControl
import mujoco.viewer


def recoder_airbot_play(save_path, obs_lst, cfg):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "obs.json"), "w") as fp:
        obj = {
            "time": [o["time"] for o in obs_lst],
            "obs": {
                "jq": [o["jq"] for o in obs_lst],
            },
        }
        json.dump(obj, fp)

    for id in cfg.obs_rgb_cam_id:
        mediapy.write_video(
            os.path.join(save_path, f"cam_{id}.mp4"),
            [o["img"][id] for o in obs_lst],
            fps=cfg.render_set["fps"],
        )


class Airbot(AirbotArm, Node):
    def __init__(self, config, sim, domain_id):

        super().__init__(config)
        context = rclpy.context.Context()
        print("domain_id", domain_id)
        rclpy.init(args=None, context=context, domain_id=domain_id)
        Node.__init__(self, "Airbot_play_node", context=context)

        self.states_feedback_publisher = self.create_publisher(
            MotorFeedback, "/" + sim + "/states_feedback", 5
        )
        self.states_feedback = MotorFeedback()
        # self.states_feedback.motor_id = list(range(self.nj))
        # self.states_feedback.position = self.sensor_joint_qpos.tolist()
        # self.states_feedback.velocity = self.sensor_joint_qvel.tolist()
        # self.states_feedback.effort = self.sensor_joint_force.tolist()
        # self.states_feedback.error_code = [0] * self.nj
        # self.states_feedback.mos_temperature = [0] * self.nj
        # self.states_feedback.motor_temperature = [0] * self.nj
        self.states_feedback.motor_id = []
        self.states_feedback.position = []
        self.states_feedback.velocity = []
        self.states_feedback.effort = []
        self.states_feedback.error_code = []
        self.states_feedback.mos_temperature = []
        self.states_feedback.motor_temperature = []

        self.control_cmd_sub = self.create_subscription(
            MotorControl, "/" + sim + "/control_command", self.control_cmd_callback, 10
        )

    def post_physics_step(self):
        # self.states_feedback.motor_id = list(range(1, self.nj+1))
        # self.states_feedback.stamp = self.get_clock().now().to_msg()
        # self.states_feedback.position = self.sensor_joint_qpos.tolist()
        # self.states_feedback.velocity = self.sensor_joint_qvel.tolist()
        # self.states_feedback.effort = self.sensor_joint_force.tolist()
        # self.states_feedback_publisher.publish(self.states_feedback)
        pass

    def control_cmd_callback(self, msg):
        if msg.mode[0] == 1:
            if msg.motor_id[0] == 7 and self.gripper_control_mode != ControlMode.MIT:
                self.switch_gripper_control_mode(ControlMode.MIT)
            elif msg.motor_id[0] != 7 and self.arm_control_mode != ControlMode.MIT:
                self.switch_arm_control_mode(ControlMode.MIT)
            self.action[msg.motor_id[0] - 1] = msg.position[0]
            self.action[self.nj + msg.motor_id[0] - 1] = msg.velocity[0]
            self.action[2 * self.nj + msg.motor_id[0] - 1] = msg.effort[0]
            self.mj_model.actuator_gainprm[msg.motor_id[0] - 1, 0] = msg.kp[0]
            self.mj_model.actuator_biasprm[msg.motor_id[0] - 1, 1] = -msg.kp[0]
            self.mj_model.actuator_gainprm[self.nj + msg.motor_id[0] - 1, 0] = msg.kd[0]
            self.mj_model.actuator_biasprm[self.nj + msg.motor_id[0] - 1, 2] = -msg.kd[
                0
            ]
        elif msg.mode[0] == 2:
            if (
                msg.motor_id[0] == 7
                and self.gripper_control_mode != ControlMode.POSITION
            ):
                self.switch_gripper_control_mode(ControlMode.POSITION)
            elif msg.motor_id[0] != 7 and self.arm_control_mode != ControlMode.POSITION:
                self.switch_arm_control_mode(ControlMode.POSITION)
            self.action[msg.motor_id[0] - 1] = msg.position[0]
            self.action[self.nj + msg.motor_id[0] - 1] = msg.velocity[0]
        # self.states_feedback.motor_id = list(range(1, self.nj+1))
        # self.states_feedback.stamp = self.get_clock().now().to_msg()
        # self.states_feedback.position = self.sensor_joint_qpos.tolist()
        # self.states_feedback.velocity = self.sensor_joint_qvel.tolist()
        # self.states_feedback.effort = self.sensor_joint_force.tolist()
        # self.states_feedback_publisher.publish(self.states_feedback)
        self.states_feedback.stamp = self.get_clock().now().to_msg()
        self.states_feedback.motor_id = msg.motor_id
        self.states_feedback.position = [self.sensor_joint_qpos[msg.motor_id[0] - 1]]
        self.states_feedback.velocity = [self.sensor_joint_qvel[msg.motor_id[0] - 1]]
        self.states_feedback.effort = [self.sensor_joint_force[msg.motor_id[0] - 1]]
        self.states_feedback.error_code = [0]
        self.states_feedback.mos_temperature = [0]
        self.states_feedback.motor_temperature = [0]
        self.states_feedback_publisher.publish(self.states_feedback)


def spin_thread_fun(node):
    rclpy.spin(node)


def main():
    parser = argparse.ArgumentParser(
        description="Run arm with specified parameters. \ne.g. python3 airbot_play_short.py --arm_type play_short --eef_type none"
    )
    parser.add_argument(
        "-s",
        "--simulation-interface",
        type=str,
        default="sim0",
        help="Simulation interface name",
    )
    parser.add_argument(
        "-a",
        "--arm_type",
        type=str,
        choices=["play_long", "play_short", "lite", "pro", "replay"],
        help="Name of the arm",
        default="play_short",
    )
    # :TODO: play_long, lite, pro, replay
    parser.add_argument(
        "-e",
        "--eef_type",
        type=str,
        choices=["G2", "E2B", "PE2", "none"],
        help="Name of the eef",
        default="none",
    )
    parser.add_argument(
        "-d",
        "--domain_id",
        type=int,
        default=os.environ.get("ROS_DOMAIN_ID", 100),
        help="ROS domain id",
    )
    parser.add_argument(
        "--no_display",
        action="store_false",
        help="Display the simulation",
    )
    # :TODO: PE2
    args = parser.parse_args()

    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg.arm_type = args.arm_type
    cfg.eef_type = args.eef_type

    cfg.enable_render = True
    cfg.obs_rgb_cam_id = [0, 1]  # -1, 0, 1, 2

    if args.no_display:
        cfg.headless = True
        print("Displaying")
    else:
        cfg.headless = True
        os.environ["MUJOCO_GL"] = "egl"
        print("Not displaying")

    exec_node = Airbot(cfg, sim=args.simulation_interface, domain_id=args.domain_id)

    spin_thread = threading.Thread(target=spin_thread_fun, args=(exec_node,))
    spin_thread.start()

    action = np.zeros(exec_node.nj * 3)
    obs = exec_node.reset()
    action[: exec_node.nj] = 0.2

    obs_lst = []
    try:
        with mujoco.viewer.launch_passive(
            exec_node.mj_model,
            exec_node.mj_data,
            key_callback=exec_node.windowKeyPressCallback,
        ) as viewer:
            while rclpy.ok() and exec_node.running:
                step_start = time.time()
                obs, pri_obs, rew, ter, info = exec_node.step()
                viewer.sync()
                time_until_next_step = exec_node.delta_t - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            if len(img_lst) < exec_node.mj_data.time * cfg.render_set["fps"]:
                obs_lst.append(obs)
    except KeyboardInterrupt:
        pass
    finally:
        recoder_airbot_play("path/to/save", obs_lst, cfg)

    exec_node.destroy_node()
    rclpy.shutdown()
    spin_thread.join()


if __name__ == "__main__":
    main()
