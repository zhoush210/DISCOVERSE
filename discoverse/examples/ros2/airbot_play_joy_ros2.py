import os
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from sensor_msgs.msg import JointState

from discoverse import DISCOVERSE_ASSERT_DIR
from discoverse.airbot_play import AirbotPlayFIK
from discoverse.envs.airbot_play_base import AirbotPlayCfg, AirbotPlayBase
from discoverse.utils.joy_stick_ros2 import JoyTeleopRos2


class AirbotPlayJoyCtl(AirbotPlayBase, Node):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        Node.__init__(self, 'Airbot_play_node')

        urdf_path = os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
        self.arm_fik = AirbotPlayFIK(urdf_path)
    
        self.tar_end_pose = np.array([0.295, -0., 0.219])
        self.tar_end_euler = np.zeros(3)
        self.tar_jq = np.zeros(self.nj)

        self.joint_state_puber = self.create_publisher(JointState, '/airbot_play/joint_states', 5)
        self.joint_state = JointState()
        self.joint_state.name = [f"joint{i+1}" for i in range(6)] + ["gripper"]

        self.joint_state.position = self.sensor_joint_qpos.tolist()
        self.joint_state.velocity = self.sensor_joint_qvel.tolist()
        self.joint_state.effort = self.sensor_joint_force.tolist()

        self.teleop = JoyTeleopRos2()
        self.sub = self.create_subscription(Joy, '/joy_throttle', self.teleop.joy_callback, 10)

    def resetState(self):
        super().resetState()
        self.tar_jq = np.zeros(self.nj)
        self.tar_end_pose = np.array([0.295, -0., 0.219])
        self.tar_end_euler = np.zeros(3)
        self.teleop.reset()

    def updateControl(self, action):
        super().updateControl(self.tar_jq)

    def teleopProcess(self):
        calc_ik = False
        tmp_arm_target_pose = self.tar_end_pose.copy()
        if self.teleop.joy_cmd.axes[0] or self.teleop.joy_cmd.axes[1] or self.teleop.joy_cmd.axes[4]:
            calc_ik = True
            tmp_arm_target_pose[0] += 0.15 * self.teleop.joy_cmd.axes[1] * self.delta_t
            tmp_arm_target_pose[1] += 0.15 * self.teleop.joy_cmd.axes[0] * self.delta_t
            tmp_arm_target_pose[2] += 0.1 * self.teleop.joy_cmd.axes[4] * self.delta_t

        el = self.tar_end_euler.copy()
        if self.teleop.joy_cmd.axes[3] or self.teleop.joy_cmd.axes[6] or self.teleop.joy_cmd.axes[7]:
            calc_ik = True
            el[0] += 0.01 * self.teleop.joy_cmd.axes[3]
            el[1] += 0.01 * self.teleop.joy_cmd.axes[7]
            el[2] += 0.01 * self.teleop.joy_cmd.axes[6]

        if calc_ik:
            rot = Rotation.from_euler('xyz', el).as_matrix()
            try:
                tarjq = self.arm_fik.properIK(self.tar_end_pose, rot, self.sensor_joint_qpos[:6])
                self.tar_end_pose[:] = tmp_arm_target_pose[:]
                self.tar_end_euler[:] = el[:]
            except ValueError:
                tarjq = None

            if not tarjq is None:
                self.tar_jq[:6] = tarjq
            else:
                self.get_logger().warn("Fail to solve inverse kinematics trans={} euler={}".format(self.tar_end_pose, self.tar_end_euler))

        if self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5]:
            self.tar_jq[6] += 1. * (self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5]) * self.delta_t
            self.tar_jq[6] = np.clip(self.tar_jq[6], 0, 1.)

    def post_physics_step(self):
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.joint_state.position = self.sensor_joint_qpos.tolist()
        self.joint_state.velocity = self.sensor_joint_qvel.tolist()
        self.joint_state.effort = self.sensor_joint_force.tolist()
        self.joint_state_puber.publish(self.joint_state)

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time = {:.3f}".format(self.mj_data.time))
        print("joint tar_q = {}".format(np.array2string(self.tar_jq, separator=', ')))
        print("joint q     = {}".format(np.array2string(self.sensor_joint_qpos, separator=', ')))
        print("joint v     = {}".format(np.array2string(self.sensor_joint_qvel, separator=', ')))

        print("target end posi  = {}".format(np.array2string(self.tar_end_pose, separator=', ')))
        print("target end euler = {}".format(np.array2string(self.tar_end_euler, separator=', ')))

        print("sensor end posi  = {}".format(np.array2string(self.sensor_endpoint_posi_local, separator=', ')))
        print("sensor end euler = {}".format(np.array2string(Rotation.from_quat(self.sensor_endpoint_quat_local[[1,2,3,0]]).as_euler("xyz"), separator=', ')))

        if self.cam_id == -1:
            print(self.free_camera)
        else:
            print(self.mj_data.camera(self.camera_names[self.cam_id]))

if __name__ == "__main__":
    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = AirbotPlayCfg()
    cfg.mjcf_file_path = "mjcf/tasks_airbot_play/laptop_close.xml"
    cfg.decimation = 4
    cfg.timestep = 0.0025
    cfg.use_gaussian_renderer = False

    exec_node = AirbotPlayJoyCtl(cfg)

    while rclpy.ok() and exec_node.running:
        exec_node.teleopProcess()
        exec_node.step()
        rclpy.spin_once(exec_node)

    exec_node.destroy_node()
    rclpy.shutdown()