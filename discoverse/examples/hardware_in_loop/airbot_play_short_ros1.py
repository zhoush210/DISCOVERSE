import numpy as np
import threading
from airbot_play_short import AirbotPlayShort, cfg

import rospy
from sensor_msgs.msg import JointState

class AirbotPlayShortROS1(AirbotPlayShort):
    def __init__(self, config):
        super().__init__(config)
        rospy.init_node('Airbot_play_short_node', anonymous=True)

        self.joint_state_puber = rospy.Publisher('/airbot_play/joint_states', JointState, queue_size=5)
        self.joint_state = JointState()
        self.joint_state.name = [f"joint{i+1}" for i in range(self.nj)]
        self.joint_state.position = self.sensor_joint_qpos.tolist()
        self.joint_state.velocity = self.sensor_joint_qvel.tolist()
        self.joint_state.effort = self.sensor_joint_force.tolist()

    def post_physics_step(self):
        self.joint_state.header.stamp = rospy.Time.now()
        self.joint_state.position = self.sensor_joint_qpos.tolist()
        self.joint_state.velocity = self.sensor_joint_qvel.tolist()
        self.joint_state.effort = self.sensor_joint_force.tolist()
        self.joint_state_puber.publish(self.joint_state)

if __name__ == "__main__":
    cfg.sync = False
    exec_node = AirbotPlayShortROS1(cfg)

    action = np.zeros(exec_node.nj*3)
    obs = exec_node.reset()
    action[:exec_node.nj] = 0.2

    frq = int(1. / exec_node.delta_t)
    print(frq)
    rate = rospy.Rate(frq)
    while not rospy.is_shutdown() and exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)
        rate.sleep()