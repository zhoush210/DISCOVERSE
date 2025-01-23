import numpy as np
import threading
from airbot_play_short import AirbotPlayShort, cfg

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class AirbotPlayShortROS2(AirbotPlayShort, Node):
    def __init__(self, config):
        super().__init__(config)
        Node.__init__(self, 'Airbot_play_short_node')

        self.joint_state_puber = self.create_publisher(JointState, '/airbot_play/joint_states', 5)
        self.joint_state = JointState()
        self.joint_state.name = [f"joint{i+1}" for i in range(self.nj)]
        self.joint_state.position = self.sensor_joint_qpos.tolist()
        self.joint_state.velocity = self.sensor_joint_qvel.tolist()
        self.joint_state.effort = self.sensor_joint_force.tolist()

    def post_physics_step(self):
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.joint_state.position = self.sensor_joint_qpos.tolist()
        self.joint_state.velocity = self.sensor_joint_qvel.tolist()
        self.joint_state.effort = self.sensor_joint_force.tolist()
        self.joint_state_puber.publish(self.joint_state)

def spin_thread(node):
    rclpy.spin(node)

if __name__ == "__main__":
    rclpy.init()
    exec_node = AirbotPlayShortROS2(cfg)

    spin_thread = threading.Thread(target=spin_thread, args=(exec_node,))
    spin_thread.start()

    action = np.zeros(exec_node.nj*3)
    obs = exec_node.reset()
    action[:exec_node.nj] = 0.2

    try:
        while rclpy.ok() and exec_node.running:
            obs, pri_obs, rew, ter, info = exec_node.step(action)
    except KeyboardInterrupt:
        pass

    exec_node.destroy_node()
    rclpy.shutdown()
    spin_thread.join()