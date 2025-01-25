import numpy as np
from discoverse.envs.mmk2_base import MMK2Base, MMK2Cfg

from discoverse.examples.s2r2025.robot_stand import cfg

"""
Round 1 Example:
    Take the disk from the second floor of the left 
    cabinet, and put it to the left side of the left table.
"""

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    cfg.use_gaussian_renderer = False
    sim_node = MMK2Base(cfg)
    obs = sim_node.reset()
    action = sim_node.init_joint_ctrl.copy()

    while sim_node.running:
        obs, _, _, _, _ = sim_node.step(action)
