import numpy as np
from discoverse.envs.airbot_play_base import AirbotPlayCfg, AirbotPlayBase

class AirbotPlayGrpc(AirbotPlayBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = AirbotPlayCfg()
    cfg.mjcf_file_path = "mjcf/tasks_airbot_play/laptop_close.xml"
    cfg.decimation = 4
    cfg.timestep = 0.002
    cfg.use_gaussian_renderer = False

    exec_node = AirbotPlayGrpc(cfg)

    action = np.zeros(exec_node.nj)
    while exec_node.running:
        exec_node.step(action)