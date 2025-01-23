import enum
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs import SimulatorBase
from discoverse.utils import BaseConfig, PIDarray

class ArmControlMode(enum.Enum):
    POSITION = 0
    MIT = 1

class AirbotPlayShort(SimulatorBase):
    def __init__(self, config: BaseConfig):
        self.nj = 6
        self.pids = PIDarray(
            kps=np.array([175.0, 225.0, 300.0, 1.5, 2.5, 1.]),
            kis=np.array([5, 25, 35, 2.5, 2.5, 2.5]),
            kds=np.array([3.0, 5.0, 7.0, 0.1, 0.25, 0.01]),
            integrator_maxs=np.array([5.0, 5.0, 5.0, 2.5, 2.5, 2.5]),
        )
        self.control_mode = ArmControlMode.POSITION
        super().__init__(config)

    def post_load_mjcf(self):
        self.sensor_joint_qpos = self.mj_data.sensordata[:self.nj]
        self.sensor_joint_qvel = self.mj_data.sensordata[self.nj:2*self.nj]
        self.sensor_joint_force = self.mj_data.sensordata[2*self.nj:3*self.nj]
        self.ctr_position = self.mj_data.ctrl[:self.nj]
        self.ctr_velcity = self.mj_data.ctrl[self.nj:2*self.nj]
        self.ctr_torque = self.mj_data.ctrl[2*self.nj:3*self.nj]

        self.mit_kps = self.mj_model.actuator_gainprm[:self.nj, 0]
        self.mit_kds = self.mj_model.actuator_gainprm[self.nj:self.nj*2, 0]

        self.swith_control_mode(self.control_mode)

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time  = {:.3f}".format(self.mj_data.time))
        print("    arm .qpos  = {}".format(np.array2string(self.sensor_joint_qpos, separator=', ')))
        print("    arm .qvel  = {}".format(np.array2string(self.sensor_joint_qvel, separator=', ')))
        print("    arm .ctrl  = {}".format(np.array2string(self.mj_data.ctrl[:self.nj], separator=', ')))
        print("    arm .force = {}".format(np.array2string(self.sensor_joint_force, separator=', ')))

    def swith_control_mode(self, target_control_mode):
        if target_control_mode == ArmControlMode.POSITION:
            self.pids.reset()
            self.mj_model.actuator_gainprm[:self.nj, 0] =  0.0 # kps
            self.mj_model.actuator_biasprm[:self.nj, 1] = -0.0 #-kps
            self.mj_model.actuator_gainprm[self.nj:self.nj*2, 0] =  0.0 # kds
            self.mj_model.actuator_biasprm[self.nj:self.nj*2, 2] = -0.0 #-kds
        elif target_control_mode == ArmControlMode.MIT:
            self.mj_model.actuator_gainprm[:self.nj, 0] =  self.mit_kps # kps
            self.mj_model.actuator_biasprm[:self.nj, 1] = -self.mit_kps #-kps
            self.mj_model.actuator_gainprm[self.nj:self.nj*2, 0] =  self.mit_kds # kds
            self.mj_model.actuator_biasprm[self.nj:self.nj*2, 2] = -self.mit_kds #-kds
        else:
            raise NotImplementedError("Invalid control mode {}".format(target_control_mode))
        self.control_mode = target_control_mode
        print("Control mode: {}".format(self.control_mode))

    def cv2WindowKeyPressCallback(self, key):
        ret = super().cv2WindowKeyPressCallback(key)
        if key == ord('m'):
            self.swith_control_mode(ArmControlMode((self.control_mode.value + 1) % len(ArmControlMode)))
        return ret

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        if self.control_mode == ArmControlMode.POSITION:
            self.ctr_torque[:] = self.pids.output(action[:self.nj] - self.sensor_joint_qpos, self.mj_model.opt.timestep)
        elif self.control_mode == ArmControlMode.MIT:
            self.ctr_position[:] = action[:self.nj]
            self.ctr_velcity[:] = action[self.nj:2*self.nj]
            self.ctr_torque[:] = action[2*self.nj:3*self.nj]
        else:
            raise NotImplementedError("Invalid control mode {}".format(self.control_mode))

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "time" : self.mj_data.time,
            "jq"   : self.sensor_joint_qpos.tolist(),
            "jv"   : self.sensor_joint_qvel.tolist(),
            "jf"   : self.sensor_joint_force.tolist(),
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

cfg = BaseConfig()
cfg.mjcf_file_path = "mjcf/hardware_in_loop/airbot_play_short.xml"
cfg.decimation = 4
cfg.timestep = 0.001
cfg.sync = True
cfg.headless = False
cfg.render_set = {
    "fps"    : 30,
    "width"  : 1280,
    "height" : 720
}

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    exec_node = AirbotPlayShort(cfg)

    action = np.zeros(exec_node.nj*3)
    obs = exec_node.reset()
    action[:exec_node.nj] = 0.2

    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)
