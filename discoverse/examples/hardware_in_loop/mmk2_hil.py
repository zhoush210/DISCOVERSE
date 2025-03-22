import enum
import mujoco
import numpy as np

from discoverse.envs import SimulatorBase
from discoverse.utils import BaseConfig, PIDController, PIDarray

class ControlMode(enum.Enum):
    POSITION = 0
    MIT = 1

class MMK2_HIL(SimulatorBase):
    dof_act = 19
    njq = 28
    njctrl = 7 * 2 * 3 + 5

    wheel_radius = 0.0838
    wheel_distance = 0.189

    def __init__(self, config: BaseConfig):

        self.pid_base_vel = PIDarray(
            kps=np.array([ 7.5 ,  7.5 ]),
            kis=np.array([  .0 ,   .0 ]),
            kds=np.array([  .0 ,   .0 ]),
            integrator_maxs=np.array([5.0, 5.0]),
        )

        ###########################################################################################################
        ######################################### 需要调参 用于位控模式 ##############################################
        ########## ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ ##########
        self.pidarr_lftarm = PIDarray(
            kps=np.array([225.0, 275.0, 350.0, 25.00, 25.00, 5.00]),
            kis=np.array([  5.0,  25.0,  35.0,  2.50,  2.50, 2.50]),
            kds=np.array([  3.0,   5.0,   7.0,  0.01,  0.01, 0.01]),
            integrator_maxs=np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]),
        )
        self.control_mode_lftarm = ControlMode.POSITION

        self.pid_lft_gripper = PIDController(1.0, 0.5, 0.1, integrator_max=0.1)
        self.control_mode_lft_gripper = ControlMode.POSITION

        self.pidarr_rgtarm = PIDarray(
            kps=np.array([225.0, 275.0, 350.0, 25.00, 25.00, 5.00]),
            kis=np.array([  5.0,  25.0,  35.0,  2.50,  2.50, 2.50]),
            kds=np.array([  3.0,   5.0,   7.0,  0.01,  0.01, 0.01]),
            integrator_maxs=np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]),
        )
        self.control_mode_rgtarm = ControlMode.POSITION
        
        self.pid_rgt_gripper = PIDController(1.0, 0.5, 0.1, integrator_max=0.1)
        self.control_mode_rgt_gripper = ControlMode.POSITION
        ########## ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ ##########
        ##################################### end 需要调参 用于位控模式 end ##########################################
        ###########################################################################################################

        super().__init__(config)

    def post_load_mjcf(self):
        try:
            self.init_joint_pose = self.mj_model.key(self.config.init_key).qpos[:self.njq]
            self.init_joint_ctrl = self.mj_model.key(self.config.init_key).ctrl[:self.njctrl]
        except (KeyError, AttributeError) as e:
            self.init_joint_pose = np.zeros(self.njq)
            self.init_joint_pose[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.init_joint_ctrl = np.zeros(self.njctrl)
        
        #######################################################################

        self.mit_init_param_kps = self.mj_model.actuator_gainprm[5:5+14, 0].copy()
        self.mit_init_param_kds = self.mj_model.actuator_gainprm[5+14:5+14*2, 1].copy()

        # control data
        self._ctr_wheel_vel = np.zeros(2) 
        
        self._ctr_slide_posi = self.mj_data.ctrl[2:3]
        self._ctr_head_posi = self.mj_data.ctrl[3:5]
        
        self._ctr_posi_arms  = self.mj_data.ctrl[5:19]
        self._ctr_vel_arms   = self.mj_data.ctrl[19:33]
        self._ctr_motor_arms = self.mj_data.ctrl[33:47]

        self._ctr_lft_arm_posi = self._ctr_posi_arms[:6]
        self._ctr_lft_arm_vel = self._ctr_vel_arms[:6]
        self._ctr_lft_arm_motor = self._ctr_motor_arms[:6]

        self._ctr_lft_gripper_posi = self._ctr_posi_arms[6:7]
        self._ctr_lft_gripper_vel = self._ctr_vel_arms[6:7]
        self._ctr_lft_gripper_motor = self._ctr_motor_arms[6:7]

        self._ctr_rgt_arm_posi = self._ctr_posi_arms[7:13]
        self._ctr_rgt_arm_vel = self._ctr_vel_arms[7:13]
        self._ctr_rgt_arm_motor = self._ctr_motor_arms[7:13]

        self._ctr_rgt_gripper_posi = self._ctr_posi_arms[13:14]
        self._ctr_rgt_gripper_vel = self._ctr_vel_arms[13:14]
        self._ctr_rgt_gripper_motor = self._ctr_motor_arms[13:14]
        
        #######################################################################
        # sensor data
        self._sensor_qpos  = self.mj_data.sensordata[:self.dof_act]
        self._sensor_qvel  = self.mj_data.sensordata[self.dof_act:2*self.dof_act]
        self._sensor_force = self.mj_data.sensordata[2*self.dof_act:3*self.dof_act]

        # base
        self._sensor_base_position    = self.mj_data.sensordata[3*self.dof_act:3*self.dof_act+3]
        self._sensor_base_orientation = self.mj_data.sensordata[3*self.dof_act+3:3*self.dof_act+7]
        self._sensor_base_linear_vel  = self.mj_data.sensordata[3*self.dof_act+7:3*self.dof_act+10]
        self._sensor_base_gyro        = self.mj_data.sensordata[3*self.dof_act+10:3*self.dof_act+13]
        self._sensor_base_acc         = self.mj_data.sensordata[3*self.dof_act+13:3*self.dof_act+16]

        # arm endpoint
        self._sensor_lftarm_ep = self.mj_data.sensordata[3*self.dof_act+16:3*self.dof_act+19]
        self._sensor_lftarm_eo = self.mj_data.sensordata[3*self.dof_act+19:3*self.dof_act+23]
        self._sensor_rgtarm_ep = self.mj_data.sensordata[3*self.dof_act+23:3*self.dof_act+26]
        self._sensor_rgtarm_eo = self.mj_data.sensordata[3*self.dof_act+26:3*self.dof_act+30]

        # wheel
        self._sensor_wheel_qpos = self._sensor_qpos[:2]
        self._sensor_wheel_qvel = self._sensor_qvel[:2]
        self._sensor_wheel_qctrl = self._sensor_force[:2]

        # slide
        self._sensor_slide_qpos = self._sensor_qpos[2:3]
        self._sensor_slide_qvel = self._sensor_qvel[2:3]
        self._sensor_slide_qctrl = self._sensor_force[2:3]

        # head
        self._sensor_head_qpos  = self._sensor_qpos[3:5]
        self._sensor_head_qvel  = self._sensor_qvel[3:5]
        self._sensor_head_qctrl = self._sensor_force[3:5]

        # left arm
        self._sensor_lft_arm_qpos  = self._sensor_qpos[5:11]
        self._sensor_lft_arm_qvel  = self._sensor_qvel[5:11]
        self._sensor_lft_arm_qctrl = self._sensor_force[5:11]

        # left gripper
        self._sensor_lft_gripper_qpos  = self._sensor_qpos[11:12]
        self._sensor_lft_gripper_qvel  = self._sensor_qvel[11:12]
        self._sensor_lft_gripper_ctrl = self._sensor_force[11:12]

        # right arm
        self._sensor_rgt_arm_qpos  = self._sensor_qpos[12:18]
        self._sensor_rgt_arm_qvel  = self._sensor_qvel[12:18]
        self._sensor_rgt_arm_qctrl = self._sensor_force[12:18]

        # right gripper
        self._sensor_rgt_gripper_qpos  = self._sensor_qpos[18:19]
        self._sensor_rgt_gripper_qvel  = self._sensor_qvel[18:19]
        self._sensor_rgt_gripper_ctrl = self._sensor_force[18:19]

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._ctr_wheel_vel[:] = 0.0
        self.mj_data.qpos[:self.njq] = self.init_joint_pose[:]
        self.mj_data.ctrl[:self.njctrl] = self.init_joint_ctrl[:]
        self.mj_model.actuator_gainprm[5:5+14, 0] =  self.mit_init_param_kps
        self.mj_model.actuator_biasprm[5:5+14, 1] = -self.mit_init_param_kps
        self.mj_model.actuator_gainprm[5+14:5+14*2, 0] =  self.mit_init_param_kds
        self.mj_model.actuator_biasprm[5+14:5+14*2, 2] = -self.mit_init_param_kds
        mujoco.mj_forward(self.mj_model, self.mj_data)

    #######################################################################
    def get_base_pose(self):
        """ return base position and orientation(wxyz) """
        return self._sensor_base_position.copy(), self._sensor_base_orientation.copy()

    def get_base_velocity(self):
        """ return base linear velocity world frame"""
        return self._sensor_base_linear_vel.copy()

    def get_base_gyro(self):
        """ return base angular velocity w.r.t imu frame """
        return self._sensor_base_gyro.copy()

    def get_base_acc(self):
        """ return base linear acceleration w.r.t imu frame """
        return self._sensor_base_acc.copy()
    
    def get_wheel_state(self):
        """ return wheel position velocity and torque """
        return self._sensor_wheel_qpos.copy(), self._sensor_wheel_qvel.copy(), self._sensor_wheel_qctrl.copy()

    def get_slide_state(self):
        """ return slide position velocity and torque """
        return self._sensor_slide_qpos.copy(), self._sensor_slide_qvel.copy(), self._sensor_slide_qctrl.copy()
    
    def get_head_state(self):
        """ return head position velocity and torque """
        return self._sensor_head_qpos.copy(), self._sensor_head_qvel.copy(), self._sensor_head_qctrl.copy()
    
    def get_left_arm_state(self):
        """ return left arm position velocity and torque """
        return self._sensor_lft_arm_qpos.copy(), self._sensor_lft_arm_qvel.copy(), self._sensor_lft_arm_qctrl.copy()
    
    def get_left_gripper_state(self):
        """ return left gripper position velocity and torque """
        return self._sensor_lft_gripper_qpos.copy(), self._sensor_lft_gripper_qvel.copy(), self._sensor_lft_gripper_ctrl.copy()
    
    def get_right_arm_state(self):
        """ return right arm position velocity and torque """
        return self._sensor_rgt_arm_qpos.copy(), self._sensor_rgt_arm_qvel.copy(), self._sensor_rgt_arm_qctrl.copy()
    
    def get_right_gripper_state(self):
        """ return right gripper position velocity and torque """
        return self._sensor_rgt_gripper_qpos.copy(), self._sensor_rgt_gripper_qvel.copy(), self._sensor_rgt_gripper_ctrl.copy()

    #######################################################################
    def set_mit_param(self, part_name, kps, kds):
        kps = np.array(kps)
        kds = np.array(kds)
        if part_name == "left_arm":
            assert len(kps) == 6, "left_arm kps should be 6-dim"
            assert len(kds) == 6, "left_arm kds should be 6-dim"
            self.mj_model.actuator_gainprm[5:5+6, 0] =  kps
            self.mj_model.actuator_biasprm[5:5+6, 1] = -kps
            self.mj_model.actuator_gainprm[5+14:5+14+6, 0] =  kds
            self.mj_model.actuator_biasprm[5+14:5+14+6, 2] = -kds
        elif part_name == "left_gripper":
            assert type(kps) is float or len(kps) == 1, "left_gripper kps should be 1-dim"
            assert type(kds) is float or len(kds) == 1, "left_gripper kds should be 1-dim"
            self.mj_model.actuator_gainprm[5+6:5+7, 0] =  kps
            self.mj_model.actuator_biasprm[5+6:5+7, 1] = -kps
            self.mj_model.actuator_gainprm[5+14+6:5+14+7, 0] =  kds
            self.mj_model.actuator_biasprm[5+14+6:5+14+7, 2] = -kds
        elif part_name == "right_arm":
            assert len(kps) == 6, "right_arm kps should be 6-dim"
            assert len(kds) == 6, "right_arm kds should be 6-dim"
            self.mj_model.actuator_gainprm[5+7:5+13, 0] =  kps
            self.mj_model.actuator_biasprm[5+7:5+13, 1] = -kps
            self.mj_model.actuator_gainprm[5+14+7:5+14+13, 0] =  kds
            self.mj_model.actuator_biasprm[5+14+7:5+14+13, 2] = -kds
        elif part_name == "right_gripper":
            assert type(kps) is float or len(kps) == 1, "right_gripper kps should be 1-dim"
            assert type(kds) is float or len(kds) == 1, "right_gripper kds should be 1-dim"
            self.mj_model.actuator_gainprm[5+13:5+14, 0] =  kps
            self.mj_model.actuator_biasprm[5+13:5+14, 1] = -kps
            self.mj_model.actuator_gainprm[5+14+13:5+14+14, 0] =  kds
            self.mj_model.actuator_biasprm[5+14+13:5+14+14, 2] = -kds
        else:
            raise ValueError("part_name should be 'left_arm', 'left_gripper', 'right_arm', or 'right_gripper'")

    ##################################
    # control functions
    # base slide head
    def set_target_base_velocity(self, target_velocity):
        """ target_velocity: [linear-vx, angular-vz] """
        assert len(target_velocity) == 2, "target_velocity should be 2-dim"

        self._ctr_wheel_vel[0] = (target_velocity[0] - target_velocity[1] * self.wheel_distance) / self.wheel_radius
        self._ctr_wheel_vel[1] = (target_velocity[0] + target_velocity[1] * self.wheel_distance) / self.wheel_radius

    def set_target_slide_position(self, target_position):
        assert len(target_position) == 1, "target_position should be 1-dim"
        self._ctr_slide_posi[:] = target_position

    def set_target_head_position(self, target_position):
        assert len(target_position) == 2, "target_position should be 2-dim"
        self._ctr_head_posi[:] = target_position
    
    ##################################
    # left arm
    def set_target_position_left_arm(self, target_position):
        assert len(target_position) == 6, "target_position should be 6-dim"
        self._ctr_lft_arm_posi[:] = target_position
        if self.control_mode_lftarm != ControlMode.POSITION:
            self.set_mit_param("left_arm", np.zeros(6), np.zeros(6))
            self.pidarr_lftarm.reset()
            self.control_mode_lftarm = ControlMode.POSITION
    
    def set_target_left_arm_MIT(self, target_position, target_velocity, target_motor):
        assert len(target_position) == 6, "target_position should be 6-dim"
        assert len(target_velocity) == 6, "target_velocity should be 6-dim"
        assert len(target_motor) == 6, "target_motor should be 6-dim"
        self.control_mode_lftarm = ControlMode.MIT
        self._ctr_lft_arm_posi[:] = target_position
        self._ctr_lft_arm_vel[:] = target_velocity
        self._ctr_lft_arm_motor[:] = target_motor

    ##################################
    # left gripper
    def set_target_position_left_gripper(self, target_position):
        assert len(target_position) == 1, "target_position should be 1-dim"
        self._ctr_lft_gripper_posi[:] = target_position
        if self.control_mode_lft_gripper != ControlMode.POSITION:
            self.set_mit_param("left_gripper", 0.0, 0.0)
            self.pid_lft_gripper.reset()
            self.control_mode_lft_gripper = ControlMode.POSITION

    def set_target_left_gripper_MIT(self, target_position, target_velocity, target_motor):
        assert len(target_position) == 1, "target_position should be 1-dim"
        assert len(target_velocity) == 1, "target_velocity should be 1-dim"
        assert len(target_motor) == 1, "target_motor should be 1-dim"
        self.control_mode_lft_gripper = ControlMode.MIT
        self._ctr_lft_gripper_posi[:] = target_position
        self._ctr_lft_gripper_vel[:] = target_velocity
        self._ctr_lft_gripper_motor[:] = target_motor

    ##################################
    # right arm
    def set_target_position_right_arm(self, target_position):
        assert len(target_position) == 6, "target_position should be 6-dim"
        self._ctr_rgt_arm_posi[:] = target_position
        if self.control_mode_rgtarm != ControlMode.POSITION:
            self.set_mit_param("right_arm", np.zeros(6), np.zeros(6))
            self.pidarr_rgtarm.reset()
            self.control_mode_rgtarm = ControlMode.POSITION

    def set_target_right_arm_MIT(self, target_position, target_velocity, target_motor):
        assert len(target_position) == 6, "target_position should be 6-dim"
        assert len(target_velocity) == 6, "target_velocity should be 6-dim"
        assert len(target_motor) == 6, "target_motor should be 6-dim"
        self.control_mode_rgtarm = ControlMode.MIT
        self._ctr_rgt_arm_posi[:] = target_position
        self._ctr_rgt_arm_vel[:] = target_velocity
        self._ctr_rgt_arm_motor[:] = target_motor

    ##################################
    # right gripper
    def set_target_position_right_gripper(self, target_position):
        assert len(target_position) == 1, "target_position should be 1-dim"
        self._ctr_rgt_gripper_posi[:] = target_position
        if self.control_mode_rgt_gripper != ControlMode.POSITION:
            self.set_mit_param("right_gripper", 0.0, 0.0)
            self.pid_rgt_gripper.reset()
            self.control_mode_rgt_gripper = ControlMode.POSITION

    def set_target_right_gripper_MIT(self, target_position, target_velocity, target_motor):
        assert len(target_position) == 1, "target_position should be 1-dim"
        assert len(target_velocity) == 1, "target_velocity should be 1-dim"
        assert len(target_motor) == 1, "target_motor should be 1-dim"
        self.control_mode_rgt_gripper = ControlMode.MIT
        self._ctr_rgt_gripper_posi[:] = target_position
        self._ctr_rgt_gripper_vel[:] = target_velocity
        self._ctr_rgt_gripper_motor[:] = target_motor

    #######################################################################
    def updateControl(self, _):
        # base velocity control
        self.mj_data.ctrl[:2] = self.pid_base_vel.output(np.clip(self._ctr_wheel_vel - self._sensor_wheel_qvel, -2.5, 2.5), self.mj_model.opt.timestep)

        if self.control_mode_lftarm == ControlMode.POSITION:
            self._ctr_lft_arm_motor[:] = self.pidarr_lftarm.output(self._ctr_lft_arm_posi - self._sensor_lft_arm_qpos, self.mj_model.opt.timestep)
        if self.control_mode_lft_gripper == ControlMode.POSITION:
            self._ctr_lft_gripper_motor[:] = self.pid_lft_gripper.output(self._ctr_lft_gripper_posi - self._sensor_lft_gripper_qpos, self.mj_model.opt.timestep)
        if self.control_mode_rgtarm == ControlMode.POSITION:
            self._ctr_rgt_arm_motor[:] = self.pidarr_rgtarm.output(self._ctr_rgt_arm_posi - self._sensor_rgt_arm_qpos, self.mj_model.opt.timestep)
        if self.control_mode_rgt_gripper == ControlMode.POSITION:
            self._ctr_rgt_gripper_motor[:] = self.pid_rgt_gripper.output(self._ctr_rgt_gripper_posi - self._sensor_rgt_gripper_qpos, self.mj_model.opt.timestep)

        self.mj_data.ctrl[:self.njctrl] = np.clip(self.mj_data.ctrl[:self.njctrl], self.mj_model.actuator_ctrlrange[:self.njctrl,0], self.mj_model.actuator_ctrlrange[:self.njctrl,1])

    def checkTerminated(self):
        return False

    def getObservation(self):
        return None

    def getPrivilegedObservation(self):
        return None

    def getReward(self):
        return None

cfg = BaseConfig()
cfg.mjcf_file_path = "mjcf/hardware_in_loop/mmk2.xml"
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
    print("从位控制模式切换到MIT模式时，务必 ！每次都要！ 调用set_mit_param函数设置该part的kp和kd，再调用set_target_xxx_MIT函数，之后才能运行exec_node.step函数")
    print("(part 包括：left_arm, left_gripper, right_arm, right_gripper)")

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    exec_node = MMK2_HIL(cfg)
    obs = exec_node.reset()

    print("初始时刻机器人状态：")
    base_posi, base_ori = exec_node.get_base_pose()
    print("    base 位置 = {}".format(base_posi))
    print("    base 方向 = {}".format(base_ori))
    base_vel = exec_node.get_base_velocity()
    print("    base 速度 = {}".format(base_vel))
    base_gyro = exec_node.get_base_gyro()
    print("    base 陀螺仪 = {}".format(base_gyro))
    base_acc = exec_node.get_base_acc()
    print("    base 加速度 = {}".format(base_acc))
    wheel_posi, wheel_vel, wheel_torque = exec_node.get_wheel_state()
    print("    轮子位置 = {}, 速度 = {}, 扭矩 = {}".format(wheel_posi, wheel_vel, wheel_torque))
    slide_posi, slide_vel, slide_torque = exec_node.get_slide_state()
    print("    滑轨位置 = {}, 速度 = {}, 扭矩 = {}".format(slide_posi, slide_vel, slide_torque))
    head_posi, head_vel, head_torque = exec_node.get_head_state()
    print("    头部位置 = {}, 速度 = {}, 扭矩 = {}".format(head_posi, head_vel, head_torque))
    left_arm_posi, left_arm_vel, left_arm_torque = exec_node.get_left_arm_state()
    print("    左臂位置 = {}".format(np.array2string(left_arm_posi, separator=', ')))
    print("       速度 = {}".format(np.array2string(left_arm_vel, separator=', ')))
    print("       扭矩 = {}".format(np.array2string(left_arm_torque, separator=', ')))
    left_gripper_posi, left_gripper_vel, left_gripper_torque = exec_node.get_left_gripper_state()
    print("    左爪位置 = {}, 速度 = {}, 扭矩 = {}".format(left_gripper_posi, left_gripper_vel, left_gripper_torque))
    right_arm_posi, right_arm_vel, right_arm_torque = exec_node.get_right_arm_state()
    print("    右臂位置 = {}".format(np.array2string(right_arm_posi, separator=', ')))
    print("       速度 = {}".format(np.array2string(right_arm_vel, separator=', ')))
    print("       扭矩 = {}".format(np.array2string(right_arm_torque, separator=', ')))
    right_gripper_posi, right_gripper_vel, right_gripper_torque = exec_node.get_right_gripper_state()
    print("    右爪位置 = {}, 速度 = {}, 扭矩 = {}".format(right_gripper_posi, right_gripper_vel, right_gripper_torque))

    print("初始时刻，各个部件的控制模式：")
    print("    左臂控制模式 = {}".format(exec_node.control_mode_lftarm))
    print("    左爪控制模式 = {}".format(exec_node.control_mode_lft_gripper))
    print("    右臂控制模式 = {}".format(exec_node.control_mode_rgtarm))
    print("    右爪控制模式 = {}".format(exec_node.control_mode_rgt_gripper))

    step_cnt = 0
    step_per_seceond = int(1./exec_node.delta_t)
    while exec_node.running:
        if step_cnt == int(step_per_seceond * 0):
            print("-" * 80)
            print("时间 = {:.1f}s".format(exec_node.mj_data.time))
            print("发送指令：")
            print("    向前移动，线速度0.5m/s，角速度0.0rad/s")
            print("    左臂设置目标为初始位置")
            print("    右臂设置目标为初始位置")
            print("    左爪设置目标为初始位置")
            print("    右爪设置目标为初始位置")
            exec_node.set_target_base_velocity([0.5, 0.0])
            exec_node.set_target_position_left_arm([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            exec_node.set_target_position_right_arm([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            exec_node.set_target_position_left_gripper([0.0])
            exec_node.set_target_position_right_gripper([0.0])
        elif step_cnt == int(step_per_seceond * 1.0):
            print("-" * 80)
            print("时间 = {:.1f}s".format(exec_node.mj_data.time))
            print("发送指令：")
            print("    停止，线速度0.0m/s，角速度0.0rad/s")
            exec_node.set_target_base_velocity([0., 0.0])
        elif step_cnt == int(step_per_seceond * 2.0):
            print("-" * 80)
            print("时间 = {:.1f}s".format(exec_node.mj_data.time))
            print("发送指令：")
            print("    原地旋转，线速度0.0m/s，角速度0.5rad/s")
            exec_node.set_target_base_velocity([0., 0.5])
        elif step_cnt == int(step_per_seceond * 3.0):
            print("-" * 80)
            print("时间 = {:.1f}s".format(exec_node.mj_data.time))
            print("    停止，线速度0.0m/s，角速度0.0rad/s")
            exec_node.set_target_base_velocity([0., 0.0])
        elif step_cnt == int(step_per_seceond * 4.0):
            print("-" * 80)
            print("时间 = {:.1f}s".format(exec_node.mj_data.time))
            print("控制升降机构，向下移动 0.3m")
            print("控制头部角度：yaw=0.5rad，pitch=0.5rad")
            exec_node.set_target_slide_position([0.3])
            exec_node.set_target_head_position([0.5, 0.5])
        elif step_cnt == int(step_per_seceond * 5.0):
            print("-" * 80)
            print("时间 = {:.1f}s".format(exec_node.mj_data.time))
            print("控制两个夹爪张开")
            exec_node.set_target_position_left_gripper([1.])
            exec_node.set_target_position_right_gripper([1.])
        elif step_cnt == int(step_per_seceond * 5.5):
            print("-" * 80)
            print("时间 = {:.1f}s".format(exec_node.mj_data.time))
            print("控制两个夹爪闭合")
            exec_node.set_target_position_left_gripper([0.])
            exec_node.set_target_position_right_gripper([0.])
        elif step_cnt == int(step_per_seceond * 6.):
            print("-" * 80)
            print("时间 = {:.1f}s".format(exec_node.mj_data.time))
            print("控制两个手的位置控制模式运动")
            print("    左臂：[0.3, 0.3, 0.5, 1., 0., -1.]")
            print("    右臂：[0.3, 0.3, 0.5, 1., 0., -1.]")
            exec_node.set_target_position_left_arm([0.3, 0.3, 0.5, 1., 0., -1.])
            exec_node.set_target_position_right_arm([0.3, 0.3, 0.5, 1., 0., -1.])
        elif step_cnt == int(step_per_seceond * 7.5):
            print("-" * 80)
            print("时间 = {:.1f}s".format(exec_node.mj_data.time))
            print("控制两个手的切换到MIT模式运动")
            print("    左臂：[0.3, 0.3, 0.5, 1., 0., -1.]")
            print("    右臂：[0.3, 0.3, 0.5, 1., 0., -1.]")
            print("注意：切换到MIT模式时，务必调用set_mit_param函数设置该part的kp和kd，再调用set_target_xxx_MIT函数，之后才能运行exec_node.step函数")

            exec_node.set_mit_param("left_arm",  [15., 15., 15., 2.5, 2.5, 2.5], [0.15, 0.175, 0.15, 0.05, 0.05, 0.05])
            exec_node.set_mit_param("right_arm", [15., 15., 15., 2.5, 2.5, 2.5], [0.15, 0.175, 0.15, 0.05, 0.05, 0.05])
            exec_node.set_target_left_arm_MIT([0.3, 0.3, 0.5, 1., 0., -1.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.])
            exec_node.set_target_right_arm_MIT([0.3, 0.3, 0.5, 1., 0., -1.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.])
        elif step_cnt == int(step_per_seceond * 9.):
            print("-" * 80)
            print("时间 = {:.1f}s".format(exec_node.mj_data.time))
            print("切换回位置控制模式运动")
            print("    左臂：[0.3, 0.3, 0.5, 1., 0., -1.]")
            print("    右臂：[0.3, 0.3, 0.5, 1., 0., -1.]")
            exec_node.set_target_position_left_arm([0.3, 0.3, 0.5, 1., 0., -1.])
            exec_node.set_target_position_right_arm([0.3, 0.3, 0.5, 1., 0., -1.])
        elif step_cnt == int(step_per_seceond * 10.):
            print("-" * 80)
            print("时间 = {:.1f}s".format(exec_node.mj_data.time))
            print("机器人状态：")
            base_posi, base_ori = exec_node.get_base_pose()
            print("    base 位置 = {}".format(base_posi))
            print("    base 方向 = {}".format(base_ori))
            base_vel = exec_node.get_base_velocity()
            print("    base 速度 = {}".format(base_vel))
            base_gyro = exec_node.get_base_gyro()
            print("    base 陀螺仪 = {}".format(base_gyro))
            base_acc = exec_node.get_base_acc()
            print("    base 加速度 = {}".format(base_acc))
            wheel_posi, wheel_vel, wheel_torque = exec_node.get_wheel_state()
            print("    轮子位置 = {}, 速度 = {}, 扭矩 = {}".format(wheel_posi, wheel_vel, wheel_torque))
            slide_posi, slide_vel, slide_torque = exec_node.get_slide_state()
            print("    滑轨位置 = {}, 速度 = {}, 扭矩 = {}".format(slide_posi, slide_vel, slide_torque))
            head_posi, head_vel, head_torque = exec_node.get_head_state()
            print("    头部位置 = {}, 速度 = {}, 扭矩 = {}".format(head_posi, head_vel, head_torque))
            left_arm_posi, left_arm_vel, left_arm_torque = exec_node.get_left_arm_state()
            print("    左臂位置 = {}".format(np.array2string(left_arm_posi, separator=', ')))
            print("       速度 = {}".format(np.array2string(left_arm_vel, separator=', ')))
            print("       扭矩 = {}".format(np.array2string(left_arm_torque, separator=', ')))
            left_gripper_posi, left_gripper_vel, left_gripper_torque = exec_node.get_left_gripper_state()
            print("    左爪位置 = {}, 速度 = {}, 扭矩 = {}".format(left_gripper_posi, left_gripper_vel, left_gripper_torque))
            right_arm_posi, right_arm_vel, right_arm_torque = exec_node.get_right_arm_state()
            print("    右臂位置 = {}".format(np.array2string(right_arm_posi, separator=', ')))
            print("       速度 = {}".format(np.array2string(right_arm_vel, separator=', ')))
            print("       扭矩 = {}".format(np.array2string(right_arm_torque, separator=', ')))
            right_gripper_posi, right_gripper_vel, right_gripper_torque = exec_node.get_right_gripper_state()
            print("    右爪位置 = {}, 速度 = {}, 扭矩 = {}".format(right_gripper_posi, right_gripper_vel, right_gripper_torque))
            exec_node.running = False

        obs, _, _, _, _ = exec_node.step()
        step_cnt += 1
