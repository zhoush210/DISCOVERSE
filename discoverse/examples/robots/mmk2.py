import numpy as np
import sys
sys.path.append("../active_slam")
from dummy_robot import DummyRobot, DummyRobotConfig

class MMK2SlamCfg(DummyRobotConfig):
    robot_name     = "mmk2"
    mjcf_file_path = "mjcf/mmk2_floor_fixed.xml"
    timestep       = 0.0025
    decimation     = 4
    init_key       = "stand"
    rb_link_list   = [
        "agv_link", "slide_link", "head_yaw_link", "head_pitch_link",
        "lft_arm_base", "lft_arm_link1", "lft_arm_link2", 
        "lft_arm_link3", "lft_arm_link4", "lft_arm_link5", "lft_arm_link6",
        "lft_finger_left_link", "lft_finger_right_link", 
        "rgt_arm_base", "rgt_arm_link1", "rgt_arm_link2", 
        "rgt_arm_link3", "rgt_arm_link4", "rgt_arm_link5", "rgt_arm_link6",
        "rgt_finger_left_link", "rgt_finger_right_link"
    ]
    gs_model_dict  = {
        "agv_link"              :   "mmk2/agv_link.ply",
        "slide_link"            :   "mmk2/slide_link.ply",
        "head_pitch_link"       :   "mmk2/head_pitch_link.ply",
        "head_yaw_link"         :   "mmk2/head_yaw_link.ply",

        "lft_arm_base"          :   "airbot_play/arm_base.ply",
        "lft_arm_link1"         :   "airbot_play/link1.ply",
        "lft_arm_link2"         :   "airbot_play/link2.ply",
        "lft_arm_link3"         :   "airbot_play/link3.ply",
        "lft_arm_link4"         :   "airbot_play/link4.ply",
        "lft_arm_link5"         :   "airbot_play/link5.ply",
        "lft_arm_link6"         :   "airbot_play/link6.ply",
        "lft_finger_left_link"  :   "airbot_play/left.ply",
        "lft_finger_right_link" :   "airbot_play/right.ply",

        "rgt_arm_base"          :   "airbot_play/arm_base.ply",
        "rgt_arm_link1"         :   "airbot_play/link1.ply",
        "rgt_arm_link2"         :   "airbot_play/link2.ply",
        "rgt_arm_link3"         :   "airbot_play/link3.ply",
        "rgt_arm_link4"         :   "airbot_play/link4.ply",
        "rgt_arm_link5"         :   "airbot_play/link5.ply",
        "rgt_arm_link6"         :   "airbot_play/link6.ply",
        "rgt_finger_left_link"  :   "airbot_play/left.ply",
        "rgt_finger_right_link" :   "airbot_play/right.ply"
    }

class MMK2SlamRobot(DummyRobot):
    pitch_joint_id = 4
    def move_camera_pitch(self, d_pitch):
        self.mj_data.ctrl[self.pitch_joint_id] -= d_pitch

    def cv2WindowKeyPressCallback(self, key):
        ret = super().cv2WindowKeyPressCallback(key)
        if self.cam_id != -1:
            self.free_camera.lookat = self.mj_model.body(self.config.robot_name).pos
        return ret

    def printMessage(self):
        print(self.mj_model.body(self.config.robot_name).pos)
        print(self.mj_model.body(self.config.robot_name).quat)
        return super().printMessage()

if __name__ == "__main__":
    cfg = MMK2SlamCfg()
    cfg.y_up = False
    
    robot_cam_id = 0
    cfg.obs_rgb_cam_id   = [robot_cam_id]
    cfg.obs_depth_cam_id = [robot_cam_id]

    cfg.timestep = 1e-3
    cfg.decimation = 4
    cfg.render_set["fps"] = 60
    cfg.render_set["width"] = 1920
    cfg.render_set["height"] = 1080
    cfg.mjcf_file_path = "mjcf/mmk2_floor_fixed.xml"

    cfg.use_gaussian_renderer = True
    cfg.gs_model_dict["background"] = "scene/Air12F/air_12f.ply"
    # cfg.gs_model_dict["background"] = "scene/solid_background/white.ply"

    robot = MMK2SlamRobot(cfg)
    robot.cam_id = robot_cam_id

    robot.mj_model.body('mmk2').pos[:] = [22.02202576, -1.78662692, -0.96]

    action = np.zeros(4)
    # if z_up:
        # action[0] : lineal_velocity_x  local m    不论yup还是zup，始终为朝前为正方向
        # action[1] : lineal_velocity_y  local m    不论yup还是zup，始终为朝左为正方向
        # action[2] : angular_velocity_z rad        不论yup还是zup，始终为从上向下看逆时针旋转为正方向
        # action[3] : camera_pitch       rad        不论yup还是zup，始终为镜头俯仰
    # elif y_up:
        # action[0] : lineal_velocity_x   local m
        # action[1] : lineal_velocity_-z  local m
        # action[2] : angular_velocity_y  rad
        # action[3] : camera_pitch        rad

    obs = robot.reset()
    rgb_cam_posi = obs["rgb_cam_posi"]
    depth_cam_posi = obs["depth_cam_posi"]
    rgb_img_0 = obs["rgb_img"][0]
    depth_img_0 = obs["depth_img"][0]

    print("rgb_cam_posi    = ", rgb_cam_posi)
    # [[posi_x, posi_y, posi_z], [quat_w, quat_x, quat_y, quat_z]]
    # [(array([0., 0., 1.]), array([ 0.49999816,  0.50000184, -0.5       , -0.5       ]))]

    print("depth_cam_posi  = ", depth_cam_posi)
    # [[posi_x, posi_y, posi_z], [quat_w, quat_x, quat_y, quat_z]]
    # [(array([0., 0., 1.]), array([ 0.49999816,  0.50000184, -0.5       , -0.5       ]))]

    print("rgb_img.shape   = ", rgb_img_0.shape  , "rgb_img.dtype    = ", rgb_img_0.dtype)
    # rgb_img.shape   =  (1080, 1920, 3) rgb_img.dtype    =  uint8

    print("depth_img.shape = ", depth_img_0.shape, "depth_img.dtype  = ", depth_img_0.dtype)
    # depth_img.shape =  (1080, 1920, 1) depth_img.dtype  =  float32

    robot.printHelp()

    ##################################################################################################################
    np.set_printoptions(precision=3, suppress=True, linewidth=350)

    def step_func(current, target, step):
        if current < target - step:
            return current + step
        elif current > target + step:
            return current - step
        else:
            return target

    # robot.mj_model.body("mmk2").pos[:] = [22.02202576, -1.78662692, -0.96      ]

    robot_nj = 21
    tj = np.zeros(robot_nj)
    target_action = np.zeros_like(tj)
    joint_move_ratio = np.zeros_like(tj)

    key_id = 0
    while robot.running:

        for i in range(robot_nj):
            tj[i] = step_func(tj[i], target_action[i], 2. * joint_move_ratio[i] * robot.config.decimation * robot.mj_model.opt.timestep)
            robot.mj_data.qpos[i] = tj[i]

        jq_cur = robot.mj_data.qpos[:robot_nj]
        if np.allclose(jq_cur, target_action, atol=2e-2):
            key_id += 1
            if key_id >= robot.mj_model.nkey:
                key_id = 0
            target_action = robot.mj_model.key(key_id).qpos[:robot_nj]
            dif = np.abs(tj - target_action)
            joint_move_ratio = dif / (np.max(dif) + 1e-6)
 
        # obs, _, _, _, _ = robot.step(action)
        robot.view()