import os
import mujoco
from scipy.spatial.transform import Rotation

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.utils import get_site_tmat

class MMK2FK:
    def __init__(self, mjcf_path=None):
        if mjcf_path is None:
            mjcf_path = os.path.join(DISCOVERSE_ROOT_DIR, "models/mjcf/mmk2_floor.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
    
    def set_base_pose(self, position, orientation):
        """
        Set the base pose of the robot.
        Params:
            position: 3D position of the base <xyz>
            orientation: 4D quaternion orientation of the base <wxyz>
        """
        self.mj_data.qpos[:3] = position
        self.mj_data.qpos[3:7] = orientation
        self.pos_modifidied = True
    
    def set_slide_joint(self, joint_angle):
        """
        Set the joint angle of the slide joint.
        Params:
            joint_angle: joint angle
        """
        self.mj_data.qpos[2] = joint_angle
        self.pos_modifidied = True
    
    def set_head_joints(self, joint_angles):
        """
        Set the joint angles of the head.
        Params:
            joint_angles: list of joint angles
        """
        assert len(joint_angles) == 2, "Head joints should have 2 joint angles."
        self.mj_data.qpos[3:5] = joint_angles
        self.pos_modifidied = True

    def set_left_arm_joints(self, joint_angles):
        """
        Set the joint angles of the left arm.
        Params:
            joint_angles: list of joint angles
        """
        assert len(joint_angles) == 6, "Left arm joints should have 6 joint angles"
        self.mj_data.qpos[5:11] = joint_angles
        self.pos_modifidied = True

    def set_right_arm_joints(self, joint_angles):
        """
        Set the joint angles of the right arm.
        Params:
            joint_angles: list of joint angles
        """
        assert len(joint_angles) == 6, "Right arm joints should have 6 joint angles"
        self.mj_data.qpos[13:19] = joint_angles
        self.pos_modifidied = True

    def forward_kinematics(self):
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.pos_modifidied = False

    def get_head_camera_pose(self):
        """
        Get the pose of the head camera w.r.t world.
        Returns:
            position: 3D position of the head camera <xyz>
            orientation: 4D quaternion orientation of the head camera <wxyz>
        """
        if self.pos_modifidied:
            self.forward_kinematics()
        tmat_head = get_site_tmat(self.mj_data, "headeye")
        position = tmat_head[:3, 3]
        orientation = Rotation.from_matrix(tmat_head[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return position, orientation

    def get_left_camera_pose(self):
        """
        Get the pose of the left camera w.r.t world.
        Returns:
            position: 3D position of the left camera <xyz>
            orientation: 4D quaternion orientation of the left camera <wxyz>"
        """
        if self.pos_modifidied:
            self.forward_kinematics()
        tmat_left = get_site_tmat(self.mj_data, "left_cam")
        position = tmat_left[:3, 3]
        orientation = Rotation.from_matrix(tmat_left[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return position, orientation
    
    def get_right_camera_pose(self):
        """
        Get the pose of the right camera w.r.t world.
        Returns:
            position: 3D position of the right camera <xyz>
            orientation: 4D quaternion orientation of the right camera <wxyz>
        """
        if self.pos_modifidied:
            self.forward_kinematics()
        tmat_right = get_site_tmat(self.mj_data, "right_cam")
        position = tmat_right[:3, 3]
        orientation = Rotation.from_matrix(tmat_right[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return position, orientation

    def get_left_endeffector_pose(self):
        """
        Get the pose of the left end effector w.r.t world.
        Returns:
            position: 3D position of the left end effector <xyz>
            orientation: 4D quaternion orientation of the left end effector <wxyz>
        """
        if self.pos_modifidied:
            self.forward_kinematics()
        tmat_left = get_site_tmat(self.mj_data, "lft_endpoint")
        position = tmat_left[:3, 3]
        orientation = Rotation.from_matrix(tmat_left[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return position, orientation

    def get_right_endeffector_pose(self):
        """
        Get the pose of the right end effector w.r.t world.
        Returns:
            position: 3D position of the right end effector <xyz>
            orientation: 4D quaternion orientation of the right end effector <wxyz>
        """
        if self.pos_modifidied:
            self.forward_kinematics()
        tmat_right = get_site_tmat(self.mj_data, "rgt_endpoint")
        position = tmat_right[:3, 3]
        orientation = Rotation.from_matrix(tmat_right[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return position, orientation

if __name__ == "__main__":
    fk = MMK2FK()
    
    fk.set_base_pose([0, 0, 0], [1, 0, 0, 0])
    fk.set_slide_joint(0)
    fk.set_head_joints([0, 0])
    fk.set_left_arm_joints([0, 0, 0, 0, 0, 0])
    fk.set_right_arm_joints([0, 0, 0, 0, 0, 0])

    print("Head camera pose: ", fk.get_head_camera_pose())
    print("Left camera pose: ", fk.get_left_camera_pose())
    print("Right camera pose: ", fk.get_right_camera_pose())

    print("Left end effector pose: ", fk.get_left_endeffector_pose())
    print("Right end effector pose: ", fk.get_right_endeffector_pose())

