import os

import numpy as np
from discoverse import DISCOVERSE_ASSERT_DIR, DISCOVERSE_ROOT_DIR

from discoverse.airbot_play.airbot_play_ik import AirbotPlayIK

class MMK2IK:
    def __init__(self, debug=False) -> None:
        self.debug = debug
        self.arm_ik = AirbotPlayIK(urdf = os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"))
        try:
            tmats = np.load(os.path.join(DISCOVERSE_ROOT_DIR, "discoverse/mmk2/mmk2_ik_tmats.npz"))
        except:
            print("Failed to load mmk2_ik_tmats.npz")
            print("Generating tmats from mjcf")
            tmats = self.generate_tmats()
            np.savez(os.path.join(DISCOVERSE_ROOT_DIR, "discoverse/mmk2/mmk2_ik_tmats.npz"), **tmats)

        self.TMat_footprint2chest = tmats["footprint2chest"]
        self.TMat_chest2lft_base = tmats["chest2lft_base"]
        self.TMat_chest2rgt_base = tmats["chest2rgt_base"]

    def generate_tmats(self, mjcf_path=None):
        import mujoco
        from discoverse.utils import get_body_tmat, get_site_tmat
        if mjcf_path is None:
            mjcf_path = os.path.join(DISCOVERSE_ASSERT_DIR, "mjcf", "mmk2_floor.xml")
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)
        tmats = {}
        tmat_footprint = get_site_tmat(mj_data, "mmk2_base")
        tmat_chest = get_body_tmat(mj_data, "slide_link")
        tmat_lft_armbase = get_body_tmat(mj_data, "lft_arm_base")
        tmat_rgt_armbase = get_body_tmat(mj_data, "rgt_arm_base")
        tmats["footprint2chest"] = np.linalg.inv(tmat_footprint) @ tmat_chest
        tmats["chest2lft_base"] = np.linalg.inv(tmat_chest) @ tmat_lft_armbase
        tmats["chest2rgt_base"] = np.linalg.inv(tmat_chest) @ tmat_rgt_armbase
        return tmats

    def armIK_wrt_footprint(self, position, rotation, arm:str, slide:float, q_ref=np.zeros(6)):
        """
        Parameters
        ----------
        position : np.ndarray
            3D position wrt mmk2 footprint
        rotation : np.ndarray
            3x3 rotation matrix wrt mmk2 footprint
        arm : str
            "l" or "r"
        slide : float
            slide value
        q_ref : np.ndarray, optional
            reference joint angles, by default np.zeros(6)
        Returns
        -------
        np.ndarray
            joint angles shape (6,)
        """
        tmat = self.TMat_footprint2chest.copy()
        tmat[2, 3] -= slide
        if arm == "l":
            Tmat_footprint2armbase = tmat @ self.TMat_chest2lft_base
        elif arm == "r":
            Tmat_footprint2armbase = tmat @ self.TMat_chest2rgt_base
        else:
            raise ValueError("Invalid arm")
        tmat_target_pose = np.eye(4)
        tmat_target_pose[:3, :3] = rotation
        tmat_target_pose[:3, 3] = position
        point3d_arm_base = (np.linalg.inv(Tmat_footprint2armbase) @ tmat_target_pose)
        posi_arm_local = point3d_arm_base[:3, 3]
        rot_arm_local = point3d_arm_base[:3, :3]
        try:
            jq = self.arm_ik.properIK(posi_arm_local, rot_arm_local, q_ref)
        except ValueError:
            if self.debug:
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("point wrt world :\n", tmat_target_pose)
                print("point_arm_base  :\n", point3d_arm_base)
                print("arm               :", arm)
                print("slide             :", slide)
                print("q_ref             :", q_ref)
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            raise ValueError(f"Invalid target position (arm-local) posi={posi_arm_local} rot={rot_arm_local}")
        return jq

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True, linewidth=500)
    mmk2_func = MMK2IK()