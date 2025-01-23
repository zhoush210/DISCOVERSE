from .controllor import PIDController, PIDarray
from .base_config import BaseConfig
from .statemachine import SimpleStateMachine

import numpy as np
from scipy.spatial.transform import Rotation

def get_site_tmat(mj_data, site_name):
    tmat = np.eye(4)
    tmat[:3,:3] = mj_data.site(site_name).xmat.reshape((3,3))
    tmat[:3,3] = mj_data.site(site_name).xpos
    return tmat

def get_body_tmat(mj_data, body_name):
    tmat = np.eye(4)
    tmat[:3,:3] = Rotation.from_quat(mj_data.body(body_name).xquat[[1,2,3,0]]).as_matrix()
    tmat[:3,3] = mj_data.body(body_name).xpos
    return tmat

def step_func(current, target, step):
    if current < target - step:
        return current + step
    elif current > target + step:
        return current - step
    else:
        return target
