import numpy as np

s2r2025_position_info = {

    "left_cabinet_range" : {
        "xmin" :  0,
        "xmax" :  0.8,
        "ymin" :  1.22,
        "ymax" :  1.52,
    },
    "right_cabinet_range" : {
        "xmin" :  1.22,
        "xmax" :  1.52,
        "ymin" :  0,
        "ymax" :  0.8,
    },
    "left_table_range" : {
        "xmin" : -0.595,
        "xmax" :  1.005,
        "ymin" : -1.65,
        "ymax" : -0.05,
    },
    "right_table_range" : {
        "xmin" : -1.4,
        "xmax" : -0.6,
        "ymin" : -1.25,
        "ymax" :  0.35,
    },
    "cabinet" : {
        "position" : [
            [[0.25, 1.35, 0.422], [0.707, 0, 0, 0.707]],
            [[0.55, 1.35, 0.422], [0.707, 0, 0, 0.707]],
            [[0.25, 1.35, 0.742], [0.707, 0, 0, 0.707]],
            [[0.55, 1.35, 0.742], [0.707, 0, 0, 0.707]],
            [[0.25, 1.35, 1.062], [0.707, 0, 0, 0.707]],
            [[0.55, 1.35, 1.062], [0.707, 0, 0, 0.707]],

            [[1.35, 0.55, 0.422], [1., 0., 0., 0.]],
            [[1.35, 0.25, 0.422], [1., 0., 0., 0.]],
            [[1.35, 0.55, 0.742], [1., 0., 0., 0.]],
            [[1.35, 0.25, 0.742], [1., 0., 0., 0.]],
            [[1.35, 0.55, 1.062], [1., 0., 0., 0.]],
            [[1.35, 0.25, 1.062], [1., 0., 0., 0.]],
        ],
        "info" : [
            "left_cabinet,second_floor,left"  ,
            "left_cabinet,second_floor,right" ,
            "left_cabinet,third_floor,left"   ,
            "left_cabinet,third_floor,right"  ,
            "left_cabinet,fourth_floor,left"  ,
            "left_cabinet,fourth_floor,right" ,

            "right_cabinet,second_floor,left" ,
            "right_cabinet,second_floor,right",
            "right_cabinet,third_floor,left"  ,
            "right_cabinet,third_floor,right" ,
            "right_cabinet,fourth_floor,left" ,
            "right_cabinet,fourth_floor,right",
        ]
    },
    "carton" : {
        "position" : [
            [[-0.05, 0., 0.137], [1, 0, 0, 0]],
            [[ 0.05, 0., 0.137], [1, 0, 0, 0]],
        ],
    },
    "disk" : {
        "position" : [
            [[-0.075, 0., 0.125], [-0.707, 0, 0.707, 0]],
            [[-0.025, 0., 0.125], [-0.707, 0, 0.707, 0]],
            [[ 0.025, 0., 0.125], [-0.707, 0, 0.707, 0]],
            [[ 0.075, 0., 0.125], [-0.707, 0, 0.707, 0]],
        ],
    },
    "sheet" : {
        "position" : [
            [[-0.075,  0.035, 0.117], [0.707, 0, -0.707, 0]],
            [[-0.075, -0.035, 0.117], [0.707, 0, -0.707, 0]],
            [[-0.025,  0.035, 0.117], [0.707, 0, -0.707, 0]],
            [[-0.025, -0.035, 0.117], [0.707, 0, -0.707, 0]],
            [[ 0.025,  0.035, 0.117], [0.707, 0, -0.707, 0]],
            [[ 0.025, -0.035, 0.117], [0.707, 0, -0.707, 0]],
            [[ 0.075,  0.035, 0.117], [0.707, 0, -0.707, 0]],
            [[ 0.075, -0.035, 0.117], [0.707, 0, -0.707, 0]],            
        ],
    },
    "table" : {
        "position" : [
            [[ 0.805, -0.9 , 0.75], [0.707, 0., 0., -0.707]],
            [[ 0.605, -0.7 , 0.75], [0.707, 0., 0., -0.707]],
            [[ 0.405, -0.9 , 0.75], [0.707, 0., 0., -0.707]],
            [[ 0.205, -0.7 , 0.75], [0.707, 0., 0., -0.707]],
            [[ 0.005, -0.9 , 0.75], [0.707, 0., 0., -0.707]],
            [[-0.195, -0.7 , 0.75], [0.707, 0., 0., -0.707]],
            [[-0.395, -0.9 , 0.75], [0.707, 0., 0., -0.707]],

            [[-0.8 , -0.25 , 0.75], [0., 0., 0., 1.]],
            [[-1.0 , -0.05 , 0.75], [0., 0., 0., 1.]],
            [[-0.8 ,  0.15 , 0.75], [0., 0., 0., 1.]],
        ],
        "info" : [
            "left_table,right,front",
            "left_table,left,right,back",
            "left_table,left,right,front",
            "left_table,left,right,back",
            "left_table,left,right,front",
            "left_table,left,right,back",
            "left_table,left,front",

            "right_table,left,right,back",
            "right_table,left,right,front",
            "right_table,left,back",
        ]
    }
}

class TaskInfo:
    round = 0
    instruction = ""
    target_box_name = ""
    target_box_qpos_id = -1
    target_prop_type = ""
    target_prop_name = ""
    target_prop_qpos_id = -1
    table_direction = ""
    drawer_layer = ""
    scoring = {
        "a" : False,
        "b" : False,
        "c" : False,
        "d" : False,
    }
    def reset(self):
        self.round = 0
        self.instruction = ""
        self.target_box_name = ""
        self.target_box_qpos_id = -1
        self.target_prop_type = ""
        self.target_prop_name = ""
        self.target_prop_qpos_id = -1
        self.table_direction = ""
        self.drawer_layer = ""
        self.scoring = {
            "a" : False,
            "b" : False,
            "c" : False,
            "d" : False
        }

def box_within_cabinet(box_posi):
    if not (s2r2025_position_info["left_cabinet_range"]["xmin"] < box_posi[0] < s2r2025_position_info["left_cabinet_range"]["xmax"] and \
        s2r2025_position_info["left_cabinet_range"]["ymin"] < box_posi[1] < s2r2025_position_info["left_cabinet_range"]["ymax"]) and \
        not (s2r2025_position_info["right_cabinet_range"]["xmin"] < box_posi[0] < s2r2025_position_info["right_cabinet_range"]["xmax"] and \
        s2r2025_position_info["right_cabinet_range"]["ymin"] < box_posi[1] < s2r2025_position_info["right_cabinet_range"]["ymax"]):
        return False
    else:
        return True

def prop_within_table(prop_posi, table_name):
    tn = table_name + "_table_range"
    return (s2r2025_position_info[tn]["xmin"] < prop_posi[0] < s2r2025_position_info[tn]["xmax"] and \
        s2r2025_position_info[tn]["ymin"] < prop_posi[1] < s2r2025_position_info[tn]["ymax"])

def prop_in_gripper(gripper_left, gripper_right, prop_posi, prop_name):
    mid_posi = (gripper_left + gripper_right) / 2
    if prop_name == "sheet":
        dst_range = 0.1
    elif prop_name == "disk":
        dst_range = 0.15
    elif prop_name == "carton":
        dst_range = 0.2
    else:
        raise ValueError("Invalid prop_name: {}".format(prop_name))
    return np.linalg.norm(mid_posi - prop_posi) < dst_range
