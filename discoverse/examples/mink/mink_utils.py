import os
import mujoco
import numpy as np

class mj_consts:
    """MuJoCo常量类，用于存储与MuJoCo框架相关的常量和映射关系。
    
    包含了支持的坐标系类型及其对应的枚举值、雅可比矩阵计算函数和位置/方向属性。
    """
    # 支持的坐标系类型
    SUPPORTED_FRAMES = ("body", "geom", "site")

    # 坐标系类型到MuJoCo对象枚举的映射
    FRAME_TO_ENUM = {
        "body": mujoco.mjtObj.mjOBJ_BODY,  # 刚体
        "geom": mujoco.mjtObj.mjOBJ_GEOM,  # 几何体
        "site": mujoco.mjtObj.mjOBJ_SITE,  # 站点
    }
    
    # 坐标系类型到对应雅可比矩阵计算函数的映射
    FRAME_TO_JAC_FUNC = {
        "body": mujoco.mj_jacBody,  # 计算刚体的雅可比矩阵
        "geom": mujoco.mj_jacGeom,  # 计算几何体的雅可比矩阵
        "site": mujoco.mj_jacSite,  # 计算站点的雅可比矩阵
    }
    
    # 坐标系类型到位置属性的映射
    FRAME_TO_POS_ATTR = {
        "body": "xpos",       # 刚体位置属性
        "geom": "geom_xpos",  # 几何体位置属性
        "site": "site_xpos",  # 站点位置属性
    }
    
    # 坐标系类型到旋转矩阵属性的映射
    FRAME_TO_XMAT_ATTR = {
        "body": "xmat",       # 刚体旋转矩阵属性
        "geom": "geom_xmat",  # 几何体旋转矩阵属性
        "site": "site_xmat",  # 站点旋转矩阵属性
    }

def mj_quat2mat(quat_wxyz):
    """将四元数转换为旋转矩阵。
    
    Args:
        quat_wxyz: 四元数，格式为[w, x, y, z]
        
    Returns:
        3x3旋转矩阵
    """
    rmat_tmp = np.zeros(9)
    mujoco.mju_quat2Mat(rmat_tmp, quat_wxyz)
    return rmat_tmp.reshape(3, 3)

def add_mocup_body_to_mjcf(mjcf_xml_path, mocap_body_xml, sensor_xml=None, keep_tmp_xml=False):
    """向MJCF模型添加运动捕捉(mocap)刚体和传感器。
    
    Args:
        mjcf_xml_path: MJCF文件路径
        mocap_body_xml: 要添加的mocap刚体的XML字符串
        sensor_xml: 要添加的传感器的XML字符串，默认为None
        keep_tmp_xml: 是否保留临时生成的XML文件，默认为False
        
    Returns:
        添加了mocap刚体和传感器的MuJoCo模型
    """
    with open(mjcf_xml_path, "r") as f:
        mjcf_xml_string = f.read()
        if mocap_body_xml is not None:
            new_mjcf_string = mjcf_xml_string.replace("</worldbody>", f"{mocap_body_xml}\n</worldbody>")
        if sensor_xml is not None:
            new_mjcf_string = new_mjcf_string.replace("</sensor>", f"{sensor_xml}\n</sensor>")

    tmp_mjcf_xml_path = mjcf_xml_path.replace(".xml", "_tmp.xml")
    with open(tmp_mjcf_xml_path, "w") as f:
        f.write(new_mjcf_string)
    m = mujoco.MjModel.from_xml_path(tmp_mjcf_xml_path)
    if not keep_tmp_xml:
        os.remove(tmp_mjcf_xml_path)
    return m

def move_mocap_to_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_name: str,
    frame_name: str,
    frame_type: str,
) -> None:
    """将mocap刚体移动到目标坐标系的位置和姿态。
    
    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        mocap_name: mocap刚体的名称
        frame_name: 目标坐标系的名称
        frame_type: 目标坐标系的类型，可以是"body"、"geom"或"site"
    
    Raises:
        KeyError: 如果指定的刚体不是mocap刚体
    """
    mocap_id = model.body(mocap_name).mocapid[0]
    if mocap_id == -1:
        raise KeyError(f"Body '{mocap_name}' is not a mocap body.")

    obj_id = mujoco.mj_name2id(model, mj_consts.FRAME_TO_ENUM[frame_type], frame_name)
    xpos = getattr(data, mj_consts.FRAME_TO_POS_ATTR[frame_type])[obj_id]
    xmat = getattr(data, mj_consts.FRAME_TO_XMAT_ATTR[frame_type])[obj_id]

    data.mocap_pos[mocap_id] = xpos.copy()
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], xmat)

def generate_mocap_xml(name, box_size=(0.05, 0.05, 0.05), arrow_length=0.05, rgba=(0.3, 0.6, 0.3, 0.2)):
    """生成mocap刚体的XML字符串。
    
    创建一个包含盒子和三个箭头（表示XYZ轴）的mocap刚体。
    
    Args:
        name: mocap刚体的名称
        box_size: 盒子的大小，默认为(0.05, 0.05, 0.05)
        arrow_length: 箭头的长度，默认为0.05
        rgba: 盒子的颜色和透明度，默认为(0.3, 0.6, 0.3, 0.2)
        
    Returns:
        包含mocap刚体定义的XML字符串
    """
    return f"""
    <body name="{name}" pos="0 0 0" quat="1 0 0 0" mocap="true">
      <site name="{name}_site" size='0.001' type='sphere'/>
      <geom name="{name}_box" type="box" size="{box_size[0]} {box_size[1]} {box_size[1]}" density="0" contype="0" conaffinity="0" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
      <geom type="cylinder" pos="{arrow_length} 0 0" euler="0 1.5708 0" size=".01 {arrow_length}" density="0" contype="0" conaffinity="0" rgba="1 0 0 .2"/>
      <geom type="cylinder" pos="0 {arrow_length} 0" euler="1.5708 0 0" size=".01 {arrow_length}" density="0" contype="0" conaffinity="0" rgba="0 1 0 .2"/>
      <geom type="cylinder" pos="0 0 {arrow_length}" euler="0 0 0"      size=".01 {arrow_length}" density="0" contype="0" conaffinity="0" rgba="0 0 1 .2"/>
    </body>
    """

def generate_mocap_sensor_xml(mocap_name, ref_name, ref_type="body"):
    """生成用于获取mocap刚体位置和方向的传感器XML字符串。
    
    Args:
        mocap_name: mocap刚体的名称
        ref_name: 参考坐标系的名称
        ref_type: 参考坐标系的类型，默认为"body"
        
    Returns:
        包含传感器定义的XML字符串
    """
    return f"""
    <framepos name="{mocap_name}_pos" objtype="site" objname="{mocap_name}_site" reftype="{ref_type}" refname="{ref_name}"/>
    <framequat name="{mocap_name}_quat" objtype="site" objname="{mocap_name}_site" reftype="{ref_type}" refname="{ref_name}"/>
    """
