import os
import time
import mujoco
import mujoco.viewer
import numpy as np

from discoverse.examples.mink.mink_utils import \
    mj_quat2mat, \
    add_mocup_body_to_mjcf, \
    move_mocap_to_frame, \
    generate_mocap_xml, \
    generate_mocap_sensor_xml

from discoverse import DISCOVERSE_ASSERT_DIR
try:
    # 尝试导入AirbotPlayFIK类（首选方法）
    from discoverse.airbot_play.airbot_play_fik import AirbotPlayFIK
except:
    # 导入失败时输出错误信息并使用备选方法
    print("Failed to import AirbotPlayFIK from discoverse.airbot_play.airbot_play_fik")
    print("Imported AirbotPlayIK_nopin instead")
    from discoverse.airbot_play.airbot_play_ik_nopin import AirbotPlayIK_nopin as AirbotPlayFIK

if __name__ == "__main__":  
    """
    Airbot Play机器人的仿真主程序
    
    该程序创建一个Airbot Play机器人模型的MuJoCo仿真环境，添加运动捕捉(mocap)目标，
    并使用逆运动学(IK)控制机器人的单臂跟踪目标位置和姿态。
    """
    # 设置numpy输出格式
    np.set_printoptions(precision=5, suppress=True, linewidth=500)

    # 加载Airbot Play机器人模型的MJCF文件
    mjcf_path = os.path.join(DISCOVERSE_ASSERT_DIR, "mjcf", "airbot_play_floor.xml")
    print("mjcf_path : " , mjcf_path)

    # 设置渲染帧率
    render_fps = 50

    # 设置末端执行器目标（mocap）名称
    mocap_name = "end_target"
    mocap_box_name = mocap_name + "_box"

    # 生成mocap刚体XML
    mocap_body_xml = generate_mocap_xml(mocap_name)
    # 生成mocap传感器XML，参考坐标系为机械臂基座
    sensor_xml = generate_mocap_sensor_xml(mocap_name, ref_name="arm_base", ref_type="body")
    # 将mocap刚体和传感器添加到模型中
    mj_model = add_mocup_body_to_mjcf(mjcf_path, mocap_body_xml, sensor_xml, keep_tmp_xml=False)
    # 计算渲染间隔，确保按照指定帧率渲染
    render_gap = int(1.0 / render_fps / mj_model.opt.timestep)

    # 创建MuJoCo数据实例
    mj_data = mujoco.MjData(mj_model)

    # 获取mocap位置和方向传感器ID
    posi_sensor_id = mj_model.sensor(f"{mocap_name}_pos").adr[0]
    quat_sensor_id = mj_model.sensor(f"{mocap_name}_quat").adr[0]

    # 获取机械臂6个关节位置传感器ID
    armjoint_snesor_ids = [mj_model.sensor(f"joint{i+1}_pos").adr[0] for i in range(6)]
    # 获取机械臂6个关节控制器ID
    armjoint_ctrl_ids = [mj_model.actuator(f"joint{i+1}").id for i in range(6)]

    # 创建Airbot Play逆运动学求解器，加载URDF模型
    arm_fik = AirbotPlayFIK(urdf = os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"))
    
    # 启动MuJoCo查看器
    with mujoco.viewer.launch_passive(
        mj_model, mj_data, 
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        while viewer.is_running():
            # 记录步骤开始时间
            step_start = time.time()

            # 执行渲染间隔次数的物理仿真步骤
            for _ in range(render_gap):
                # 执行物理仿真步骤
                mujoco.mj_step(mj_model, mj_data)

            # 获取目标在机械臂基座坐标系中的位置
            t_posi_local = mj_data.sensordata[posi_sensor_id:posi_sensor_id+3]
            # 获取目标在机械臂基座坐标系中的旋转矩阵
            t_rmat_local = mj_quat2mat(mj_data.sensordata[quat_sensor_id:quat_sensor_id+4])

            # 获取当前关节位置作为参考值
            q_ref = mj_data.sensordata[armjoint_snesor_ids].copy()
            try:
                # 计算逆运动学解，获取关节角度
                jq = arm_fik.properIK(t_posi_local, t_rmat_local, q_ref)
                # 控制关节执行计算出的角度
                mj_data.ctrl[armjoint_ctrl_ids] = jq
                # 设置目标框为绿色（表示IK计算成功）
                mj_model.geom(mocap_box_name).rgba = (0.3, 0.6, 0.3, 0.2)
            except ValueError as e:
                # 捕获逆运动学计算异常
                print(e)
                # 将目标移动到末端执行器位置
                move_mocap_to_frame(mj_model, mj_data, mocap_name, "endpoint", "site")
                # 设置目标框为红色（表示IK计算失败）
                mj_model.geom(mocap_box_name).rgba = (0.6, 0.3, 0.3, 0.2)

            # 同步查看器状态
            viewer.sync()
            
            # 计算下一步开始前需要等待的时间，保证帧率稳定
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)