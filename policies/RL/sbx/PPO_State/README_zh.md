# SBX PPO 强化学习训练框架

本目录包含使用 PPO (Proximal Policy Optimization) 算法进行强化学习训练的框架代码。这个框架专为 DISCOVERSE 环境设计，特别是针对机械臂抓取任务。

[English version see README_en.md](README.md)

## 安装
参考https://github.com/araffin/sbx，用的是v0.20.0

## 文件结构

- `env.py`: 环境类定义，包含观察空间、动作空间、奖励计算等
- `train.py`: 训练脚本，用于创建和训练 PPO 模型

## 环境配置与自定义

### 导入说明

在 `env.py` 中，我们从示例目录导入了 `SimNode` 和 `cfg`：

```python
from discoverse.examples.tasks_mmk2.pick_kiwi import SimNode, cfg
```

这里的 `pick_kiwi.py` 是位于 `discoverse/examples/tasks_mmk2/` 目录下的示例文件。您可以根据自己的任务需求，替换为其他任务文件。

### 环境配置

环境配置主要在 `Env` 类的 `__init__` 方法中设置：

```python
# 环境配置
cfg.use_gaussian_renderer = False  # 关闭高斯渲染器
cfg.init_key = "pick"  # 初始化模式为"抓取"
cfg.gs_model_dict["plate_white"] = "object/plate_white.ply"  # 定义白色盘子的模型路径
cfg.gs_model_dict["kiwi"] = "object/kiwi.ply"  # 定义奇异果的模型路径
cfg.gs_model_dict["background"] = "scene/tsimf_library_1/point_cloud.ply"  # 定义背景的模型路径
cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"  # MuJoCo 环境文件路径
cfg.obj_list = ["plate_white", "kiwi"]  # 环境中包含的对象列表
cfg.sync = True  # 是否同步更新
cfg.headless = not render  # 根据render参数决定是否显示渲染画面
```

您可以根据自己的任务需求修改这些配置，例如更改模型路径、对象列表等。

### 重点自定义部分

#### 1. 观察空间 (Observation Space)

观察空间定义了智能体能够感知到的环境状态。在 `_get_obs` 方法中自定义：

```python
def _get_obs(self):
    # 获取机械臂状态
    qpos = self.mj_data.qpos.copy()  # 关节位置
    qvel = self.mj_data.qvel.copy()  # 关节速度

    # 获取猕猴桃和盘子的位置
    tmat_kiwi = get_body_tmat(self.mj_data, "kiwi")  # 获取奇异果的变换矩阵
    tmat_plate = get_body_tmat(self.mj_data, "plate_white")  # 获取白色盘子的变换矩阵

    kiwi_pos = np.array([tmat_kiwi[1, 3], tmat_kiwi[0, 3], tmat_kiwi[2, 3]])  # 奇异果的位置
    plate_pos = np.array([tmat_plate[1, 3], tmat_plate[0, 3], tmat_plate[2, 3]])  # 盘子的位置

    # 组合观测
    obs = np.concatenate([
        qpos,
        qvel,
        kiwi_pos,
        plate_pos
    ]).astype(np.float32)  # 将关节状态和目标物体位置组合为观测值

    return obs
```

根据您的任务，您可能需要包含不同的状态信息，例如：
- 机械臂的关节角度和速度
- 目标物体的位置和姿态
- 障碍物的位置
- 传感器读数等

#### 2. 奖励函数 (Reward Function)

奖励函数是强化学习中最关键的部分之一，它定义了任务的目标。在 `_compute_reward` 方法中自定义：

```python
def _compute_reward(self):
    # 获取位置信息
    tmat_kiwi = get_body_tmat(self.mj_data, "kiwi")  # 奇异果的变换矩阵
    tmat_plate = get_body_tmat(self.mj_data, "plate_white")  # 盘子的变换矩阵
    tmat_rgt_arm = get_body_tmat(self.mj_data, "rgt_arm_link6")  # 右臂末端效应器的变换矩阵

    kiwi_pos = np.array([tmat_kiwi[1, 3], tmat_kiwi[0, 3], tmat_kiwi[2, 3]])  # 奇异果的位置
    plate_pos = np.array([tmat_plate[1, 3], tmat_plate[0, 3], tmat_plate[2, 3]])  # 盘子的位置
    rgt_arm_pos = np.array([tmat_rgt_arm[1, 3], tmat_rgt_arm[0, 3], tmat_rgt_arm[2, 3]])  # 右臂末端的位置

    # 计算距离
    distance_to_kiwi = np.linalg.norm(rgt_arm_pos - kiwi_pos)  # 右臂末端到奇异果的距离
    kiwi_to_plate = np.linalg.norm(kiwi_pos - plate_pos)  # 奇异果到盘子的距离

    # 计算各种奖励
    # 接近奖励：鼓励机械臂靠近奇异果
    approach_reward = 0.0
    if distance_to_kiwi < 0.1:
        approach_reward = 1.0
    elif distance_to_kiwi < 0.2:
        approach_reward = 0.5
    else:
        approach_reward = -distance_to_kiwi

    # 放置奖励：鼓励将奇异果放到盘子上
    place_reward = 0.0
    if kiwi_to_plate < 0.02:
        place_reward = 10.0
    elif kiwi_to_plate < 0.1:
        place_reward = 2.0
    else:
        place_reward = -kiwi_to_plate

    # 步数惩罚：每一步都有一定的惩罚
    step_penalty = -0.01 * self.current_step

    # 动作幅度惩罚：惩罚较大的控制信号
    action_magnitude = np.mean(np.abs(self.mj_data.ctrl))
    action_penalty = -0.1 * action_magnitude

    # 总奖励
    total_reward = (
        approach_reward +
        place_reward +
        step_penalty +
        action_penalty
    )

    # 记录日志
    self.info = {
        "rewards/approach_reward": approach_reward,
        "rewards/place_reward": place_reward,
        "rewards/step_penalty": step_penalty,
        "rewards/action_penalty": action_penalty,
        "info/distance_to_kiwi": distance_to_kiwi,
        "info/kiwi_to_plate": kiwi_to_plate,
        "info/action_magnitude": action_magnitude
    }

    return total_reward
```

设计奖励函数时，请考虑：
- 稀疏奖励 vs 密集奖励：稀疏奖励只在完成任务时给予，而密集奖励在每一步都提供反馈
- 奖励塑形：通过中间奖励引导智能体学习
- 惩罚项：对不希望的行为施加惩罚，如过大的动作、过多的步数等

#### 3. 终止条件 (Termination Condition)

终止条件定义了何时结束一个回合。在 `_check_termination` 方法中自定义：

```python
def _check_termination(self):
    # 检查是否完成任务
    tmat_kiwi = get_body_tmat(self.mj_data, "kiwi")  # 奇异果的变换矩阵
    tmat_plate = get_body_tmat(self.mj_data, "plate_white")  # 盘子的变换矩阵

    kiwi_pos = np.array([tmat_kiwi[1, 3], tmat_kiwi[0, 3], tmat_kiwi[2, 3]])  # 奇异果的位置
    plate_pos = np.array([tmat_plate[1, 3], tmat_plate[0, 3], tmat_plate[2, 3]])  # 盘子的位置

    # 如果奇异果成功放置在盘子上
    if np.linalg.norm(kiwi_pos - plate_pos) < 0.02:
        return True  # 任务完成，终止环境
    return False
```

## 训练与测试

### 训练模型

使用 `train.py` 脚本训练模型：

```bash
python train.py --render --total_timesteps 1000000 --batch_size 64
```

### TensorBoard 可视化

在训练过程中，我们使用 TensorBoard 来记录和可视化关键指标：

1. **启动 TensorBoard**：
   ```bash
   tensorboard --logdir data/PPO_State/logs_[timestamp]
   ```
   其中 `[timestamp]` 是训练开始时的时间戳。

2. **查看训练指标**：
   - 在浏览器中打开 http://localhost:6006
   - 可视化以下关键指标：
     - 奖励：总奖励、各奖励分量（接近奖励、放置奖励等）
     - 损失函数：策略损失、值函数损失
     - 其他指标：动作熵、学习率、状态值估计

3. **使用技巧**：
   - 使用平滑滑动条调整曲线显示
   - 对比不同训练运行的性能
   - 通过 HISTOGRAM 标签页分析参数分布
   - 使用 SCALARS 标签页跟踪训练进度

#### 训练参数说明：
- `--render`: 在训练过程中显示渲染画面（可选）
- `--seed`: 随机种子，默认为42
- `--total_timesteps`: 总训练步数，默认为1000000
- `--batch_size`: 批次大小，默认为64
- `--n_steps`: 每次更新所收集的轨迹长度，默认为2048
- `--learning_rate`: 学习率，默认为3e-4
- `--log_dir`: 日志目录，默认为自动生成的时间戳目录
- `--model_path`: 预训练模型路径，用于继续训练（可选）
- `--eval_freq`: 评估频率，默认为10000步
- `--log_interval`: 日志记录间隔，默认为10

#### 特色功能：

1. **训练进度条**：训练过程中会显示实时进度条，包括已训练步数和已用时间。

2. **训练恢复**：支持从之前的检查点继续训练，只需指定 `--model_path` 参数：
   ```bash
   python train.py --model_path data/PPO/logs_20230101_120000/best_model/best_model.zip
   ```

3. **自动保存**：训练过程中会自动保存性能最佳的模型，并在训练结束时保存最终模型。

### 测试模型

使用 `train.py` 脚本测试已训练的模型：

```bash
python train.py --test --model_path path/to/model.zip --render
```

#### 测试参数说明：
- `--test`: 启用测试模式
- `--model_path`: 模型路径（必需）
- `--render`: 在测试过程中显示渲染画面（可选）
- `--episodes`: 测试回合数，默认为10
- `--deterministic`: 使用确定性策略进行测试（可选）
- `--seed`: 随机种子，默认为42

## 自定义任务的步骤

1. **创建任务环境**：
   - 复制 `env.py` 并根据您的任务需求进行修改
   - 调整观察空间、奖励函数和终止条件

2. **配置环境**：
   - 修改 `cfg` 配置，包括模型路径、对象列表等
   - 根据需要调整渲染设置

3. **调整训练参数**：
   - 在 `train.py` 中修改 PPO 算法的超参数
   - 调整训练步数、批次大小等

4. **训练与评估**：
   - 运行训练脚本并监控训练进度
   - 定期评估模型性能并保存最佳模型

## 提示与技巧

1. **奖励设计**：
   - 奖励函数是强化学习中最关键的部分
   - 尝试不同的奖励组合，找到最有效的设计
   - 使用奖励分量来引导智能体学习复杂任务

2. **观察空间**：
   - 只包含任务相关的信息，避免无关信息干扰学习
   - 考虑使用相对位置而非绝对位置

3. **超参数调优**：
   - 学习率、批次大小、更新频率等超参数对训练效果有显著影响
   - 使用网格搜索或贝叶斯优化来寻找最佳超参数

4. **可视化与调试**：
   - 使用 TensorBoard 等工具可视化训练过程
   - 记录详细的奖励分量，便于分析和调试

5. **域随机化**：
   - 在训练中引入随机性，提高模型的泛化能力
   - 随机化初始状态、物体位置、物理参数等

---
[English version see README.md](README.md)
