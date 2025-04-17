# 基于视觉的PPO算法 (PPO_Vision)

本项目实现了一个基于视觉输入的近端策略优化（Proximal Policy Optimization, PPO）强化学习算法，用于控制机械臂完成抓取和放置任务。

[English version see README.md](README.md)

## 安装

参考 https://github.com/araffin/sbx，使用 v0.20.0 版本。

## 算法特点

与基于状态的PPO算法（PPO_State）相比，PPO_Vision的主要区别在于：

1. **视觉输入**：使用摄像头图像作为观察空间，而不是直接使用机械臂关节状态和物体位置。图像大小为84x84的RGB图像。
2. **CNN特征提取器**：使用卷积神经网络(CNN)处理图像输入，提取视觉特征。网络结构包含3层卷积层和1层全连接层。
3. **端到端学习**：从像素到动作的端到端学习，无需手动特征工程。
4. **域随机化**：通过环境重置时的域随机化增强模型泛化能力。

## 环境设置

环境基于DISCOVERSE框架中的MMK2TaskBase，任务是控制机械臂抓取奇异果并放置到盘子上。

### 观察空间
- 类型：RGB图像
- 维度：(3, 84, 84)
- 数据类型：uint8（0-255）

### 动作空间
- 类型：机械臂关节角度控制
- 维度：与机械臂自由度相同
- 范围：由actuator_ctrlrange确定

### 奖励设计
- 接近奖励：机械臂末端到目标物体的距离
- 抓取奖励：成功抓取目标物体
- 放置奖励：将物体放置到指定位置
- 步数惩罚：每步的时间惩罚
- 动作幅度惩罚：防止动作过大

## CNN特征提取器

```python
class CNNFeatureExtractor(torch.nn.Module):
    def __init__(self, observation_space):
        super(CNNFeatureExtractor, self).__init__()
        # 输入是(3, 84, 84)的RGB图像
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, 512),
            torch.nn.ReLU()
        )
        self._features_dim = 512
    def forward(self, observations):
        return self.linear(self.cnn(observations))
    @property
    def features_dim(self):
        return self._features_dim
```

## 使用方法

### 训练模型

```bash
python train.py --render --total_timesteps 1000000 --batch_size 64 --learning_rate 3e-4
```

### 测试模型

```bash
python train.py --test --model_path path/to/model.zip --render --episodes 10 --deterministic
```

### TensorBoard 可视化

在训练过程中，我们使用 TensorBoard 来记录和可视化关键指标：

1. **启动 TensorBoard**：
   ```bash
   tensorboard --logdir data/PPO_Vision/logs_[timestamp]
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

### 主要参数

- `--render`: 是否显示渲染画面
- `--total_timesteps`: 总训练步数
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--model_path`: 模型路径（用于测试或继续训练）
- `--episodes`: 测试回合数
- `--deterministic`: 是否使用确定性策略进行测试

## 文件结构

- `env.py`: 环境定义，包含基于视觉的观察空间、动作空间、奖励计算等
- `train.py`: 训练和测试脚本，用于创建和训练PPO模型
- `README.md`: 项目说明文档

---
[English version see README.md](README_zh.md)
