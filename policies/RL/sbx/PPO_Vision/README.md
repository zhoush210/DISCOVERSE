# PPO_Vision: Vision-Based PPO Algorithm

This project implements a vision-based Proximal Policy Optimization (PPO) reinforcement learning algorithm to control a robotic arm for grasping and placing tasks.

[中文版请见 README_zh.md](README_zh.md)

## Installation

Refer to https://github.com/araffin/sbx, using version v0.20.0.

## Algorithm Features

Compared with the state-based PPO algorithm (PPO_State), the main differences of PPO_Vision are:

1. **Vision Input**: Uses camera images as the observation space instead of directly using joint states and object positions. The image size is 84x84 RGB.
2. **CNN Feature Extractor**: Uses a Convolutional Neural Network (CNN) to process image input and extract visual features. The network consists of 3 convolutional layers and 1 fully connected layer.
3. **End-to-End Learning**: End-to-end learning from pixels to actions, no manual feature engineering required.
4. **Domain Randomization**: Enhances model generalization via domain randomization during environment resets.

## Environment Setup

The environment is based on DISCOVERSE's MMK2TaskBase. The task is to control a robotic arm to grasp a kiwi and place it on a plate.

### Observation Space
- Type: RGB image
- Shape: (3, 84, 84)
- Data type: uint8 (0-255)

### Action Space
- Type: Robotic arm joint angle control
- Dimension: Same as the robot's degrees of freedom
- Range: Determined by actuator_ctrlrange

### Reward Design
- Approach reward: Distance from the arm's end-effector to the target object
- Grasp reward: Successfully grasping the object
- Place reward: Placing the object at the target location
- Step penalty: Time penalty per step
- Action magnitude penalty: Prevents excessive actions

## CNN Feature Extractor

```python
class CNNFeatureExtractor(torch.nn.Module):
    def __init__(self, observation_space):
        super(CNNFeatureExtractor, self).__init__()
        # Input is (3, 84, 84) RGB image
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

## Usage

### Training

```bash
python train.py --render --total_timesteps 1000000 --batch_size 64 --learning_rate 3e-4
```

### Testing

```bash
python train.py --test --model_path path/to/model.zip --render --episodes 10 --deterministic
```

### TensorBoard Visualization

During training, TensorBoard is used to log and visualize key metrics:

1. **Start TensorBoard**:
   ```bash
   tensorboard --logdir data/PPO_Vision/logs_[timestamp]
   ```
   `[timestamp]` is the time when training started.

2. **View Training Metrics**:
   - Open http://localhost:6006 in your browser
   - Visualize key metrics:
     - Rewards: total reward, reward components (approach, place, etc.)
     - Losses: policy loss, value loss
     - Others: action entropy, learning rate, value estimates

3. **Tips**:
   - Use the smoothing slider to adjust curve display
   - Compare performance of different runs
   - Analyze parameter distributions in the HISTOGRAM tab
   - Track training progress in the SCALARS tab

### Main Parameters

- `--render`: Whether to display rendering
- `--total_timesteps`: Total training steps
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--model_path`: Model path (for testing or continued training)
- `--episodes`: Number of test episodes
- `--deterministic`: Whether to use a deterministic policy for testing

## File Structure

- `env.py`: Environment definition, including vision-based observation/action space, reward calculation, etc.
- `train.py`: Training and testing script for PPO
- `README.md`: Project documentation

---
[中文版请见 README_zh.md](README_zh.md)