# PPO_Vision: Vision-Based Proximal Policy Optimization Algorithm

This repository implements a vision-based Proximal Policy Optimization (PPO) algorithm for robotic manipulation tasks within the DISCOVERSE environment. The implementation focuses on learning directly from visual observations for robust robotic control.

## Installation

Refer to https://github.com/araffin/sbx, using version v0.20.0

## Project Structure

- `env.py`: Environment implementation with vision-based observation space, action space, and reward calculation
- `train.py`: Training script with CNN feature extractor and PPO model configuration
- `inference.py`: Inference script for model evaluation and deployment

## Environment Architecture

### Vision Space Configuration

The observation space utilizes stacked RGB images:

```python
obs_shape = (3, 84, 84, 4)  # (channels, height, width, stack_size)
self.observation_space = spaces.Box(
    low=np.zeros(obs_shape, dtype=np.float32),
    high=np.ones(obs_shape, dtype=np.float32),
    dtype=np.float32
)
```

### Action Space

The action space is defined by the robot's joint control ranges:

```python
self.action_space = spaces.Box(
    low=ctrl_range[:, 0],    # Minimum joint angles
    high=ctrl_range[:, 1],   # Maximum joint angles
    dtype=np.float32
)
```

### CNN Feature Extractor

A custom CNN architecture processes visual observations:

```python
class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] * observation_space.shape[3]
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten()
        )
```

### Reward Function Design

The reward function incorporates multiple components for effective learning:

1. **Approach Reward**: Encourages the end-effector to approach the target object
   ```python
   if distance_to_kiwi < 0.05:
       approach_reward = 2.0
   else:
       approach_reward = -distance_to_kiwi
   ```

2. **Placement Reward**: Rewards successful object placement
   ```python
   if kiwi_to_plate < 0.02:  # Successful placement
       place_reward = 10.0
   elif kiwi_to_plate < 0.1:  # Near placement
       place_reward = 2.0
   else:
       place_reward = -kiwi_to_plate
   ```

3. **Step Penalty**: Encourages efficient task completion
   ```python
   step_penalty = -0.01 * self.current_step
   ```

4. **Action Magnitude Penalty**: Promotes smooth control
   ```python
   action_penalty = -0.1 * action_magnitude
   ```

## Training Configuration

### Command-line Arguments

```bash
python train.py [
    --render              # Enable environment rendering
    --seed INT            # Random seed (default: 42)
    --total_timesteps INT # Total training steps (default: 1000000)
    --batch_size INT      # Batch size (default: 64)
    --n_steps INT         # Steps per update (default: 2048)
    --learning_rate FLOAT # Learning rate (default: 3e-4)
    --log_dir STR        # Log directory path
    --model_path STR     # Pre-trained model path
    --log_interval INT   # Logging frequency (default: 10)
]
```

### PPO Model Parameters

```python
model = PPO(
    "MlpPolicy",
    env,
    n_steps=n_steps,        # Trajectory length per update
    batch_size=batch_size,  # Training batch size
    n_epochs=10,           # Update iterations
    gamma=0.99,            # Discount factor
    learning_rate=3e-4,    # Learning rate
    clip_range=0.2,        # PPO clipping parameter
    ent_coef=0.01,         # Entropy coefficient
    policy_kwargs=policy_kwargs,  # CNN feature extractor
    tensorboard_log=log_dir # TensorBoard logging
)
```

## Model Evaluation

### Inference Configuration

```bash
python inference.py [
    --model_path STR      # Model path (default: "data\\PPO_Vision\\logs_20250514_140005\\final_model.zip")
    --render              # Enable rendering (default: True)
    --episodes INT        # Number of test episodes (default: 10)
    --deterministic      # Use deterministic policy
    --seed INT           # Random seed (default: 42)
]
```

### Performance Metrics

The inference script provides detailed performance metrics:
- Episode rewards
- Success rate
- Average completion time
- Standard deviation of performance

## TensorBoard Visualization

Training progress can be monitored through TensorBoard:

```bash
tensorboard --logdir data/PPO_Vision/logs_[timestamp]
```

Key metrics include:
- Total rewards
- Component rewards (approach, placement)
- Policy loss
- Value function loss
- Action entropy

## Implementation Details

### Environment Initialization

```python
cfg.use_gaussian_renderer = False
cfg.gs_model_dict = {
    "plate_white": "object/plate_white.ply",
    "kiwi": "object/kiwi.ply",
    "background": "scene/tsimf_library_1/point_cloud.ply"
}
cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"
cfg.obs_rgb_cam_id = [0]  # Use first camera
```

### Task Success Criteria

```python
distance = np.hypot(
    tmat_kiwi[0, 3] - tmat_plate_white[0, 3],
    tmat_kiwi[1, 3] - tmat_plate_white[1, 3]
)
terminated = distance < 0.018  # Task completion threshold
```