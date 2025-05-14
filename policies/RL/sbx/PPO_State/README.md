# PPO_State: State-Based Proximal Policy Optimization Algorithm

This repository implements a state-based Proximal Policy Optimization (PPO) algorithm for robotic manipulation tasks within the DISCOVERSE environment. The implementation focuses on efficient learning from state observations for precise robotic control.

## Installation

Refer to https://github.com/araffin/sbx, using version v0.20.0

## Project Structure

- `env.py`: Environment implementation with state-based observation space, action space, and reward calculation
- `train.py`: Training script with PPO model configuration and training loop
- `inference.py`: Inference script for model evaluation and deployment

## Environment Architecture

### State Space Configuration

The observation space encompasses comprehensive state information:

```python
obs = np.concatenate([
    qpos,           # Joint positions
    qvel,           # Joint velocities
    kiwi_pos,       # Target object position
    plate_pos       # Goal position
]).astype(np.float32)
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

### Reward Function Design

The reward function incorporates multiple components for effective learning:

1. **Approach Reward**: Encourages the end-effector to approach the target object
   ```python
   approach_reward = (1 - np.tanh(2 * distance_to_kiwi)) * w_approach
   ```

2. **Placement Reward**: Rewards successful object placement
   ```python
   place_reward = (1 - np.tanh(2 * kiwi_to_plate)) * w_place
   ```

3. **Step Penalty**: Encourages efficient task completion
   ```python
   step_penalty = -w_step * self.current_step
   ```

4. **Action Magnitude Penalty**: Promotes smooth control
   ```python
   action_penalty = -w_action * action_magnitude
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
    tensorboard_log=log_dir # TensorBoard logging
)
```

## Model Evaluation

### Inference Configuration

```bash
python inference.py [
    --model_path STR      # Model path (default: "data\\PPO_State\\logs_20250514_132028\\final_model.zip")
    --render              # Enable rendering (default: True)
    --episodes INT        # Number of test episodes (default: 10000)
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
tensorboard --logdir data/PPO_State/logs_[timestamp]
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
cfg.init_key = "pick"
cfg.gs_model_dict = {
    "plate_white": "object/plate_white.ply",
    "kiwi": "object/kiwi.ply",
    "background": "scene/tsimf_library_1/point_cloud.ply"
}
cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"
```

### Task Success Criteria

```python
distance = np.hypot(
    tmat_kiwi[0, 3] - tmat_plate_white[0, 3],
    tmat_kiwi[1, 3] - tmat_plate_white[1, 3]
)
terminated = distance < 0.018  # Task completion threshold
```
