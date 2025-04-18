# SBX PPO Reinforcement Learning Training Framework

This directory contains the framework code for reinforcement learning training using the PPO (Proximal Policy Optimization) algorithm. This framework is designed for the DISCOVERSE environment, especially for robotic arm grasping tasks.

[中文版请见 README_zh.md](README_zh.md)

## Installation
Refer to https://github.com/araffin/sbx, using version v0.20.0

## File Structure

- `env.py`: Environment class definition, including observation space, action space, reward calculation, etc.
- `train.py`: Training script for creating and training PPO models

## Environment Configuration and Customization

### Import Instructions

In `env.py`, we import `SimNode` and `cfg` from the example directory:

```python
from discoverse.examples.tasks_mmk2.pick_kiwi import SimNode, cfg
```

Here, `pick_kiwi.py` is an example file located in the `discoverse/examples/tasks_mmk2/` directory. You can replace it with other task files according to your needs.

### Environment Configuration

The environment configuration is mainly set in the `__init__` method of the `Env` class:

```python
# Environment configuration
cfg.use_gaussian_renderer = False  # Disable Gaussian renderer
cfg.init_key = "pick"  # Initialization mode: "pick"
cfg.gs_model_dict["plate_white"] = "object/plate_white.ply"  # Path to white plate model
cfg.gs_model_dict["kiwi"] = "object/kiwi.ply"  # Path to kiwi model
cfg.gs_model_dict["background"] = "scene/tsimf_library_1/point_cloud.ply"  # Path to background model
cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"  # MuJoCo environment file path
cfg.obj_list = ["plate_white", "kiwi"]  # List of objects in the environment
cfg.sync = True  # Whether to update synchronously
cfg.headless = not render  # Whether to show rendering based on render parameter
```

You can modify these configurations as needed, such as changing model paths or object lists.

### Key Customization Parts

#### 1. Observation Space

The observation space defines the environmental states accessible to the agent. Customize in the `_get_obs` method:

```python
def _get_obs(self):
    # Get robot arm state
    qpos = self.mj_data.qpos.copy()  # Joint positions
    qvel = self.mj_data.qvel.copy()  # Joint velocities

    # Get positions of kiwi and plate
    tmat_kiwi = get_body_tmat(self.mj_data, "kiwi")  # Kiwi transformation matrix
    tmat_plate = get_body_tmat(self.mj_data, "plate_white")  # Plate transformation matrix

    kiwi_pos = np.array([tmat_kiwi[1, 3], tmat_kiwi[0, 3], tmat_kiwi[2, 3]])  # Kiwi position
    plate_pos = np.array([tmat_plate[1, 3], tmat_plate[0, 3], tmat_plate[2, 3]])  # Plate position

    # Combine observations
    obs = np.concatenate([
        qpos,
        qvel,
        kiwi_pos,
        plate_pos
    ]).astype(np.float32)

    return obs
```

Depending on your task, you may need to include different state information, such as:
- Joint angles and velocities of the robot arm
- Position and pose of the target object
- Position of obstacles
- Sensor readings, etc.

#### 2. Reward Function

The reward function is one of the most critical parts of reinforcement learning, defining the task objective. Customize in the `_compute_reward` method:

```python
def _compute_reward(self):
    # Get position information
    tmat_kiwi = get_body_tmat(self.mj_data, "kiwi")  # Kiwi transformation matrix
    tmat_plate = get_body_tmat(self.mj_data, "plate_white")  # Plate transformation matrix
    tmat_rgt_arm = get_body_tmat(self.mj_data, "rgt_arm_link6")  # Right arm end-effector transformation matrix

    kiwi_pos = np.array([tmat_kiwi[1, 3], tmat_kiwi[0, 3], tmat_kiwi[2, 3]])  # Kiwi position
    plate_pos = np.array([tmat_plate[1, 3], tmat_plate[0, 3], tmat_plate[2, 3]])  # Plate position
    rgt_arm_pos = np.array([tmat_rgt_arm[1, 3], tmat_rgt_arm[0, 3], tmat_rgt_arm[2, 3]])  # Right arm end position

    # Calculate distances
    distance_to_kiwi = np.linalg.norm(rgt_arm_pos - kiwi_pos)  # Distance from arm to kiwi
    kiwi_to_plate = np.linalg.norm(kiwi_pos - plate_pos)  # Distance from kiwi to plate

    # Calculate rewards
    # Approach reward: encourage arm to approach kiwi
    approach_reward = 0.0
    if distance_to_kiwi < 0.1:
        approach_reward = 1.0
    elif distance_to_kiwi < 0.2:
        approach_reward = 0.5
    else:
        approach_reward = -distance_to_kiwi

    # Placement reward: encourage placing kiwi onto plate
    place_reward = 0.0
    if kiwi_to_plate < 0.02:
        place_reward = 10.0
    elif kiwi_to_plate < 0.1:
        place_reward = 2.0
    else:
        place_reward = -kiwi_to_plate

    # Step penalty: penalize each step
    step_penalty = -0.01 * self.current_step

    # Action magnitude penalty: penalize large control signals
    action_magnitude = np.mean(np.abs(self.mj_data.ctrl))
    action_penalty = -0.1 * action_magnitude

    # Total reward
    total_reward = (
        approach_reward +
        place_reward +
        step_penalty +
        action_penalty
    )

    # For logging
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

When designing reward functions, consider:
- Sparse vs. dense rewards: Sparse rewards are only given upon task completion, while dense rewards provide feedback at every step
- Reward shaping: Use intermediate rewards to guide learning
- Penalties: Penalize undesired behaviors, such as large actions or excessive steps

#### 3. Termination Condition

Termination conditions define when an episode ends. Customize in the `_check_termination` method:

```python
def _check_termination(self):
    # Check if task is completed
    tmat_kiwi = get_body_tmat(self.mj_data, "kiwi")
    tmat_plate = get_body_tmat(self.mj_data, "plate_white")

    kiwi_pos = np.array([tmat_kiwi[1, 3], tmat_kiwi[0, 3], tmat_kiwi[2, 3]])
    plate_pos = np.array([tmat_plate[1, 3], tmat_plate[0, 3], tmat_plate[2, 3]])

    # If kiwi is successfully placed on the plate
    if np.linalg.norm(kiwi_pos - plate_pos) < 0.02:
        return True  # Task complete
    return False
```

## Training and Testing

### Training the Model

Use the `train.py` script to train the model:

```bash
python train.py --render --total_timesteps 1000000 --batch_size 64
```

### TensorBoard visualization

During the training process, we use TensorBoard to record and visualize key metrics:

1. **Launch TensorBoard**:
    ```bash
    tensorboard --logdir data/PPO_State/logs_[timestamp]
    ```
    Among them, 'timestamp' is the timestamp at the beginning of training.

2. **View training metrics**:
    -Open in browser http://localhost:6006
    -Visualize the following key indicators:
    -Rewards: Total rewards, individual reward components (proximity rewards, placement rewards, etc.)
    -Loss functions: policy loss, value function loss
    -Other indicators: action entropy, learning rate, state value estimation

3. **Usage tips**:
    -Adjust curve display using a smooth slider
    -Compare the performance of different training runs
    -Analyze parameter distribution through the HISTOGRAM tab
    -Track training progress using SCALARS tabs

#### Training Parameters:
- `--render`: Show rendering during training (optional)
- `--seed`: Random seed, default is 42
- `--total_timesteps`: Total training steps, default is 1000000
- `--batch_size`: Batch size, default is 64
- `--n_steps`: Trajectory length per update, default is 2048
- `--learning_rate`: Learning rate, default is 3e-4
- `--log_dir`: Log directory, default is a timestamped directory
- `--model_path`: Pretrained model path for continued training (optional)
- `--eval_freq`: Evaluation frequency, default is 10000 steps
- `--log_interval`: Logging interval, default is 10

#### Features:

1. **Training Progress Bar**: Real-time progress bar during training, showing trained steps and elapsed time.

2. **Training Resume**: Supports resuming from previous checkpoints by specifying the `--model_path` parameter:
   ```bash
   python train.py --model_path data/PPO/logs_20230101_120000/best_model/best_model.zip
   ```

3. **Auto-Save**: Automatically saves the best-performing model during training and the final model at the end.

### Testing the Model

Use the `train.py` script to test the trained model:

```bash
python train.py --test --model_path path/to/model.zip --render
```

#### Testing Parameters:
- `--test`: Enable test mode
- `--model_path`: Model path (required)
- `--render`: Show rendering during testing (optional)
- `--episodes`: Number of test episodes, default is 10
- `--deterministic`: Use deterministic policy for testing (optional)
- `--seed`: Random seed, default is 42

## Steps to Customize Your Task

1. **Create Task Environment**:
   - Copy `env.py` and modify it according to your task
   - Adjust observation space, reward function, and termination conditions

2. **Configure Environment**:
   - Modify `cfg` configuration, including model paths, object list, etc.
   - Adjust rendering settings as needed

3. **Adjust Training Parameters**:
   - Modify PPO hyperparameters in `train.py`
   - Adjust training steps, batch size, etc.

4. **Train and Evaluate**:
   - Run the training script and monitor progress
   - Periodically evaluate model performance and save the best model

## Tips and Tricks

1. **Reward Design**:
   - The reward function is the most critical part of reinforcement learning
   - Try different reward combinations to find the most effective design
   - Use reward components to guide the agent to learn complex tasks

2. **Observation Space**:
   - Only include task-relevant information to avoid distracting the agent
   - Consider using relative positions instead of absolute positions

3. **Hyperparameter Tuning**:
   - Learning rate, batch size, and update frequency significantly affect training
   - Use grid search or Bayesian optimization to find the best hyperparameters

4. **Visualization and Debugging**:
   - Use tools like TensorBoard to visualize the training process
   - Log detailed reward components for analysis and debugging

5. **Domain Randomization**:
   - Introduce randomness during training to improve model generalization
   - Randomize initial states, object positions, physical parameters, etc.

---
[中文版请见 README_zh.md](README_zh.md)
