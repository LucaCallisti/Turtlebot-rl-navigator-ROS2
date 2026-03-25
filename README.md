# TurtleBot RL Navigator (ROS2 + PPO)

Autonomous navigation project for TurtleBot3 in Gazebo using reinforcement learning (PPO with Stable-Baselines3) and a ROS2 inference node.


https://github.com/user-attachments/assets/1f7ac3a6-e61c-4aae-a9e9-ff086e04a31c


## Project Description

This repository implements a full training-to-deployment loop:

1. A Gymnasium environment (`NavEnv`) connected to ROS2 topics.
2. PPO training with `VecNormalize`, checkpoints, and periodic evaluation.
3. A lightweight ROS2 inference node that loads the trained policy and publishes velocity commands.

The robot navigates toward a fixed goal while avoiding collisions, using LiDAR and odometry.

## Main Repository Layout

```text
robotics_project/
├── README.md
├── launch/
│   └── turtlebot3_world.launch.py
├── worlds/
│   └── turtlebot3_world.world
├── src/
│   └── turtlebot_rl/
└── rl_ros2_nav/
      ├── rl_gym_env/
      │   ├── constants.py
      │   └── env.py
      ├── training/
      │   ├── callbacks/save_best_callback.py
      │   ├── config/ppo_config.yaml
      │   ├── evaluate.py
      │   └── train.py
      ├── ros2_ws/
      │   └── src/rl_inference_node/policy_node.py
      ├── models/
      └── logs/
```

## Environment Details 

### Observation (state)

State size is 38:

- `0:35` -> 36 downsampled LiDAR values, normalized to `[0, 1]`
- `36` -> normalized distance to goal (`dist / 5.0`, clipped to `1.0`)
- `37` -> normalized heading error (`heading_error / pi`, in `[-1, 1]`)

Constants:

- `LIDAR_SAMPLES = 36`
- `MAX_LIDAR_RANGE = 3.5`
- Goal position: `(GOAL_X, GOAL_Y) = (1.5, 1.0)`

### Action space

Continuous 2D action:

- Linear velocity: `[0.0, 0.22]` m/s (forward only)
- Angular velocity: `[-2.0, 2.0]` rad/s

### Reward function

Per step logic:

- Collision if `min_lidar < 0.12` -> reward `-200`, episode ends
- Goal reached if `distance < 0.25` -> reward `+300`, episode ends
- Otherwise:
   - progress reward: `150 * (previous_distance - current_distance)`
   - step penalty: `-0.1`
   - total reward: `progress - 0.1`

In formula form (non-terminal step):

$$
r_t = 150 \cdot (d_{t-1} - d_t) - 0.1
$$

### Episode reset and random initial position

By default, `NavEnv` is created with `random_spawn=True` in `training/train.py`.

At reset:

- The robot is first stopped.
- A random spawn is sampled in a square area around the origin:
   - `x, y ~ U(-SPAWN_AREA_SIZE/2, SPAWN_AREA_SIZE/2)`
   - with `SPAWN_AREA_SIZE = 3.0`
   - random yaw in `[0, 2*pi]`
- Up to 20 spawn attempts are made, checking safety via LiDAR.
- Final fallback on last attempt is `(0.0, -0.5)`.

### Termination and truncation

- `done=True` on goal or collision
- `truncated=True` when step count reaches `MAX_STEPS = 500`
- `info["termination"]` is set to `goal_reached`, `collision`, or `timeout`

## How To Start Training

### 1) Start Gazebo 

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch /home/luca/robotics_project/launch/turtlebot3_world.launch.py
```

### 2) Run training 

```bash
python3 training/train.py
```

Default parameters are saved in rl_ros2_nav/training/config/ppo_config.yaml

## Inference

With Gazebo already running:

```bash
python3 rl_ros2_nav/ros2_ws/src/rl_inference_node/policy_node.py
```

Default files used by the inference node:

- `rl_ros2_nav/models/best/best_model.zip`
- `rl_ros2_nav/models/best/best_model_vecnormalize.pkl`

