# TurtleBot3 RL Navigator - ROS2 Integration Project

A **ROS2-based autonomous navigation system** for TurtleBot3 that combines reinforcement learning with real-time sensor integration. This project demonstrates end-to-end ROS2 development: from environment simulation to policy training and deployment as a production-ready inference node.

## 🎯 Project Overview

This project showcases practical **ROS2 skills** through building a complete autonomous navigation pipeline:

- **ROS2 Node Development**: Custom nodes for sensor data handling and control
- **Message Publishing/Subscription**: LiDAR (LaserScan), Odometry, and velocity command integration
- **Service Clients**: Gazebo world reset for environment management
- **PyTorch Model Inference**: Real-time policy execution without SB3 overhead
- **Sensor Fusion**: Combining LiDAR and odometry for navigation decisions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         TurtleBot3 (Gazebo)                │
│  ┌──────────────────────────────────────┐  │
│  │   /scan (LaserScan)                 │  │
│  │   /odom (Odometry)                  │  │
│  │   /cmd_vel (Twist)                  │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
        ▲                ▲              │
        │                │              │
   Subscribes        Subscribes    Publishes
        │                │              │
        └────────────────┼──────────────┘
                         │
              ┌──────────▼──────────┐
              │  Inference Node      │
              │  (policy_node.py)    │
              │                      │
              │ • Normalize obs      │
              │ • Load PyTorch model │
              │ • Execute policy     │
              │ • Publish commands   │
              └──────────────────────┘
```

## 📊 Training Pipeline

```
NavEnv (Gymnasium) ──► VecNormalize ──► PPO (SB3) ──► Best Model
  │                         │              │              │
  ├─ 36 LiDAR rays    ├─ Obs normalization │              │
  ├─ Goal distance    │  (mean/std)     ├─ 500k steps   │
  ├─ Goal angle       │                 │               │
  └─ ROS2 topics      └─ Reward norm    └─ Checkpoint   └──► Inference
```

## ⚙️ Key Components

### 1. **Navigation Environment** (`rl_gym_env/env.py`)
- Gymnasium-compliant environment wrapping TurtleBot3 in Gazebo
- **Observation space**: 36 downsampled LiDAR rays + distance/angle to goal
- **Action space**: Linear velocity [0.0, 0.22] m/s, Angular velocity [-2.0, 2.0] rad/s
- ROS2 integration: subscribes to `/scan` and `/odom`, publishes to `/cmd_vel`

### 2. **Training Script** (`training/train.py`)
- PPO algorithm with 500k training steps
- Vectorized environment and VecNormalize for observation normalization
- Checkpoint and best-model callbacks
- Configuration from YAML

### 3. **Inference Node** (`ros2_ws/src/rl_inference_node/policy_node.py`)
- Pure ROS2 node (no SB3 dependency at runtime)
- Loads trained PyTorch policy directly
- Applies saved VecNormalize statistics
- 10 Hz control loop with collision detection
- Status publishing to `/rl_status`

## 🚀 Quick Start

### Prerequisites

```bash
# ROS2 Humble
source /opt/ros/humble/setup.bash

# Python 3.10+
python3 --version

# Key dependencies (install in virtual environment)
pip install gymnasium stable-baselines3 torch numpy pyyaml
```

### Setup

1. **Clone and build the ROS2 workspace**:
   ```bash
   cd ~/robotics_project/rl_ros2_nav
   source /opt/ros/humble/setup.bash
   colcon build
   source install/setup.bash
   ```

2. **Setup TurtleBot3 Gazebo** (in separate terminal):
   ```bash
   export TURTLEBOT3_MODEL=burger
   ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
   ```

### Training

```bash
cd ~/robotics_project/rl_ros2_nav
python3 training/train.py
```

Logs and checkpoints are saved to `models/` and `logs/` directories. Best model is stored in `models/best/`.

### Inference

1. **Launch Gazebo with TurtleBot3** (see Setup step 2)

2. **Run the inference node**:
   ```bash
   cd ~/robotics_project/rl_ros2_nav/ros2_ws
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   
   ros2 run rl_inference_node policy_node
   ```

3. **Monitor behavior**:
   ```bash
   # Watch status messages
   ros2 topic echo /rl_status
   
   # Visualize in RViz
   rviz2
   ```

## 📋 Configuration

Edit `training/config/ppo_config.yaml` to customize training:

```yaml
policy: "MlpPolicy"              # Network architecture
learning_rate: 3e-4              # Adam optimizer LR
n_steps: 2048                     # Rollout buffer size
batch_size: 64                    # Mini-batch size
n_epochs: 10                      # PPO epochs per update
gamma: 0.99                       # Discount factor
gae_lambda: 0.95                  # GAE parameter
checkpoint_freq: 50000            # Save checkpoint every N steps
```

## 🧠 Environment Details

**Goal**: Navigate TurtleBot3 to position (1.5, 1.0) while avoiding obstacles.

**Observation** (38-dim vector):
- **[0:36]** - Normalized 360° LiDAR scan downsampled to 36 rays
- **[36]** - Normalized distance to goal
- **[37]** - Normalized heading error (-1 to 1)

**Reward**:
- ✅ **+100** - Reached goal (distance < 0.25 m)
- ❌ **-100** - Collision (LiDAR < 0.12 m)
- ⏱️ **-1** - Per step (encourages quick navigation)
- 📍 -1/(max_steps) for each step of distance decrease

**Episode Termination**:
- Goal reached
- Collision detected
- Max steps (500) exceeded

## 📈 Results

After training:
- **Success rate**: ~85-90% on test episodes
- **Average episode length**: ~150-200 steps
- **Training time**: ~2-3 hours (1 GPU)
- **Model size**: ~50KB (PyTorch .pth)

## 🔧 ROS2 Skills Demonstrated

✅ **Node Creation & Lifecycle**: Custom ROS2 nodes with proper initialization  
✅ **Pub/Sub Communication**: Multi-topic subscriptions and message publishing  
✅ **Message Types**: LaserScan, Odometry, Twist, String, Marker  
✅ **Service Clients**: Gazebo world reset service  
✅ **Timers & Callbacks**: Asynchronous control loops at fixed rate  
✅ **Colcon Build System**: Workspace organization and building  
✅ **Logging**: ROS2 logger for debugging and monitoring  

## 📦 Project Structure

```
robotics_project/
├── src/turtlebot_rl/              # ROS2 package root
│   ├── setup.py
│   ├── package.xml
│   └── test/
├── rl_ros2_nav/
│   ├── rl_gym_env/                # Gymnasium environment
│   │   ├── env.py
│   │   └── constants.py
│   ├── training/                  # Training pipeline
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── config/ppo_config.yaml
│   │   └── callbacks/
│   ├── models/                    # Trained policies
│   │   └── best/
│   ├── logs/                      # TensorBoard logs
│   └── ros2_ws/
│       └── src/
│           └── rl_inference_node/
│               └── policy_node.py
└── README.md
```

