import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
import math
import time

from rl_gym_env.constants import (
    LIDAR_SAMPLES, MAX_LIDAR_RANGE,
    GOAL_X, GOAL_Y, COLLISION_DIST, MAX_STEPS
)

class NavEnv(gym.Env):
    """
    Gymnasium environment that wraps a TurtleBot3 in Gazebo via ROS2.

    Observation (38,):
        [0:36]  – 36 downsampled LiDAR ranges (normalised 0→1)
        [36]    – distance to goal (normalised)
        [37]    – angle to goal in radians (normalised -1→1)

    Action (2,)  continuous, Box:
        [0]  linear  velocity  in [-0.22, 0.22] m/s
        [1]  angular velocity  in [-2.0,  2.0]  rad/s
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # ── Spaces ───────────────────────────────────────────
        obs_size = LIDAR_SAMPLES + 2   # 36 lidar + dist + angle
        low  = np.concatenate([
            np.zeros(LIDAR_SAMPLES, dtype=np.float32),   # lidar:      [0.0, 1.0]
            np.array([0.0, -1.0], dtype=np.float32)      # dist, angle: dist ≥ 0, angle in [-1,1]
        ])
        high = np.ones(obs_size, dtype=np.float32)       # all upper bounds are 1.0

        self.observation_space = gym.spaces.Box(
            low=low, high=high,
            shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -2.0], dtype=np.float32),
            high=np.array([0.22,  2.0], dtype=np.float32)
        )
        # Note: linear velocity is non-negative (forward only) to simplify learning and to teach robot to rotate and then move forward.

        # ── ROS2 init ────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()    
        self.node = Node("nav_env_node")    # creating a ROS2 node for this environment

        self._scan    = None   # latest LaserScan message
        self._odom    = None   # latest Odometry message

        # register ROS2 listeners (Message type = LaserScan, topic = /scan, callback = self._scan_cb, queue size = 10)
        self.node.create_subscription(
            LaserScan, "/scan", self._scan_cb, 10)
        self.node.create_subscription(
            Odometry, "/odom", self._odom_cb, 10)
        # register ROS2 publisher that can send messages for cmd_vel (Message type = Twist (is for velocity command), topic = /cmd_vel, queue size = 10)
        self._cmd_pub = self.node.create_publisher(
            Twist, "/cmd_vel", 10)
        # client for reset_world service (used to reset Gazebo world, which teleports robot back to spawn position)
        self._reset_client = self.node.create_client(Empty, "/reset_world")
        # publisher for RViz marker to visualise the goal position
        self._marker_pub = self.node.create_publisher(
            Marker, "/goal_marker", 10)

        # ── Episode state ────────────────────────────────────
        self._step_count = 0
        self._prev_dist  = None

    # ── ROS2 callbacks ───────────────────────────────────────

    def _scan_cb(self, msg):
        self._scan = msg

    def _odom_cb(self, msg):
        self._odom = msg

    def _spin_once(self):
        """Process pending ROS2 callbacks (non-blocking)."""
        rclpy.spin_once(self.node, timeout_sec=0.1)

    # ── Observation builder ──────────────────────────────────

    def _get_obs(self):
        self._spin_once()

        # --- LiDAR ---
        if self._scan is None:
            lidar = np.ones(LIDAR_SAMPLES, dtype=np.float32)
        else:
            raw   = np.array(self._scan.ranges, dtype=np.float32)
            raw   = np.nan_to_num(raw, nan=MAX_LIDAR_RANGE, posinf=MAX_LIDAR_RANGE)
            raw   = np.clip(raw, 0.0, MAX_LIDAR_RANGE)
            # downsample: take every N-th ray
            step  = len(raw) // LIDAR_SAMPLES
            lidar = raw[::step][:LIDAR_SAMPLES]
            lidar = lidar / MAX_LIDAR_RANGE   # normalise 0→1

        # --- Goal relative position ---
        if self._odom is None:
            dist, angle_norm = 1.0, 0.0
        else:
            rx = self._odom.pose.pose.position.x
            ry = self._odom.pose.pose.position.y
            dist  = math.hypot(GOAL_X - rx, GOAL_Y - ry)

            # yaw from quaternion
            q  = self._odom.pose.pose.orientation
            yaw = math.atan2(
                2*(q.w*q.z + q.x*q.y),
                1 - 2*(q.y**2 + q.z**2)
            )
            angle_to_goal = math.atan2(GOAL_Y - ry, GOAL_X - rx)
            heading_err   = angle_to_goal - yaw
            # wrap to [-π, π]
            heading_err   = math.atan2(
                math.sin(heading_err), math.cos(heading_err))
            angle_norm    = heading_err / math.pi   # normalise -1→1
            dist          = min(dist / 5.0, 1.0)    # normalise, cap at 5 m

        obs = np.concatenate([
            lidar,
            [np.float32(dist), np.float32(angle_norm)]
        ]).astype(np.float32)
        return obs

    # ── Reward ───────────────────────────────────────────────

    def _compute_reward(self, obs):
        min_lidar = obs[:LIDAR_SAMPLES].min() * MAX_LIDAR_RANGE
        dist      = obs[LIDAR_SAMPLES] * 5.0   # un-normalise
        angle_rad = obs[LIDAR_SAMPLES + 1] * math.pi

        # Collision
        if min_lidar < COLLISION_DIST:
            return -200.0, True   # (reward, done)

        # Goal reached
        if dist < 0.25:
            return +300.0, True

        # Progress reward: positive when getting closer
        progress = 0.0
        if self._prev_dist is not None:
            progress = (self._prev_dist - dist) * 150.0
        self._prev_dist = dist
        heading_bonus = (math.cos(angle_rad) + 1.0) / 2.0 * 0.5

        # Small penalty per step (encourages speed)
        step_penalty = -0.1

        return progress + step_penalty, False

    # ── Gym API ──────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Stop the robot
        self._publish_velocity(0.0, 0.0)
        time.sleep(0.1)  # small delay to ensure robot has stopped

        # Reset Gazebo world — teleports robot back to spawn position
        if self._reset_client.wait_for_service(timeout_sec=2.0):
            req = Empty.Request()
            self._reset_client.call_async(req)
            time.sleep(0.5)   # wait for world to reset
        else:
            self.node.get_logger().warn("reset_world service not available")

        # Reset counters
        self._step_count = 0
        self._prev_dist  = None

        # Wait for fresh sensor data
        for _ in range(20):
            self._spin_once()
            if self._scan is not None and self._odom is not None:
                break
            time.sleep(0.05)

        obs = self._get_obs()
        self._prev_dist = obs[LIDAR_SAMPLES] * 5.0
        self._publish_goal_marker()
        return obs, {}

    def step(self, action):
        # Publish action to /cmd_vel
        lin, ang = float(action[0]), float(action[1])
        self._publish_velocity(lin, ang)

        # Wait one control cycle (5 Hz)
        time.sleep(0.05)
        self._spin_once()

        obs              = self._get_obs()
        reward, done     = self._compute_reward(obs)
        self._step_count += 1
        truncated        = self._step_count >= MAX_STEPS

        info = {}
        if done:
            min_lidar = obs[:LIDAR_SAMPLES].min() * MAX_LIDAR_RANGE
            dist      = obs[LIDAR_SAMPLES] * 5.0
            if dist < 0.25:
                info["termination"] = "goal_reached"
                info["is_success"]  = True    # EvalCallback reads this automatically
            elif min_lidar < COLLISION_DIST:
                info["termination"] = "collision"
                info["is_success"]  = False
        elif truncated:
            info["termination"]  = "timeout"
            info["is_success"]   = False

        return obs, reward, done, truncated, info

    def _publish_velocity(self, linear, angular):
        msg             = Twist()
        msg.linear.x    = float(linear)
        msg.angular.z   = float(angular)
        self._cmd_pub.publish(msg)

    def close(self):
        self._publish_velocity(0.0, 0.0)
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def _publish_goal_marker(self):
        m = Marker()
        m.header.frame_id = "odom"
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(GOAL_X)
        m.pose.position.y = float(GOAL_Y)
        m.pose.position.z = 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = 0.3
        m.scale.y = 0.3
        m.scale.z = 0.2
        m.color.r = 1.0
        m.color.g = 0.2
        m.color.b = 0.0
        m.color.a = 0.8
        self._marker_pub.publish(m)