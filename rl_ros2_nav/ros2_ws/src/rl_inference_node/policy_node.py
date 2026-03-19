import os
import sys

sys.path.append(os.path.expanduser("~/robotics_project/rl_ros2_nav"))
from rl_gym_env.constants import (
    LIDAR_SAMPLES, MAX_LIDAR_RANGE,
    GOAL_X, GOAL_Y, COLLISION_DIST
)

import pickle
import zipfile
import io
import numpy as np
import math
import torch
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String



# ── Model paths ──────────────────────────────────────────────
MODEL_PATH    = "/home/luca/robotics_project/rl_ros2_nav/models/best/best_model.zip"
VECNORM_PATH  = "/home/luca/robotics_project/rl_ros2_nav/models/best/best_model_vecnormalize.pkl"

class InferenceNode(Node):
    """
    Pure ROS2 inference node.
    Loads the trained policy as a raw PyTorch network —
    no SB3 dependency at runtime.
    """

    def __init__(self):
        super().__init__("inference_node")

        # ── Load normalisation statistics ────────────────────
        # VecNormalize saves obs_rms (RunningMeanStd) in the .pkl file.
        # We only need mean and variance to normalise observations.
        with open(VECNORM_PATH, "rb") as f:
            vecnorm_data = pickle.load(f)
        self._obs_mean = vecnorm_data.obs_rms.mean.astype(np.float32)
        self._obs_var  = vecnorm_data.obs_rms.var.astype(np.float32)
        self._clip_obs = 10.0
        self.get_logger().info("VecNormalize stats loaded")

        # ── Load policy network from SB3 zip ─────────────────
        # SB3 saves the PyTorch policy inside a zip as 'policy.pth'.
        # We extract and load it directly without instantiating SB3.
        with zipfile.ZipFile(MODEL_PATH, "r") as zf:
            with zf.open("policy.pth") as f:
                weights = torch.load(
                    io.BytesIO(f.read()),
                    map_location="cpu"
                )
        self._policy = torch.nn.Sequential(
            torch.nn.Linear(38, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 2),
        )
        self._policy[0].weight.data = weights["mlp_extractor.policy_net.0.weight"]
        self._policy[0].bias.data   = weights["mlp_extractor.policy_net.0.bias"]
        self._policy[2].weight.data = weights["mlp_extractor.policy_net.2.weight"]
        self._policy[2].bias.data   = weights["mlp_extractor.policy_net.2.bias"]
        self._policy[4].weight.data = weights["action_net.weight"]
        self._policy[4].bias.data   = weights["action_net.bias"]

        self._policy.eval()

        # ── Sensor state ─────────────────────────────────────
        self._scan = None
        self._odom = None

        # ── ROS2 pub/sub ─────────────────────────────────────
        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.create_subscription(Odometry,  "/odom", self._odom_cb, 10)
        self._cmd_pub    = self.create_publisher(Twist,  "/cmd_vel",   10)
        self._status_pub = self.create_publisher(String, "/rl_status", 10)

        # ── Control loop at 10 Hz ────────────────────────────
        self.create_timer(0.1, self._control_loop)
        self.get_logger().info(
            f"Inference node running — goal: ({GOAL_X}, {GOAL_Y})")

    # ── Callbacks ─────────────────────────────────────────────

    def _scan_cb(self, msg):
        self._scan = msg

    def _odom_cb(self, msg):
        self._odom = msg

    # ── Normalisation ─────────────────────────────────────────

    def _normalize_obs(self, obs):
        """Apply saved VecNormalize statistics to raw observation."""
        obs_norm = (obs - self._obs_mean) / np.sqrt(self._obs_var + 1e-8)
        return np.clip(obs_norm, -self._clip_obs, self._clip_obs)

    # ── Observation builder ───────────────────────────────────

    def _get_obs(self):
        # --- LiDAR ---
        if self._scan is None:
            lidar = np.ones(LIDAR_SAMPLES, dtype=np.float32)
        else:
            raw   = np.array(self._scan.ranges, dtype=np.float32)
            raw   = np.nan_to_num(raw, nan=MAX_LIDAR_RANGE,
                                  posinf=MAX_LIDAR_RANGE)
            raw   = np.clip(raw, 0.0, MAX_LIDAR_RANGE)
            step  = len(raw) // LIDAR_SAMPLES
            lidar = raw[::step][:LIDAR_SAMPLES]
            lidar = lidar / MAX_LIDAR_RANGE

        # --- Goal relative position ---
        if self._odom is None:
            dist, angle_norm = 1.0, 0.0
        else:
            rx = self._odom.pose.pose.position.x
            ry = self._odom.pose.pose.position.y
            dist = math.hypot(GOAL_X - rx, GOAL_Y - ry)

            q   = self._odom.pose.pose.orientation
            yaw = math.atan2(
                2*(q.w*q.z + q.x*q.y),
                1 - 2*(q.y**2 + q.z**2)
            )
            angle_to_goal = math.atan2(GOAL_Y - ry, GOAL_X - rx)
            heading_err   = angle_to_goal - yaw
            heading_err   = math.atan2(
                math.sin(heading_err), math.cos(heading_err))
            angle_norm    = heading_err / math.pi
            dist          = min(dist / 5.0, 1.0)

        return np.concatenate([
            lidar,
            [np.float32(dist), np.float32(angle_norm)]
        ]).astype(np.float32)

    # ── Control loop ──────────────────────────────────────────

    def _control_loop(self):
        if self._scan is None or self._odom is None:
            self.get_logger().info(
                "Waiting for sensor data...", once=True)
            return

        # 1. Build raw observation from sensors
        obs = self._get_obs()

        # 2. Normalise using saved VecNormalize statistics
        obs_norm = self._normalize_obs(obs)

        # 3. Run PyTorch policy — no SB3 involved
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0)
            action     = self._policy(obs_tensor)

        # Clamp to action space bounds defined in env.py
        lin = float(action[0][0].clamp(0.0,  0.22))
        ang = float(action[0][1].clamp(-2.0,  2.0))

        # 4. Check termination conditions
        dist      = obs[LIDAR_SAMPLES] * 5.0
        min_lidar = obs[:LIDAR_SAMPLES].min() * MAX_LIDAR_RANGE

        if dist < 0.25:
            self.get_logger().info("Goal reached!")
            self._publish_velocity(0.0, 0.0)
            self._publish_status("GOAL_REACHED")
            return

        if min_lidar < COLLISION_DIST:
            self.get_logger().warn("Collision detected — stopping!")
            self._publish_velocity(0.0, 0.0)
            self._publish_status("COLLISION")
            return

        # 5. Publish velocity command
        self._publish_velocity(lin, ang)
        self._publish_status(
            f"RUNNING | dist: {dist:.2f}m | "
            f"lin: {lin:.2f} ang: {ang:.2f}")

    def _publish_velocity(self, linear, angular):
        msg           = Twist()
        msg.linear.x  = linear
        msg.angular.z = angular
        self._cmd_pub.publish(msg)

    def _publish_status(self, text):
        msg      = String()
        msg.data = text
        self._status_pub.publish(msg)

    def destroy_node(self):
        self._publish_velocity(0.0, 0.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down — stopping robot")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()