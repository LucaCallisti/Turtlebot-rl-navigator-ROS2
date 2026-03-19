import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

# Add ROS2 workspace to Python path
sys.path.append(os.path.expanduser(
    "~/robotics_project/rl_ros2_nav/ros2_ws/src/rl_gym_env"))
from rl_gym_env.env import NavEnv

# ── Config ───────────────────────────────────────────────────
MODELS_DIR   = os.path.expanduser("~/robotics_project/rl_ros2_nav/models")
N_EPISODES   = 20       # number of test episodes
DETERMINISTIC = True    # no exploration during evaluation


def make_env():
    return NavEnv()


def evaluate(model_path=None, vecnorm_path=None):

    # ── 1. Load environment ──────────────────────────────────
    env = make_vec_env(make_env, n_envs=1)
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False      # freeze normalisation statistics
    env.norm_reward = False   # raw reward for readability

    # ── 2. Load model ────────────────────────────────────────
    model = PPO.load(model_path, env=env)
    print(f"\nModel loaded: {model_path}")
    print(f"VecNormalize: {vecnorm_path}")
    print(f"Running {N_EPISODES} evaluation episodes...\n")

    # ── 3. Run episodes ──────────────────────────────────────
    rewards      = []
    lengths      = []
    successes    = []
    collisions   = []

    for ep in range(N_EPISODES):
        obs          = env.reset()
        done         = False
        ep_reward    = 0.0
        ep_length    = 0
        goal_reached = False
        collided     = False

        while not done:
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_length += 1

        episode_info = info[0]  
        if episode_info.get("is_success", False):
            goal_reached = True
        elif episode_info.get("termination") == "collision":
            collided = True

        rewards.append(ep_reward)
        lengths.append(ep_length)
        successes.append(goal_reached)
        collisions.append(collided)

        status = "GOAL" if goal_reached else "COLLISION" if collided else "TIMEOUT"
        print(f"Episode {ep+1:2d} | reward: {ep_reward:8.1f} | "
              f"length: {ep_length:4d} | {status}")

    # ── 4. Summary ───────────────────────────────────────────
    print("\n" + "="*50)
    print(f"Results over {N_EPISODES} episodes:")
    print(f"  Mean reward    : {np.mean(rewards):8.1f} ± {np.std(rewards):.1f}")
    print(f"  Mean length    : {np.mean(lengths):8.1f} steps")
    print(f"  Success rate   : {np.mean(successes)*100:5.1f}%")
    print(f"  Collision rate : {np.mean(collisions)*100:5.1f}%")
    print(f"  Timeout rate   : {(1-np.mean(successes)-np.mean(collisions))*100:5.1f}%")
    print("="*50)

    env.close()


if __name__ == "__main__":
    # By default loads the best model saved by EvalCallback
    model_path  = os.path.join(MODELS_DIR, "ppo_nav_500000_steps.zip")
    vecnorm_path = os.path.join(MODELS_DIR, "ppo_nav_vecnormalize_500000_steps.pkl")

    # Fallback to final model if best does not exist yet
    if not os.path.exists(model_path):
        print("Best model not found, loading final model...")
        model_path   = os.path.join(MODELS_DIR, "ppo_nav_final.zip")
        vecnorm_path = os.path.join(MODELS_DIR, "vec_normalize_final.pkl")

    evaluate(model_path, vecnorm_path)