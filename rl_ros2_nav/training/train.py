import os
import sys

sys.path.append(os.path.expanduser(
    "~/robotics_project/rl_ros2_nav"))
sys.path.append(os.path.expanduser(
    "~/robotics_project/rl_ros2_nav/ros2_ws/src/rl_gym_env"))

import yaml
import rclpy
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from training.callbacks.save_best_callback import SaveBestWithNormalizeCallback
from stable_baselines3.common.vec_env import VecNormalize

# Add ROS2 workspace to Python path
sys.path.append(os.path.expanduser(
    "~/robotics_project/rl_ros2_nav/ros2_ws/src/rl_gym_env"))
from rl_gym_env.env import NavEnv

# ── Config ───────────────────────────────────────────────────
CONFIG_PATH = os.path.expanduser(
    "~/robotics_project/rl_ros2_nav/training/config/ppo_config.yaml")
MODELS_DIR  = os.path.expanduser("~/robotics_project/rl_ros2_nav/models")
LOGS_DIR    = os.path.expanduser("~/robotics_project/rl_ros2_nav/logs")

MODEL_TO_LOAD = "/home/luca/robotics_project/rl_ros2_nav/models/ppo_nav_30000_steps.zip"
VECNORM_TO_LOAD = "/home/luca/robotics_project/rl_ros2_nav/models/ppo_nav_vecnormalize_30000_steps.pkl"
MODEL_TO_LOAD = None
VECNORM_TO_LOAD = None

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env():
    """Factory function required by make_vec_env."""
    return NavEnv()


def train():
    cfg = load_config(CONFIG_PATH)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    # ── Initialize W&B ──────────────────────────────────────
    wandb.init(
        project="turtlebot-rl-navigation",
        name=None,  # Auto-generated name with timestamp
        config=cfg,
        sync_tensorboard=True,
    )

    resuming = os.path.exists(MODEL_TO_LOAD) and os.path.exists(VECNORM_TO_LOAD) if MODEL_TO_LOAD and VECNORM_TO_LOAD else False
    

    # ── 1. Create vectorised environments ───────────────────
    raw_env      = make_vec_env(make_env, n_envs=1)
    raw_eval_env = make_vec_env(make_env, n_envs=1)

    if resuming:
        print(f"Resuming from checkpoint: {MODEL_TO_LOAD}")
        env      = VecNormalize.load(VECNORM_TO_LOAD, raw_env)
        eval_env = VecNormalize.load(VECNORM_TO_LOAD, raw_eval_env)
    else:
        print("No checkpoint found, starting from scratch...")
        env      = VecNormalize(raw_env,      norm_obs=True, norm_reward=True,  clip_obs=10.0)
        eval_env = VecNormalize(raw_eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    env.training    = True
    env.norm_reward = True
    eval_env.training    = False
    eval_env.norm_reward = False

    # ── 2. Callbacks ─────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=cfg["checkpoint_freq"],
        save_path=MODELS_DIR,
        name_prefix="ppo_nav",
        save_vecnormalize=True,   
    )


    eval_cb = SaveBestWithNormalizeCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, "best"),
        log_path=LOGS_DIR,
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
    )

    # ── W&B Callback for live logging ────────────────────────
    run = wandb.init(
        project="turtlebot-rl-navigation",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    wandb_cb = WandbCallback(
        gradient_save_freq=0,  # Disable gradient logging
        model_save_path=None,  # Don't save model to W&B (we save locally)
        verbose=0,
    )

    # ── 3. Instantiate PPO model ─────────────────────────────
    if resuming:
        model = PPO.load(MODEL_TO_LOAD, env=env, tensorboard_log=LOGS_DIR, device="auto")
    else:
        model = PPO(
            policy="MlpPolicy",           
            env=env,
            learning_rate=cfg["learning_rate"],
            n_steps=cfg["n_steps"],       
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],     
            gamma=cfg["gamma"],           
            gae_lambda=cfg["gae_lambda"], 
            clip_range=cfg["clip_range"], 
            ent_coef=cfg["ent_coef"],     
            verbose=1,                    
            tensorboard_log=LOGS_DIR,
            device="auto",                
        )

    print("\n--- Training started ---")
    print(f"Total timesteps : {cfg['total_timesteps']:,}")
    print(f"Checkpoint every: {cfg['checkpoint_freq']:,} steps")
    print(f"Log dir         : {LOGS_DIR}")
    print(f"Models dir      : {MODELS_DIR}\n")

    # ── 4. Run training ──────────────────────────────────────
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=[checkpoint_cb, eval_cb, wandb_cb],
        progress_bar=True,
    )

    # ── Finish W&B run ──────────────────────────────────────
    run.finish()

    # ── 5. Save final model ──────────────────────────────────
    final_path = os.path.join(MODELS_DIR, "ppo_nav_final")
    model.save(final_path)
    env.save(os.path.join(MODELS_DIR, "vec_normalize_final.pkl"))
    print(f"\nFinal model saved to: {final_path}.zip")


if __name__ == "__main__":
    train()