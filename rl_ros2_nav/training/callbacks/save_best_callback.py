import os
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize


class SaveBestWithNormalizeCallback(EvalCallback):
    """
    Extends EvalCallback to also save VecNormalize statistics
    whenever a new best model is found.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        # Store reward before parent runs
        old_best = self.best_mean_reward

        # Run standard EvalCallback logic
        result = super()._on_step()

        # If best reward improved → also save VecNormalize
        if self.best_mean_reward > old_best:
            env = self.training_env
            if isinstance(env, VecNormalize):
                path = os.path.join(
                    self.best_model_save_path,
                    "best_model_vecnormalize.pkl"
                )
                env.save(path)
                print(f"\nNew best model — VecNormalize saved to {path}")

        return result