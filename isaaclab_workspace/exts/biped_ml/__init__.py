import gymnasium as gym
from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv

from .env_cfg import BipedEnvCfg

# Register the custom Cassie environment so Stable Baselines 3 or rl_games can spawn it natively
gym.register(
    id="Isaac-Cassie-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv", # The entry point is the core Isaac wrapper
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:BipedEnvCfg"},
)
