import argparse
import os

from omni.isaac.lab.app import AppLauncher

# Must launch the Isaac Sim application BEFORE any deep omniverse/isaac imports
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of simulated environments to run in parallel.")
# Append AppLauncher cli args (headless, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Gymnasium and Isaac Lab Wrappers AFTER AppLauncher
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import omni.isaac.lab_tasks  # Import official tasks to ensure registry is ready
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper

# Import our custom environment registry
import biped_ml

def main():
    """Main training loop."""
    print("[INFO] Setting up Stable Baselines 3 in Isaac Lab...")
    
    # Create the environment. Because physics are batched on the GPU, we spawn `num_envs` robots simultaneously!
    env = gym.make("Isaac-Cassie-v0", env_cfg={"scene": {"num_envs": args_cli.num_envs}})
    
    # Wrap the raw Isaac Lab environment into a compatible SB3 vectorized environment
    # This automatically syncs Isaac's batched Pytorch Tensors with SB3's expected types!
    vec_env = Sb3VecEnvWrapper(env)
    
    # Standard PPO initialization
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=24,             # Steps per environment before a mini-batch update
        batch_size=1024,        # Batch size for optimization
        n_epochs=5,
        learning_rate=3e-4,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="logs/sb3/",
        device="cuda",          # Ensure PPO uses the GPU
    )

    # Save model checkpoints periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path='logs/sb3/checkpoints/',
        name_prefix='cassie_biped'
    )

    print("[INFO] Starting PPO Training...")
    # Typically robot learning takes millions of steps.
    model.learn(total_timesteps=5_000_000, callback=checkpoint_callback)

    print("[INFO] Saving final policy...")
    model.save("cassie_biped_final.zip")

    print("[INFO] Cleaning up environments...")
    vec_env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
