import os
# Must import the custom environment
from biped_env import BipedalEnv

# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    """
    Main Training Loop for the BD-style Bipedal Robot using PPO.
    """
    print("[INFO] Starting Isaac Sim Application...")
    # Initialize our custom environment wrapping Isaac Sim
    # headless=False means you will see the Omniverse viewer UI
    env = BipedalEnv(headless=False)
    
    # Create a checkpoint callback to save the model every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path='./logs/',
        name_prefix='bd_biped_model'
    )
    
    print("[INFO] Initializing PPO Agent...")
    # Initialize Stable Baselines PPO Policy
    # We use MlpPolicy since our observations are simply an array of floats
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/tensorboard/")
    
    print("[INFO] Beginning Training (100k timesteps)...")
    # Train the agent (100k is a test amout; real training takes millions of steps)
    model.learn(total_timesteps=100000, callback=checkpoint_callback)
    
    print("[INFO] Training Complete! Saving Final Model...")
    model.save("final_bd_biped_model")
    
    print("[INFO] Viewing learned behavior for 500 steps...")
    obs = env.reset()
    for i in range(500):
        # Predict the action given the state
        action, _states = model.predict(obs, deterministic=True)
        # Act in the environment
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
            
    print("[INFO] Closing environment.")
    env.close()

if __name__ == "__main__":
    main()
