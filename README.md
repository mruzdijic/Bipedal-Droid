# Bipedal ML Foundation (Isaac Lab Version)

This repository contains the foundational code for training a bipedal robot using Deep Reinforcement Learning (DRL) via **Stable Baselines3** within the **NVIDIA Isaac Lab** ecosystem.

## Backbone

This project explicitly adheres to the following pipeline:
- **Environment Framework:** NVIDIA Isaac Lab (built on top of Isaac Sim).
- **Robot Asset:** The high-fidelity **Cassie** biped model directly from the regular Isaac Sim Nucleus server.
- **Commands:** Cartesian Coordinates `(vx, vy, yaw_rate)`.
- **Sensors Simulated:**
  - IMU (Base linear acceleration)
  - Gyroscope (Base angular velocity)
  - Joint Encoders (Joint positions & velocities)
- **RL Framework:** Stable Baselines 3 (PPO algorithm).

## Project Structure

```
Bipedal-Droid/
├── README.md
├── install_instructions.md                   (Crucial steps for the Isaac Lab script!)
└── isaaclab_workspace/                       (The Isaac Lab extension scaffolding)
    └── exts/
        └── biped_ml/                         (Our custom RL environment)
            ├── __init__.py                   (Registers env with OpenAI Gym)
            ├── env_cfg.py                    (ManagerBasedRLEnv Config for Cassie)
            └── train_sb3.py                  (SB3 Training Execution Script)
```

## Quick Start
Please refer to `install_instructions.md` to configure your primary laptop. You must clone the official Isaac Lab repository and link it.