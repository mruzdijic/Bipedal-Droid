# Bipedal ML Foundation (BD-Droid Style)

This repository contains the foundational code for training and simulating a bipedal robot using Deep Reinforcement Learning (DRL) via Stable Baselines3 within **NVIDIA Omniverse Isaac Sim**, alongside **ROS2 Humble** integration for hybrid learning/teleop capabilities.

## Robot Configuration

The biped mimics a Star Wars BD droid with digitigrade legs.

- **DOF:** 12 total (6 DOF per leg).
  - Hip: 3 DOF (Roll, Pitch, Yaw)
  - Knee: 1 DOF (Pitch)
  - Ankle: 2 DOF (Pitch, Roll)
- **Geometry Dimensions:**
  - Torso: Main chassis
  - Upper Leg (Thigh): 100mm
  - Lower Leg (Shin): 100mm
  - Foot: 50-60mm

## Project Structure

```
Bipedal-Droid/
├── README.md
├── install_instructions.md          (Commands for setup on a secondary machine)
├── urdf/
│   └── biped.urdf                   (Definition of our BD-style robot)
└── isaac_scripts/                   (Future scripts for physics, ROS2 bridges, & RL)
    ├── biped_env.py
    ├── train.py
    └── biped_ros2_bridge.py
```

## Quick Start
Please refer to `install_instructions.md` to configure your PC to run Isaac Sim and ROS2 properly for this workspace.

## Backbone
Use a model from IsaacSim Assets
Movement based on relative data and polar coordinates/values. move R, angle theta
Determine sensors for application
- IMU
- Gyroscope
- Lidar maybe?
- servos which already have encoders (duh)
Stable Baselines - Python Library (PPO?) - explore other options
Building in Isaac Lab - must run on primary laptop
- Read over documentation (follow isaaclab installation, not isaacsim installation)

Stable Baselines link
  https://stable-baselines3.readthedocs.io/en/master/guide