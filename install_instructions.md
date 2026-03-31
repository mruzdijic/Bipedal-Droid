# Isaac Lab Installation Instructions

This project targets **Isaac Lab**, the official GPU-accelerated robot learning framework built *on top* of Omniverse Isaac Sim by NVIDIA. To run the environment and train the Cassie biped, follow these strict installation instructions.

> [!IMPORTANT]
> Because you are compiling and cloning Isaac Lab, you MUST have the Omniverse Launcher and an NVIDIA driver (version 535+) installed on your primary laptop/desktop running Ubuntu 22.04 LTS natively or WSL2 on Windows 11.

## 1. Install Isaac Sim
You cannot run Isaac Lab without the base engine.
1. Download the **NVIDIA Omniverse Launcher**.
2. Go to the "Exchange" tab, search for **Isaac Sim**.
3. Install version **4.1.0** (or the latest version compatible with the current Isaac Lab `main` branch).

## 2. Clone Isaac Lab 
Open a terminal in your workspace directory and clone the official repository:

```bash
# Clone the repository
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Create a symbolic link to your Isaac Sim installation so Isaac Lab knows where it is
ln -s ~/.local/share/ov/pkg/isaac-sim-4.1.0 _isaac_sim
```

## 3. Install Isaac Lab Extensions
NVIDIA provides a built-in bash script to resolve all pip dependencies, build the C++ extensions, and configure the Python environments natively.

Run the installer:
```bash
./isaaclab.sh --install
```

## 4. Install RL Framework (Stable Baselines 3)
Isaac Lab supports multiple Reinforcement Learning libraries via extras. Since this project uses Stable Baselines 3 (SB3), run:

```bash
./isaaclab.sh -p -m pip install "stable-baselines3[extra]"
```

## 5. Running the Project
Because Isaac Lab dynamically links to Isaac Sim's bundled python, you **cannot** run python scripts with the standard `python script.py`.

You must wrap all execution with the `isaaclab.sh -p` command.

Example to run the training script:
```bash
cd /path/to/Bipedal-Droid/isaaclab_workspace/exts/biped_ml
/path/to/IsaacLab/isaaclab.sh -p train_sb3.py
```
