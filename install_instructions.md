# Install Instructions for Bipedal ML (Secondary Computer)

To set up a completely fresh secondary computer (highly recommended to use **Ubuntu 22.04 LTS** for maximum compatibility with ROS2 Humble and Isaac Sim), follow these CLI commands. 

> [!NOTE]
> If you are on Windows 11, you can install ROS2 natively, but managing Isaac Sim + ROS2 + Stable Baselines is easiest either in Ubuntu natively, or using WSL2 on Windows. The following assumes a Linux/Ubuntu 22.04 environment.

## 1. Setup ROS2 Humble

Run these commands in a terminal to install ROS2 Humble:

```bash
# Set locale
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Setup Sources
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2
sudo apt update
sudo apt install -y ros-humble-desktop

# Install Colcon and dependencies (for building your own ROS packages)
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-vcstool

# Initialize rosdep
sudo rosdep init
rosdep update
```

Next, add the source file to your bashrc:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 2. Omniverse Isaac Sim

NVIDIA Omniverse is distributed via the standalone "Omniverse Launcher". Unfortunately, this cannot be strictly installed entirely via terminal, as it requires login.

1. **Download:** Go to [NVIDIA Omniverse](https://developer.nvidia.com/omniverse) and download the **Omniverse Launcher**.
2. **Install Isaac Sim:** Once logged in, go to the "Exchange" tab, search for **Isaac Sim** and install version 2023.1.0 or newer.

> [!TIP]
> After downloading Isaac Sim, map the path! Typically it installs here:
> `~/.local/share/ov/pkg/isaac_sim-2023.1.0`

## 3. Install Machine Learning Python Dependencies

Isaac Sim ships with its own Python environment (`python.sh`). You must install `stable-baselines3` and `ros2` Python bridges into its bundled python!

Open your terminal and run:

```bash
# Create an alias so you don't have to type the full path every time
# (Change the version number if you downloaded a newer version of Isaac Sim)
export ISAAC_PATH=~/.local/share/ov/pkg/isaac_sim-2023.1.0

# Install PyTorch (usually comes pre-installed in Isaac Sim but verify)
$ISAAC_PATH/python.sh -m pip install torch torchvision torchaudio

# Install Stable Baselines3 (our RL Engine)
$ISAAC_PATH/python.sh -m pip install stable-baselines3[extra]

# Install basic ML/data libs
$ISAAC_PATH/python.sh -m pip install typing-extensions numpy pandas matplotlib tensorboard
```

## 4. Run the ROS2 Extention Enable Script

Once Isaac Sim is running (via `$ISAAC_PATH/isaac-sim.sh`), navigate to: 
`Window -> Extensions`
Search for **ROS2 Bridge** (`omni.isaac.ros2_bridge`) and toggle the "Autoload" switch to True.

You are now fully configured for Reinforcement Learning Bipedal simulation and ROS2 teleop!
