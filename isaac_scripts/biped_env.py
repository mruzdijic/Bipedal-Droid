import gym
from gym import spaces
import numpy as np
import os

# Isaac Sim Imports
from omni.isaac.kit import SimulationApp

class BipedalEnv(gym.Env):
    """
    A foundational Custom Gym Environment for a BD-Style Bipedal Robot in Isaac Sim.
    """
    def __init__(self, headless=False):
        super().__init__()
        
        # 1. Start the Simulation Application
        self.sim_app = SimulationApp({"headless": headless})
        
        # Imports that require the simulation app to be running
        from omni.isaac.core import World
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.articulations import Articulation
        
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # 2. Define Action and Observation Spaces
        # 12 Joints total (6 per leg)
        self.num_actions = 12
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        
        # Observations: 12 joint positions, 12 joint velocities, 3 base lin vel, 3 base ang vel (30 total)
        self.num_observations = 30
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observations,), dtype=np.float32)

        # 3. Load the URDF Robot into the Stage
        # Path to the locally generated URDF
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "..", "urdf", "biped.urdf")
        
        # Isaac Sim needs an importer to convert URDF to USD (Universal Scene Description)
        from omni.isaac.urdf import _urdf
        urdf_interface = _urdf.acquire_urdf_interface()
        
        # Import config
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.make_default_prim = True
        
        # Import the URDF as a USD in memory and add to stage
        dest_path = "/World/Biped"
        urdf_interface.parse_urdf(urdf_path, import_config, dest_path)
        
        # Wrap it with the Articulation standard Isaac Class
        self.robot = self.world.scene.add(
            Articulation(prim_path=dest_path, name="biped_robot")
        )

        self.world.reset()
        
    def reset(self):
        """Reset the environment to its initial state."""
        self.world.reset()
        
        # Reset robot to default joint positions (0 for all)
        default_dof_pos = np.zeros(self.num_actions)
        self.robot.set_joint_positions(default_dof_pos)
        
        # Run one step to update physics
        self.world.step(render=True)
        return self._get_obs()

    def step(self, action):
        """Apply the action (joint position targets), step physics, and calculate reward."""
        from omni.isaac.core.utils.types import ArticulationAction
        
        # Scale action from [-1, 1] to actual joint limits (e.g. [-1.57, 1.57] radians)
        # For simplicity in this foundation, we just apply the action as a direct position target
        target_positions = action * 1.5 
        
        self.robot.apply_action(ArticulationAction(joint_positions=target_positions))
        
        # Step physics simulator
        self.world.step(render=True)
        
        # Get new observations
        obs = self._get_obs()
        
        # Calculate Reward (Simple template: reward for height, penalty for falling)
        base_pos, _ = self.robot.get_world_pose()
        height = base_pos[2]
        reward = height  # Replace with complex reward function
        
        # Check termination (if it falls down)
        done = bool(height < 0.1)
        
        info = {}
        return obs, reward, done, info

    def _get_obs(self):
        """Gathers sensory data for the neural network."""
        import numpy as np
        
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        base_lin_vel = self.robot.get_linear_velocity()
        base_ang_vel = self.robot.get_angular_velocity()
        
        obs = np.concatenate([
            joint_positions, 
            joint_velocities, 
            base_lin_vel, 
            base_ang_vel
        ], dtype=np.float32)
        
        return obs

    def close(self):
        self.sim_app.close()
