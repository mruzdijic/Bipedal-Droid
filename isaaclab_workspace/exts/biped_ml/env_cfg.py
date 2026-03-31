"""
Bipedal ML: Cassie Configuration for Isaac Lab.
This defines the entire RL MDP (Markov Decision Process) setup including 
Sensors (Observations), Rewards, Commands, and Actions for the Cassie Biped.
"""

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass

import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg

# Import the pre-defined high-fidelity Cassie configuration from the lab assets!
try:
    from omni.isaac.lab_assets.cassie import CASSIE_CFG
except ImportError:
    # Fallback placeholder if asset string changes in future Isaac Lab updates
    print("[WARNING] Could not load CASSIE_CFG directly from lab_assets. Proceeding with dummy ArticulationCfg. You may need to update the asset path.")
    CASSIE_CFG = ArticulationCfg(spawn=None, init_state=None)

@configclass
class BipedSceneCfg(InteractiveSceneCfg):
    """Configuration for the Stage."""
    # Basic ground plane terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )
    # The Robot
    robot: ArticulationCfg = CASSIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class BipedCommandsCfg:
    """Configurations for the Cartesian Velocity Commands (vx, vy, yaw_rate)."""
    velocity_command = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.02, # 2% chance of robot being commanded to stand perfectly still 
        rel_heading_envs=1.0,
        heading_command=False,  # Use yaw rate instead of absolute heading
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class BipedActionsCfg:
    """Configuration for actions."""
    # We use joint PD targets as our action space for the RL neural network.
    joint_positions = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"], # Control all joints
        scale=0.5, # the NN output is scaled before adding to the default joint pos
        use_default_offset=True,
    )


@configclass
class BipedObservationsCfg:
    """Configurations for the Observations (Sensors)."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Standard Observation vector fed to the Actor Network."""
        
        # 1. Internal Encoders (Proprioception)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=mdp.UniformNoiseCfg(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=mdp.UniformNoiseCfg(n_min=-1.5, n_max=1.5))
        
        # 2. IMU / Gyroscope (Base State)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=mdp.UniformNoiseCfg(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=mdp.UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        
        # 3. Task Context (The Cartesian velocities we want it to achieve)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "velocity_command"})
        
        # 4. Action Recurrence
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # Tell the environment to flatten this and concatenate it into a 1D tensor setup
            self.enable_corruption = True
            self.concatenate_terms = True

    # The group we feed into SB3
    policy: PolicyCfg = PolicyCfg()


@configclass
class BipedRewardsCfg:
    """Configuration for Reward functions."""
    # Reward for walking in the commanded direction
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "velocity_command", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "velocity_command", "std": 0.5}
    )
    
    # Penalties for erratic behavior
    penalize_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    penalize_action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # Penalty if the base link collides with the ground (it fell over)
    # Cassie specifically has a "pelvis" link.
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class BipedTerminationsCfg:
    """Configuration for when to reset an environment (e.g., robot fell down)."""
    # Terminate if the robot survives for max steps (success)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Terminate if the pelvis hits the ground
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="pelvis")},
    )


@configclass
class BipedEnvCfg(ManagerBasedRLEnvCfg):
    """The High-Level Environment Class linking all configurations together."""
    scene: BipedSceneCfg = BipedSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: BipedObservationsCfg = BipedObservationsCfg()
    actions: BipedActionsCfg = BipedActionsCfg()
    commands: BipedCommandsCfg = BipedCommandsCfg()
    rewards: BipedRewardsCfg = BipedRewardsCfg()
    terminations: BipedTerminationsCfg = BipedTerminationsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # General physics settings
        self.sim.dt = 0.005 # 200 Hz Physics
        self.decimation = 4 # Controller updates at 50Hz (200Hz / 4)
        self.episode_length_s = 20.0 # Each episode maxes out at 20 seconds
        
        # Override Isaac Sim Viewer to look at a specific robot rather than the whole 4096 cluster
        self.viewer.eye = (3.0, 3.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
