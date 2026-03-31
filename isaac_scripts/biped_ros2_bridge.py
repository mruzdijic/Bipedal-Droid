import os
from omni.isaac.kit import SimulationApp

def main():
    print("[INFO] Starting Isaac Sim Application...")
    simulation_app = SimulationApp({"headless": False})

    # Ensure the ROS2 Bridge inside Isaac Sim is enabled programmatically
    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension("omni.isaac.ros2_bridge")

    # Wait for app to be ready
    simulation_app.update()

    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    import omni.graph.core as og

    world = World()
    world.scene.add_default_ground_plane()

    print("[INFO] Loading URDF into Stage...")
    # Get the URDF File
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "..", "urdf", "biped.urdf")
    
    from omni.isaac.urdf import _urdf
    urdf_interface = _urdf.acquire_urdf_interface()
    import_config = _urdf.ImportConfig()
    import_config.make_default_prim = True
    
    dest_path = "/World/Biped"
    urdf_interface.parse_urdf(urdf_path, import_config, dest_path)

    robot = world.scene.add(Robot(prim_path=dest_path, name="biped_robot"))

    print("[INFO] Generating ROS2 Action Graphs...")
    # Create the ROS2 Joint State Publisher node and the clock
    try:
        keys = og.Controller.Keys
        (ros2_graph, list_of_nodes, _, _) = og.Controller.edit(
            {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ROS2Context", "omni.isaac.ros2_bridge.ROS2Context"),
                    ("ROS2PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ("ROS2PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "ROS2PublishClock.inputs:execIn"),
                    ("ROS2Context.outputs:context", "ROS2PublishClock.inputs:context"),
                    ("OnPlaybackTick.outputs:tick", "ROS2PublishJointState.inputs:execIn"),
                    ("ROS2Context.outputs:context", "ROS2PublishJointState.inputs:context"),
                ],
                keys.SET_VALUES: [
                    ("ROS2PublishJointState.inputs:targetPrim", [dest_path]),
                    # Topic name
                    ("ROS2PublishJointState.inputs:topicName", "/joint_states"),
                ],
            },
        )
        print("[INFO] Successfully Built ROS2 Graph! Robot state publishing to `/joint_states` topic.")
    except Exception as e:
        print(f"[ERROR] Could not build ROS2 Action Graph: {e}")

    world.reset()

    print("[INFO] Simulation initialized. You can now connect via RViz or ROS2 terminals.")
    print("Press Ctrl+C to stop.")
    
    # Standalone simulation loop
    while simulation_app.is_running():
        world.step(render=True)
        
    simulation_app.close()

if __name__ == "__main__":
    main()
