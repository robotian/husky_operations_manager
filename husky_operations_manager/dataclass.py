from dataclasses import dataclass, field
from geometry_msgs.msg import PoseStamped

@dataclass
class WPFStatus:
    """
    WPFStatus class represents the status of a way point follower process.

    Attributes:
        status (int): The current status of the way point follower process.
        task (str): The task associated with the way point follower process.
        current_node_id (int): The ID of the current node in the way point follower.
        target_node_id (int): The ID of the target node in the way point follower.
    """

    status: int
    task: str
    current_node_id: int
    target_node_id: int


@dataclass
class DockingFeedback:
    """DockingStatus class represents the status of a docking process.

    Attributes:
        status (int): The current status of the docking process.
        task (str): The task associated with the docking process.
    """

    status: int
    task: str
    docking_location: str
    feedback_message: str
    docking_time: float
    num_retries: int

@dataclass
class DockPose:
    x: float
    y: float
    theta: float

@dataclass
class Docks:
    name: str
    type: str
    frame: str
    pose: DockPose


@dataclass
class DockPluginConfig:
    plugin_name: str
    plugin: str
    docking_threshold: float
    staging_x_offset: float
    staging_yaw_offset: float
    use_external_detection_pose: bool
    external_detection_timeout: float
    external_detection_translation_x: float
    external_detection_translation_y: float
    external_detection_rotation_roll: float
    external_detection_rotation_pitch: float
    external_detection_rotation_yaw: float
    filter_coef: float
    detector_service_name: str
    detector_service_timeout: float
    subscribe_toggle: bool
    use_battery_status: bool
    use_stall_detection: bool
    stall_velocity_threshold: float
    stall_effort_threshold: float
    charging_threshold: float
    rotate_to_dock: bool
    dock_direction: str                        

 
@dataclass
class DockInstanceConfig:
    instance_name: str
    type: str
    frame: str
    pose: list
    id: str        
    dock_x: float  
    dock_y: float  
    dock_theta: float 
 
 
@dataclass
class DockingConfig:
    # Raw lists from docking_server
    dock_plugins: list
    docks: list
 
    # Fetched configs keyed by name
    plugin_configs: dict        # plugin_name → DockPluginConfig
    dock_configs: dict          # dock_name   → DockInstanceConfig
 
    # Top-level params
    base_frame: str
    fixed_frame: str
    controller_frequency: float
    initial_perception_timeout: float
    wait_charge_timeout: float
    dock_approach_timeout: float
    undock_linear_tolerance: float
    undock_angular_tolerance: float
    max_retries: int
    dock_backwards: bool
    dock_prestaging_tolerance: float
 
    # Controller params
    controller_k_phi: float
    controller_k_delta: float
    controller_v_linear_min: float
    controller_v_linear_max: float
    controller_v_angular_max: float
    controller_slowdown_radius: float
    controller_use_collision_detection: bool
    controller_costmap_topic: str
    controller_footprint_topic: str
    controller_transform_tolerance: float
    controller_projection_time: float
    controller_simulation_time_step: float
    controller_dock_collision_threshold: float
    controller_rotate_to_heading_angular_vel: float
    controller_rotate_to_heading_max_angular_accel: float

@dataclass
class ManipulatorTaskFeedback:
    """Feedback data for a harvesting arm task."""
    status: int
    task: str
    arm_task: str
    arm_pose: PoseStamped = field(default_factory=PoseStamped)
    feedback_message: str = ""
    execution_time: float = 0.0
    num_retries: int = 0