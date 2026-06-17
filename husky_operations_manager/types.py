from dataclasses import dataclass, field
from geometry_msgs.msg import PoseStamped


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
class DockInstanceConfig:
    instance_name: str
    type: str
    frame: str
    pose: DockPose


@dataclass
class DriveConfig:
    detection_topic: str
    odom_topic: str
    base_frame: str
    cmd_vel_rate: float
    ex_tolerance: float
    stop_lookahead: float
    ex_coast_gate: float
    ex_angular_gate: float
    k_rho: float
    v_linear_min: float
    v_linear_max: float
    v_angular_max: float
    departure_clearance: float
    no_detection_distance: float


@dataclass
class ReverseDriveConfig:
    dock_names: list[str]
    dock_configs: dict
    plugin_name: str
    staging_x_offset: float
    staging_yaw_offset: float
    base_frame: str
    controller_frequency: float
    v_linear_min: float
    v_angular_max: float
    linear_tolerance: float
    angular_tolerance: float
    dock_backwards: bool


@dataclass
class NavigationFeedback:
    """
    NavigationFeedback class represents the status of a way point follower process.

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
class ManipulatorTaskFeedback:
    """Feedback data for a harvesting arm task."""

    status: int
    task: str
    arm_task: str
    feedback_message: str
    execution_time: float
    num_retries: int
    arm_pose: PoseStamped = field(default_factory=PoseStamped)


@dataclass
class UnloaderFeedback:
    """Feedback data for an unloader carriage goal."""

    status: int
    target: str  # 'HOME' or 'END'
    progress_percent: float
    step_count: int
    at_home_limit: bool
    at_end_limit: bool
    feedback_message: str
