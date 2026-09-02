from dataclasses import dataclass

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
class DockInstanceConfig:
    instance_name: str
    type: str
    frame: str
    pose: DockPose


@dataclass
class ReverseDriveConfig:
    # Every dock the client can reverse to, keyed by instance name. The caller
    # selects one per run via drive_to_staging(dock_name) — the client holds no
    # dock identity between runs. Each dock's `type` doubles as its plugin name.
    dock_configs: dict[str, DockInstanceConfig]
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
class ManipulatorTaskFeedback:
    """Feedback data for a harvesting arm task."""

    status: int
    task: str
    arm_task: str
    feedback_message: str
    execution_time: float
    num_retries: int
    arm_pose: PoseStamped


@dataclass
class DriveConfig:
    timeout: float
    base_frame: str
    tf_base_frame: str
    tf_detection_frame: str
    v_linear: float
    v_angular: float
    tf_polling_rate: float    
    tolerance: float

