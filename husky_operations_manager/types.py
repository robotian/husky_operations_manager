from dataclasses import dataclass

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
    v_linear_min: float
    v_linear_max: float
    v_angular_max: float
    departure_clearance: float
    no_detection_distance: float

    # --- PD target-pose controller (drive.py) ---
    ang_tol: float  # rad — final-heading tolerance (~3deg)
    k_v_p: float
    k_v_d: float
    k_omega_p: float
    k_omega_d: float
    k_beta_p: float
    k_beta_d: float
    a_max: float  # m/s^2
    alpha_max: float  # rad/s^2
    backward_distance_threshold: float  # m — allow reverse approach below this distance
    same_bush_threshold: float  # m — CONTROLLING re-lock accepted only within this of the currently locked target
    
    # --- Camera mount / row geometry (drive.py) ---
    # cam_tx/cam_ty/arm_tx_offset are resolved via TF by the owning node
    # (base_link -> camera frame / arm frame) before DriveConfig is built —
    # DriveClient itself has no TF dependency, just consumes the floats.
    cam_tx: float  # m — camera behind base_link
    cam_ty: float  # m — camera right of base_link
    bushrow_theta: float  # rad — row orientation in odom frame
    # TF-derived: arm_0_base_link.tx = -0.214 m relative to base_link.
    # Declared for parity with the reference sim, which documents this offset
    # but never applies it in the target-pose calc — kept unused here too.
    arm_tx_offset: float


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
    arm_pose: PoseStamped


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
