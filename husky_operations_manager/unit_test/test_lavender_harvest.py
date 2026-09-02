"""
test_lavender_harvest.py

Standalone, hardcoded-sequence harvest run across a farm's rows: for each row in
ROWS, drive to its start, harvest every bush along it (camera-guided), then move
to the next row. Low battery or a full load interrupts the current row, docks,
resolves the condition, and resumes where it left off. When every row in ROWS
has been attempted, the node docks at the unloading station, unloads, undocks,
and stops.

No status_server / Task-message dependency — this node drives its own sequence
end to end. All tuning values are module constants below, not YAML; field
tuning is a one-file edit that takes effect on the next run under
--symlink-install, no rebuild.

-------------------------------------------------------------------------------
State model
-------------------------------------------------------------------------------

  This node is a prototype for the production husky_operations_manager once
  DriveClient lands there, so it uses production's state model rather than its
  own vocabulary:

    * `self.current_status` (RobotStatusEnum) is the ONLY state. It moves only
      through `_transition_status()`.
    * Everything else is context — it never decides *what runs*, only which
      exit a state takes.
    * `_process_action_clients()` is the single arbiter: exactly one motion
      client is serviced per tick, in priority order. The manipulator is polled
      alongside, because arm motion overlaps robot motion.

  Camera driving reuses START_MOVING / MOVING / DESTINATION_REACHED. The
  `motion_source` context field says whether Nav2 or DriveClient owns the
  current MOVING, and therefore what the following DESTINATION_REACHED means:

    'nav'   -> arrived at a row start   -> arm READY, then scan()
    'drive' -> parked at a bush         -> harvest

Harvesting is trigger-and-wait only: one START_HARVEST goal per bush. Cut
counting and any per-cut loading sequence are owned by another package and are
out of scope here.

No time.sleep() anywhere — every wait is polled from a single monitor timer, so
battery, pose, and TF stay live throughout.

Run:
  ros2 run husky_operations_manager test_lavender_harvest \\
    --ros-args -r __ns:=/a200_0284 -r /tf:=tf -r /tf_static:=tf_static
"""

import math

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener

from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.action_clients.drive import DriveClient
from husky_operations_manager.action_clients.manipulator import ArmCommand, ManipulatorTaskActionClient
from husky_operations_manager.action_clients.navigation import NavigationActionClient
from husky_operations_manager.action_clients.reverse_drive import ReverseDriveClient
from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.action_clients.unloader import UnloaderActionClient
from husky_operations_manager.robot_enums import (
    NavigationStatus,
    ReverseDriveStatus,
    RobotStatusEnum,
    TaskEnum,
)
from husky_operations_manager.types import DockInstanceConfig, DockPose, DriveConfig, ReverseDriveConfig
from status_interfaces.action import OperateUnloader
from status_interfaces.msg import DockGoal, RobotStatus, SubTask, Task, UndockGoal, WayPoint

# =============================================================================
# GLOBAL CONFIG — edit here instead of a YAML file
# =============================================================================

# --- TF frames ---
BASE_FRAME = 'base_link'
CAMERA_FRAME = 'camera_1_color_optical_frame'
# Frame DriveClient builds bush targets in AND reads the robot pose in. Those two
# must match: the PD controller subtracts one from the other, so a target in map
# against a pose in odom put the whole map->odom offset into the position error.
ODOM_FRAME = 'map'
# Frame every navigation goal is stamped in (navigation.py stamps 'map'), and
# therefore the frame ROWS and row_progress_pose are compared in. The pose topic
# publishes in the odom frame, so TF closes the gap — see _robot_pose_in_frame.
NAV_FRAME = 'map'

STATIC_DRIVE_PARAMS = {
    'detection_topic': 'sensors/camera_1/detection/image_annotated/detection_pose',
    'odom_topic': 'platform/odom/filtered',
    'base_frame': BASE_FRAME,
    'cmd_vel_rate': 10.0,  # Hz
    'ex_tolerance': 0.02,  # m
    'v_linear_min': 0.05,  # m/s
    'v_linear_max': 0.125,  # m/s
    'v_angular_max': 0.15,  # rad/s
    'departure_clearance': 0.2,  # m
    'no_detection_distance': 1.5,  # m — row end assumed after this distance
    'ang_tol': 0.05,  # rad
    'k_v_p': 0.2,
    'k_v_d': 0.07,
    'k_omega_p': 0.4,
    'k_omega_d': 0.1,
    'k_beta_p': 1.0,
    'k_beta_d': 0.4,
    'a_max': 0.05,  # m/s^2
    'alpha_max': 0.3,  # rad/s^2w
    'backward_distance_threshold': 1.0,  # m
    'same_bush_threshold': 0.25,  # m
    'controlling_timeout': 30.0,  # s
    'max_controlling_retries': 3,
    'controlling_retry_delay': 5.0,  # s
}

# --- Job identity ---
# Filled into every Task this node builds for itself. In production the task
# server owns both.
ROBOT_ID = 0
CROP_TYPE = 'lavender'

# --- Subscribed topics ---
TOPIC_BATTERY = 'platform/bms/state'
TOPIC_ESTOP = 'platform/emergency_stop'

# --- Timing ---
TIMER_PERIOD_SEC = 0.2  # s — monitor loop period; every polled wait derives from this

# s — a sensor reading older than this is treated as no reading at all. Set
# from the slowest of the three topics, not the fastest.
SENSOR_STALE_SEC = 5.0

# --- Deadlines ---
# Every state that waits on something external has one. Without them a dead
# action server or a silent topic parks the robot indefinitely with nothing in
# the log to say why. Each is counted in monitor ticks, never slept.
TIMEOUT_INIT_SEC = 60.0          # pose + battery + camera TF + map->odom
TIMEOUT_DRIVE_READY_SEC = 15.0   # DriveClient odom/TF before a scan or resume
TIMEOUT_NAV_SEC = 180.0          # one Nav2 goal
TIMEOUT_DRIVE_SEC = 120.0        # one camera-guided leg (30s x3 retries + delays)
TIMEOUT_DOCKING_SEC = 180.0      # includes the staging approach
TIMEOUT_UNDOCKING_SEC = 25.0     # 2x the max_undocking_time sent in the goal
TIMEOUT_HARVEST_SEC = 120.0      # one real cut cycle
TIMEOUT_UNLOAD_STEP_SEC = 60.0   # one carriage END or HOME traverse
TIMEOUT_ERROR_SETTLE_SEC = 10.0  # cancels to land during recovery

RETRY_DELAY_SEC = 5.0  # s — backoff before re-sending a failed nav or dock goal

# --- Row alignment ---
# Nav2 reports arrival on position first, which says nothing useful about which
# way the robot ends up pointing, and SCANNING then drives straight along
# whatever heading it happens to hold (drive.py keeps _cmd_angular_z at 0.0). So
# the robot is turned onto the row heading before the camera drive may start.
# This also covers the move between rows 2 and 3, which is a pure 180 deg turn
# with no translation — the one Nav2 can call SUCCEEDED without turning at all.
ALIGN_ANGULAR_TOL_RAD = 0.05  # rad (~2.9 deg) — matches DriveClient's ang_tol
ALIGN_ANGULAR_MAX = 0.15      # rad/s — matches DriveClient's v_angular_max
ALIGN_ANGULAR_MIN = 0.05      # rad/s — below this the base will not turn at all
ALIGN_GAIN = 0.8              # rad/s of turn per rad of heading error
TIMEOUT_ALIGN_SEC = 45.0      # s — a 180 deg turn at ALIGN_ANGULAR_MAX takes ~21s

# Charging has no fixed deadline — the battery decides when it is done. What is
# bounded is progress: if the percentage stops rising, the robot is not actually
# charging (charger off, bad contact, BMS wedged) and waiting longer will not
# help. EPSILON exists because the BMS reports fractional percent, so noise
# alone must not count as progress and mask a real stall.
CHARGE_NO_PROGRESS_TIMEOUT_SEC = 120.0
CHARGE_PROGRESS_EPSILON = 0.1  # %

# --- Hardware simulation ---
# Only the cut cycle and the unloader carriage are stood in for. GO_STOW and
# GO_READY are always sent to the real manipulator server, so the stow gate
# that guards every base motion reflects where the arm actually is.
SIMULATE_HARVEST = True
SIMULATE_UNLOADER = True

SIM_HARVEST_SEC = 30.0   # s — stand-in for one START_HARVEST cut cycle
SIM_UNLOADER_SEC = 4.0   # s — stand-in for one carriage END or HOME traverse

# --- Thresholds ---
BATTERY_LOW_THRESHOLD = 50.0   # % — below this, interrupt the row and go charge
BATTERY_FULL_THRESHOLD = 99.0  # % — ends the charging interrupt
LOAD_INCREMENT = 20.0          # % per confirmed harvest — SIMULATED, nothing weighs the bin
DOCKING_THRESHOLD_M = 1.0     # m — startup "at dock" distance check
NAV_CLOSE_ENOUGH_M = 0.25      # m — ABORT within this of the target may count as success
NAV_HEADING_TOL_RAD = 0.15     # rad (~8.6deg) — ...but only if the heading also matches

# --- Retries ---
MAX_HARVEST_RETRIES = 3
MAX_DOCKING_RETRIES = 3
MAX_NAVIGATION_RETRIES = 3
MAX_RECOVERY_ATTEMPTS = 3  # ERROR recoveries before parking in ABNORMAL for good

# --- Docks ---
CHARGING_DOCK_NAME = 'husky_charger'
UNLOADING_DOCK_NAME = 'unloading_station'

DOCK_CONFIGS: dict[str, DockInstanceConfig] = {
    'husky_charger': DockInstanceConfig(
        instance_name='husky_charger',
        type='simple_charging_dock',
        frame='map',
        pose=DockPose(x=-6.001, y=2.194, theta=0.0), # -6.001, 2.194, 0.0
    ),
    'unloading_station': DockInstanceConfig(
        instance_name='unloading_station',
        type='simple_charging_dock',
        frame='map',
        pose=DockPose(x=-6.001, y=2.194, theta=0.0), # NOTE: for testing only
        # pose=DockPose(x=-6.772, y=8.290, theta=1.571), # -6.772, 8.290, 1.571
        # pose=DockPose(x=0.85, y=1.60, theta=1.571),
    ),
}

MOTION_CONFIG = {
    'dock_configs': DOCK_CONFIGS,
    'staging_x_offset': -1.5,
    'staging_yaw_offset': 0.0,
    'base_frame': 'base_link',
    'controller_frequency': 50.0,
    'v_linear_min': 0.15,
    'v_angular_max': 0.25,
    'linear_tolerance': 0.05,
    'angular_tolerance': 0.1,
    'dock_backwards': False,
}

POST_UNLOAD_DELAY_SEC = 5.0  # s — wait at END before returning carriage HOME

# --- Rows ---
# Field data, taken from the farm_node DB table. Each row is
# (start_node_id, start_x, start_y, end_node_id, end_x, end_y). Heading down
# the row is computed from start -> end at run time.
# ROWS = [
#     (1, -1.6, 0.5, 3, 1.0, 0.5),      # row b1
#     (2, 1.0, -1.26, 4, -1.6, -1.26),  # row b2
# ]

# NOTE: Below values are wrt map frame while the values in db are wrt base_mocap frame for testing in lab
#
# TODO: node 2 and node 3 currently hold the SAME coordinates (-1.089, 0.299).
# Surveyed values needed for at least one of them. While they stay identical,
# _next_row_target(1) navigates to the point the robot already occupies after
# row b1 ends, so the goal is ~0 translation and a 141 deg rotation. Nav2 checks
# position first, so it can report SUCCEEDED without ever turning — and SCANNING
# drives straight along the robot's current heading (drive.py,
# _cmd_angular_z = 0.0), so row b2 would rescan row b1 over harvested bushes.
# _is_at_nav_target() now also checks heading, which stops the ABORT-within-
# tolerance path from hiding this, but it cannot fix a genuine Nav2 SUCCEEDED.
# ROWS = [
#     (1, 1.622, 1.219, 2, -1.089, 0.299),  # row b1
#     (3, -1.089, 0.299, 4, 0.601, 2.937),  # row b2  <-- node 3 == node 2, see TODO
# ]

# Note these corrds are exact location before and after the bush
ROWS = [
    (1, -9.243, 3.98, 2, -13.422, 3.86),  # row b1
    (3, -12.953, 5.64, 4, -9.539, 5.67),
    (5, -9.539, 5.67, 6, -12.953, 5.64),  # row b2
]


def _row_heading(row: tuple) -> float:
    """atan2(end - start) for a ROWS entry — the direction of travel down that row."""
    _, sx, sy, _, ex, ey = row
    return math.atan2(ey - sy, ex - sx)


def _row_start_waypoint(row: tuple) -> WayPoint:
    """Build the nav target WayPoint for a ROWS entry's start point, heading down the row."""
    node_id, x, y, _, _, _ = row
    return _make_waypoint(node_id, x, y, _row_heading(row))


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _pose_to_waypoint(pose: tuple[float, float, float]) -> WayPoint:
    """A resume point has no node in the farm table, hence node_id -1."""
    x, y, theta = pose
    return _make_waypoint(-1, x, y, theta)


def _normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# =============================================================================
# TASK / SUBTASK CONSTRUCTION
# =============================================================================
#
# In production a task server builds these messages and publishes them; this
# node subscribes and executes. Here the same messages are built locally so the
# node can run on its own, but they are the real Task and SubTask types with
# every field a server would fill in.
#
# Building them through these factories rather than inline at each goal-send
# means swapping to a real subscriber later replaces one function — _build_task
# — and nothing else. It also keeps the fields consistent: previously each send
# site set whichever two fields it happened to need and left the rest at their
# defaults.


def _make_waypoint(node_id: int, x: float, y: float, theta: float) -> WayPoint:
    wp = WayPoint()
    wp.node_id = node_id
    wp.x = x
    wp.y = y
    wp.theta = theta
    return wp


def _make_subtask(
    sub_task_id: int,
    sub_task_type: int,
    description: str,
    waypoints: list[WayPoint] | None = None,
    dock_goal: DockGoal | None = None,
    undock_goal: UndockGoal | None = None,
) -> SubTask:
    """Build one SubTask. Only the fields that type actually uses are set."""
    subtask = SubTask()
    subtask.sub_task_id = sub_task_id
    subtask.type = sub_task_type
    subtask.description = description
    if waypoints is not None:
        subtask.data = waypoints
    if dock_goal is not None:
        subtask.dock_goal = dock_goal
    if undock_goal is not None:
        subtask.undock_goal = undock_goal
    return subtask


def _make_dock_goal(dock_id: str) -> DockGoal:
    dock_goal = DockGoal()
    dock_goal.use_dock_id = True
    dock_goal.dock_id = dock_id
    dock_goal.navigate_to_staging_pose = True
    return dock_goal


def _make_undock_goal(dock_type: str, max_undocking_time: float) -> UndockGoal:
    return UndockGoal(dock_type=dock_type, max_undocking_time=max_undocking_time)


def _max_undocking_time() -> float:
    """How long backing out to the staging pose should take, plus 25%."""
    staging_x_offset = MOTION_CONFIG['staging_x_offset']
    v_linear_min = MOTION_CONFIG['v_linear_min']
    return (abs(staging_x_offset) / max(v_linear_min, 0.01)) * 1.25


# Readable names for log lines. The message only carries the integers.
SUB_TASK_NAMES = {
    SubTask.MOVING: 'move',
    SubTask.HARVESTING: 'harvest',
    SubTask.DOCKING: 'dock',
    SubTask.CHARGING: 'charge',
    SubTask.LOADING: 'load',
    SubTask.UNLOADING: 'unload',
    SubTask.UNDOCKING: 'undock',
}


def _make_task(
    task_id: int,
    task_type: TaskEnum,
    description: str,
    sub_tasks: list[SubTask],
    crop_load: float,
    target_node_id: int = -1,
) -> Task:
    """Build the Task envelope a server would otherwise publish."""
    task = Task()
    task.task_id = task_id
    task.task_type = task_type.value
    task.assigned_robot_id = ROBOT_ID
    task.target_node_id = target_node_id
    task.job_schedule = ''  # set by the scheduler in production
    task.description = description
    task.crop_type = CROP_TYPE
    task.crop_load = crop_load
    task.sub_tasks = sub_tasks
    return task


class TestLavenderHarvestNode(Node):
    """Hardcoded, standalone run: dock check -> per-row camera harvest, with
    battery/load interrupts -> final dock/unload/undock once every row is done."""

    def __init__(self):
        super().__init__('test_lavender_harvest')

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'Node namespace: {self.namespace}')

        self._init_state_variables()
        self._init_sensor_data()
        self._init_subscriptions()
        self._validate_config()

        # RobotStatus is republished here purely so NavigationActionClient has a
        # live current-position to build its first goal pose from — nothing else
        # in this node reads it back.
        self.robot_state_pub = self.create_publisher(
            RobotStatus, f'{self.namespace}/status/robot', 10)

        # Same topic and message type DriveClient and ReverseDriveClient use. The
        # single-mover interlock is what stops two of them commanding at once.
        self.align_cmd_pub = self.create_publisher(
            TwistStamped, f'{self.namespace}/cmd_vel', 10)

        # --- Action clients ---
        self.navigation = NavigationActionClient(self)
        self.docking_action_client = DockingActionClient(self)
        self.undocking_action_client = UndockingActionClient(self)
        self.manipulator_client = ManipulatorTaskActionClient(self)
        self.unloader_client = UnloaderActionClient(self)
        self.reverse_drive_client = ReverseDriveClient(self, ReverseDriveConfig(**MOTION_CONFIG))

        # --- DriveClient — built once pose and camera TF both resolve ---
        self.drive_client: DriveClient | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # One startup timer, not two. Until it sets is_initialized, timer_callback
        # publishes status and does nothing else — so no phase can advance while
        # DriveClient is still missing.
        self.init_check_timer = self.create_timer(TIMER_PERIOD_SEC, self._initial_check_timer)
        self.timer = self.create_timer(TIMER_PERIOD_SEC, self.timer_callback)

        self.get_logger().info('test_lavender_harvest started')

    # =========================================================================
    # INITIALISATION
    # =========================================================================

    def _init_state_variables(self) -> None:
        """Initialise every state-tracking variable to its boot default."""
        # --- Robot state: the only true state in this node ---
        self.current_status = RobotStatusEnum.IDLE
        self.previous_status = RobotStatusEnum.IDLE

        # --- Startup ---
        self.is_initialized = False           # True once every init gate has passed
        self.is_at_docking_station = False    # True if booted within DOCKING_THRESHOLD_M
        self.startup_undock_complete = False  # True once the boot sequence is finished
        self.init_ticks = 0                   # counted against TIMEOUT_INIT_SEC
        self.pose_valid = False               # True while the pose resolves into NAV_FRAME

        # --- Dock selection (this node owns active_dock; the client does not) ---
        self.active_dock: DockInstanceConfig | None = None

        # --- Task ---
        # Which of HARVESTING/CHARGING/UNLOADING the current status chain serves.
        # CHARGING and UNLOADING share START_DOCKING..DONE_UNDOCKING, so this and
        # active_dock are what tell the two apart.
        self.current_task_type: TaskEnum | None = None

        # The Task message the node built for itself, standing in for the one a
        # task server would publish. Every goal sent while it is current draws
        # its SubTask from here rather than making one up at the send site.
        self.current_task: Task | None = None
        self.next_task_id = 1
        self.next_sub_task_id = 1
        self.farm_exhausted = False
        self.run_complete = False

        # --- Motion ownership: who owns the current MOVING / DESTINATION_REACHED ---
        self.motion_source: str | None = None  # 'nav', 'align' or 'drive'
        self.pending_move: str | None = None   # 'scan' or 'resume'
        self.nav_target_wp: WayPoint | None = None

        # --- Row bookkeeping (no production counterpart) ---
        self.row_index = 0
        self.row_over = False
        # Furthest point covered in the current row, written every time the robot
        # parks at a bush — not only after a harvest completes. A row is driven
        # in one direction, so a single pose describes progress fully; resume
        # after any interrupt returns here rather than to the row start.
        self.row_progress_pose: tuple[float, float, float] | None = None
        self.park_count_snapshot = 0
        self.bushes_visited = 0
        self.bushes_harvested = 0
        self.current_load_status = 0.0

        # --- Retry counters ---
        self.navigation_retry_count = 0
        self.docking_retry_count = 0
        self.harvest_retry_count = 0
        self.recovery_attempts = 0

        # --- Retry backoff: ticks to wait before re-sending a failed goal ---
        self.nav_retry_wait_ticks = 0
        self.dock_retry_wait_ticks = 0

        # --- Deadlines ---
        # One counter for the current status, reset by _transition_status. States
        # with sub-steps (unloading, recovery, init) count separately below.
        self.state_ticks = 0
        self.unload_step_ticks = 0
        self.error_settle_ticks = 0

        # --- Charging progress monitor ---
        self.charge_last_pct: float | None = None
        self.charge_stall_ticks = 0

        # --- Reverse drive ---
        self.reverse_drive_active = False

        # --- Row alignment (turn in place onto the row heading) ---
        self.align_active = False
        self.align_target_theta: float | None = None

        # --- Unloader: END -> wait -> HOME is three steps inside one UNLOADING
        # state, which RobotStatusEnum cannot express on its own. ---
        self.unload_step: str | None = None  # 'END', 'WAIT', 'HOME'
        self.unload_goal_sent = False
        self.unload_wait_ticks = 0

        # --- Goal-sent latches ---
        self.nav_goal_sent = False
        self.dock_goal_sent = False
        self.undock_goal_sent = False

        # --- Arm state ---
        # Boot value is UNKNOWN, not GO_STOW: nothing has confirmed where the arm
        # actually is. Both startup paths gate on a confirmed STOW before the
        # robot is allowed to move.
        self.last_confirmed_arm_command: str = ArmCommand.UNKNOWN
        self.arm_stow_pending = False
        self.arm_ready_pending = False

        # --- Hardware simulation ---
        # Tick counters standing in for an action in flight. Counted against the
        # monitor period rather than slept, so the loop keeps serving callbacks
        # and an e-stop pre-empts a simulated wait the same as a real one.
        self.sim_harvest_ticks = 0
        self.sim_unloader_ticks = 0

    def _init_sensor_data(self) -> None:
        # All three start as None, not as default-constructed messages. An empty
        # BatteryState reads 0% and an empty Bool reads "not stopped", so a dead
        # or QoS-mismatched topic would look like valid data saying "flat
        # battery, safe to drive". None is the only value that means "nothing
        # has told us yet", and every reader below treats it as unsafe.
        #
        # Each carries the time it was received rather than a header stamp:
        # std_msgs/Bool has no header, so receipt time is the only measure of
        # freshness the three can share.
        # The robot pose is not among them: it comes from TF, so its freshness is
        # the age of the transform rather than the age of a message.
        self.battery_status: BatteryState | None = None
        self.battery_stamp: Time | None = None
        self.estop_status: Bool | None = None
        self.estop_stamp: Time | None = None

    def _init_subscriptions(self) -> None:
        # There is no pose subscription. The robot pose is read from TF instead,
        # because docks, rows and navigation goals are all map-frame while the
        # odometry topic publishes in the odom frame — comparing the two silently
        # measured distances across frames, which in the field put the whole
        # map->odom offset (about 10 m) into the camera drive's position error.
        #
        # QoS matches the publishers (qos_profile_sensor_data, BEST_EFFORT): a
        # RELIABLE subscriber against a BEST_EFFORT publisher connects to nothing
        # and reports no error.
        self.battery_sub = self.create_subscription(
            BatteryState, f'{self.namespace}/{TOPIC_BATTERY}',
            self._battery_callback, qos_profile_sensor_data)
        self.estop_sub = self.create_subscription(
            Bool, f'{self.namespace}/{TOPIC_ESTOP}',
            self._estop_callback, qos_profile_sensor_data)

    def _battery_callback(self, msg: BatteryState) -> None:
        self.battery_status = msg
        self.battery_stamp = self.get_clock().now()

    def _estop_callback(self, msg: Bool) -> None:
        self.estop_status = msg
        self.estop_stamp = self.get_clock().now()

    def _is_fresh(self, stamp: Time | None) -> bool:
        """True when a reading arrived recently enough to be acted on."""
        if stamp is None:
            return False
        return (self.get_clock().now() - stamp).nanoseconds / 1e9 <= SENSOR_STALE_SEC

    def _pose_tf_fresh(self) -> bool:
        """True when TF can still place the robot, recently enough to act on."""
        try:
            tf = self._tf_buffer.lookup_transform(
                NAV_FRAME, BASE_FRAME, Time(), timeout=Duration(seconds=0.05)
            )
        except TransformException:
            return False

        stamp = Time.from_msg(tf.header.stamp)
        if stamp.nanoseconds == 0:
            return True  # static transform, it has no age to check
        return self._is_fresh(stamp)

    def _stale_sensors(self) -> list[str]:
        """Names of the readings that have gone quiet. Empty is good."""
        stale = []
        if not self._pose_tf_fresh():
            stale.append('pose')
        if not self._is_fresh(self.battery_stamp):
            stale.append('battery')
        if not self._is_fresh(self.estop_stamp):
            stale.append('e-stop')
        return stale

    # =========================================================================
    # STARTUP
    # =========================================================================

    def _validate_config(self) -> None:
        """
        Check the hardcoded tables before anything can move.

        A bad row or dock entry does not fail loudly at run time — it produces a
        goal that looks valid and drives the robot somewhere wrong. Catching it
        at boot is the only cheap point.
        """
        problems: list[str] = []
        warnings: list[str] = []

        if not ROWS:
            problems.append('ROWS is empty — there is nothing to harvest')

        for index, row in enumerate(ROWS):
            _, sx, sy, _, ex, ey = row
            if math.isclose(sx, ex, abs_tol=1e-6) and math.isclose(sy, ey, abs_tol=1e-6):
                problems.append(
                    f'ROWS[{index}] starts and ends at the same point '
                    f'({sx:.3f}, {sy:.3f}) — its heading cannot be computed'
                )

        if not DOCK_CONFIGS:
            problems.append('DOCK_CONFIGS is empty — the robot cannot dock or undock')

        for name in (CHARGING_DOCK_NAME, UNLOADING_DOCK_NAME):
            if name not in DOCK_CONFIGS:
                problems.append(f"DOCK_CONFIGS has no entry named '{name}'")

        docks = list(DOCK_CONFIGS.values())
        for i, first in enumerate(docks):
            for second in docks[i + 1:]:
                if (math.isclose(first.pose.x, second.pose.x, abs_tol=1e-6)
                        and math.isclose(first.pose.y, second.pose.y, abs_tol=1e-6)):
                    warnings.append(
                        f"docks '{first.instance_name}' and '{second.instance_name}' "
                        f'share the same position ({first.pose.x:.3f}, {first.pose.y:.3f}) '
                        '— the nearest-dock choice between them is arbitrary'
                    )

        for message in warnings:
            self.get_logger().warning(f'Config warning: {message}')

        if problems:
            for message in problems:
                self.get_logger().error(f'Config error: {message}')
            self.get_logger().error(
                'Configuration is not safe to drive with — halting before startup'
            )
            self._transition_status(RobotStatusEnum.ABNORMAL)
            return

        self.get_logger().info(
            f'Config checked: {len(ROWS)} rows, {len(DOCK_CONFIGS)} docks'
        )

    def _action_servers(self) -> dict:
        """
        Every action server this node sends goals to, keyed by the name used in
        logs. Simulated hardware is left out — nothing is sent to it.
        """
        servers = {
            'navigation (NavigateThroughPoses)': self.navigation.client,
            'docking (dock_robot)': self.docking_action_client.client,
            'undocking (undock_robot)': self.undocking_action_client.client,
            'arm (manipulator_action_server)': self.manipulator_client.client,
        }
        if not SIMULATE_UNLOADER:
            servers['unloader (operate_unloader)'] = self.unloader_client.client
        return servers

    def _missing_servers(self) -> list[str]:
        """Names of the action servers that are not up. Empty is good.

        server_is_ready() answers from the discovery graph without blocking,
        unlike wait_for_server(), which parks the whole monitor loop for its
        timeout while it waits.
        """
        return [
            name for name, client in self._action_servers().items()
            if not client.server_is_ready()
        ]

    def _require_server(self, name: str, client) -> bool:
        """
        Confirm a server is there before sending it anything.

        A missing action server is not a transient fault. Retrying spent the
        whole recovery budget re-sending goals into nothing, and moved the arm
        through a full stow on every attempt. It also blocked the monitor loop
        for the client's wait_for_server timeout each time, which made the
        sensor freshness check report pose, battery and e-stop as stale when
        they were arriving normally.
        """
        if client.server_is_ready():
            return True

        self.get_logger().error(
            f'{name} is not running. Retrying will not bring it back, so the node '
            'is stopping here. Start that server, then restart this node.'
        )
        self._transition_status(RobotStatusEnum.ABNORMAL)
        return False

    def _initial_check_timer(self) -> None:
        """
        Fires until every precondition for driving is satisfied, then runs the
        dock check, builds DriveClient, and cancels itself.

        Battery is one of the gates. Without it the first task pick reads a
        default 0% and sends the robot to the charger, where it waits for a full
        charge that a silent BMS never reports. The action servers are another:
        starting a run without them just burns the recovery budget on goals that
        cannot be delivered.
        """
        if self.current_status == RobotStatusEnum.ABNORMAL:
            return

        self.init_ticks += 1
        elapsed = self.init_ticks * TIMER_PERIOD_SEC

        waiting_for: list[str] = []
        if not self._tf_buffer.can_transform(NAV_FRAME, BASE_FRAME, Time()):
            waiting_for.append(f'robot pose TF ({NAV_FRAME} -> {BASE_FRAME})')
        if self.battery_status is None:
            waiting_for.append(f"battery state (topic '{TOPIC_BATTERY}')")
        if self.estop_status is None:
            waiting_for.append(f"emergency stop state (topic '{TOPIC_ESTOP}')")
        if not self._tf_buffer.can_transform(BASE_FRAME, CAMERA_FRAME, Time()):
            waiting_for.append(f'camera TF ({BASE_FRAME} -> {CAMERA_FRAME})')

        missing = self._missing_servers()
        if missing:
            waiting_for.append(f'action servers [{", ".join(missing)}]')

        if waiting_for:
            self._init_wait_or_give_up(elapsed, waiting_for)
            return

        # The dock check needs map->odom, which can lag camera TF at boot. Resolve
        # it before building anything so the timer can simply fire again.
        if not self._check_initial_position():
            self._init_wait_or_give_up(elapsed, [f'TF into the dock frame ({NAV_FRAME})'])
            return

        # bushrow_theta starts at the real row-0 heading rather than a placeholder.
        # set_bushrow_theta() still updates it per row before every scan()/resume().
        drive_config = DriveConfig(
            **STATIC_DRIVE_PARAMS,
            camera_frame=CAMERA_FRAME,
            odom_frame=ODOM_FRAME,
            bushrow_theta=_row_heading(ROWS[0]),
        )
        self.drive_client = DriveClient(self, drive_config)

        self.init_check_timer.cancel()
        self.is_initialized = True
        self.get_logger().info(
            f'Startup checks passed in {elapsed:.1f}s — pose, battery, e-stop and '
            'camera TF all present, DriveClient built'
        )

    def _init_wait_or_give_up(self, elapsed: float, waiting_for: list[str]) -> None:
        """Log what startup is still missing, or halt once the budget is spent."""
        missing = ', '.join(waiting_for)

        if elapsed > TIMEOUT_INIT_SEC:
            self.get_logger().error(
                f'Startup gave up after {elapsed:.0f}s. Still missing: {missing}. '
                'The robot will not move — check those topics and the TF tree.'
            )
            self.init_check_timer.cancel()
            self._transition_status(RobotStatusEnum.ABNORMAL)
            return

        self.get_logger().info(
            f'Startup waiting for: {missing} ({elapsed:.0f}s of '
            f'{TIMEOUT_INIT_SEC:.0f}s)',
            throttle_duration_sec=2.0,
        )

    def _robot_pose_in_frame(self, target_frame: str) -> tuple[float, float, float] | None:
        """
        Robot pose (x, y, yaw) in `target_frame`, read from TF.

        One lookup, no odometry message involved. Everything this node compares
        against — dock poses, row coordinates, navigation goals — lives in the
        map frame, and TF is what knows where base_link is inside it.

        Returns None while TF cannot resolve the pair, so the caller can retry
        rather than compare two frames as though they were one.
        """
        target = target_frame.lstrip('/')
        try:
            # Time() asks for the latest available transform rather than one at a
            # specific stamp, which would be rejected as an extrapolation while
            # map->odom is still being filled in at boot.
            tf = self._tf_buffer.lookup_transform(
                target, BASE_FRAME, Time(), timeout=Duration(seconds=0.2)
            )
        except TransformException as e:
            self.get_logger().warning(
                f"TF lookup '{BASE_FRAME}' -> '{target}' failed: {e}",
                throttle_duration_sec=2.0,
            )
            return None

        return (
            tf.transform.translation.x,
            tf.transform.translation.y,
            _yaw_from_quaternion(tf.transform.rotation),
        )

    def _check_initial_position(self) -> bool:
        """
        Pick the nearest dock and decide whether the boot sequence must undock.

        False means TF was not ready — call again on the next tick. Every dock
        must resolve before deciding: picking a "nearest" dock from a partial set
        would silently select the wrong one at boot.
        """
        # Docks may not all declare the same frame, so resolve the robot once per
        # distinct frame and compare each dock inside its own frame.
        robot_xy_by_frame: dict[str, tuple[float, float]] = {}
        candidates: list[tuple[DockInstanceConfig, float]] = []

        for dock in DOCK_CONFIGS.values():
            frame = dock.frame.lstrip('/')
            if frame not in robot_xy_by_frame:
                robot_pose = self._robot_pose_in_frame(frame)
                if robot_pose is None:
                    self.get_logger().warning(
                        f"Dock '{dock.instance_name}' frame '{frame}' unresolved — "
                        'deferring position check',
                        throttle_duration_sec=2.0,
                    )
                    return False
                robot_xy_by_frame[frame] = robot_pose[:2]
            robot_x, robot_y = robot_xy_by_frame[frame]
            candidates.append((dock, math.hypot(robot_x - dock.pose.x, robot_y - dock.pose.y)))

        nearest_dock, nearest_dist = min(candidates, key=lambda candidate: candidate[1])
        robot_x, robot_y = robot_xy_by_frame[nearest_dock.frame.lstrip('/')]

        self.get_logger().info(
            f"Initial Robot position: ({robot_x:.3f}, {robot_y:.3f}) in "
            f"'{nearest_dock.frame}' | Nearest dock: '{nearest_dock.instance_name}' | "
            f'Distance to dock: {nearest_dist:.3f}m'
        )

        self.active_dock = nearest_dock
        self.is_at_docking_station = nearest_dist <= DOCKING_THRESHOLD_M

        if self.is_at_docking_station:
            self.get_logger().info('Robot at dock — will stow and undock before rows')
        else:
            self.get_logger().info('Robot away from dock — will stow before driving')

        return True

    def _handle_startup(self) -> None:
        """
        Drive the boot sequence. Both exits gate on a confirmed STOW first, so
        the robot never drives with the arm in an unknown configuration.

        At a dock:    IDLE -(STOW)-> START_UNDOCKING -> UNDOCKING -> DONE_UNDOCKING
        Away from it: IDLE -(STOW)-> startup complete, normal task execution
        """
        if self.current_status == RobotStatusEnum.IDLE:
            if not self._ensure_arm(ArmCommand.GO_STOW):
                return  # hold in IDLE until the arm is confirmed stowed

            if self.is_at_docking_station:
                self.get_logger().info('Arm stowed — starting startup undocking')
                self._transition_status(RobotStatusEnum.START_UNDOCKING)
            else:
                self.get_logger().info('Arm stowed — skipping undock, ready for rows')
                self.startup_undock_complete = True
            return

        if self.current_status == RobotStatusEnum.START_UNDOCKING:
            self._send_undock_goal('Startup undocking')
            return

        if self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info('Startup undocking complete')
            self.startup_undock_complete = True
            self._transition_status(RobotStatusEnum.IDLE)

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def timer_callback(self) -> None:
        """
        Main control loop.

        Order matters here. The emergency stop is evaluated before the
        initialisation gate, because "not started up yet" must never be a reason
        to ignore the stop button. Anything that can move the robot sits below
        both.

        Each tick:
          1. Build a RobotStatus from current sensor readings
          2. Emergency stop, then terminal states, then the init gate
          3. Route: ERROR -> recovery, boot incomplete -> startup,
             otherwise -> task execution
          4. Check the current state against its deadline
          5. Stamp final state and publish
        """
        robot_status = RobotStatus()
        robot_status.header.stamp = self.get_clock().now().to_msg()
        robot_status.robot_namespace = self.namespace.replace('/', '')
        self._set_sensor_fields(robot_status)

        if self.run_complete:
            self._publish_status(robot_status, 'Run complete')
            return

        if self.current_status == RobotStatusEnum.ABNORMAL:
            self._publish_status(robot_status, 'Halted — needs a person')
            return

        # --- Emergency stop, above everything that can move ------------------
        # An unknown stop state is not a clear one. Hold rather than declaring
        # EMERGENCY_STOP: that state exits through ERROR, and a topic that has
        # not spoken yet at boot is not a fault worth spending a recovery on.
        if self.estop_status is None:
            self._publish_status(robot_status, 'Waiting for emergency stop state')
            self.get_logger().warning(
                f"No message on '{TOPIC_ESTOP}' yet — holding until the emergency "
                'stop state is known',
                throttle_duration_sec=2.0,
            )
            return

        if self.estop_status.data:
            self._publish_status(robot_status, 'Emergency stop')
            if self.current_status != RobotStatusEnum.EMERGENCY_STOP:
                self.get_logger().error('EMERGENCY STOP pressed — stopping everything')
                self._cancel_active_motion()
                self._transition_status(RobotStatusEnum.EMERGENCY_STOP)
            return

        if self.current_status == RobotStatusEnum.EMERGENCY_STOP:
            self._resume_from_last_good_state('Emergency stop released')
            self._publish_status(robot_status, self._status_description())
            return

        # Nothing may advance before DriveClient exists and the dock check has run.
        if not self.is_initialized:
            self._publish_status(robot_status, 'Starting up')
            return

        # Past this point the robot is allowed to drive, so a sensor that has
        # gone quiet is a fault, not a wait.
        stale = self._stale_sensors()
        if stale and self.current_status != RobotStatusEnum.ERROR:
            self.get_logger().error(
                f'Lost {" and ".join(stale)} for more than {SENSOR_STALE_SEC:.0f}s '
                '— stopping'
            )
            self._transition_status(RobotStatusEnum.ERROR)

        if self.current_status == RobotStatusEnum.ERROR:
            self._handle_error_recovery()
        elif not self.startup_undock_complete:
            self._process_action_clients()
            self._handle_startup()
        else:
            self._handle_task_execution()

        self._check_state_deadline()
        self._publish_status(robot_status, self._status_description())

    def _resume_from_last_good_state(self, reason: str) -> None:
        """
        Drop everything in flight and restart the current task from the last
        point the robot is known to have reached.

        Row progress, load and which task was running are all kept — only the
        in-flight goals are thrown away. The arm is deliberately marked unknown:
        whatever was moving when the stop landed may have stopped anywhere, so
        it gets re-stowed before the robot drives again.
        """
        self.get_logger().info(f'{reason} — cancelling everything and restarting the task')

        self._cancel_active_motion()
        self._reset_all_clients()
        self._clear_goal_latches()

        self.last_confirmed_arm_command = ArmCommand.UNKNOWN
        self.arm_stow_pending = False
        self.arm_ready_pending = False

        if self.current_task_type is None:
            self.get_logger().info('No task was running — going idle')
            self._transition_status(RobotStatusEnum.IDLE)
            return

        target = self._next_nav_target()
        self.get_logger().info(
            f'Restarting {self.current_task_type.name} | row {self.row_index} | '
            f'returning to ({target.x:.3f}, {target.y:.3f})'
        )
        self._transition_status(RobotStatusEnum.JOB_START)

    def _check_state_deadline(self) -> None:
        """
        Fail a state that has waited longer than it ever should.

        Without this a dead action server or a client that stops reporting
        leaves the robot parked in a normal-looking state with nothing in the
        log to say it is stuck.
        """
        limit = self._state_deadline_sec()
        if limit is None:
            self.state_ticks = 0
            return

        self.state_ticks += 1
        waited = self.state_ticks * TIMER_PERIOD_SEC
        if waited <= limit:
            return

        self.get_logger().error(
            f'{self.current_status.name} has been waiting {waited:.0f}s '
            f'(limit {limit:.0f}s) — giving up on it'
        )
        self._transition_status(RobotStatusEnum.ERROR)

    def _state_deadline_sec(self) -> float | None:
        """How long the current state is allowed to wait, or None if it may wait."""
        status = self.current_status

        if status == RobotStatusEnum.MOVING:
            if self.motion_source == 'nav':
                return TIMEOUT_NAV_SEC
            if self.motion_source == 'align':
                return TIMEOUT_ALIGN_SEC
            return TIMEOUT_DRIVE_SEC

        # Covers both sources. The drive branch waits on odom/TF; the nav branch
        # can stall too, when a previous goal is somehow still active and the
        # send is skipped every tick.
        if status == RobotStatusEnum.START_MOVING:
            return TIMEOUT_DRIVE_READY_SEC

        return {
            RobotStatusEnum.DOCKING: TIMEOUT_DOCKING_SEC,
            RobotStatusEnum.UNDOCKING: TIMEOUT_UNDOCKING_SEC,
            RobotStatusEnum.HARVESTING: TIMEOUT_HARVEST_SEC,
        }.get(status)

    def _set_sensor_fields(self, robot_status: RobotStatus) -> None:
        pct = self._battery_pct()
        robot_status.battery_level = pct if pct is not None else 0.0
        robot_status.load_status = self.current_load_status

        # NavigationActionClient reads this back as the goal list's first waypoint
        # and stamps the goal 'map', so it must be published in NAV_FRAME. Raw
        # odometry here would put the start of every path in the wrong place.
        robot_pose = self._robot_pose_in_frame(NAV_FRAME)
        self.pose_valid = robot_pose is not None
        if robot_pose is None:
            return

        robot_x, robot_y, robot_yaw = robot_pose
        robot_status.topo_map_position.x = robot_x
        robot_status.topo_map_position.y = robot_y
        robot_status.topo_map_orientation.z = math.sin(robot_yaw / 2.0)
        robot_status.topo_map_orientation.w = math.cos(robot_yaw / 2.0)

    def _publish_status(self, robot_status: RobotStatus, description: str) -> None:
        # A status with no resolved position is worse than no status at all:
        # NavigationActionClient reads the position back as the start of the next
        # path, and an unset field reads as the map origin.
        if not self.pose_valid:
            self.get_logger().warning(
                f"Robot position not resolvable in '{NAV_FRAME}' — withholding "
                'status so no path is planned from the wrong place',
                throttle_duration_sec=2.0,
            )
            return

        robot_status.status = self.current_status.value
        robot_status.task = description
        robot_status.load_status = self.current_load_status
        self.robot_state_pub.publish(robot_status)

    def _status_description(self) -> str:
        """Human-readable sentence for RobotStatus.task, matching production's use
        of that field. The machine-readable state goes in RobotStatus.status."""
        if not self.startup_undock_complete:
            return 'Startup sequence'
        if self.current_task_type == TaskEnum.CHARGING_TASK:
            return f'Charging at {CHARGING_DOCK_NAME}'
        if self.current_task_type == TaskEnum.UNLOADING_TASK:
            return f'Unloading at {UNLOADING_DOCK_NAME}'
        if self.current_task_type == TaskEnum.HARVESTING_TASK:
            return f'Harvesting row {self.row_index}'
        return 'Idle'

    def _battery_pct(self) -> float | None:
        """Battery percentage, or None while nothing has reported one."""
        if self.battery_status is None:
            return None
        pct = self.battery_status.percentage
        return pct * 100.0 if pct <= 1.0 else pct

    def _battery_is_low(self) -> bool:
        """True only when the battery is *known* to be below the threshold.

        An unknown battery is not low — it is unknown, and the init gate already
        refuses to let the robot move until a real reading has arrived.
        """
        pct = self._battery_pct()
        return pct is not None and pct < BATTERY_LOW_THRESHOLD

    def _transition_status(self, new_status: RobotStatusEnum) -> None:
        if new_status == self.current_status:
            return
        self.previous_status = self.current_status
        self.current_status = new_status
        # Deadlines are per-state, so entering a new one starts a fresh clock.
        self.state_ticks = 0
        self.get_logger().info(
            f'Status: {self.previous_status.name} -> {self.current_status.name}'
        )

    # =========================================================================
    # ACTION CLIENT ARBITRATION
    # =========================================================================

    def _process_action_clients(self) -> None:
        """
        Poll the action clients and hand the tick to exactly one motion client,
        in priority order:

          1. NavigationActionClient — NavigateThroughPoses
          2. DockingActionClient    — dock_robot
          3. UndockingActionClient  — undock_robot
          4. ReverseDriveClient     — TF closed-loop reverse (undocking fallback)
          5. DriveClient            — camera-guided row driving

        No discriminator flag is needed between 1 and 5: DriveClient reports IDLE
        whenever it is not driving, so it can only win the funnel while active.

        The manipulator is polled separately because arm motion overlaps robot
        motion — its completion is what unblocks the motion states.
        """
        nav_status = self.navigation.get_navigation_status()
        dock_status = self.docking_action_client.get_status()
        undock_status = self.undocking_action_client.get_status()
        drive_active = False
        if self.drive_client is not None:
            drive_feedback = self.drive_client.get_status()
            drive_active = drive_feedback.status != drive_feedback.IDLE

        if nav_status != NavigationStatus.IDLE:
            self._handle_navigation()
        elif dock_status != RobotStatusEnum.IDLE:
            self._handle_docking()
        elif undock_status != RobotStatusEnum.IDLE:
            self._handle_undocking()
        elif self.reverse_drive_active:
            self._handle_reverse_drive()
        elif self.align_active:
            self._handle_align()
        elif drive_active:
            self._handle_drive()

        arm_harvest_active = (
            self.current_status == RobotStatusEnum.HARVESTING
            and not self.arm_stow_pending
            and not self.arm_ready_pending
        )
        if self.arm_stow_pending or self.arm_ready_pending or arm_harvest_active:
            self._handle_manipulator()

    def _cancel_active_motion(self) -> None:
        """
        Stop everything that can move the robot. Non-blocking.

        Docking and undocking are in this list because Nav2's dock server drives
        the base itself. Cancelling only navigation left the robot driving into
        or out of a dock with the state machine frozen. The manipulator is here
        for the same reason at the other end — an arm mid-motion keeps moving
        until its goal is cancelled.
        """
        stopped: list[str] = []

        if self.navigation.is_navigation_active():
            self.navigation.cancel_goal()
            stopped.append('navigation')
        if self.docking_action_client.get_status() != RobotStatusEnum.IDLE:
            self.docking_action_client.cancel_goal()
            stopped.append('docking')
        if self.undocking_action_client.get_status() != RobotStatusEnum.IDLE:
            self.undocking_action_client.cancel_goal()
            stopped.append('undocking')
        if self.reverse_drive_active:
            self.reverse_drive_client.cancel()
            stopped.append('reverse drive')
        if self.align_active:
            self._publish_align_cmd(0.0)
            self.align_active = False
            stopped.append('row alignment')
        if self.drive_client is not None and self.drive_client.is_active():
            self.drive_client.cancel()
            stopped.append('camera drive')
        if self.manipulator_client.get_status() not in (
            RobotStatusEnum.IDLE, RobotStatusEnum.DONE_HARVESTING
        ):
            self.manipulator_client.cancel_goal()
            stopped.append('arm')

        if stopped:
            self.get_logger().warning(f'Cancelling: {", ".join(stopped)}')
        else:
            self.get_logger().info('Nothing was moving')

    def _busy_motion_clients(self, starting: str) -> list[str]:
        """
        Names of motion clients that are not idle, ignoring the one about to
        start.

        Each client publishes cmd_vel from its own timer, so two of them active
        at once means two sets of velocity commands fighting over the base. The
        priority chain in _process_action_clients only decides which one this
        node *watches* — it does not stop the others from driving.
        """
        busy: list[str] = []

        if starting != 'nav' and self.navigation.get_navigation_status() != NavigationStatus.IDLE:
            busy.append('navigation')
        if starting != 'dock' and self.docking_action_client.get_status() != RobotStatusEnum.IDLE:
            busy.append('docking')
        if starting != 'undock' and self.undocking_action_client.get_status() != RobotStatusEnum.IDLE:
            busy.append('undocking')
        if starting != 'reverse' and self.reverse_drive_client.get_status() != ReverseDriveStatus.IDLE:
            busy.append('reverse drive')
        if starting != 'align' and self.align_active:
            busy.append('row alignment')
        if starting != 'drive' and self.drive_client is not None:
            feedback = self.drive_client.get_status()
            if feedback.status != feedback.IDLE:
                busy.append('camera drive')

        return busy

    def _may_start_motion(self, starting: str) -> bool:
        """Refuse to start a second mover while another one is still running."""
        busy = self._busy_motion_clients(starting)
        if not busy:
            return True

        self.get_logger().error(
            f'Refusing to start {starting} — {" and ".join(busy)} still running'
        )
        return False

    def _reset_all_clients(self) -> None:
        """Return every action client to idle. Local state only."""
        self.navigation.reset()
        self.docking_action_client.reset()
        self.undocking_action_client.reset()
        self.manipulator_client.reset()
        self.unloader_client.reset()
        self.reverse_drive_client.reset()
        if self.drive_client is not None:
            self.drive_client.reset()
        self.reverse_drive_active = False

    def _clear_goal_latches(self) -> None:
        """Forget which goals were sent so the next attempt starts clean."""
        self.nav_goal_sent = False
        self.dock_goal_sent = False
        self.undock_goal_sent = False
        self.unload_goal_sent = False
        self.unload_step = None
        self.motion_source = None
        self.pending_move = None
        self.nav_retry_wait_ticks = 0
        self.dock_retry_wait_ticks = 0
        self.align_active = False
        self.align_target_theta = None
        self.sim_harvest_ticks = 0
        self.sim_unloader_ticks = 0
        self.unload_step_ticks = 0
        self.navigation_retry_count = 0
        self.docking_retry_count = 0
        self.harvest_retry_count = 0

    # =========================================================================
    # ACTION CLIENT HANDLERS
    # =========================================================================

    def _handle_navigation(self) -> None:
        """Map NavigationStatus onto the shared MOVING / DESTINATION_REACHED pair."""
        if self.motion_source != 'nav':
            return

        # Backoff between attempts. The client is deliberately left un-reset while
        # this counts down, so its status stays non-idle and this handler keeps
        # being reached. Re-sending a goal the instant Nav2 rejected it just
        # spends the whole retry budget in under a second.
        if self.nav_retry_wait_ticks > 0:
            self.nav_retry_wait_ticks -= 1
            if self.nav_retry_wait_ticks == 0:
                self.navigation.reset()
                self.nav_goal_sent = False
                self.get_logger().info('Retrying navigation now')
                self._transition_status(RobotStatusEnum.START_MOVING)
            return

        status = self.navigation.get_navigation_status()

        if status == NavigationStatus.SUCCEEDED:
            self.navigation.reset()
            self.nav_goal_sent = False
            self.navigation_retry_count = 0
            self._transition_status(RobotStatusEnum.DESTINATION_REACHED)
            return

        if status in (
            NavigationStatus.SENDING,
            NavigationStatus.ACCEPTED,
            NavigationStatus.ACTIVE,
        ):
            return  # still in flight

        if status == NavigationStatus.CANCELED:
            # Somebody else stopped this goal. Previously this fell through every
            # branch and returned, leaving the node in MOVING for good with
            # nothing in the log.
            self.get_logger().warning(
                'Navigation goal was cancelled from outside — dropping it and '
                'restarting the task'
            )
            self.navigation.reset()
            self.nav_goal_sent = False
            self._transition_status(RobotStatusEnum.IDLE)
            return

        if status not in (
            NavigationStatus.ABORTED,
            NavigationStatus.ERROR,
            NavigationStatus.FAILED,
            NavigationStatus.REJECTED,
        ):
            self.get_logger().error(
                f'Navigation reported {status.name}, which this node has no rule '
                'for — treating it as a failure rather than waiting forever'
            )
            self.navigation.reset()
            self.nav_goal_sent = False
            self._transition_status(RobotStatusEnum.ERROR)
            return

        if self._is_at_nav_target():
            self.get_logger().info(
                f'Navigation {status.name} but within position and heading tolerance — treating as arrived'
            )
            self.navigation.reset()
            self.nav_goal_sent = False
            self.navigation_retry_count = 0
            self._transition_status(RobotStatusEnum.DESTINATION_REACHED)
            return

        if self.navigation_retry_count < MAX_NAVIGATION_RETRIES:
            self.navigation_retry_count += 1
            self.nav_retry_wait_ticks = max(1, int(RETRY_DELAY_SEC / TIMER_PERIOD_SEC))
            self.get_logger().warning(
                f'Navigation {status.name} — waiting {RETRY_DELAY_SEC:.0f}s then '
                f'retrying ({self.navigation_retry_count}/{MAX_NAVIGATION_RETRIES})'
            )
            return

        self.navigation.reset()
        self.nav_goal_sent = False
        self.get_logger().error(f'Navigation failed after {MAX_NAVIGATION_RETRIES} retries')
        self._transition_status(RobotStatusEnum.ERROR)

    def _is_at_nav_target(self) -> bool:
        """
        True when an ABORTED goal actually left the robot where it was asked to be.

        Position alone is not enough: a goal that is mostly a rotation (see the
        node 2 == node 3 TODO above) can end within centimetres of the target
        while pointing the wrong way, and SCANNING drives along the robot's
        current heading. Heading is checked too.
        """
        if not self.nav_target_wp:
            return False

        # Goal poses are stamped NAV_FRAME by NavigationActionClient, so the robot
        # has to be expressed there too — raw odometry would measure the wrong gap.
        robot_pose = self._robot_pose_in_frame(NAV_FRAME)
        if robot_pose is None:
            return False

        robot_x, robot_y, robot_yaw = robot_pose
        dist = math.hypot(robot_x - self.nav_target_wp.x, robot_y - self.nav_target_wp.y)
        if dist > NAV_CLOSE_ENOUGH_M:
            return False

        heading_error = abs(_normalize_angle(robot_yaw - self.nav_target_wp.theta))
        if heading_error > NAV_HEADING_TOL_RAD:
            self.get_logger().warning(
                f'Within {dist:.3f}m of target but heading is off by '
                f'{math.degrees(heading_error):.1f}deg — not treating as arrived'
            )
            return False

        return True

    def _handle_drive(self) -> None:
        """
        Map DriveClient feedback onto the same MOVING / DESTINATION_REACHED pair
        that Nav2 uses. Only acts while MOVING is owned by the drive client —
        DriveClient sits in STOPPED through the whole harvest, so without this
        guard the same stop would be handled on every tick.
        """
        if self.current_status != RobotStatusEnum.MOVING or self.motion_source != 'drive':
            return

        feedback = self.drive_client.get_status()

        if feedback.status in (feedback.CANCELED, feedback.ERROR):
            self.get_logger().error(f'DriveClient reported status {feedback.status} during row traversal')
            self._transition_status(RobotStatusEnum.ERROR)
            return

        if feedback.status == feedback.ABORTED:
            # ABORTED is DriveClient's own retry wait, not a failure: it returns to
            # CONTROLLING after controlling_retry_delay and only escalates to ERROR
            # once max_controlling_retries is spent. Treating it as fatal here would
            # throw away every configured retry.
            self.get_logger().info('DriveClient ABORTED — internal retry in progress', throttle_duration_sec=5.0)
            return

        if feedback.status != feedback.STOPPED:
            return  # SCANNING / CONTROLLING / DEPARTING — keep waiting

        if feedback.harvested_count == self.park_count_snapshot:
            # Nothing new found since the snapshot — the row is finished.
            # harvested_count is cumulative across rows, which is why the
            # comparison is against a per-move snapshot rather than zero.
            self.get_logger().info('No detection — row over')
            self.drive_client.reset()
            self.row_over = True
            self._transition_status(RobotStatusEnum.DONE_HARVESTING)
            return

        self.bushes_visited += 1
        self.harvest_retry_count = 0

        # Row progress is recorded here, at the park, rather than after the
        # harvest completes. An interrupt between parking and finishing the cut
        # would otherwise resume from the row start and re-drive bushes already
        # covered.
        parked_pose = self._robot_pose_in_frame(NAV_FRAME)
        if parked_pose is not None:
            self.row_progress_pose = parked_pose
        else:
            self.get_logger().warning(
                f"Parked at a bush but could not work out where in '{NAV_FRAME}' — "
                'if this row is interrupted it will restart from the beginning'
            )

        self.get_logger().info(f'Parked at bush {self.bushes_visited} of this run')
        self._transition_status(RobotStatusEnum.DESTINATION_REACHED)

    def _handle_docking(self) -> None:
        # Backoff between attempts, same reasoning as navigation: the client is
        # left un-reset so this handler keeps being reached while it counts down.
        if self.dock_retry_wait_ticks > 0:
            self.dock_retry_wait_ticks -= 1
            if self.dock_retry_wait_ticks == 0:
                self.docking_action_client.reset()
                self.dock_goal_sent = False
                self.get_logger().info('Retrying docking now')
                self._transition_status(RobotStatusEnum.START_DOCKING)
            return

        status = self.docking_action_client.get_status()

        if status == RobotStatusEnum.DONE_DOCKING:
            self.docking_action_client.reset()
            self.dock_goal_sent = False
            self.docking_retry_count = 0
            self._transition_status(RobotStatusEnum.DONE_DOCKING)
            return

        if status in (RobotStatusEnum.START_DOCKING, RobotStatusEnum.DOCKING):
            self._transition_status(RobotStatusEnum.DOCKING)
            return

        if status == RobotStatusEnum.IDLE:
            return

        if status != RobotStatusEnum.ERROR:
            self.get_logger().error(
                f'Docking reported {status.name}, which this node has no rule for '
                '— treating it as a failure rather than waiting forever'
            )
            self.docking_action_client.reset()
            self.dock_goal_sent = False
            self._transition_status(RobotStatusEnum.ERROR)
            return

        if self.docking_retry_count < MAX_DOCKING_RETRIES:
            self.docking_retry_count += 1
            self.dock_retry_wait_ticks = max(1, int(RETRY_DELAY_SEC / TIMER_PERIOD_SEC))
            self.get_logger().warning(
                f'Docking failed — waiting {RETRY_DELAY_SEC:.0f}s then retrying '
                f'({self.docking_retry_count}/{MAX_DOCKING_RETRIES})'
            )
            return

        self.docking_action_client.reset()
        self.dock_goal_sent = False
        self.get_logger().error(f'Docking failed after {MAX_DOCKING_RETRIES} retries')
        self._transition_status(RobotStatusEnum.ERROR)

    def _handle_undocking(self) -> None:
        status = self.undocking_action_client.get_status()

        if status == RobotStatusEnum.DONE_UNDOCKING:
            self.undocking_action_client.reset()
            self.undock_goal_sent = False
            self._transition_status(RobotStatusEnum.DONE_UNDOCKING)
            return

        if status in (RobotStatusEnum.START_UNDOCKING, RobotStatusEnum.UNDOCKING):
            self._transition_status(RobotStatusEnum.UNDOCKING)
            return

        if status == RobotStatusEnum.IDLE:
            return

        if status != RobotStatusEnum.ERROR:
            self.get_logger().error(
                f'Undocking reported {status.name}, which this node has no rule '
                'for — treating it as a failure rather than waiting forever'
            )
            self.undocking_action_client.reset()
            self.undock_goal_sent = False
            self._transition_status(RobotStatusEnum.ERROR)
            return

        self.undocking_action_client.reset()
        self.undock_goal_sent = False
        self._start_reverse_drive('Undocking action failed')

    def _start_reverse_drive(self, reason: str) -> None:
        """Undocking did not work — back out using the TF closed-loop drive instead."""
        dock_name = self.active_dock.instance_name if self.active_dock else ''

        if not self._may_start_motion('reverse'):
            self._transition_status(RobotStatusEnum.ERROR)
            return

        self.get_logger().warning(
            f"{reason} — reversing to the '{dock_name}' staging pose instead"
        )

        if not self.reverse_drive_client.drive_to_staging(dock_name):
            self.get_logger().error(
                f"Reverse drive refused to start for dock '{dock_name}' — "
                'the robot cannot get clear on its own'
            )
            self._transition_status(RobotStatusEnum.ERROR)
            return

        self.reverse_drive_active = True
        self._transition_status(RobotStatusEnum.UNDOCKING)

    def _handle_align(self) -> None:
        """
        Turn in place until the robot faces along the row, then hand over to the
        camera drive.

        Runs on the monitor tick rather than a timer of its own. At 5 Hz and
        ALIGN_ANGULAR_MAX the robot turns about 1.7 deg per tick, comfortably
        inside the 2.9 deg tolerance, so the coarse rate cannot overshoot the
        gate — and an emergency stop pre-empts the turn like any other motion.
        """
        if self.current_status != RobotStatusEnum.MOVING or self.motion_source != 'align':
            return

        if self.align_target_theta is None:
            self._publish_align_cmd(0.0)
            self.align_active = False
            self.get_logger().error('Asked to line up with the row but no heading was set')
            self._transition_status(RobotStatusEnum.ERROR)
            return

        robot_pose = self._robot_pose_in_frame(NAV_FRAME)
        if robot_pose is None:
            # Do not keep turning blind — hold still until TF comes back. The
            # MOVING deadline catches it if it never does.
            self._publish_align_cmd(0.0)
            return

        error = _normalize_angle(self.align_target_theta - robot_pose[2])

        if abs(error) <= ALIGN_ANGULAR_TOL_RAD:
            self._publish_align_cmd(0.0)
            self.align_active = False
            self.align_target_theta = None
            self.get_logger().info(
                f'Lined up with the row — facing {math.degrees(robot_pose[2]):.0f} degrees, '
                f'{math.degrees(abs(error)):.1f} degrees off'
            )
            self.motion_source = 'drive'
            self._transition_status(RobotStatusEnum.START_MOVING)
            return

        omega = max(-ALIGN_ANGULAR_MAX, min(ALIGN_ANGULAR_MAX, ALIGN_GAIN * error))
        # A command below the base's own deadband turns nothing at all, so the
        # last couple of degrees would never close.
        if abs(omega) < ALIGN_ANGULAR_MIN:
            omega = math.copysign(ALIGN_ANGULAR_MIN, omega)

        self._publish_align_cmd(omega)
        self.get_logger().info(
            f'Turning onto the row — {math.degrees(error):.0f} degrees to go',
            throttle_duration_sec=2.0,
        )

    def _publish_align_cmd(self, omega: float) -> None:
        """Rotation-only velocity command. No linear component, ever."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.twist.angular.z = omega
        self.align_cmd_pub.publish(msg)

    def _handle_reverse_drive(self) -> None:
        status = self.reverse_drive_client.get_status()

        if status == ReverseDriveStatus.DONE:
            self.get_logger().info('Reverse drive complete')
            self.reverse_drive_client.reset()
            self.reverse_drive_active = False
            self._transition_status(RobotStatusEnum.DONE_UNDOCKING)
            return

        if status in (ReverseDriveStatus.ERROR, ReverseDriveStatus.CANCELED):
            self.reverse_drive_client.reset()
            self.reverse_drive_active = False
            self.get_logger().error('Reverse drive failed')
            self._transition_status(RobotStatusEnum.ERROR)

    def _handle_manipulator(self) -> None:
        """
        Resolve arm goals. The manipulator reports DONE_HARVESTING for every
        command it completes — STOW, READY and START_HARVEST alike — so the
        pending flags are what say which one just finished.
        """
        # STOW and READY always go to the real arm, so the gate that stops the
        # robot driving with the arm out reflects where the arm actually is.
        if self.arm_stow_pending:
            self._resolve_arm_gate(self.manipulator_client.get_status(), ArmCommand.GO_STOW)
            return

        if self.arm_ready_pending:
            self._resolve_arm_gate(self.manipulator_client.get_status(), ArmCommand.GO_READY)
            return

        if self.current_status != RobotStatusEnum.HARVESTING:
            return

        status = (
            self._sim_harvest_status() if SIMULATE_HARVEST
            else self.manipulator_client.get_status()
        )

        if status == RobotStatusEnum.ERROR:
            self.manipulator_client.reset()
            if self.harvest_retry_count < MAX_HARVEST_RETRIES:
                self.harvest_retry_count += 1
                self.get_logger().warning(
                    f'Harvest failed — retrying ({self.harvest_retry_count}/{MAX_HARVEST_RETRIES})'
                )
                self._transition_status(RobotStatusEnum.START_HARVESTING)
                return
            self.get_logger().error(f'Harvest failed after {MAX_HARVEST_RETRIES} retries')
            self._transition_status(RobotStatusEnum.ERROR)
            return

        if status != RobotStatusEnum.DONE_HARVESTING:
            return  # still in flight

        if not SIMULATE_HARVEST:
            self.manipulator_client.reset()

        self.bushes_harvested += 1
        # SIMULATED: nothing weighs the bin. This assumed increment is what
        # triggers the unload interrupt.
        self.current_load_status = min(self.current_load_status + LOAD_INCREMENT, 100.0)

        # A finished bush is real progress, so the recovery budget starts over.
        self._note_progress()

        self.get_logger().info(
            f'Harvest done | {self.bushes_harvested} bushes cut so far | '
            f'load now {self.current_load_status:.0f}% (estimated, nothing weighs the bin)'
        )
        self._transition_status(RobotStatusEnum.DONE_HARVESTING)

    def _sim_harvest_status(self) -> RobotStatusEnum:
        """
        Stand in for the cut cycle when no START_HARVEST goal was sent.

        Counted in monitor ticks rather than on a timer of its own. A separate
        ROS timer would keep running through an emergency stop and then report a
        finished harvest from whatever state the robot had moved into; counting
        ticks here means the stop pre-empts the wait by construction.
        """
        self.sim_harvest_ticks += 1
        remaining = SIM_HARVEST_SEC - self.sim_harvest_ticks * TIMER_PERIOD_SEC

        if remaining > 0:
            self.get_logger().info(
                f'Pretending to harvest — {remaining:.0f}s left',
                throttle_duration_sec=5.0,
            )
            return RobotStatusEnum.HARVESTING

        self.sim_harvest_ticks = 0
        return RobotStatusEnum.DONE_HARVESTING

    def _note_progress(self) -> None:
        """
        Clear the recovery budget after something actually went right.

        The budget exists to stop the node driving into the same fault forever.
        Counting it across a whole farm run instead would halt a healthy robot
        after three unrelated hiccups hours apart.
        """
        if self.recovery_attempts:
            self.get_logger().info(
                f'Progress made — clearing {self.recovery_attempts} earlier '
                'recovery attempt(s)'
            )
        self.recovery_attempts = 0

    def _resolve_arm_gate(self, status: RobotStatusEnum, target: str) -> None:
        if status == RobotStatusEnum.DONE_HARVESTING:
            self.manipulator_client.reset()
            self.last_confirmed_arm_command = target
            self.arm_stow_pending = False
            self.arm_ready_pending = False
            self.get_logger().info(f'Arm {target} confirmed')
            return

        if status == RobotStatusEnum.ERROR:
            self.manipulator_client.reset()
            self.arm_stow_pending = False
            self.arm_ready_pending = False
            self.get_logger().error(f'Arm {target} goal failed')
            self._transition_status(RobotStatusEnum.ERROR)

    def _ensure_arm(self, target: str) -> bool:
        """
        Idempotent arm gate. Returns True only when `target` is confirmed.

        Safe to call on every tick: it sends one goal, then reports False until
        _handle_manipulator resolves it.
        """
        if self.last_confirmed_arm_command == target:
            return True

        if self.arm_stow_pending or self.arm_ready_pending:
            return False  # already sent, waiting on _handle_manipulator

        if not self._require_server(
            'the arm (manipulator_action_server)', self.manipulator_client.client
        ):
            return False

        # An arm gate is not part of the task a server would send — it is this
        # node's own safety move before driving — so it gets its own subtask
        # rather than borrowing the task's harvesting one.
        subtask = self._new_subtask(SubTask.HARVESTING, f'Arm gate: {target}')

        if target == ArmCommand.GO_STOW:
            sent = self.manipulator_client.send_stow_goal(subtask)
            self.arm_stow_pending = sent
        else:
            sent = self.manipulator_client.send_ready_goal(subtask)
            self.arm_ready_pending = sent

        if not sent:
            self.get_logger().error(
                f"Could not ask the arm to '{target}' — is the manipulator action "
                'server running? Not moving until the arm answers.'
            )
            self._transition_status(RobotStatusEnum.ERROR)
            return False

        self.get_logger().info(f"Asked the arm to '{target}' — waiting for it")
        return False

    # =========================================================================
    # TASK EXECUTION
    # =========================================================================

    def _handle_task_execution(self) -> None:
        """
        Run after the boot sequence finishes.

          1. Poll the action clients so async results land on the tick they arrive
          2. IDLE / JOB_DONE -> pick the next task
          3. anything else   -> run the current task's handler
        """
        self._process_action_clients()

        if self.current_status in (RobotStatusEnum.IDLE, RobotStatusEnum.JOB_DONE):
            self._handle_task_start()
            return

        handler = {
            TaskEnum.HARVESTING_TASK: self._task_harvesting,
            TaskEnum.CHARGING_TASK: self._task_charging,
            TaskEnum.UNLOADING_TASK: self._task_unloading,
        }.get(self.current_task_type)

        if handler:
            handler()
        else:
            self.get_logger().warning(f'No handler for task type: {self.current_task_type}')

    def _handle_task_start(self) -> None:
        """
        Pick the next task. This is the internal sequencer that stands in for the
        JobPublisher the production node subscribes to.

        Priority: low battery, then full load, then whatever the farm needs next.
        """
        if self.farm_exhausted and self.current_task_type == TaskEnum.UNLOADING_TASK:
            self._finish_run()
            return

        if self._battery_is_low():
            self.get_logger().warning(
                f'Battery down to {self._battery_pct():.0f}% — going to charge first'
            )
            self._start_task(TaskEnum.CHARGING_TASK)
            return

        if self.current_load_status >= 100.0:
            self.get_logger().warning('Load full (simulated) — unloading before continuing')
            self._start_task(TaskEnum.UNLOADING_TASK)
            return

        if self.farm_exhausted:
            self.get_logger().info('All rows attempted — final unload')
            self._start_task(TaskEnum.UNLOADING_TASK)
            return

        self._start_task(TaskEnum.HARVESTING_TASK)

    def _start_task(self, task_type: TaskEnum) -> None:
        self.current_task_type = task_type
        self.current_task = self._build_task(task_type)
        self.docking_retry_count = 0
        self.navigation_retry_count = 0

        steps = ' then '.join(
            SUB_TASK_NAMES.get(sub.type, str(sub.type))
            for sub in self.current_task.sub_tasks
        )
        self.get_logger().info(
            f'Task {self.current_task.task_id}: {self.current_task.description} '
            f'({steps})'
        )
        self._transition_status(RobotStatusEnum.JOB_START)

    def _build_task(self, task_type: TaskEnum) -> Task:
        """
        Compose the Task to run next.

        This is the stand-in for the task server. When tasks start arriving from
        outside, this method is what a subscription replaces — everything
        downstream already reads the Task message rather than making up its own
        goals.
        """
        task_id = self.next_task_id
        self.next_task_id += 1

        if task_type == TaskEnum.HARVESTING_TASK:
            row = ROWS[self.row_index]
            sub_tasks = [
                self._new_subtask(
                    SubTask.MOVING,
                    f'Drive to row {self.row_index}',
                    waypoints=self._nav_waypoints(self._next_nav_target()),
                ),
                self._new_subtask(SubTask.HARVESTING, f'Harvest the bushes in row {self.row_index}'),
            ]
            return _make_task(
                task_id, task_type, f'Harvest row {self.row_index}',
                sub_tasks, self.current_load_status, target_node_id=row[0],
            )

        charging = task_type == TaskEnum.CHARGING_TASK
        dock_name = CHARGING_DOCK_NAME if charging else UNLOADING_DOCK_NAME
        dock = DOCK_CONFIGS[dock_name]
        middle_type = SubTask.CHARGING if charging else SubTask.UNLOADING
        description = 'Charge the battery' if charging else 'Empty the bin'

        sub_tasks = [
            self._new_subtask(
                SubTask.DOCKING, f'Dock at {dock_name}',
                dock_goal=_make_dock_goal(dock_name),
            ),
            self._new_subtask(middle_type, description),
            self._new_subtask(
                SubTask.UNDOCKING, f'Undock from {dock_name}',
                undock_goal=_make_undock_goal(dock.type, _max_undocking_time()),
            ),
        ]
        return _make_task(
            task_id, task_type, description, sub_tasks, self.current_load_status,
        )

    def _new_subtask(
        self,
        sub_task_type: int,
        description: str,
        waypoints: list[WayPoint] | None = None,
        dock_goal: DockGoal | None = None,
        undock_goal: UndockGoal | None = None,
    ) -> SubTask:
        """Build a SubTask with the next id in sequence."""
        sub_task_id = self.next_sub_task_id
        self.next_sub_task_id += 1
        return _make_subtask(
            sub_task_id, sub_task_type, description, waypoints, dock_goal, undock_goal
        )

    def _subtask_of(self, sub_task_type: int) -> SubTask | None:
        """The current task's subtask of that type, or None if there isn't one."""
        if self.current_task is None:
            return None
        for subtask in self.current_task.sub_tasks:
            if subtask.type == sub_task_type:
                return subtask
        return None

    def _nav_waypoints(self, target: WayPoint) -> list[WayPoint]:
        """
        Two-waypoint list for a navigation subtask.

        NavigationActionClient throws away index 0's x/y and substitutes the
        robot's live position, so only the last entry has to be a real target.
        Its heading is copied onto the placeholder so both agree.
        """
        return [_make_waypoint(-1, 0.0, 0.0, target.theta), target]

    def _finish_run(self) -> None:
        self.get_logger().info(
            '\n══════════════════════════════════════════════\n'
            '  Farm run complete\n'
            f'  bushes_visited={self.bushes_visited} bushes_harvested={self.bushes_harvested}\n'
            '══════════════════════════════════════════════'
        )
        self.run_complete = True
        self._transition_status(RobotStatusEnum.JOB_DONE)
        rclpy.shutdown()

    # =========================================================================
    # TASK: HARVESTING
    # =========================================================================

    def _task_harvesting(self) -> None:
        """
        One row, bush by bush.

          JOB_START           -> nav to the row start (or back to the last bush)
          START_MOVING        -> issue the Nav2 goal, or scan()/resume()
          MOVING              -> owned by _handle_navigation / _handle_drive
          DESTINATION_REACHED -> 'nav': arm READY then scan()
                                 'drive': parked at a bush, harvest it
          START_HARVESTING    -> send START_HARVEST
          HARVESTING          -> owned by _handle_manipulator
          DONE_HARVESTING     -> stow and end the row, or move to the next bush
        """
        if self.current_status == RobotStatusEnum.JOB_START:
            # Nav2 drives the base, so the arm must be stowed before it starts —
            # the same gate charging and unloading already use. It used to hold
            # here only by accident of the order the states happen to run in.
            if not self._ensure_arm(ArmCommand.GO_STOW):
                return

            self.row_over = False
            self.motion_source = 'nav'
            self.nav_target_wp = self._next_nav_target()
            self.nav_goal_sent = False
            resuming = 'resuming at' if self.row_progress_pose is not None else 'starting at'
            self.get_logger().info(
                f'Row {self.row_index} — {resuming} '
                f'({self.nav_target_wp.x:.3f}, {self.nav_target_wp.y:.3f}), '
                f'facing {math.degrees(self.nav_target_wp.theta):.0f} degrees'
            )
            self._transition_status(RobotStatusEnum.START_MOVING)
            return

        if self.current_status == RobotStatusEnum.START_MOVING:
            self._start_move()
            return

        if self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            if not self._ensure_arm(ArmCommand.GO_READY):
                return

            if self.motion_source == 'nav':
                # At the row start. Set the row heading before the first scan —
                # the single choke point every path funnels through.
                row_theta = _row_heading(ROWS[self.row_index])
                self.drive_client.set_bushrow_theta(row_theta)
                # Turn onto that heading before scanning. Nav2 arrives on
                # position, and SCANNING drives straight along whatever heading
                # the robot is left holding.
                self.align_target_theta = row_theta
                self.pending_move = 'scan'
                self.motion_source = 'align'
                self._transition_status(RobotStatusEnum.START_MOVING)
                return

            self._transition_status(RobotStatusEnum.START_HARVESTING)
            return

        if self.current_status == RobotStatusEnum.START_HARVESTING:
            if SIMULATE_HARVEST:
                self.sim_harvest_ticks = 0
                self.get_logger().warning(
                    f'No harvest goal sent (simulated) — pretending to cut for '
                    f'{SIM_HARVEST_SEC:.0f}s'
                )
                self._transition_status(RobotStatusEnum.HARVESTING)
                return

            if not self._require_server(
                'the arm (manipulator_action_server)', self.manipulator_client.client
            ):
                return

            subtask = self._subtask_of(SubTask.HARVESTING)
            if subtask is None:
                subtask = self._new_subtask(SubTask.HARVESTING, 'Executing harvest sequence')
            if not self.manipulator_client.send_harvesting_goal(subtask):
                self.get_logger().error(
                    'Could not ask the arm to harvest — is the manipulator action '
                    'server running?'
                )
                self._transition_status(RobotStatusEnum.ERROR)
                return
            self._transition_status(RobotStatusEnum.HARVESTING)
            return

        if self.current_status == RobotStatusEnum.DONE_HARVESTING:
            self._after_harvest()

    def _start_move(self) -> None:
        """Issue whichever motion the current motion_source owns."""
        if self.motion_source == 'nav':
            if self.nav_goal_sent:
                return
            if self.navigation.is_navigation_active():
                self.get_logger().warning('Navigation send skipped — already active')
                return
            if not self._may_start_motion('nav'):
                self._transition_status(RobotStatusEnum.ERROR)
                return
            if not self._require_server(
                'Nav2 (NavigateThroughPoses)', self.navigation.client
            ):
                return
            if not self._send_nav_goal():
                self.get_logger().error(
                    'Could not send the navigation goal — is Nav2 running?'
                )
                self._transition_status(RobotStatusEnum.ERROR)
                return
            self.nav_goal_sent = True
            self._transition_status(RobotStatusEnum.MOVING)
            return

        if self.motion_source == 'align':
            if not self._may_start_motion('align'):
                self._transition_status(RobotStatusEnum.ERROR)
                return
            self.align_active = True
            self.get_logger().info(
                f'Turning to face along row {self.row_index} '
                f'({math.degrees(self.align_target_theta or 0.0):.0f} degrees)'
            )
            self._transition_status(RobotStatusEnum.MOVING)
            return

        not_ready: list[str] = []
        if not self.drive_client.is_ready():
            not_ready.append(f'robot pose TF ({ODOM_FRAME} -> {BASE_FRAME})')
        if not self.drive_client.is_camera_tf_ready():
            not_ready.append('camera TF')
        if not_ready:
            self.get_logger().info(
                f'Camera drive not ready yet — waiting for {" and ".join(not_ready)}',
                throttle_duration_sec=2.0,
            )
            return

        if not self._may_start_motion('drive'):
            self._transition_status(RobotStatusEnum.ERROR)
            return

        # Snapshot before starting: harvested_count is cumulative across rows, so
        # "did we find a new bush" is only meaningful against this baseline.
        self.park_count_snapshot = self.drive_client.get_status().harvested_count

        if self.pending_move == 'scan':
            self.drive_client.scan()
        else:
            # resume() departs and re-enters SCANNING on its own — do not also
            # call scan() here (drive.py ends DEPARTING with scan()).
            self.drive_client.resume()

        self._transition_status(RobotStatusEnum.MOVING)

    def _after_harvest(self) -> None:
        """Row over, interrupt needed, or straight on to the next bush."""
        if self.row_over:
            if not self._ensure_arm(ArmCommand.GO_STOW):
                return
            self._advance_row()
            return

        if self._battery_is_low() or self.current_load_status >= 100.0:
            if not self._ensure_arm(ArmCommand.GO_STOW):
                return
            why = 'battery low' if self._battery_is_low() else 'bin full'
            self.get_logger().info(f'Pausing this row — {why}')
            self._transition_status(RobotStatusEnum.JOB_DONE)
            return

        self.pending_move = 'resume'
        self.motion_source = 'drive'
        self._transition_status(RobotStatusEnum.START_MOVING)

    def _advance_row(self) -> None:
        next_index = self.row_index + 1
        self.row_progress_pose = None
        self.row_over = False

        # A finished row is real progress, so the recovery budget starts over.
        self._note_progress()

        if next_index < len(ROWS):
            self.row_index = next_index
            self.get_logger().info(
                f'Row {self.row_index - 1} finished — moving on to row {next_index} '
                f'of {len(ROWS) - 1}'
            )
        else:
            self.farm_exhausted = True
            self.get_logger().info('Last row finished — heading to the unloading dock')

        self._transition_status(RobotStatusEnum.JOB_DONE)

    def _next_nav_target(self) -> WayPoint:
        """Resume at the furthest point already covered in this row, otherwise
        start at the row's start point."""
        if self.row_progress_pose is not None:
            return _pose_to_waypoint(self.row_progress_pose)
        return _row_start_waypoint(ROWS[self.row_index])

    def _send_nav_goal(self) -> bool:
        # The MOVING subtask belongs to the current task. Its waypoints are
        # refreshed here rather than reused, because the target differs between
        # the first attempt, a retry, and a resume part-way down a row.
        subtask = self._subtask_of(SubTask.MOVING)
        if subtask is None:
            subtask = self._new_subtask(SubTask.MOVING, 'Navigate')
        subtask.data = self._nav_waypoints(self.nav_target_wp)

        task = _make_task(
            self.current_task.task_id if self.current_task else 0,
            self.current_task_type or TaskEnum.HARVESTING_TASK,
            subtask.description,
            [subtask],
            self.current_load_status,
            target_node_id=self.nav_target_wp.node_id,
        )
        return self.navigation.send_goal(task)

    # =========================================================================
    # TASK: CHARGING
    # =========================================================================

    def _task_charging(self) -> None:
        """
        JOB_START -(STOW)-> START_DOCKING -> DOCKING -> DONE_DOCKING
                  -> START_CHARGING -> CHARGING -> DONE_CHARGING
                  -> START_UNDOCKING -> UNDOCKING -> DONE_UNDOCKING -> JOB_DONE
        """
        if self.current_status == RobotStatusEnum.JOB_START:
            if not self._ensure_arm(ArmCommand.GO_STOW):
                return
            self.active_dock = DOCK_CONFIGS[CHARGING_DOCK_NAME]
            self.dock_goal_sent = False
            self._transition_status(RobotStatusEnum.START_DOCKING)
            return

        if self.current_status == RobotStatusEnum.START_DOCKING:
            self._send_dock_goal()
            return

        if self.current_status == RobotStatusEnum.DONE_DOCKING:
            self._transition_status(RobotStatusEnum.START_CHARGING)
            return

        if self.current_status == RobotStatusEnum.START_CHARGING:
            self.charge_last_pct = self._battery_pct()
            self.charge_stall_ticks = 0
            self.get_logger().info(
                f'Docked at the charger at {self._battery_pct():.0f}% — waiting for '
                f'{BATTERY_FULL_THRESHOLD:.0f}%'
            )
            self._transition_status(RobotStatusEnum.CHARGING)
            return

        if self.current_status == RobotStatusEnum.CHARGING:
            self._monitor_charging()
            return

        if self.current_status == RobotStatusEnum.DONE_CHARGING:
            self._note_progress()
            self.undock_goal_sent = False
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            return

        if self.current_status == RobotStatusEnum.START_UNDOCKING:
            self._send_undock_goal(f'Undock from {self.active_dock.instance_name}')
            return

        if self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info('Charged and clear of the dock — back to work')
            self._transition_status(RobotStatusEnum.JOB_DONE)

    def _monitor_charging(self) -> None:
        """
        Wait for a full battery, but only while the battery is actually filling.

        There is no fixed deadline here — the pack decides how long it takes.
        What is bounded is progress: a charger that is off, a bad contact or a
        wedged BMS all look identical from here, and none of them get better by
        waiting longer. The epsilon matters because the BMS reports fractional
        percent, so drifting noise must not be mistaken for charging.
        """
        pct = self._battery_pct()
        if pct is None:
            return  # staleness is handled in timer_callback

        if pct >= BATTERY_FULL_THRESHOLD:
            self.get_logger().info(f'Battery full at {pct:.1f}%')
            self._transition_status(RobotStatusEnum.DONE_CHARGING)
            return

        if self.charge_last_pct is None or pct > self.charge_last_pct + CHARGE_PROGRESS_EPSILON:
            self.charge_last_pct = pct
            self.charge_stall_ticks = 0
        else:
            self.charge_stall_ticks += 1

        stalled_for = self.charge_stall_ticks * TIMER_PERIOD_SEC
        if stalled_for > CHARGE_NO_PROGRESS_TIMEOUT_SEC:
            self.get_logger().error(
                f'Battery has sat at {pct:.1f}% for {stalled_for:.0f}s — it is not '
                'charging. Check the charger and the dock contacts.'
            )
            self._transition_status(RobotStatusEnum.ERROR)
            return

        self.get_logger().info(
            f'Charging: {pct:.1f}% of {BATTERY_FULL_THRESHOLD:.0f}%',
            throttle_duration_sec=10.0,
        )

    # =========================================================================
    # TASK: UNLOADING
    # =========================================================================

    def _task_unloading(self) -> None:
        """
        Same dock/undock chain as charging, with the carriage sequence in the
        middle. END -> wait -> HOME is three steps inside one UNLOADING state,
        which RobotStatusEnum cannot express, so unload_step tracks them.
        """
        if self.current_status == RobotStatusEnum.JOB_START:
            if not self._ensure_arm(ArmCommand.GO_STOW):
                return
            self.active_dock = DOCK_CONFIGS[UNLOADING_DOCK_NAME]
            self.dock_goal_sent = False
            self._transition_status(RobotStatusEnum.START_DOCKING)
            return

        if self.current_status == RobotStatusEnum.START_DOCKING:
            self._send_dock_goal()
            return

        if self.current_status == RobotStatusEnum.DONE_DOCKING:
            self._transition_status(RobotStatusEnum.START_UNLOADING)
            return

        if self.current_status == RobotStatusEnum.START_UNLOADING:
            self.unload_step = 'END'
            self.unload_goal_sent = False
            self.unload_wait_ticks = 0
            self.unload_step_ticks = 0
            self._transition_status(RobotStatusEnum.UNLOADING)
            return

        if self.current_status == RobotStatusEnum.UNLOADING:
            self._run_unload_sequence()
            return

        if self.current_status == RobotStatusEnum.DONE_UNLOADING:
            self._note_progress()
            self.undock_goal_sent = False
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            return

        if self.current_status == RobotStatusEnum.START_UNDOCKING:
            self._send_undock_goal(f'Undock from {self.active_dock.instance_name}')
            return

        if self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info('Unloading complete')
            self._transition_status(RobotStatusEnum.JOB_DONE)

    def _sim_unloader_status(self) -> RobotStatusEnum:
        """
        Stand in for UnloaderActionClient.get_status() with no carriage present.

        Reports UNLOADING while the tick count runs, DONE_UNLOADING once it
        expires — the same two values the real client reports, so the caller
        needs no sim branch.
        """
        self.sim_unloader_ticks += 1
        if self.sim_unloader_ticks * TIMER_PERIOD_SEC < SIM_UNLOADER_SEC:
            return RobotStatusEnum.UNLOADING

        self.sim_unloader_ticks = 0
        return RobotStatusEnum.DONE_UNLOADING

    def _run_unload_sequence(self) -> None:
        if self.unload_step == 'WAIT':
            self.unload_wait_ticks += 1
            if self.unload_wait_ticks * TIMER_PERIOD_SEC >= POST_UNLOAD_DELAY_SEC:
                self.get_logger().info('Wait over — sending the carriage back home')
                self.unload_step = 'HOME'
                self.unload_goal_sent = False
                self.unload_step_ticks = 0
            return

        # END and HOME each get their own clock. The whole UNLOADING state cannot
        # share one, because the wait in the middle is deliberate.
        self.unload_step_ticks += 1
        waited = self.unload_step_ticks * TIMER_PERIOD_SEC
        if waited > TIMEOUT_UNLOAD_STEP_SEC:
            self.get_logger().error(
                f'Unloader carriage has not reached {self.unload_step} after '
                f'{waited:.0f}s — giving up on it'
            )
            self.unloader_client.reset()
            self._transition_status(RobotStatusEnum.ERROR)
            return

        target = (
            OperateUnloader.Goal.END if self.unload_step == 'END'
            else OperateUnloader.Goal.HOME
        )

        if not self.unload_goal_sent:
            if SIMULATE_UNLOADER:
                self.sim_unloader_ticks = 0
                self.unload_goal_sent = True
                self.get_logger().warning(
                    f'No unloader goal sent (simulated) — pretending the carriage '
                    f'moves to {self.unload_step} over {SIM_UNLOADER_SEC:.0f}s'
                )
                return
            if not self._require_server(
                'the unloader (operate_unloader)', self.unloader_client.client
            ):
                return
            if not self.unloader_client.send_goal(target):
                self.get_logger().error(
                    f'Unloader rejected the goal to go to {self.unload_step}'
                )
                self._transition_status(RobotStatusEnum.ERROR)
                return
            self.unload_goal_sent = True
            return

        status = (
            self._sim_unloader_status() if SIMULATE_UNLOADER
            else self.unloader_client.get_status()
        )

        if status == RobotStatusEnum.ERROR:
            self.get_logger().error(f'Unloader failed on the way to {self.unload_step}')
            self.unloader_client.reset()
            self._transition_status(RobotStatusEnum.ERROR)
            return

        if status != RobotStatusEnum.DONE_UNLOADING:
            return

        if not SIMULATE_UNLOADER:
            self.unloader_client.reset()

        if self.unload_step == 'END':
            self.get_logger().info(
                f'Carriage at the end — waiting {POST_UNLOAD_DELAY_SEC:.0f}s for the '
                'load to clear before bringing it home'
            )
            self.unload_step = 'WAIT'
            self.unload_wait_ticks = 0
            return

        self.current_load_status = 0.0
        self.unload_step = None
        self.get_logger().info('Carriage home — bin recorded as empty')
        self._transition_status(RobotStatusEnum.DONE_UNLOADING)

    # =========================================================================
    # SHARED DOCK / UNDOCK HELPERS
    # =========================================================================

    def _send_dock_goal(self) -> None:
        if self.dock_goal_sent:
            return

        if not self._may_start_motion('dock'):
            self._transition_status(RobotStatusEnum.ERROR)
            return

        if not self._require_server(
            'the docking server (dock_robot)', self.docking_action_client.client
        ):
            return

        dock_id = self.active_dock.instance_name

        subtask = self._subtask_of(SubTask.DOCKING)
        if subtask is None:
            subtask = self._new_subtask(SubTask.DOCKING, f'Dock at {dock_id}')
        # active_dock can change between building the task and sending the goal
        # (error recovery re-picks it), so the goal is rebuilt from it here.
        subtask.description = f'Dock at {dock_id}'
        subtask.dock_goal = _make_dock_goal(dock_id)

        self.get_logger().info(f"Driving to dock '{dock_id}'")
        if not self.docking_action_client.send_docking_goal(subtask):
            self.get_logger().error(f'Failed to send docking goal for {dock_id}')
            self._transition_status(RobotStatusEnum.ERROR)
            return

        self.dock_goal_sent = True
        self._transition_status(RobotStatusEnum.DOCKING)

    def _send_undock_goal(self, description: str) -> None:
        if self.undock_goal_sent:
            return

        if not self._may_start_motion('undock'):
            self._transition_status(RobotStatusEnum.ERROR)
            return

        # A missing undock server is the one case that does not halt: backing out
        # is a TF closed-loop drive on cmd_vel and needs no server at all, which
        # is exactly what the fallback is for.
        if not self.undocking_action_client.client.server_is_ready():
            self._start_reverse_drive('Undock server is not running')
            return

        # Startup undocking runs before any task exists, so there may be nothing
        # to draw the subtask from.
        subtask = self._subtask_of(SubTask.UNDOCKING)
        if subtask is None:
            subtask = self._new_subtask(SubTask.UNDOCKING, description)
        subtask.description = description
        subtask.undock_goal = _make_undock_goal(
            self.active_dock.type, _max_undocking_time()
        )

        if not self.undocking_action_client.send_undocking_goal(subtask):
            # The undock server never took the goal, which is exactly the case
            # the reverse-drive fallback exists for. Previously this went
            # straight to ERROR and the fallback was only reachable when a goal
            # was accepted and then aborted.
            self._start_reverse_drive('Undock server did not accept the goal')
            return

        self.undock_goal_sent = True
        self._transition_status(RobotStatusEnum.UNDOCKING)

    # =========================================================================
    # ERROR RECOVERY
    # =========================================================================

    def _handle_error_recovery(self) -> None:
        """
        Cancel whatever was moving, clear pending arm goals, return to IDLE.

        Two deliberate differences from the production node's version:
          * no time.sleep() — cancellation is polled across ticks, so the loop
            keeps battery, pose and TF live while it settles
          * capped — production waits for an operator to send a new task, but
            this node re-picks its own work, so an uncapped recovery would drive
            straight back into the same fault forever
        """
        if self.recovery_attempts >= MAX_RECOVERY_ATTEMPTS:
            self.get_logger().error(
                f'Tried to recover {MAX_RECOVERY_ATTEMPTS} times without getting '
                'anywhere — stopping for good. A person needs to look at this.'
            )
            self._transition_status(RobotStatusEnum.ABNORMAL)
            return

        # Cancel once on entry, not on every tick.
        if self.error_settle_ticks == 0:
            self._cancel_active_motion()

        self.error_settle_ticks += 1
        settling = self.error_settle_ticks * TIMER_PERIOD_SEC

        # Poll instead of sleeping: hold in ERROR until the cancels land, but not
        # forever — a cancel whose callback never fires would hold here for good.
        still_moving: list[str] = []
        if self.navigation.is_navigation_active():
            still_moving.append('navigation')
        if self.reverse_drive_active and self.reverse_drive_client.is_active():
            still_moving.append('reverse drive')
        if self.drive_client is not None and self.drive_client.is_active():
            still_moving.append('camera drive')

        if still_moving:
            if settling > TIMEOUT_ERROR_SETTLE_SEC:
                self.get_logger().error(
                    f'{" and ".join(still_moving)} did not stop within '
                    f'{TIMEOUT_ERROR_SETTLE_SEC:.0f}s — cannot confirm the robot is '
                    'stationary, stopping for good'
                )
                self.error_settle_ticks = 0
                self._transition_status(RobotStatusEnum.ABNORMAL)
                return
            self.get_logger().info(
                f'Waiting for {" and ".join(still_moving)} to stop',
                throttle_duration_sec=1.0,
            )
            return

        # Everything has stopped. Do not start over while a sensor is still
        # missing: recovery would return to IDLE, the missing sensor would fault
        # again on the next tick, and the whole budget would be gone in seconds.
        # Hold here instead — stopped, cancelled and saying so — until it returns.
        stale = self._stale_sensors()
        if stale:
            self.get_logger().error(
                f'Stopped and waiting for {" and ".join(stale)} to come back',
                throttle_duration_sec=5.0,
            )
            return

        self.error_settle_ticks = 0
        self.recovery_attempts += 1
        self.get_logger().warning(
            f'Recovery attempt {self.recovery_attempts} of {MAX_RECOVERY_ATTEMPTS} '
            '— everything stopped, starting over from idle'
        )

        self._reset_all_clients()
        self._clear_goal_latches()

        # The arm is marked unknown on purpose. reset() only clears this node's
        # copy of the client state; a goal that was already running on the arm
        # may have left it anywhere, so it gets re-stowed before the robot moves.
        self.last_confirmed_arm_command = ArmCommand.UNKNOWN
        self.arm_stow_pending = False
        self.arm_ready_pending = False

        # Drop the task too. _handle_task_start builds a fresh one from whatever
        # the battery, the load and the row index say once we are back in IDLE.
        self.current_task_type = None
        self.current_task = None

        self._transition_status(RobotStatusEnum.IDLE)


def main(args=None):
    rclpy.init(args=args)
    node = TestLavenderHarvestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
