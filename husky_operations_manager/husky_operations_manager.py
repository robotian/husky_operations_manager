import json
import math

import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import BatteryState, Imu, NavSatFix
from status_interfaces.msg import RobotStatus, SubTask, Task, UndockGoal, WayPoint
from std_msgs.msg import Bool

from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.action_clients.manipulator import ArmCommand, ManipulatorTaskActionClient
from husky_operations_manager.action_clients.navigation import NavigationActionClient
from husky_operations_manager.action_clients.reverse_drive_client import ReverseDriveClient
from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.dataclass import DockInstanceConfig, DockPose, ReverseDriveConfig
from husky_operations_manager.enum import (
    NavigationStatus,
    OnlineFlagEnum,
    ReverseDriveStatus,
    RobotStatusEnum,
)


def is_no_move(task: Task | None) -> bool:
    """A charging/unloading task whose move sub-task (the first) only names the start node: nothing to drive."""
    if task is None or task.task_type not in (Task.CHARGING_TASK, Task.UNLOADING_TASK) or not task.sub_tasks:
        return False
    move = task.sub_tasks[0]
    nodes = {wp.node_id for wp in move.data} if move.type == SubTask.MOVING else set()
    return len(nodes) == 1


class HuskyOperationsManager(Node):
    """
    ROS2 node that manages the full operational lifecycle of a Clearpath Husky robot.

    Responsibilities:
      - Fetch live docking configuration from docking_server at start
      - Detect whether robot starts at dock and perform startup undocking if needed
      - Execute HARVESTING, CHARGING, and UNLOADING tasks from the job publisher
      - Manage navigation, docking, undocking, harvesting, loading, and unloading subtasks
      - Recover from undocking failures via closed-loop reverse drive to staging pose
      - Gate arm configuration (STOW/READY) around harvesting and undocking operations
      - Publish RobotStatus on /status/robot at 1Hz
    """

    def __init__(self):
        super().__init__('husky_operations_manager')

        # Strip trailing slash so topic names are well-formed under any namespace
        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'Node namespace: {self.namespace}')

        self._declare_parameter()
        self._get_paramters()
        self._init_state_variables()
        self._init_sensor_data()
        self._init_subscriptions()

        self.robot_state_pub = self.create_publisher(RobotStatus, f'{self.namespace}/status/robot', 10)

        # Motion / dock configuration is read synchronously from ROS params
        # (config.yaml) — see _build_motion_config. These values must be kept in
        # sync with docking_server manually.
        self.motion_config: ReverseDriveConfig = self._build_motion_config()

        # Name of the dock the current operation targets. Set by
        # check_initial_position (startup) and _resolve_task_dock_name (per task).
        self.active_dock_name: str | None = None

        self.get_logger().info(
            f'Motion config loaded | '
            f'docks={list(self.motion_config.dock_configs.keys())} | '
            f"charging='{self._dock_for_charging}' unloading='{self._dock_for_unloading}' | "
            f'staging_x_offset={self.motion_config.staging_x_offset} | '
            f'v_linear_min={self.motion_config.v_linear_min} | '
            f'dock_backwards={self.motion_config.dock_backwards}'
        )

        # All action clients depend on motion_config
        self.navigation = NavigationActionClient(self)
        self.docking_action_client = DockingActionClient(self)
        self.undocking_action_client = UndockingActionClient(self)
        self.manipulator_client = ManipulatorTaskActionClient(self)
        # ReverseDriveClient is the fallback when undock_robot action fails.
        # It drives the robot in reverse using TF closed-loop feedback.
        self.reverse_drive_client = ReverseDriveClient(self, self.motion_config)

        # Delay the initial position check to allow the pose subscription to
        # receive its first message before comparing against the dock position.
        self.init_check_timer = self.create_timer(
            self.timing_initial_position_check_delay, self._initial_position_check_timer
        )
        self.timer = self.create_timer(self.timing_timer_period, self.timer_callback)

        self.get_logger().info('Husky Operations Manager ready.')

    # --- Typed parameter reads (avoid `Parameter.value` being `Any | None`) ---

    def _p_str(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _p_int(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _p_float(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _p_bool(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def _p_str_list(self, name: str) -> list[str]:
        return list(self.get_parameter(name).get_parameter_value().string_array_value)

    def _p_float_list(self, name: str) -> list[float]:
        return [float(v) for v in self.get_parameter(name).get_parameter_value().double_array_value]

    def _build_motion_config(self) -> ReverseDriveConfig:
        """
        Build a ReverseDriveConfig from the `motion.*` ROS params in config.yaml.

        motion.dock_names lists the dock instance names; the per-dock leaves
        (type / frame / pose) are declared here after that list is read.
        pose is a [x, y, theta] double array.
        """
        m = 'motion'
        dock_names = self._p_str_list(f'{m}.dock_names')

        dock_configs: dict[str, DockInstanceConfig] = {}
        for name in dock_names:
            p = f'{m}.dock_configs.{name}'
            self.declare_parameter(f'{p}.type', 'simple_charging_dock')
            self.declare_parameter(f'{p}.frame', 'map')
            self.declare_parameter(f'{p}.pose', [0.0, 0.0, 0.0])
            pose = self._p_float_list(f'{p}.pose')
            dock_configs[name] = DockInstanceConfig(
                instance_name=name,
                type=self._p_str(f'{p}.type'),
                frame=self._p_str(f'{p}.frame'),
                pose=DockPose(x=pose[0], y=pose[1], theta=pose[2]),
            )

        self._dock_for_charging = self._p_str(f'{m}.dock_for_charging')
        self._dock_for_unloading = self._p_str(f'{m}.dock_for_unloading')

        return ReverseDriveConfig(
            dock_configs=dock_configs,
            staging_x_offset=self._p_float(f'{m}.staging_x_offset'),
            staging_yaw_offset=self._p_float(f'{m}.staging_yaw_offset'),
            base_frame=self._p_str(f'{m}.base_frame'),
            controller_frequency=self._p_float(f'{m}.controller_frequency'),
            v_linear_min=self._p_float(f'{m}.v_linear_min'),
            v_angular_max=self._p_float(f'{m}.v_angular_max'),
            linear_tolerance=self._p_float(f'{m}.linear_tolerance'),
            angular_tolerance=self._p_float(f'{m}.angular_tolerance'),
            dock_backwards=self._p_bool(f'{m}.dock_backwards'),
        )

    def _dock(self, name: str | None) -> DockInstanceConfig | None:
        """Look up a dock instance config by name, or None if unknown/unset."""
        if not name:
            return None
        return self.motion_config.dock_configs.get(name)

    def _resolve_task_dock_name(self, task_type: int | None) -> str | None:
        """Map a task type to its configured dock name (motion.dock_for_*)."""
        if task_type == Task.CHARGING_TASK:
            # the dock the charging task docked at (its DockGoal.dock_id), else the configured default
            if self._last_docked_id and self._last_docked_id in self.motion_config.dock_configs:
                return self._last_docked_id
            return self._dock_for_charging
        if task_type == Task.UNLOADING_TASK:
            return self._dock_for_unloading
        return None

    # =========================================================================
    # PARAMS AND VARIABLES INITIALIZATION
    # =========================================================================

    def _declare_parameter(self):
        """Declare all ROS2 parameters with default values. Overridden by YAML at launch."""
        self.declare_parameter('navigation.max_retries', 3)
        self.declare_parameter('navigation.retry_delay', 5.0)
        self.declare_parameter('docking.max_retries', 2)
        self.declare_parameter('docking.retry_delay', 3.0)
        self.declare_parameter('docking.threshold', 0.25)
        # Hard floor only: below it the robot gives its task up (FAILED:battery) and waits for a charging task. When
        # to go charging is the fleet server's decision (status_server's battery thresholds), not the robot's.
        self.declare_parameter('battery.critical_threshold', 10.0)
        self.declare_parameter('battery.full_threshold', 99.0)
        self.declare_parameter('loading.increment', 20.0)
        self.declare_parameter('timing.timer_period', 1.0)
        self.declare_parameter('timing.initial_position_check_delay', 2.0)
        self.declare_parameter('timing.job_done_hold', 5.0)
        self.declare_parameter('server.action_server_timeout', 5.0)

        # Startup "am I docked?" check. The robot map pose is projected into each
        # dock's frame; it is treated as docked if it sits in the staging corridor
        # (staging_x_offset - x_tol <= local_x <= x_tol), close to the dock axis
        # (|local_y| <= lat_tol) and roughly aligned (|yaw - dock.theta| <= yaw_tol).
        self.declare_parameter('startup.dock_x_tol', 0.30)
        self.declare_parameter('startup.dock_lat_tol', 0.40)
        self.declare_parameter('startup.dock_yaw_tol', 0.35)
        # The decision is committed only after this many consecutive TF pose
        # samples agree within pose_stable_eps metres.
        self.declare_parameter('startup.pose_stable_count', 3)
        self.declare_parameter('startup.pose_stable_eps', 0.05)
        # Startup undocking (undock_robot + reverse-drive fallback) attempts before
        # giving up and holding in ERROR for manual intervention.
        self.declare_parameter('startup.undock_max_retries', 3)

        # TF pose staleness: warn (throttled) if the map->base transform is older
        # than this. The transform is still used.
        self.declare_parameter('pose.tf_max_age', 1.0)

        # Max time to wait for an arm goal's result (node clock): a named move (STOW / READY) and a harvest
        # (cut_stem: several MoveIt plans and servo moves per patch -- minutes). A goal the server rejects (busy) or
        # can't take (server not up yet) is re-sent send_retries times, send_retry_delay apart.
        self.declare_parameter('arm.stow_timeout', 120.0)
        self.declare_parameter('arm.ready_timeout', 120.0)
        self.declare_parameter('arm.harvest_timeout', 900.0)
        self.declare_parameter('arm.send_retries', 3)
        self.declare_parameter('arm.send_retry_delay', 5.0)
        # How long a failed task is reported (RobotStatus.task = FAILED:<kind>:<task_id>:<detail>, status ERROR)
        # before the node goes back to IDLE: the fleet server polls at 1 Hz.
        self.declare_parameter('error.report_hold', 5.0)
        # false: skip the startup "am I docked?" check (the robot never starts in a dock, e.g. a sim spawn)
        self.declare_parameter('startup.check_dock', True)

        # Motion / dock config (replaces the old docking_server fetch).
        # Per-dock leaves (motion.dock_configs.<name>.*) are declared in
        # _build_motion_config once motion.dock_names is known.
        self.declare_parameter('motion.dock_names', ['husky_charger'])
        self.declare_parameter('motion.dock_for_charging', 'husky_charger')
        self.declare_parameter('motion.dock_for_unloading', 'unloading_station')
        self.declare_parameter('motion.staging_x_offset', -1.5)
        self.declare_parameter('motion.staging_yaw_offset', 0.0)
        self.declare_parameter('motion.base_frame', 'base_link')
        self.declare_parameter('motion.controller_frequency', 50.0)
        self.declare_parameter('motion.v_linear_min', 0.15)
        self.declare_parameter('motion.v_angular_max', 0.25)
        self.declare_parameter('motion.linear_tolerance', 0.05)
        self.declare_parameter('motion.angular_tolerance', 0.1)
        self.declare_parameter('motion.dock_backwards', False)

    def _get_paramters(self):
        """Read all declared parameters into instance variables."""
        self.navigation_max_retries = self._p_int('navigation.max_retries')
        self.navigation_retry_delay = self._p_float('navigation.retry_delay')
        self.docking_max_retries = self._p_int('docking.max_retries')
        self.docking_retry_delay = self._p_float('docking.retry_delay')
        self.docking_threshold = self._p_float('docking.threshold')
        self.battery_critical_threshold = self._p_float('battery.critical_threshold')
        self.battery_full_threshold = self._p_float('battery.full_threshold')
        self.loading_increment = self._p_float('loading.increment')
        self.timing_timer_period = self._p_float('timing.timer_period')
        self.timing_initial_position_check_delay = self._p_float('timing.initial_position_check_delay')
        self.timing_job_done_hold = self._p_float('timing.job_done_hold')
        self._last_task_msg_ns: int | None = None  # node clock of the last Task message (any task)
        self.server_action_server_timeout = self._p_float('server.action_server_timeout')

        self.startup_dock_x_tol = self._p_float('startup.dock_x_tol')
        self.startup_dock_lat_tol = self._p_float('startup.dock_lat_tol')
        self.startup_dock_yaw_tol = self._p_float('startup.dock_yaw_tol')
        self.startup_pose_stable_count = self._p_int('startup.pose_stable_count')
        self.startup_pose_stable_eps = self._p_float('startup.pose_stable_eps')
        self.startup_undock_max_retries = self._p_int('startup.undock_max_retries')
        self.pose_tf_max_age = self._p_float('pose.tf_max_age')
        self.arm_stow_timeout = self._p_float('arm.stow_timeout')
        self.arm_ready_timeout = self._p_float('arm.ready_timeout')
        self.arm_harvest_timeout = self._p_float('arm.harvest_timeout')
        self.arm_send_retries = self._p_int('arm.send_retries')
        self.arm_send_retry_delay = self._p_float('arm.send_retry_delay')
        self.error_report_hold = self._p_float('error.report_hold')
        self.startup_check_dock = self._p_bool('startup.check_dock')

        self.get_logger().debug(
            f'Parameters loaded | '
            f'nav_retries={self.navigation_max_retries} nav_delay={self.navigation_retry_delay}s | '
            f'dock_retries={self.docking_max_retries} dock_delay={self.docking_retry_delay}s '
            f'dock_threshold={self.docking_threshold}m | '
            f'battery_critical={self.battery_critical_threshold}% battery_full={self.battery_full_threshold}% | '
            f'load_increment={self.loading_increment}% | '
            f'timer={self.timing_timer_period}s'
        )

    def _init_state_variables(self):
        """
        Initialise all state-tracking variables to their boot defaults.

        Grouped by concern:
          - Startup flags: track whether boot sequence is complete
          - Robot state: current and previous RobotStatusEnum value
          - Task management: current task, subtask, and index tracking
          - Retry counters: navigation and docking retries (reset on success)
          - Job simulation: timer used for placeholder harvesting/loading/unloading
          - Undocking: subtask stored for retry, task type that triggered undocking
          - Reverse drive: flag true while ReverseDriveClient is actively reversing
          - Arm state: tracks confirmed arm configuration and pending goal flags
        """
        # --- Startup ---
        self.is_initialized = False  # True after check_initial_position runs
        self.is_at_docking_station = False  # True if robot started in a dock's staging corridor
        self.startup_undock_complete = False  # True once startup undocking path is done
        # Also used to distinguish startup vs task
        # context in _handle_undocking and
        # _handle_reverse_drive
        # TF pose stability gate for the startup docked/not-docked decision
        self._pose_stable_samples = 0
        self._last_stable_xy: tuple[float, float] | None = None
        # Startup-undock attempt counter + hard-hold flag. Each fresh
        # IDLE -> START_UNDOCKING counts as one attempt; past the ceiling the node
        # holds in ERROR (error recovery does not bounce it back to IDLE).
        self._startup_undock_attempts = 0
        self._startup_undock_failed = False

        # --- Robot state ---
        self.current_status = RobotStatusEnum.IDLE
        self.previous_status = RobotStatusEnum.IDLE

        # --- Task management ---
        self.current_task: Task | None = None
        self.current_sub_task: SubTask | None = None
        # Index into current_task.sub_tasks list; incremented externally by JobPublisher
        # publishing the next subtask. Reset to 0 on each new task.
        self.current_sub_task_index = 0

        # Used to detect new tasks/subtasks arriving on the /status/task topic
        self.last_handled_task_id: int | None = None
        self.last_handled_task_type: int | None = None
        self.last_handled_subtask_type: int | None = None
        # De-dupes the "New Task" log line — separate from last_handled_task_id,
        # which only updates once the task actually starts.
        self._last_logged_task_id: int | None = None

        # Current topological map node the robot occupies; updated on DESTINATION_REACHED
        self.current_node_id = 0
        # Load level 0-100%; incremented by loading_increment per harvest cycle,
        # reset to 0 after successful unloading
        self.current_load_status = 0.0
        # One-shot: seed current_load_status from the first Task message's
        # crop_load so a restart keeps the basket level. Never re-seeded (a
        # harvest/unload owns the value after that).
        self._load_seeded = False

        # --- Retry counters ---
        self.navigation_retry_count = 0  # Reset to 0 on success or after max reached
        self.docking_retry_count = 0  # Reset to 0 on success or after max reached

        # --- Non-blocking dwell deadlines ---
        # key -> rclpy Time deadline. Armed on first _wait_ready(key, delay) call,
        # cleared once the delay has elapsed. Replaces time.sleep() in the retry
        # and dwell paths so the executor keeps spinning.
        self._dwell_deadlines: dict = {}

        # --- Battery current EMA (for remaining-runtime estimate) ---
        self._battery_current_ema: float | None = None

        # --- Charge current EMA (for charge-completion ETA) ---
        # Kept separate from _battery_current_ema, which tracks discharge
        # magnitude. Reset to None at the start of every charging session.
        self._charge_current_ema: float | None = None

        # --- Job duration (used for unloading/charging timers) ---
        self.job_start_time = None  # rclpy clock time when current job phase started
        self.job_duration = 0.0  # Duration in seconds for current phase

        # --- Undocking ---
        # Stores the charging or unloading SubTask so _subtask_undocking can send the
        # correct UndockGoal. UNDOCKING is not in the received subtask list for
        # CHARGING_TASK or UNLOADING_TASK — it is triggered internally.
        self.last_undocking_subtask: SubTask | None = None
        # Tracks which task type triggered undocking (CHARGING_TASK or UNLOADING_TASK)
        self.undocking_after_task_type: int | None = None
        # timer_callback runs the state machine and then publishes once, so a
        # status set and cleared inside one tick never reaches the wire. The
        # server frees the dock on DONE_UNDOCKING, so that status has to be
        # published: this flag holds it for one tick before JOB_DONE.
        self.done_undocking_published: bool = False

        # --- Reverse drive ---
        # Set True when ReverseDriveClient.drive_to_staging() starts.
        # Set False in _handle_reverse_drive when DONE, ERROR, or CANCELED.
        # Used as the priority-4 check in _process_action_clients.
        self.reverse_drive_active: bool = False

        # --- Arm state ---
        # Tracks the last arm command the manipulator server REPORTED as complete.
        # NOTE: trust-on-report — set purely from the action returning success,
        # with no joint-state / arm-pose confirmation. A server that reports OK
        # without the arm reaching the configuration will be believed. Closed-loop
        # verification needs the arm's joint-states interface (out of scope here).
        # Boot value UNKNOWN forces an explicit STOW before the first gated move.
        self.last_confirmed_arm_command: str = ArmCommand.UNKNOWN
        # True while waiting for a STOW goal to complete via _handle_manipulator.
        # Gates undocking, moving, docking and error recovery.
        self.arm_stow_pending: bool = False
        # True while waiting for a READY goal to complete via _handle_manipulator.
        # Gates _subtask_harvesting from advancing past DESTINATION_REACHED.
        self.arm_ready_pending: bool = False
        # True when the STOW itself failed during error recovery: the arm is untrusted, the node holds in ERROR
        # (reporting FAILED:arm_stow) until restarted.
        self._error_from_arm: bool = False
        # The failure the node is reporting / recovering from: (kind, task_id, detail) -> RobotStatus.task
        # 'FAILED:<kind>:<task_id>:<detail>' while in ERROR (see _fail_current_task). The fleet server closes that
        # task and sends a new task_id; republishes of the failed id are ignored (last_handled_task_id keeps it).
        self._failure: tuple[str, int, str] | None = None
        # True for a low-battery failure: after the report the node stays in ERROR until a CHARGING_TASK arrives.
        self._failure_hold: bool = False
        # Arm goal sends that were rejected (busy) / not taken (server down), re-sent up to arm.send_retries
        self._arm_send_attempts: int = 0
        # The dock the robot last docked at (DockGoal.dock_id): the one it undocks from after charging
        self._last_docked_id: str | None = None
        self._arm_goal_timeout_current: float = 0.0

        self.get_logger().debug('State variables initialised')

    def _init_sensor_data(self):
        """Initialise sensor data containers with empty default messages."""
        self.battery_status = BatteryState()
        self.task = Task()
        self.gps_status = NavSatFix()
        # Robot pose in the map frame, refreshed from TF each timer tick.
        # None until the first successful map -> base_frame lookup.
        self.pose_status: PoseStamped | None = None
        self.imu_status = Imu()
        self.estop_status = Bool()
        # True once at least one TF pose lookup has succeeded.
        self._pose_received = False

    def _init_subscriptions(self):
        """Create all ROS2 subscriptions and the TF listener."""
        # Robot pose is taken from TF (map -> base frame), refreshed each timer
        # tick by _update_pose_from_tf. Replaces the ground_truth/pose topic.
        self._pose_reference_frame = 'map'
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Battery state — used for low-battery detection and charge completion checks
        self.battery_sub = self.create_subscription(
            BatteryState,
            f'{self.namespace}/platform/bms/state',
            lambda msg: setattr(self, 'battery_status', msg),
            qos_profile_sensor_data,
        )

        # IMU — stored but not currently used in state machine logic
        self.imu_sub = self.create_subscription(
            Imu,
            f'{self.namespace}/sensors/gps_0/imu',
            lambda msg: setattr(self, 'imu_status', msg),
            qos_profile_sensor_data,
        )

        # Emergency stop — mapped to online_flag in published RobotStatus
        self.estop_sub = self.create_subscription(
            Bool,
            f'{self.namespace}/platform/emergency_stop',
            lambda msg: setattr(self, 'estop_status', msg),
            qos_profile_sensor_data,
        )

        # Task topic — published by JobPublisher with HARVESTING, CHARGING, or UNLOADING tasks
        self.task_sub = self.create_subscription(Task, f'{self.namespace}/status/task', self._task_callback, 10)

        self.get_logger().debug('Subscriptions initialised')

    # =========================================================================
    # MAIN CALLBACK METHODS
    # =========================================================================

    def _task_callback(self, msg: Task):
        """
        Handle incoming Task messages from JobPublisher.

        Detects whether the arriving message represents a new task, a new subtask
        on the same task, or a repeated publish of the same subtask (no-op).

        New task:    reset subtask index and subtask tracking
        New subtask: reset subtask index; if currently JOB_DONE, transition to IDLE
                     so _handle_task_start can pick up the next subtask
        """
        # The operator's STOP (status_server stop_robot: task_id -1, description STOP, no job_schedule): whatever
        # runs is cancelled and the task given up (FAILED:stopped, which the server treats as released, not as an
        # attempt); nothing starts until a task with a real id arrives. Ignored while idle with no task, and while
        # in ERROR: the server republishes STOP every tick, and failing again there cancelled the recovery's stow.
        if msg.task_id < 0 and msg.description == 'STOP' and not msg.job_schedule:
            if self.current_status not in (RobotStatusEnum.ERROR, RobotStatusEnum.EMERGENCY_STOP) and (
                    self.current_status != RobotStatusEnum.IDLE or self.current_task is not None):
                self.get_logger().warning('STOP from the operator: cancelling the current task')
                self._fail_current_task('stopped', 'by the operator')
            return

        subtasks_summary = [(st.sub_task_id, st.type, st.description) for st in msg.sub_tasks]
        self.get_logger().debug(
            f'Received task | ID: {msg.task_id} | Type: {msg.task_type} | '
            f'Target Node: {msg.target_node_id} | SubTasks: {subtasks_summary}'
        )

        # A new task is detected when the task_id changes OR when task_type changes
        # on the same id (e.g. fallback from HARVESTING to CHARGING on low battery)
        is_new_task = (
            msg.task_id != self.last_handled_task_id
            or self.current_task
            and msg.task_type != self.current_task.task_type
        )

        # A new subtask is detected when the first subtask type in the message differs
        # from the last one we processed — JobPublisher advances the subtask index
        is_new_subtask = False
        if isinstance(msg.sub_tasks, list) and len(msg.sub_tasks) > 0:
            first_subtask = msg.sub_tasks[0]
            if isinstance(first_subtask, SubTask):
                is_new_subtask = first_subtask.type != self.last_handled_subtask_type

        self.get_logger().debug(
            f'Task callback | id={msg.task_id} type={msg.task_type} | '
            f'is_new_task={is_new_task} is_new_subtask={is_new_subtask} | '
            f'last_task_id={self.last_handled_task_id} '
            f'last_subtask_type={self.last_handled_subtask_type}'
        )

        if is_new_task and msg.task_id != self.last_handled_task_id:
            # A charging task (the server's low-battery decision) arriving mid-move: cancel the navigation; its
            # CANCELED result takes the node to IDLE, where the new task starts (_handle_task_start).
            if (msg.task_type == Task.CHARGING_TASK and self.current_task is not None
                    and self.current_task.task_type != Task.CHARGING_TASK
                    and self.current_status in (RobotStatusEnum.START_MOVING, RobotStatusEnum.MOVING)):
                self.get_logger().warning(f'Charging task {msg.task_id} preempts the move of task '
                                          f'{self.current_task.task_id}: cancelling navigation')
                try:
                    self.navigation.cancel_goal()
                except Exception as e:  # noqa: BLE001
                    self.get_logger().warning(f'Navigation cancel raised: {e}')

        if is_new_task:
            # Log once per task_id, not on every republish. last_handled_task_id
            # only updates once the task actually starts (_handle_task_start), so
            # while the node holds in ERROR (e.g. low battery) a stuck task_id can
            # otherwise re-log "New Task" on every JobPublisher republish.
            if msg.task_id != self._last_logged_task_id:
                self.get_logger().info(f'New Task: {msg.description} (ID: {msg.task_id})')
                self._last_logged_task_id = msg.task_id
            self.current_sub_task_index = 0
            self.last_handled_subtask_type = None
            # self.current_load_status = msg.crop_load
        elif is_new_subtask:
            self.get_logger().info(f'New Subtask for task ID: {msg.task_id}')
            self.current_sub_task_index = 0
            # If the node completed the previous subtask and is waiting at JOB_DONE,
            # receiving a new subtask means the job publisher wants us to continue
            if self.current_status == RobotStatusEnum.JOB_DONE:
                self._transition_status(RobotStatusEnum.IDLE)

        # Seed load once from the first task message (server is the source of
        # truth for basket level across a restart). A dedicated one-shot flag —
        # not last_handled_task_id, which only advances once a task actually
        # starts, so while the node is held before start it would otherwise
        # re-clobber current_load_status on every republish.
        if not self._load_seeded:
            self.current_load_status = msg.crop_load
            self._load_seeded = True
            self.get_logger().info(f'Load seeded from task: {self.current_load_status:.1f}')

        self.task = msg
        self._last_task_msg_ns = self.get_clock().now().nanoseconds

    def timer_callback(self):
        """
        Main control loop — fires at timing_timer_period (default 1Hz).

        Each tick:
          1. Build a fresh RobotStatus from current sensor readings
          2. Route to the correct handler based on current state:
               ERROR/ABNORMAL  → _handle_error_recovery
               startup not done → _handle_startup_undocking
               otherwise       → _handle_task_execution
          3. Stamp the final status values and publish to /status/robot
        """
        robot_status = RobotStatus()
        robot_status.header.stamp = self.get_clock().now().to_msg()
        robot_status.robot_namespace = self.namespace.replace(r'/', '')

        # Refresh pose from TF, then update sensor fields before routing so all
        # paths see fresh data
        self._update_pose_from_tf()
        self._set_battery_status(robot_status)
        self._set_estop_status(robot_status)
        self._set_location_status(robot_status)

        self.get_logger().debug(
            f'Timer tick | status={self.current_status.name} | '
            f'startup_undock_complete={self.startup_undock_complete} | '
            f'reverse_drive_active={self.reverse_drive_active} | '
            f'arm_stow_pending={self.arm_stow_pending} | '
            f'arm_ready_pending={self.arm_ready_pending} | '
            f"last_confirmed_arm='{self.last_confirmed_arm_command}' | "
            f'battery={self._normalize_battery(self.battery_status.percentage):.1f}%'
        )

        # Emergency stop (platform/emergency_stop, real robots): everything is cancelled and the node latches in
        # EMERGENCY_STOP; when it is released the task counts as failed (FAILED:estop) and the server reassigns.
        if self.estop_status.data:
            if self.current_status != RobotStatusEnum.EMERGENCY_STOP:
                self.get_logger().error('Emergency stop: cancelling navigation, docking and arm goals')
                self._cancel_everything()
                self._transition_status(RobotStatusEnum.EMERGENCY_STOP)
            robot_status.task = 'ESTOP'
        elif self.current_status == RobotStatusEnum.EMERGENCY_STOP:
            self._fail_current_task('estop', 'released')

        if self.current_status == RobotStatusEnum.EMERGENCY_STOP:
            pass
        # ERROR and ABNORMAL are handled before startup check so a fault during
        # startup undocking is also covered by _handle_error_recovery
        elif self.current_status in [RobotStatusEnum.ERROR, RobotStatusEnum.ABNORMAL]:
            self._handle_error_recovery(robot_status)
        elif not self.startup_undock_complete:
            self._handle_startup_undocking(robot_status)
        else:
            self._handle_task_execution(robot_status)

        # Write final state into the outgoing message. Until the startup check ran (it waits for a stable map pose)
        # the robot reports INITIALIZING, not IDLE: the server assigns it nothing and the web UI shows why.
        if not self.is_initialized and self.current_status == RobotStatusEnum.IDLE:
            robot_status.status = RobotStatusEnum.INITIALIZING.value
            robot_status.task = 'INITIALIZING: waiting for the map pose' if not self._pose_received else 'INITIALIZING'
        else:
            robot_status.status = self.current_status.value
        robot_status.current_node_id = self.current_node_id
        robot_status.load_status = self.current_load_status
        self.robot_state_pub.publish(robot_status)

    # =========================================================================
    # POSE (TF)
    # =========================================================================

    def _update_pose_from_tf(self):
        """
        Refresh self.pose_status from the map -> base_frame transform.

        Called each timer tick. On a lookup failure the last good pose is kept
        and _pose_received is left unchanged.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                self._pose_reference_frame,
                self.motion_config.base_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except Exception as e:  # noqa: BLE001 - any TF failure: keep last good pose
            self.get_logger().warning(
                f'Pose TF lookup {self._pose_reference_frame} -> '
                f'{self.motion_config.base_frame} failed: {e}',
                throttle_duration_sec=2.0,
            )
            return

        # Staleness check — warn only, the transform is still used
        stamp = Time.from_msg(tf.header.stamp)
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9
        if age > self.pose_tf_max_age:
            self.get_logger().warning(
                f'Pose TF is stale: {age:.2f}s old (> {self.pose_tf_max_age:.1f}s) — '
                f'{self._pose_reference_frame} -> {self.motion_config.base_frame}',
                throttle_duration_sec=2.0,
            )

        pose = PoseStamped()
        pose.header = tf.header
        pose.pose.position.x = tf.transform.translation.x
        pose.pose.position.y = tf.transform.translation.y
        pose.pose.position.z = tf.transform.translation.z
        pose.pose.orientation = tf.transform.rotation
        self.pose_status = pose
        self._pose_received = True

    # =========================================================================
    # STARTUP AND INITIALIZATION
    # =========================================================================

    def _initial_position_check_timer(self):
        """
        Fires every timing_initial_position_check_delay until the check completes.

        Holds until a TF pose is available AND has been stable for
        startup_pose_stable_count consecutive samples, then runs
        check_initial_position and cancels itself once it has initialised.
        """
        self._update_pose_from_tf()
        if not self._pose_received:
            self.get_logger().warning('Waiting for pose TF...', throttle_duration_sec=2.0)
            return
        if not self._pose_is_stable():
            self.get_logger().warning(
                f'Waiting for stable pose '
                f'({self._pose_stable_samples}/{self.startup_pose_stable_count})',
                throttle_duration_sec=2.0,
            )
            return
        self.check_initial_position()
        if self.is_initialized:
            self.init_check_timer.cancel()

    def _pose_is_stable(self) -> bool:
        """
        True once the TF pose has moved less than startup_pose_stable_eps between
        startup_pose_stable_count consecutive samples. Any larger jump resets the
        counter. Guards the startup docked/not-docked decision against a
        not-yet-converged localisation estimate.
        """
        if self.pose_status is None:
            return False
        p = self.pose_status.pose.position
        if self._last_stable_xy is None:
            self._last_stable_xy = (p.x, p.y)
            self._pose_stable_samples = 1
            return False
        moved = math.hypot(p.x - self._last_stable_xy[0], p.y - self._last_stable_xy[1])
        self._last_stable_xy = (p.x, p.y)
        if moved <= self.startup_pose_stable_eps:
            self._pose_stable_samples += 1
        else:
            self._pose_stable_samples = 1
        return self._pose_stable_samples >= self.startup_pose_stable_count

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        """Extract the yaw (Z) angle from a geometry_msgs Quaternion."""
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _project_into_dock_frame(self, rx, ry, ryaw, dock):
        """
        Return (local_x, local_y, local_yaw) of the robot expressed in the dock's
        frame. local_x points from the dock along its heading (dock -> approach),
        local_y is left of it, local_yaw is the heading error vs the dock.
        """
        dx, dy, dth = dock.pose.x, dock.pose.y, dock.pose.theta
        c, s = math.cos(dth), math.sin(dth)
        ox, oy = rx - dx, ry - dy
        local_x = c * ox + s * oy
        local_y = -s * ox + c * oy
        local_yaw = math.atan2(math.sin(ryaw - dth), math.cos(ryaw - dth))
        return local_x, local_y, local_yaw

    def check_initial_position(self):
        """
        Determine whether the robot starts at a docking station.

        Projects the robot's map pose into each dock's frame. The robot is
        treated as docked at a dock when it sits in that dock's staging corridor:
          staging_x_offset - dock_x_tol <= local_x <= dock_x_tol
          |local_y|  <= dock_lat_tol
          |local_yaw| <= dock_yaw_tol
        If several docks match, the one with local_x closest to 0 (nearest the
        contact pose) wins.

        Sets startup_undock_complete=True if the robot is NOT in any dock's
        corridor, so the node skips the startup undocking sequence.
        """
        if self.is_initialized:
            return

        if not self.startup_check_dock:
            self.is_initialized = True
            self.startup_undock_complete = True
            self.get_logger().info('startup.check_dock is false — skipping the dock check, ready for tasks')
            return

        if not self._pose_received or self.pose_status is None:
            self.get_logger().warning('check_initial_position called before pose TF — deferring')
            return

        p = self.pose_status.pose
        rx, ry = p.position.x, p.position.y
        ryaw = self._yaw_from_quaternion(p.orientation)
        staging_x = self.motion_config.staging_x_offset

        docked_name: str | None = None
        docked_lx = 0.0
        for name, dock in self.motion_config.dock_configs.items():
            lx, ly, lyaw = self._project_into_dock_frame(rx, ry, ryaw, dock)
            lo = min(staging_x - self.startup_dock_x_tol, self.startup_dock_x_tol)
            hi = max(staging_x - self.startup_dock_x_tol, self.startup_dock_x_tol)
            in_corridor = (
                lo <= lx <= hi
                and abs(ly) <= self.startup_dock_lat_tol
                and abs(lyaw) <= self.startup_dock_yaw_tol
            )
            self.get_logger().debug(
                f"Dock '{name}' | local=({lx:.3f}, {ly:.3f}, {math.degrees(lyaw):.1f}deg) | "
                f'in_corridor={in_corridor}'
            )
            if in_corridor and (docked_name is None or abs(lx) < abs(docked_lx)):
                docked_name = name
                docked_lx = lx

        self.get_logger().info(
            f'Initial robot pose: ({rx:.3f}, {ry:.3f}, {math.degrees(ryaw):.1f}deg) | '
            f"docked at '{docked_name}'" + (f' (local_x={docked_lx:.3f}m)' if docked_name else '')
        )

        if docked_name is not None:
            self.is_at_docking_station = True
            self.active_dock_name = docked_name
            self.get_logger().info(f"Robot in staging corridor of dock '{docked_name}' — will undock before tasks")
        else:
            self.is_at_docking_station = False
            self.startup_undock_complete = True
            self.get_logger().info('Robot not at any dock — ready for tasks')

        self.is_initialized = True

    def _handle_startup_undocking(self, robot_status: RobotStatus):
        """
        Drive the startup undocking state machine.

        Called from timer_callback while startup_undock_complete is False.

        Tick 1 (IDLE):            transition to START_UNDOCKING
        Tick 2 (START_UNDOCKING): compute UndockGoal from motion_config,
                                  STOW check runs inside _subtask_undocking,
                                  send to UndockingActionClient, transition to UNDOCKING
        Tick 3+ (else):           hand off to _handle_undocking or _handle_reverse_drive
                                  depending on reverse_drive_active flag
        """
        if not self.is_initialized or self.startup_undock_complete:
            return

        if self.active_dock_name is None:
            self.get_logger().warning('_handle_startup_undocking called but active_dock_name is not set')
            return

        self.get_logger().debug(
            f'Startup undocking | current_status={self.current_status.name} | '
            f'reverse_drive_active={self.reverse_drive_active}'
        )

        if self.current_status == RobotStatusEnum.IDLE:
            if self._startup_undock_attempts >= self.startup_undock_max_retries:
                self._startup_undock_failed = True
                self.get_logger().error(
                    f'Startup undocking failed {self._startup_undock_attempts}x '
                    f'(max {self.startup_undock_max_retries}) — holding in ERROR for manual '
                    'intervention. Check the dock config (undock_robot dock_type must be a '
                    'registered plugin) and the reverse-drive path.',
                    throttle_duration_sec=5.0,
                )
                self._transition_status(RobotStatusEnum.ERROR)
                return
            self._startup_undock_attempts += 1
            self.get_logger().info(
                f'Starting startup undocking '
                f'(attempt {self._startup_undock_attempts}/{self.startup_undock_max_retries})...'
            )
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            robot_status.task = 'Startup: Preparing to undock'

        elif self.current_status == RobotStatusEnum.START_UNDOCKING:
            robot_status.task = 'Startup: Undocking'

            # Compute max_undocking_time from motion config rather than hardcoding.
            # Formula: distance to staging pose / minimum speed * safety factor 1.25
            dock = self._dock(self.active_dock_name)
            if dock is None:
                self.get_logger().error(f"Startup undocking — unknown dock '{self.active_dock_name}' — ERROR")
                self._transition_status(RobotStatusEnum.ERROR)
                return

            dock_type = dock.type
            staging_x_offset = self.motion_config.staging_x_offset
            v_linear_min = self.motion_config.v_linear_min
            max_undocking_time = (abs(staging_x_offset) / max(v_linear_min, 0.01)) * 1.25

            self.get_logger().debug(
                f"Startup UndockGoal | dock='{self.active_dock_name}' type='{dock_type}' | "
                f'staging_x_offset={staging_x_offset} | '
                f'v_linear_min={v_linear_min} | '
                f'max_undocking_time={max_undocking_time:.1f}s'
            )

            startup_subtask = SubTask()
            startup_subtask.type = SubTask.UNDOCKING
            startup_subtask.description = 'Startup Undocking'
            startup_subtask.undock_goal = UndockGoal(dock_type=dock_type, max_undocking_time=max_undocking_time)

            # Store so the STOW gate inside _subtask_undocking can reference it
            self.last_undocking_subtask = startup_subtask

            if self.arm_stow_pending:
                self._handle_manipulator(robot_status)

            self._subtask_undocking()

        else:
            # Tick 3+: undock_robot goal has been sent. Route to the correct handler.
            # If ReverseDriveClient is active (fallback after undocking failure),
            # monitor it; otherwise monitor the undocking action client.
            if self.reverse_drive_active:
                self._handle_reverse_drive(robot_status)
            else:
                self._handle_undocking(robot_status)

    # =========================================================================
    # TASK EXECUTION
    # =========================================================================

    def _handle_task_execution(self, robot_status: RobotStatus):
        """
        Main task execution handler — called from timer_callback when
        startup_undock_complete is True and status is not ERROR/ABNORMAL.

        Order of operations each tick:
          1. Validate task exists — go IDLE if not
          2. Check battery — go ERROR if low (except during CHARGING_TASK)
          3. Process active action clients (navigation/docking/undocking/reverse drive/arm)
          4. Refresh current_sub_task from the subtask list
          5. Route: IDLE/JOB_DONE → _handle_task_start, otherwise → _execute_current_subtask
        """
        # No task or empty schedule means JobPublisher has nothing for us
        if not self.task or not self.task.description or not self.task.job_schedule:
            if self.current_status != RobotStatusEnum.IDLE:
                self._transition_status(RobotStatusEnum.IDLE)
            return

        self.current_task = self.task

        if self._check_and_handle_low_battery():
            return

        # Action client handlers run before subtask handlers so that async results
        # (navigation success, docking complete, arm command complete, etc.) are
        # processed on the same tick they arrive, preventing a one-tick delay in
        # state transitions
        self._process_action_clients(robot_status)
        self._update_current_subtask()

        robot_status.crop_type = self.current_task.crop_type
        robot_status.target_node_id = self.current_task.target_node_id
        robot_status.task = (
            self.current_sub_task.description if self.current_sub_task else self.current_task.description
        )

        self.get_logger().debug(
            f'Task execution | status={self.current_status.name} | '
            f'task_id={self.current_task.task_id} task_type={self.current_task.task_type} | '
            f'subtask_index={self.current_sub_task_index} | '
            f'subtask_type={self.current_sub_task.type if self.current_sub_task else "None"}'
        )

        if self.current_status in (RobotStatusEnum.IDLE, RobotStatusEnum.JOB_DONE):
            self._handle_task_start()
        else:
            self._execute_current_subtask()

    def _check_and_handle_low_battery(self) -> bool:
        """
        Check battery and interrupt task if level is critical.

        Returns True if the caller should abort the current tick (battery is
        low). Returns False to continue normally.

        NOTE: this only runs while current_status != ERROR/ABNORMAL — timer_callback
        routes those statuses to _handle_error_recovery instead, which is why the
        exit from a low-battery ERROR is handled there (via _failure_hold),
        not here. Do not try to recover from ERROR in this method: it will never see
        a CHARGING_TASK because it is not called while in ERROR.
        """
        if not self.task or not self.current_task:
            return False

        battery_pct = self._normalize_battery(self.battery_status.percentage)

        self.get_logger().debug(
            f'Battery check | pct={battery_pct:.1f}% | '
            f'critical_threshold={self.battery_critical_threshold}% | '
            f'task_type={self.task.task_type}'
        )

        # The hard floor: everything is cancelled (a cut too), the task is reported failed (FAILED:battery) and the
        # node waits in ERROR for a charging task. Above the floor the fleet server decides when to send one.
        if self.task.task_type != Task.CHARGING_TASK and battery_pct <= self.battery_critical_threshold:
            self.get_logger().warning(
                f'Battery critical: {battery_pct:.1f}% <= {self.battery_critical_threshold}% | '
                f'task_type={self.task.task_type} | status={self.current_status.name}'
            )
            self._fail_current_task('battery', f'{battery_pct:.0f}%', hold=True)
            return True

        return False

    def _process_action_clients(self, robot_status: RobotStatus):
        """
        Poll all action clients and route to the appropriate handler.

        Priority order for motion clients (only the first active client is handled
        per tick):
          1. NavigationActionClient   — NavigateThroughPoses
          2. DockingActionClient      — dock_robot
          3. UndockingActionClient    — undock_robot
          4. ReverseDriveClient       — TF closed-loop reverse (fallback for undocking)

        The manipulator client is polled independently after the motion clients
        because arm commands run in parallel and do not block robot motion, but
        their completion does unblock state transitions in the subtask handlers.

        The manipulator is polled when:
          - arm_stow_pending is True  (STOW goal in flight)
          - arm_ready_pending is True (READY goal in flight)
          - status is HARVESTING with no pending flags (START_HARVEST goal in flight)
        """
        nav_status = self.navigation.get_navigation_status()
        dock_status = self.docking_action_client.get_status()
        undock_status = self.undocking_action_client.get_status()

        self.get_logger().debug(
            f'Action clients | nav={nav_status.name} | '
            f'dock={dock_status.name} | '
            f'undock={undock_status.name} | '
            f'reverse_drive_active={self.reverse_drive_active} | '
            f'arm_stow_pending={self.arm_stow_pending} | '
            f'arm_ready_pending={self.arm_ready_pending}'
        )

        if nav_status != NavigationStatus.IDLE:
            self._handle_navigation(robot_status)
        elif dock_status != RobotStatusEnum.IDLE:
            self._handle_docking(robot_status)
        elif undock_status != RobotStatusEnum.IDLE:
            self._handle_undocking(robot_status)
        elif self.reverse_drive_active:
            self._handle_reverse_drive(robot_status)

        # Arm is polled independently — it runs in parallel with the robot motion
        # state machine and its completion unblocks harvesting and undocking flows.
        arm_harvest_active = (
            self.current_status == RobotStatusEnum.HARVESTING
            and not self.arm_stow_pending
            and not self.arm_ready_pending
        )

        if self.arm_stow_pending or self.arm_ready_pending or arm_harvest_active:
            self._handle_manipulator(robot_status)

    def _update_current_subtask(self):
        """
        Refresh current_sub_task from the sub_tasks list using current_sub_task_index.

        current_sub_task_index is reset to 0 by _task_callback on new task/subtask.
        It is the responsibility of the external JobPublisher to advance which subtask
        it publishes — the node always reads index 0 of the incoming sub_tasks list
        and current_sub_task_index tracks locally which subtask we are on.
        """
        if not self.current_task:
            return

        if isinstance(self.current_task.sub_tasks, list):
            if self.current_sub_task_index < len(self.current_task.sub_tasks):
                self.current_sub_task = self.current_task.sub_tasks[self.current_sub_task_index]
            else:
                # Index out of range — all subtasks exhausted
                self.current_sub_task = None

        self.get_logger().debug(
            f'Current subtask | index={self.current_sub_task_index} | '
            f'type={self.current_sub_task.type if self.current_sub_task else "None"} | '
            f"desc='{self.current_sub_task.description if self.current_sub_task else 'None'}'"
        )

    def _handle_task_start(self):
        """
        Initialise a new task or clear a completed one.

        Called when current_status is IDLE or JOB_DONE.

        New task (task_id or task_type changed):
          - Cache the new task identifiers to prevent re-triggering on repeated publishes
          - Clear undocking state in case the previous task left stale values
          - Transition to JOB_START to kick off subtask execution

        Same task still publishing after JOB_DONE:
          - Transition to IDLE so the node remains idle until a genuinely new task arrives
        """
        if not self.current_task:
            return

        if (
            self.current_task.task_id != self.last_handled_task_id
            or self.current_task.task_type != self.last_handled_task_type
        ):
            self.get_logger().info(
                f'Starting Task: {self.current_task.description} | '
                f'ID: {self.current_task.task_id} | '
                f'Current Node: {self.current_node_id} | '
                f'Target Node: {self.current_task.target_node_id}'
            )
            self.get_logger().debug(
                f'Task start | task_type={self.current_task.task_type} | '
                f'last_task_id={self.last_handled_task_id} | '
                f'last_task_type={self.last_handled_task_type} | '
                f'num_subtasks={len(self.current_task.sub_tasks) if isinstance(self.current_task.sub_tasks, list) else 0}'
            )

            # Cache to prevent re-triggering JOB_START on repeated publishes
            self._arm_send_attempts = 0
            self.last_handled_task_id = self.current_task.task_id
            self.last_handled_task_type = self.current_task.task_type
            self.last_handled_subtask_type = None
            # Clear any stale undocking state from a previous task
            self.last_undocking_subtask = None
            self.undocking_after_task_type = None
            self.done_undocking_published = False
            self._transition_status(RobotStatusEnum.JOB_START)

        elif self.current_status == RobotStatusEnum.JOB_DONE:
            # The server republishes the task every tick until it has seen JOB_DONE; both loops run at 1 Hz, so a
            # JOB_DONE held for a single tick was missed now and then and the task never closed (the robot then
            # sat IDLE ignoring republishes of it). Hold JOB_DONE while republishes keep coming: IDLE once they
            # stopped for timing.job_done_hold s (the server saw it) or a new task arrived (above).
            if self._last_task_msg_ns is not None and \
                    (self.get_clock().now().nanoseconds - self._last_task_msg_ns) * 1e-9 < self.timing_job_done_hold:
                return
            self.get_logger().debug(
                f'JOB_DONE — task {self.current_task.task_id} no longer published — transitioning to IDLE'
            )
            self._transition_status(RobotStatusEnum.IDLE)
            self.current_task = None

    def _execute_current_subtask(self):
        """
        Route to the correct subtask handler based on current_sub_task.type.
        """
        if not isinstance(self.current_sub_task, SubTask):
            self.get_logger().debug(
                f'_execute_current_subtask called but no valid subtask | status={self.current_status.name}'
            )
            return

        # Log only on subtask type change to avoid flooding at 1Hz
        if self.current_sub_task.type != self.last_handled_subtask_type:
            self.get_logger().info(f'Executing: {self.current_sub_task.description}')
            self.last_handled_subtask_type = self.current_sub_task.type

        task_handler_map = {
            SubTask.MOVING: self._subtask_moving,
            SubTask.HARVESTING: self._subtask_harvesting,
            SubTask.DOCKING: self._subtask_docking,
            SubTask.CHARGING: self._subtask_charging,
            SubTask.UNLOADING: self._subtask_unloading,
        }

        handler = task_handler_map.get(self.current_sub_task.type)
        if handler:
            handler()
        else:
            self.get_logger().warning(f'Unknown subtask type: {self.current_sub_task.type} — no handler registered')

    # =========================================================================
    # ACTION CLIENT HANDLERS
    # =========================================================================

    def _handle_navigation(self, robot_status: RobotStatus):
        """
        Monitor NavigationActionClient and update RobotStatus.

        ACTIVE    → ensure status is MOVING
        SUCCEEDED → reset client, update current_node_id, → DESTINATION_REACHED
        ABORTED/ERROR → retry logic via _handle_navigation_retry
        CANCELED  → reset client, → IDLE
        """
        nav_status = self.navigation.get_navigation_status()
        wpf_status = self.navigation.get_current_status()

        # Prefer live waypoint-following status for the published task field
        if wpf_status:
            robot_status.task = wpf_status.task
            robot_status.current_node_id = wpf_status.current_node_id
            robot_status.target_node_id = wpf_status.target_node_id
        else:
            robot_status.task = self.current_task.description if self.current_task else ''
            robot_status.current_node_id = self.current_node_id
            robot_status.target_node_id = self.current_task.target_node_id if self.current_task else -1

        self.get_logger().debug(
            f'Navigation handler | nav_status={nav_status.name} | '
            f'current_node={self.current_node_id} | '
            f'target_node={robot_status.target_node_id} | '
            f'retry_count={self.navigation_retry_count}/{self.navigation_max_retries}'
        )

        if nav_status == NavigationStatus.ACTIVE:
            self._transition_status(RobotStatusEnum.MOVING)

        elif nav_status == NavigationStatus.SUCCEEDED:
            self.get_logger().info('Navigation complete')
            self.navigation_retry_count = 0
            self.navigation.reset()
            # Advance tracked node to the waypoint just reached
            self.current_node_id = wpf_status.target_node_id if wpf_status else self.current_node_id
            self._transition_status(RobotStatusEnum.DESTINATION_REACHED)

        elif nav_status in [NavigationStatus.ABORTED, NavigationStatus.ERROR]:
            self.get_logger().debug(
                f'Navigation aborted/error | nav_status={nav_status.name} | retry_count={self.navigation_retry_count}'
            )
            self._handle_navigation_retry()

        elif nav_status == NavigationStatus.CANCELED:
            self.get_logger().info('Navigation canceled')
            self.navigation.reset()
            self.navigation_retry_count = 0
            # Nav2 is still winding the cancelled goal down: the next goal waits navigation.retry_delay (a new
            # follow_path in the same tick left the a200's MPPI controller failing every cycle, 2026-10-13)
            self._wait_ready('post_cancel', self.navigation_retry_delay)
            self._transition_status(RobotStatusEnum.IDLE)

    def _handle_navigation_retry(self):
        """
        Handle a failed NavigateThroughPoses goal.

        First checks if the robot is already close enough to the target
        (within 0.25m) — if so, treats it as a success. This handles the
        case where Nav2 aborts near the goal but the robot has effectively arrived.

        Otherwise retries up to navigation_max_retries times with a delay,
        then transitions to ERROR.
        """
        self.get_logger().warning(
            f'Navigation failed | retry {self.navigation_retry_count + 1}/{self.navigation_max_retries} | '
            f'current_node={self.current_node_id}',
            throttle_duration_sec=2.0,
        )

        # If robot is physically at the target despite nav failure, treat as success
        if self._is_robot_at_target():
            self.get_logger().info('Robot already at target — treating navigation as complete')
            wpf_status = self.navigation.get_current_status()
            self.navigation_retry_count = 0
            self.navigation.reset()
            self.current_node_id = wpf_status.target_node_id if wpf_status else self.current_node_id
            self._transition_status(RobotStatusEnum.DESTINATION_REACHED)
            return

        if self.navigation_retry_count < self.navigation_max_retries:
            # Non-blocking dwell — re-checked on the next timer tick
            if not self._wait_ready('nav_retry', self.navigation_retry_delay):
                return
            self.navigation_retry_count += 1
            self.get_logger().info(
                f'Retrying navigation | attempt {self.navigation_retry_count}/{self.navigation_max_retries}'
            )
            self._retry_navigation()
        else:
            self.get_logger().error(
                f'Navigation failed after {self.navigation_max_retries} retries | '
                f'current_node={self.current_node_id} | '
                f'target_node={self.current_task.target_node_id if self.current_task else "None"}'
            )
            self.navigation.reset()
            self.navigation_retry_count = 0
            self._fail_current_task(
                'nav', f'{self.navigation_max_retries} retries, target_node='
                       f'{self.current_task.target_node_id if self.current_task else "?"}')

    def _retry_navigation(self):
        """
        Re-send the NavigateThroughPoses goal for the current task.

        Guards against sending a new goal when the previous one is still
        active (ACTIVE or SENDING) — can happen if the retry delay is too short.
        """
        if not self.current_task:
            self.get_logger().error('Navigation retry failed — no current task')
            return

        nav_status = self.navigation.get_navigation_status()
        self.get_logger().debug(
            f'Retry navigation | nav_status={nav_status.name} | '
            f'attempt={self.navigation_retry_count}/{self.navigation_max_retries}'
        )

        # Do not send a new goal if the previous one has not terminated yet
        if nav_status in [NavigationStatus.ACTIVE, NavigationStatus.SENDING]:
            self.get_logger().warning(f'Navigation still active ({nav_status.name}) — skipping retry until complete')
            return

        self.navigation.reset()

        if self.navigation.send_goal(self.current_task):
            self._transition_status(RobotStatusEnum.MOVING)
        else:
            self.get_logger().error(
                f'Navigation retry send_goal failed | '
                f'attempt={self.navigation_retry_count}/{self.navigation_max_retries}'
            )
            self._fail_current_task('nav', 'navigate_through_poses not available')

    def _handle_docking(self, robot_status: RobotStatus):
        """
        Monitor DockingActionClient and update RobotStatus.

        DOCKING      → ensure status is DOCKING
        DONE_DOCKING → reset client, → DONE_DOCKING (next subtask picks up)
        ERROR        → retry via _handle_docking_retry
        """
        status = self.docking_action_client.get_status()
        feedback = self.docking_action_client.get_feedback()

        if feedback:
            robot_status.task = feedback.task
            robot_status.current_node_id = self.current_node_id
        else:
            robot_status.task = 'Docking in progress'

        self.get_logger().debug(
            f'Docking handler | status={status.name} | '
            f'retry_count={self.docking_retry_count}/{self.docking_max_retries}'
        )

        if status == RobotStatusEnum.DOCKING:
            self._transition_status(RobotStatusEnum.DOCKING)

        elif status == RobotStatusEnum.DONE_DOCKING:
            self.get_logger().info('Docking complete')
            if self.current_sub_task and self.current_sub_task.dock_goal.dock_id:
                self._last_docked_id = self.current_sub_task.dock_goal.dock_id
            self._transition_status(RobotStatusEnum.DONE_DOCKING)
            self.docking_action_client.reset()
            self.docking_retry_count = 0

        elif status == RobotStatusEnum.ERROR:
            if not self.current_sub_task or self.current_sub_task.type != SubTask.DOCKING \
                    or self.current_status not in (RobotStatusEnum.START_DOCKING, RobotStatusEnum.DOCKING):
                # a late result of an earlier dock goal (the subtask has moved on): not a reason to dock again
                self.get_logger().warning(f'Docking client error outside the docking subtask (status '
                                          f'{self.current_status.name}): ignored')
                self.docking_action_client.reset()
                return
            self.get_logger().debug(
                f'Docking error — triggering retry | retry_count={self.docking_retry_count}/{self.docking_max_retries}'
            )
            self._handle_docking_retry()

    def _handle_docking_retry(self):
        """Retry dock_robot goal up to docking_max_retries, then go to ERROR."""
        dock_id = (
            self.current_sub_task.dock_goal.dock_id
            if self.current_sub_task and self.current_sub_task.dock_goal
            else 'unknown'
        )
        self.get_logger().error(
            f"Docking failed | retry {self.docking_retry_count + 1}/{self.docking_max_retries} | dock_id='{dock_id}'",
            throttle_duration_sec=2.0,
        )

        if self.docking_retry_count < self.docking_max_retries:
            # Non-blocking dwell — re-checked on the next timer tick
            if not self._wait_ready('dock_retry', self.docking_retry_delay):
                return
            self.docking_retry_count += 1
            self.get_logger().info(f'Retrying docking | attempt {self.docking_retry_count}/{self.docking_max_retries}')
            self._retry_docking()
        else:
            self.get_logger().error(f'Docking failed after {self.docking_max_retries} retries')
            self.docking_action_client.reset()
            self.docking_retry_count = 0
            self._fail_current_task('dock', f'{self.docking_max_retries} retries, dock_id={dock_id}')

    def _retry_docking(self):
        """Re-send the dock_robot goal using the current subtask's DockGoal."""
        if not self.current_sub_task:
            self.get_logger().error('Docking retry failed — no current subtask')
            return
        self.get_logger().debug(
            f'Retry docking | attempt={self.docking_retry_count}/{self.docking_max_retries} | '
            f"subtask='{self.current_sub_task.description}'"
        )
        if self.docking_action_client.send_docking_goal(self.current_sub_task):
            self._transition_status(RobotStatusEnum.DOCKING)
        else:
            self.get_logger().error(
                f'Docking retry send_goal failed | attempt={self.docking_retry_count}/{self.docking_max_retries}'
            )
            self._fail_current_task('dock', 'dock_robot not available')

    def _handle_undocking(self, robot_status: RobotStatus):
        """
        Monitor UndockingActionClient and update RobotStatus.

        Context-aware DONE_UNDOCKING handling:
          startup context (startup_undock_complete=False):
            Set startup_undock_complete=True, clear task IDs, → IDLE
            Next timer_callback tick enters _handle_task_execution.

          task context (startup_undock_complete=True):
            Stay at DONE_UNDOCKING. _subtask_charging or _subtask_unloading
            will call _subtask_undocking on the next tick which transitions
            to JOB_DONE.

        On ERROR:
          Delegate to _handle_undocking_retry which triggers ReverseDriveClient.
        """
        status = self.undocking_action_client.get_status()
        feedback = self.undocking_action_client.get_feedback()

        if feedback:
            robot_status.task = feedback.task
            robot_status.current_node_id = self.current_node_id
        else:
            robot_status.task = 'Undocking in progress'

        self.get_logger().debug(
            f'Undocking handler | status={status.name} | startup_undock_complete={self.startup_undock_complete}'
        )

        if status == RobotStatusEnum.UNDOCKING:
            self._transition_status(RobotStatusEnum.UNDOCKING)

        elif status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info('Undocking complete')
            self._transition_status(RobotStatusEnum.DONE_UNDOCKING)
            self.undocking_action_client.reset()

            if not self.startup_undock_complete:
                # Startup context: robot has left the dock, ready for tasks
                self.get_logger().debug('Undocking DONE — startup context: setting startup_undock_complete=True → IDLE')
                self.startup_undock_complete = True
                # Clear task IDs so the first real task is treated as new
                self.last_handled_task_id = None
                self.last_handled_task_type = None
                self._transition_status(RobotStatusEnum.IDLE)
                self.get_logger().info('Robot ready for tasks')
            else:
                # Task context: stay at DONE_UNDOCKING — _subtask_undocking
                # will pick this up on the next tick and transition to JOB_DONE
                self.get_logger().debug(
                    'Undocking DONE — task context: staying at DONE_UNDOCKING for _subtask_undocking'
                )

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().debug(
                f'Undocking error — triggering reverse drive fallback | '
                f'startup_undock_complete={self.startup_undock_complete}'
            )
            self._handle_undocking_retry()

    def _handle_undocking_retry(self):
        """
        Undocking action failed — start ReverseDriveClient as fallback.

        The undock_robot action is ABORTED (status 6, mapped to ERROR=94).
        ReverseDriveClient drives the robot in reverse using TF closed-loop
        feedback to reach the pre-computed staging pose.

        Safety: If dock_backwards=True in motion_config, reversing is unsafe
        and the node transitions to ERROR for manual recovery.
        """
        self.undocking_action_client.reset()
        self.get_logger().warning(
            f"Undocking failed — starting reverse drive to staging pose | dock='{self.active_dock_name or 'unknown'}'"
        )

        if self.active_dock_name and self.reverse_drive_client.drive_to_staging(self.active_dock_name):
            # drive_to_staging returns True when REVERSING begins successfully
            self.reverse_drive_active = True
            self.get_logger().debug(f'ReverseDriveClient started | reverse_drive_active={self.reverse_drive_active}')
            self._transition_status(RobotStatusEnum.UNDOCKING)
        else:
            # Returns False when dock_backwards=True, dock unknown, or already reversing
            self.get_logger().error(
                f'ReverseDriveClient refused to start — '
                f"dock='{self.active_dock_name}' "
                f'dock_backwards={self.motion_config.dock_backwards}'
            )
            self._fail_current_task('undock', f'undock_robot failed, no reverse drive (dock={self.active_dock_name})')

    def _handle_reverse_drive(self, robot_status: RobotStatus):
        """
        Monitor ReverseDriveClient and map ReverseDriveStatus to RobotStatusEnum.

        Same context-aware logic as _handle_undocking for the DONE branch:
          startup context → startup_undock_complete=True → DONE_UNDOCKING → IDLE
          task context    → DONE_UNDOCKING (let _subtask_undocking → JOB_DONE)

        REVERSING → UNDOCKING (stay, robot is moving)
        DONE      → context-dependent (see above)
        ERROR     → ERROR
        CANCELED  → IDLE
        """
        robot_status.task = 'Reverse drive to staging pose'
        status = self.reverse_drive_client.get_status()

        self.get_logger().debug(
            f'Reverse drive handler | status={status.name} | startup_undock_complete={self.startup_undock_complete}'
        )

        if status == ReverseDriveStatus.REVERSING:
            self._transition_status(RobotStatusEnum.UNDOCKING)

        elif status == ReverseDriveStatus.DONE:
            self.get_logger().info('Reverse drive complete — undocking done')
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()

            if not self.startup_undock_complete:
                # Startup context: robot reached staging pose, ready for tasks
                self.get_logger().debug(
                    'Reverse drive DONE — startup context: setting startup_undock_complete=True → IDLE'
                )
                self.startup_undock_complete = True
                self.last_handled_task_id = None
                self.last_handled_task_type = None
                self._transition_status(RobotStatusEnum.DONE_UNDOCKING)
                self._transition_status(RobotStatusEnum.IDLE)

                self.get_logger().info('Robot ready for tasks')
            else:
                # Task context: stay at DONE_UNDOCKING for _subtask_undocking
                self.get_logger().debug(
                    'Reverse drive DONE — task context: staying at DONE_UNDOCKING for _subtask_undocking'
                )
                self._transition_status(RobotStatusEnum.DONE_UNDOCKING)

        elif status == ReverseDriveStatus.ERROR:
            self.get_logger().error(
                f'Reverse drive failed — transitioning to ERROR | '
                f'startup_undock_complete={self.startup_undock_complete}'
            )
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()
            self._fail_current_task('undock', 'reverse drive to the staging pose failed')

        elif status == ReverseDriveStatus.CANCELED:
            self.get_logger().warning(
                f'Reverse drive canceled — transitioning to IDLE | '
                f'startup_undock_complete={self.startup_undock_complete}'
            )
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()
            self._transition_status(RobotStatusEnum.IDLE)

    # =========================================================================
    # ARM SAFETY
    # =========================================================================

    def _arm_stow_ref(self) -> SubTask:
        """A SubTask to attach to a STOW goal when no task subtask is in play."""
        ref = self.current_sub_task or self.last_undocking_subtask
        if isinstance(ref, SubTask):
            return ref
        st = SubTask()
        st.description = 'Arm STOW gate'
        return st

    def _arm_goal_watch(self, active: bool, timeout: float | None = None):
        """Arm or clear the arm-goal result timeout (key 'arm_goal'); timeout per command (stow / ready / harvest)."""
        if active:
            self._arm_goal_timeout_current = timeout if timeout is not None else self.arm_stow_timeout
            self._dwell_deadlines.setdefault(
                'arm_goal', self.get_clock().now() + Duration(seconds=self._arm_goal_timeout_current)
            )
        else:
            self._dwell_deadlines.pop('arm_goal', None)

    def _arm_send_ready(self) -> bool:
        """False while a re-send of a rejected / untaken arm goal still waits for arm.send_retry_delay."""
        if 'arm_send_retry' not in self._dwell_deadlines:
            return True
        return self._wait_ready('arm_send_retry', self.arm_send_retry_delay)

    def _arm_send_failed(self, what: str):
        """An arm goal could not be sent, or was rejected / not taken: re-send it later (the state handler sends
        again once _arm_send_ready), or give the task up after arm.send_retries."""
        err = self.manipulator_client.last_error or 'send failed'
        retryable = self.manipulator_client.last_rejected or 'not available' in err
        self.manipulator_client.reset()
        if retryable and self._arm_send_attempts < self.arm_send_retries:
            self._arm_send_attempts += 1
            self.get_logger().warning(
                f'Arm {what} goal: {err} — retry {self._arm_send_attempts}/{self.arm_send_retries} '
                f'in {self.arm_send_retry_delay:.0f} s')
            self._wait_ready('arm_send_retry', self.arm_send_retry_delay)  # arms the dwell
            return
        self._fail_current_task('harvest' if what == 'harvest' else 'arm', f'{what}: {err}')

    def _arm_stow_gate(self) -> bool:
        """
        Shared arm-safety gate. Returns True when the arm is confirmed in STOW.

        Otherwise dispatches a STOW goal once, sets arm_stow_pending and the goal
        timeout, and returns False so the caller holds and re-checks next tick.
        _handle_manipulator clears the flag on completion.

        Used by _subtask_moving, _subtask_docking and _handle_error_recovery. The
        undock flow keeps its own callback-driven variant in _subtask_undocking
        for same-tick continuation.
        """
        if self.last_confirmed_arm_command == ArmCommand.GO_STOW:
            return True
        if not self.arm_stow_pending:
            if not self._arm_send_ready():
                return False
            self.get_logger().info(
                f"Arm not in STOW (last='{self.last_confirmed_arm_command}') — sending STOW"
            )
            if self.manipulator_client.send_stow_goal(self._arm_stow_ref()):
                self.arm_stow_pending = True
                self._arm_goal_watch(True, self.arm_stow_timeout)
            else:
                self._arm_send_failed('stow')
        else:
            self.get_logger().debug('Arm STOW in progress — waiting')
        return False

    def _handle_manipulator(self, robot_status: RobotStatus | None = None):
        """
        Monitor ManipulatorTaskActionClient for pending STOW, READY, or START_HARVEST goals.

        Called from _process_action_clients every tick while any arm operation is pending,
        and directly from _handle_startup_undocking / _handle_error_recovery.

        Context-aware completion handling:
          STOW completion:
            - START_UNDOCKING context → arm is safe, re-enter _subtask_undocking to proceed
            - DONE_HARVESTING context → arm is safe, _subtask_harvesting will advance to
                                        JOB_DONE on the next tick
            - START_MOVING / START_DOCKING / ERROR / ABNORMAL → gate clears, caller re-checks
          READY completion:
            - DESTINATION_REACHED context → arm is prepared, _subtask_harvesting will
                                            advance to START_HARVESTING on the next tick
          START_HARVEST completion (DONE_HARVESTING result from manipulator):
            - HARVESTING context → harvest cycle complete, transition to DONE_HARVESTING
                                   so _subtask_harvesting can begin STOW on next tick

        A rejected / untaken arm goal is re-sent (arm.send_retries); any other arm
        failure, or a goal exceeding its timeout (arm.*_timeout), fails the task
        (_fail_current_task: FAILED:arm / FAILED:harvest) and recovery stows the arm.
        """
        arm_status = self.manipulator_client.get_status()

        self.get_logger().debug(
            f'Manipulator handler | arm_status={arm_status.name} | '
            f'current_status={self.current_status.name} | '
            f'arm_stow_pending={self.arm_stow_pending} | '
            f'arm_ready_pending={self.arm_ready_pending} | '
            f"last_confirmed_arm='{self.last_confirmed_arm_command}'"
        )

        # ----------------------------------------------------------------
        # Arm goal result timeout
        # ----------------------------------------------------------------
        deadline = self._dwell_deadlines.get('arm_goal')
        if deadline is not None and self.get_clock().now() >= deadline and arm_status != RobotStatusEnum.ERROR:
            self.get_logger().error(
                f'Arm goal timed out after {self._arm_goal_timeout_current:.0f}s | '
                f'context={self.current_status.name} | '
                f'stow_pending={self.arm_stow_pending} ready_pending={self.arm_ready_pending}'
            )
            harvesting = (self.current_status == RobotStatusEnum.HARVESTING
                          and not self.arm_stow_pending and not self.arm_ready_pending)
            self.arm_stow_pending = False
            self.arm_ready_pending = False
            self._arm_goal_watch(False)
            try:
                self.manipulator_client.cancel_goal()
            except Exception:  # noqa: BLE001
                pass
            self.manipulator_client.reset()
            self._fail_current_task('harvest' if harvesting else 'arm',
                                    f'timed out after {self._arm_goal_timeout_current:.0f}s')
            return

        # ----------------------------------------------------------------
        # STOW completion
        # ----------------------------------------------------------------
        if self.arm_stow_pending and arm_status == RobotStatusEnum.DONE_HARVESTING:
            self.get_logger().info(f'Arm STOW confirmed | context={self.current_status.name}')
            self._arm_send_attempts = 0
            self.arm_stow_pending = False
            self.last_confirmed_arm_command = ArmCommand.GO_STOW
            self._arm_goal_watch(False)
            self.manipulator_client.reset()

            if self.current_status == RobotStatusEnum.START_UNDOCKING:
                # Undocking flow was gated on STOW — arm is now safe, proceed
                self.get_logger().debug('STOW confirmed in START_UNDOCKING context — re-entering _subtask_undocking')
                self._subtask_undocking()
            elif self.current_status == RobotStatusEnum.DONE_HARVESTING:
                # Harvest flow was gated on STOW — _subtask_harvesting will see
                # last_confirmed_arm_command == GO_STOW and advance to JOB_DONE
                # on the next timer tick. Nothing to call here.
                self.get_logger().debug(
                    'STOW confirmed in DONE_HARVESTING context — '
                    '_subtask_harvesting will advance to JOB_DONE on next tick'
                )
            elif self.current_status in (
                RobotStatusEnum.START_MOVING,
                RobotStatusEnum.START_DOCKING,
                RobotStatusEnum.ERROR,
                RobotStatusEnum.ABNORMAL,
            ):
                # Gated by _arm_stow_gate — the calling handler re-checks
                # last_confirmed_arm_command == GO_STOW on the next tick.
                self.get_logger().debug(
                    f'STOW confirmed in {self.current_status.name} context — gate clears next tick'
                )
            else:
                self.get_logger().warning(
                    f'STOW confirmed in unexpected context={self.current_status.name} — no action taken'
                )

        # ----------------------------------------------------------------
        # READY completion
        # ----------------------------------------------------------------
        elif self.arm_ready_pending and arm_status == RobotStatusEnum.DONE_HARVESTING:
            self.get_logger().info(f'Arm READY confirmed | context={self.current_status.name}')
            self._arm_send_attempts = 0
            self.arm_ready_pending = False
            self.last_confirmed_arm_command = ArmCommand.GO_READY
            self._arm_goal_watch(False)
            self.manipulator_client.reset()

            if self.current_status == RobotStatusEnum.DESTINATION_REACHED:
                # Harvest flow was gated on READY — _subtask_harvesting will see
                # last_confirmed_arm_command == GO_READY and advance to START_HARVESTING
                # on the next timer tick. Nothing to call here.
                self.get_logger().debug(
                    'READY confirmed in DESTINATION_REACHED context — '
                    '_subtask_harvesting will advance to START_HARVESTING on next tick'
                )
            else:
                self.get_logger().warning(
                    f'READY confirmed in unexpected context={self.current_status.name} — no action taken'
                )

        # ----------------------------------------------------------------
        # START_HARVEST completion
        # ----------------------------------------------------------------
        elif (
            not self.arm_stow_pending
            and not self.arm_ready_pending
            and arm_status == RobotStatusEnum.DONE_HARVESTING
            and self.current_status == RobotStatusEnum.HARVESTING
        ):
            fb = self.manipulator_client.get_feedback()
            self.get_logger().info('Harvest goal complete — transitioning to DONE_HARVESTING'
                                   + (f' ({fb.feedback_message})' if fb else ''))
            self._arm_send_attempts = 0
            self._arm_goal_watch(False)
            self.manipulator_client.reset()
            self._transition_status(RobotStatusEnum.DONE_HARVESTING)

        # ----------------------------------------------------------------
        # ERROR — any arm command failure
        # ----------------------------------------------------------------
        elif arm_status == RobotStatusEnum.ERROR:
            err = self.manipulator_client.last_error or 'failed'
            self.get_logger().error(
                f'Arm command failed: {err} | context={self.current_status.name} | '
                f'stow_pending={self.arm_stow_pending} | '
                f'ready_pending={self.arm_ready_pending} | '
                f"last_confirmed='{self.last_confirmed_arm_command}'"
            )
            harvesting = (self.current_status == RobotStatusEnum.HARVESTING
                          and not self.arm_stow_pending and not self.arm_ready_pending)
            what = 'harvest' if harvesting else ('stow' if self.arm_stow_pending else 'ready')
            self.arm_stow_pending = False
            self.arm_ready_pending = False
            self._arm_goal_watch(False)
            if harvesting:
                # back to the send state: _arm_send_failed either re-sends (busy / server down) or fails the task
                self._transition_status(RobotStatusEnum.START_HARVESTING)
            self._arm_send_failed(what)

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    def _handle_error_recovery(self, robot_status: RobotStatus | None = None):
        """
        Attempt to recover from ERROR or ABNORMAL status.

        Cancels any active navigation goal (non-blocking wait for the cancel to
        propagate), brings the arm to STOW, resets subtask state, → IDLE.

        Arm-caused ERROR: if the ERROR came from an arm failure the arm is
        untrusted — recovery does NOT re-command it and the node holds in ERROR
        for manual intervention (restart).

        Low-battery ERROR: the arm is still brought to STOW (safe to sit idle),
        but the node holds in ERROR — it does NOT transition to IDLE — until a
        CHARGING_TASK is the current task. This method is the only place that
        exits a low-battery ERROR; _check_and_handle_low_battery never runs while
        current_status == ERROR (timer_callback routes ERROR here instead), so it
        cannot see a CHARGING_TASK arrive.
        """
        if self.current_status not in [RobotStatusEnum.ERROR, RobotStatusEnum.ABNORMAL]:
            return

        # The failure report goes out on every tick in ERROR, from the first one (the fleet server reads it)
        if self._failure is not None and robot_status is not None:
            robot_status.task = self._failure_report()

        if self._error_from_arm:
            if robot_status is not None:
                robot_status.task = self._failure_report('arm_stow')
            self.get_logger().error(
                'ERROR: the arm could not be stowed — arm untrusted, holding for manual '
                'intervention (restart node)',
                throttle_duration_sec=5.0,
            )
            return

        if self._startup_undock_failed:
            if robot_status is not None:
                robot_status.task = self._failure_report('startup_undock')
            self.get_logger().error(
                'Startup undocking exhausted its retries — holding in ERROR for manual '
                'intervention (restart node)',
                throttle_duration_sec=5.0,
            )
            return

        self.get_logger().warning(
            f'Error recovery | status={self.current_status.name} | '
            f'task_id={self.last_handled_task_id} | '
            f'subtask_type={self.last_handled_subtask_type}',
            throttle_duration_sec=2.0,
        )

        nav_status = self.navigation.get_navigation_status()
        self.get_logger().debug(f'Error recovery | nav_status={nav_status.name}')

        if nav_status in [NavigationStatus.ACTIVE, NavigationStatus.ACCEPTED]:
            # Cancel once (when the dwell is first armed), then wait non-blocking
            # for Nav2 to process the cancellation before resetting state.
            if 'err_recovery' not in self._dwell_deadlines:
                self.get_logger().info('Cancelling active navigation during error recovery')
                try:
                    self.navigation.cancel_goal()
                except Exception as e:  # noqa: BLE001 - defensive: cancel must never abort recovery
                    self.get_logger().warning(f'Navigation cancel raised exception: {e}')
            if not self._wait_ready('err_recovery', self.navigation_retry_delay):
                return

        # Clear any lingering dwell deadline from the cancel path above
        self._dwell_deadlines.pop('err_recovery', None)

        # A pending READY is abandoned; recovery drives the arm to STOW instead.
        self.arm_ready_pending = False

        # Arm safety gate — hold in ERROR until the arm is confirmed STOW.
        if not self._arm_stow_gate():
            self._handle_manipulator(robot_status)
            return
        self.manipulator_client.reset()

        # The report stays up for error.report_hold so the fleet server (1 Hz) sees it a few times.
        if self._failure is not None and not self._wait_ready('fail_report', self.error_report_hold):
            return

        # Low battery: the arm is stowed, but stay in ERROR until the server sends a new CHARGING_TASK.
        # Recovering to IDLE would let _check_and_handle_low_battery fail the next task at once.
        if self._failure_hold:
            failed_id = self._failure[1] if self._failure else None
            if self.task and self.task.task_type == Task.CHARGING_TASK and self.task.task_id != failed_id:
                self.get_logger().info('Recovering from low battery — CHARGING_TASK received')
                self._failure_hold = False
            else:
                self.get_logger().warning(
                    'Holding in ERROR — battery critical, waiting for a CHARGING_TASK',
                    throttle_duration_sec=5.0,
                )
                return

        # Clean up and idle. last_handled_task_id keeps the failed task's id: its republishes don't restart it; a
        # task with a new id does.
        self._failure = None
        self._arm_send_attempts = 0
        self.current_sub_task = None
        self.current_sub_task_index = 0
        self.current_task = None
        self.task = Task()
        self.get_logger().debug('Error recovery — arm stowed, reset subtask state, transitioning to IDLE')
        self._transition_status(RobotStatusEnum.IDLE)

    def _failure_report(self, kind: str | None = None) -> str:
        """RobotStatus.task while in ERROR: 'FAILED:<kind>:<task_id>:<detail>'."""
        if self._failure is None:
            task_id = self.last_handled_task_id if self.last_handled_task_id is not None else -1
            return f'FAILED:{kind or "other"}:{task_id}:'
        k, task_id, detail = self._failure
        return f'FAILED:{kind or k}:{task_id}:{detail}'

    def _cancel_everything(self):
        """Cancel every goal in flight (navigation, docking, undocking, reverse drive, arm); never raises."""
        for name, cancel in (('navigation', self.navigation.cancel_goal),
                             ('docking', self.docking_action_client.cancel_goal),
                             ('undocking', self.undocking_action_client.cancel_goal),
                             ('arm', self.manipulator_client.cancel_goal)):
            try:
                cancel()
            except Exception as e:  # noqa: BLE001 - a cancel must never abort the failure path
                self.get_logger().warning(f'{name} cancel raised: {e}')
        if self.reverse_drive_active:
            self.reverse_drive_active = False
            try:
                self.reverse_drive_client.reset()
            except Exception as e:  # noqa: BLE001
                self.get_logger().warning(f'reverse drive reset raised: {e}')
        self.arm_stow_pending = False
        self.arm_ready_pending = False
        self._arm_goal_watch(False)
        self._dwell_deadlines.pop('arm_send_retry', None)
        self._dwell_deadlines.pop('fail_report', None)

    def _fail_current_task(self, kind: str, detail: str, hold: bool = False):
        """Give the current task up: cancel what runs, go to ERROR and report
        'FAILED:<kind>:<task_id>:<detail>' in RobotStatus.task (kind: nav, dock, undock, arm, harvest, battery,
        estop) until the server, which closes the task and sends a new task_id, has had error.report_hold to see
        it; then recover (stow) to IDLE. hold: stay in ERROR after the report until a CHARGING_TASK arrives.
        While already recovering (status ERROR), an arm failure means the stow failed: the arm is untrusted and
        the node holds (FAILED:arm_stow) until restarted."""
        task_id = self.current_task.task_id if self.current_task else (self.task.task_id if self.task else -1)
        detail = (detail or '').replace(':', ';').replace('\n', ' ')[:120]
        if self.current_status in (RobotStatusEnum.ERROR, RobotStatusEnum.ABNORMAL) and kind in ('arm', 'harvest'):
            self.get_logger().error(f'Stow failed during error recovery ({detail}): holding in ERROR')
            self._error_from_arm = True
            self._failure = ('arm_stow', task_id, detail)
            return
        self.get_logger().error(f'Task {task_id} failed ({kind}): {detail}')
        self._cancel_everything()
        self._failure = (kind, task_id, detail)
        self._failure_hold = hold
        self._transition_status(RobotStatusEnum.ERROR)

    # =========================================================================
    # SUBTASK HANDLERS
    # =========================================================================

    def _subtask_moving(self):
        """
        Handle the MOVING subtask.

        State progression:
          JOB_START → START_MOVING → send NavigateThroughPoses → MOVING
          (navigation result handled by _handle_navigation via _process_action_clients)

        Guards against sending a duplicate goal if navigation is already active —
        can happen if timer fires multiple times while waiting for goal acceptance.

        Arm safety: the arm must be confirmed STOW before a navigation goal is
        sent — navigating with the arm extended risks collisions / tip-over.
        """
        if not self.current_task:
            return

        self.get_logger().debug(
            f'_subtask_moving | status={self.current_status.name} | '
            f'current_node={self.current_node_id} | '
            f'target_node={self.current_task.target_node_id}'
        )

        if self.current_status == RobotStatusEnum.JOB_START:
            self._transition_status(RobotStatusEnum.START_MOVING)

        elif self.current_status == RobotStatusEnum.START_MOVING:
            # after a cancelled navigation, give Nav2 a moment before the next goal
            if 'post_cancel' in self._dwell_deadlines and not self._wait_ready('post_cancel', self.navigation_retry_delay):
                return
            # Arm safety gate — hold here until the arm is confirmed STOW
            if not self._arm_stow_gate():
                return
            if is_no_move(self.current_task):
                # A dock task's move outside a lane is the start node twice (status_server _build_dock_task): the
                # docking server drives to its own staging pose. Navigating to that node's exact pose can mean a
                # goal a few cm behind the robot, which a forward-only MPPI can't reach: it turned on the spot.
                self.get_logger().info('No move: the docking server drives to its staging pose')
                self._transition_status(RobotStatusEnum.DESTINATION_REACHED)
                return
            # Guard: do not send a second goal if the previous one is still being accepted
            if self.navigation.is_navigation_active():
                self.get_logger().warning(
                    f'Navigation send skipped — already active | '
                    f'nav_status={self.navigation.get_navigation_status().name}'
                )
                return
            self.get_logger().info(f'Starting navigation: {self.current_node_id} → {self.current_task.target_node_id}')
            if self.navigation.send_goal(self.current_task):
                self._transition_status(RobotStatusEnum.MOVING)
            else:
                self.get_logger().error(
                    f'Failed to send navigation goal | '
                    f'task_id={self.current_task.task_id} | '
                    f'target_node={self.current_task.target_node_id}'
                )
                self._fail_current_task('nav', 'navigate_through_poses not available')

    def _subtask_docking(self):
        """
        Handle the DOCKING subtask.

        State progression:
          DESTINATION_REACHED → START_DOCKING → 1s delay → send dock_robot → DOCKING
          (docking result handled by _handle_docking via _process_action_clients)

        The 1s delay gives Nav2 time to decelerate fully before the docking
        controller starts, reducing the chance of a dock alignment failure.

        Arm safety: the arm must be confirmed STOW before docking begins.
        """
        dock_id = (
            self.current_sub_task.dock_goal.dock_id
            if self.current_sub_task and self.current_sub_task.dock_goal
            else 'None'
        )
        self.get_logger().debug(f"_subtask_docking | status={self.current_status.name} | dock_id='{dock_id}'")

        if self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            self._transition_status(RobotStatusEnum.START_DOCKING)

        elif self.current_status == RobotStatusEnum.START_DOCKING:
            # Arm safety gate — hold here until the arm is confirmed STOW
            if not self._arm_stow_gate():
                return
            # Brief non-blocking pause to let the robot fully stop before docking.
            # Re-checked on the next timer tick until the delay has elapsed.
            if not self._wait_ready('dock_start', 1.0):
                return
            self.get_logger().info('Started docking')
            if self.docking_action_client.send_docking_goal(self.current_sub_task):
                self._transition_status(RobotStatusEnum.DOCKING)
            else:
                self.get_logger().error(
                    f'Failed to send docking goal | '
                    f"subtask='{self.current_sub_task.description if self.current_sub_task else 'None'}'"
                )
                self._fail_current_task('dock', 'dock_robot not available')

    def _subtask_undocking(self):
        """
        Handle the UNDOCKING subtask.

        This method is called in two ways:
          1. Directly by _subtask_charging and _subtask_unloading when their
             respective tasks complete — UNDOCKING is an internal trigger, not
             a received subtask for CHARGING_TASK or UNLOADING_TASK.
          2. Via _handle_startup_undocking during the startup sequence.
          3. Via _execute_current_subtask if UNDOCKING appears as a standalone
             received subtask type (future use).

        Arm safety gate:
          Before sending the undocking goal the arm must be confirmed in STOW.
          If not, a STOW goal is dispatched and this method returns early.
          _handle_manipulator will call back into this method once STOW is confirmed.

        UndockGoal source:
          current_sub_task if available (standalone subtask case),
          otherwise last_undocking_subtask (stored by _subtask_charging/unloading
          or _handle_startup_undocking).

        State progression:
          START_UNDOCKING → (STOW gate) → send undock_robot → UNDOCKING
          DONE_UNDOCKING  → clear stored subtask → JOB_DONE
          (undocking result handled by _handle_undocking via _process_action_clients)
        """
        self.get_logger().debug(
            f'_subtask_undocking | status={self.current_status.name} | '
            f'last_undocking_subtask={"set" if self.last_undocking_subtask else "None"} | '
            f'undocking_after_task_type={self.undocking_after_task_type} | '
            f"last_confirmed_arm='{self.last_confirmed_arm_command}' | "
            f'arm_stow_pending={self.arm_stow_pending}'
        )

        if self.current_status == RobotStatusEnum.START_UNDOCKING:
            # ---- Arm safety gate ----
            # Arm must be confirmed STOW before undocking begins. _handle_manipulator
            # re-enters this method once STOW is confirmed (START_UNDOCKING context).
            if not self._arm_stow_gate():
                return

            # ---- Arm confirmed STOW — proceed with undocking ----
            self.get_logger().info('Starting undocking — arm confirmed STOW')

            undock_subtask = self.current_sub_task if self.current_sub_task else self.last_undocking_subtask
            dock_type = (
                undock_subtask.undock_goal.dock_type if undock_subtask and undock_subtask.undock_goal else 'None'
            )

            self.get_logger().debug(
                f'Undocking subtask | '
                f'source={"current_sub_task" if self.current_sub_task else "last_undocking_subtask"} | '
                f"dock_type='{dock_type}'"
            )

            # Docked in this process: the docking server remembers that dock and its plugin type, so an empty
            # dock_type undocks from it; the type the task names is only needed for the startup undock.
            dock_type_override = '' if self._last_docked_id else None
            if self.undocking_action_client.send_undocking_goal(undock_subtask, dock_type=dock_type_override):
                self._transition_status(RobotStatusEnum.UNDOCKING)
            else:
                self.get_logger().error(
                    f'Failed to send undocking goal | undocking_after_task_type={self.undocking_after_task_type}'
                )
                self._fail_current_task('undock', 'undock_robot not available')

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            # Hold for one tick so DONE_UNDOCKING is actually published.
            # timer_callback publishes once, after the state machine has run,
            # so moving straight to JOB_DONE here would mean the server never
            # sees DONE_UNDOCKING and never frees the dock.
            if not self.done_undocking_published:
                self.done_undocking_published = True
                self.get_logger().debug('Undocking done — publishing DONE_UNDOCKING before JOB_DONE')
                return

            # Undocking succeeded — clean up stored state and mark task complete
            self.get_logger().info('Undocking done')
            self.get_logger().debug(
                f'Undocking done — clearing state | undocking_after_task_type={self.undocking_after_task_type}'
            )
            self.last_undocking_subtask = None
            self.undocking_after_task_type = None
            self.done_undocking_published = False
            self._transition_status(RobotStatusEnum.JOB_DONE)

    def _subtask_harvesting(self):
        """
        Handle the HARVESTING subtask.

        State progression:
          DESTINATION_REACHED → (READY gate) → DESTINATION_REACHED (arm confirmed)
          → START_HARVESTING  → send START_HARVEST goal → HARVESTING
          → (manipulator result via _handle_manipulator) → DONE_HARVESTING
          → (STOW gate) → DONE_HARVESTING (arm confirmed) → JOB_DONE

        Arm configuration gates:
          - GO_READY must be confirmed before START_HARVESTING is entered.
            Arm is brought to READY at DESTINATION_REACHED so it is prepared
            on arrival without being extended during transit over uneven terrain.
          - GO_STOW must be confirmed before JOB_DONE is entered.
            Arm is always stowed after harvesting to keep it safe between tasks
            and to satisfy the mandatory STOW requirement before any undocking.
        """
        self.get_logger().debug(
            f'_subtask_harvesting | status={self.current_status.name} | '
            f'arm_ready_pending={self.arm_ready_pending} | '
            f'arm_stow_pending={self.arm_stow_pending} | '
            f"last_confirmed_arm='{self.last_confirmed_arm_command}'"
        )

        if self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            # Ensure arm is in READY before harvesting begins.
            # Hold here until _handle_manipulator confirms READY.
            if self.last_confirmed_arm_command != ArmCommand.GO_READY:
                if not self.arm_ready_pending:
                    if not self._arm_send_ready():
                        return
                    self.get_logger().info(
                        f"Arm not in READY (last='{self.last_confirmed_arm_command}') — sending READY before harvesting"
                    )
                    if self.manipulator_client.send_ready_goal(self.current_sub_task):
                        self.arm_ready_pending = True
                        self._arm_goal_watch(True, self.arm_ready_timeout)
                    else:
                        self._arm_send_failed('ready')
                else:
                    self.get_logger().debug('Arm READY already in progress — waiting')
                return  # Hold at DESTINATION_REACHED until arm_ready_pending clears

            # Arm confirmed in READY — proceed to harvesting
            self._transition_status(RobotStatusEnum.START_HARVESTING)

        elif self.current_status == RobotStatusEnum.START_HARVESTING:
            # Guard: do not send a second goal if manipulator is already active
            if self.manipulator_client.get_status() == RobotStatusEnum.HARVESTING:
                self.get_logger().warning('Harvest goal send skipped — manipulator already active')
                return
            if not self._arm_send_ready():
                return
            self.get_logger().info('Sending harvest goal to manipulator (START HARVEST = cut_stem)')
            if self.manipulator_client.send_harvesting_goal(self.current_sub_task):
                self._transition_status(RobotStatusEnum.HARVESTING)
                self._arm_goal_watch(True, self.arm_harvest_timeout)
            else:
                self._arm_send_failed('harvest')

        elif self.current_status == RobotStatusEnum.HARVESTING:
            # The cut runs for minutes; its result arrives through _handle_manipulator (DONE_HARVESTING, or a
            # failure / the harvest_timeout). Nothing to do here but wait.
            self.get_logger().debug('Harvesting in progress — waiting for the manipulator result')

        elif self.current_status == RobotStatusEnum.DONE_HARVESTING:
            # Update load before checking arm — load must be recorded regardless of
            # whether STOW has been sent yet. Guard with arm_stow_pending to prevent
            # re-incrementing on subsequent ticks while waiting for STOW confirmation.
            if not self.arm_stow_pending and self.last_confirmed_arm_command != ArmCommand.GO_STOW:
                new_load = min(self.current_load_status + self.loading_increment, 100.0)
                self.get_logger().debug(
                    f'Load update | {self.current_load_status:.1f}% → {new_load:.1f}% (+{self.loading_increment:.1f}%)'
                )
                self.current_load_status = new_load
                self.get_logger().info(f'Load status: {self.current_load_status:.1f}%')

                # Always stow the arm after harvesting before transitioning to JOB_DONE.
                # Hold here until _handle_manipulator confirms STOW.
                if not self._arm_send_ready():
                    return
                self.get_logger().info('Harvesting done — sending arm to STOW')
                if self.manipulator_client.send_stow_goal(self.current_sub_task):
                    self.arm_stow_pending = True
                    self._arm_goal_watch(True, self.arm_stow_timeout)
                else:
                    self._arm_send_failed('stow')
                return  # Hold until STOW confirmed

            # STOW confirmed — advance to JOB_DONE
            if self.last_confirmed_arm_command == ArmCommand.GO_STOW and not self.arm_stow_pending:
                self.get_logger().info('Arm stowed after harvest — transitioning to JOB_DONE')
                self._transition_status(RobotStatusEnum.JOB_DONE)

    def _subtask_charging(self):
        """
        Handle the CHARGING subtask.

        State progression:
          DONE_DOCKING   → START_CHARGING → CHARGING
          CHARGING       → poll battery until >= battery_full_threshold → DONE_CHARGING
          DONE_CHARGING  → store last_undocking_subtask, internally trigger undocking
          DONE_UNDOCKING → delegate to _subtask_undocking → JOB_DONE

        UNDOCKING is triggered internally here — it is NOT in the received subtask
        list for CHARGING_TASK. The UndockGoal comes from charging_sub_task.undock_goal
        stored as last_undocking_subtask.
        """
        battery_pct = self._normalize_battery(self.battery_status.percentage)
        self.get_logger().debug(
            f'_subtask_charging | status={self.current_status.name} | '
            f'battery={battery_pct:.1f}% | '
            f'full_threshold={self.battery_full_threshold}%'
        )

        if self.current_status == RobotStatusEnum.DONE_DOCKING:
            self._transition_status(RobotStatusEnum.START_CHARGING)

        elif self.current_status == RobotStatusEnum.START_CHARGING:
            self._transition_status(RobotStatusEnum.CHARGING)
            self._charge_current_ema = None  # Fresh ETA estimate per charging session
            self.get_logger().info('Charging started')

        elif self.current_status == RobotStatusEnum.CHARGING:
            # Throttled to avoid flooding logs at 1Hz during the charging wait
            eta = self._estimate_charge_eta(battery_pct)
            eta_str = eta if eta is not None else '--:--:--'
            self.get_logger().info(
                f'Battery charging: {battery_pct:.1f}% | ETA {eta_str}',
                throttle_duration_sec=10.0,
            )
            if battery_pct >= self._charge_target():
                self.get_logger().info(f'Battery charged: {battery_pct:.1f}% (target {self._charge_target():.0f}%)')
                self._transition_status(RobotStatusEnum.DONE_CHARGING)

        elif self.current_status == RobotStatusEnum.DONE_CHARGING:
            # Store the charging subtask so _subtask_undocking can retrieve its UndockGoal.
            # The STOW gate inside _subtask_undocking will fire if arm is not already stowed.
            self.active_dock_name = self._resolve_task_dock_name(
                self.current_task.task_type if self.current_task else None
            )
            self.get_logger().debug(
                f"DONE_CHARGING — dock='{self.active_dock_name}', "
                'storing last_undocking_subtask and triggering undocking'
            )
            self.last_undocking_subtask = self.current_sub_task
            self.undocking_after_task_type = Task.CHARGING_TASK
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            # Call immediately — next tick would call this handler again on CHARGING subtask
            # but current_status is now START_UNDOCKING so _subtask_undocking handles it
            self._subtask_undocking()

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            # _handle_undocking set status to DONE_UNDOCKING (task context).
            # Delegate to _subtask_undocking to transition to JOB_DONE.
            self.get_logger().debug('DONE_UNDOCKING in charging context — delegating to _subtask_undocking')
            self._subtask_undocking()

    def _subtask_unloading(self):
        """
        Handle the UNLOADING subtask.

        State progression:
          DONE_DOCKING   → START_UNLOADING → UNLOADING → DONE_UNLOADING
          DONE_UNLOADING → reset load, store last_undocking_subtask, trigger undocking
          DONE_UNDOCKING → delegate to _subtask_undocking → JOB_DONE

        UNDOCKING is triggered internally here — same pattern as _subtask_charging.
        current_load_status is reset to 0.0 when unloading completes.
        The STOW gate inside _subtask_undocking will fire if arm is not already stowed.

        TODO: Replace simulated timer with actual unloading hardware integration:
          - Connect to unloading mechanism
          - Monitor unload confirmation sensors
          - Verify complete unload before proceeding
        """
        self.get_logger().debug(
            f'_subtask_unloading | status={self.current_status.name} | current_load={self.current_load_status:.1f}%'
        )

        if self.current_status == RobotStatusEnum.DONE_DOCKING:
            self._transition_status(RobotStatusEnum.START_UNLOADING)

        elif self.current_status == RobotStatusEnum.START_UNLOADING:
            self._transition_status(RobotStatusEnum.UNLOADING)
            self.get_logger().info('Unloading started')
            # Start simulated unload timer (replace with hardware signal)
            self.job_start_time = self.get_clock().now()
            self.job_duration = 4.0  # seconds

        elif self.current_status == RobotStatusEnum.UNLOADING:
            if self.job_start_time:
                elapsed = (self.get_clock().now() - self.job_start_time).nanoseconds / 1e9
                if elapsed >= self.job_duration:
                    self.get_logger().info('Unloading complete (simulated)')
                    self.get_logger().debug(
                        f'Unloading timer done | elapsed={elapsed:.2f}s duration={self.job_duration:.1f}s'
                    )
                    self._transition_status(RobotStatusEnum.DONE_UNLOADING)
                    self.job_start_time = None

        elif self.current_status == RobotStatusEnum.DONE_UNLOADING:
            # Reset load to 0 and store undocking subtask before triggering undocking.
            # The STOW gate inside _subtask_undocking will fire if arm is not already stowed.
            self.active_dock_name = self._resolve_task_dock_name(
                self.current_task.task_type if self.current_task else None
            )
            self.get_logger().debug(
                f"DONE_UNLOADING — dock='{self.active_dock_name}', "
                'storing last_undocking_subtask and triggering undocking'
            )
            self.last_undocking_subtask = self.current_sub_task
            self.undocking_after_task_type = Task.UNLOADING_TASK
            self.current_load_status = 0.0
            self.get_logger().info('Unloading done, starting undocking')
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            self._subtask_undocking()

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            # _handle_undocking set status to DONE_UNDOCKING (task context).
            # Delegate to _subtask_undocking to transition to JOB_DONE.
            self.get_logger().debug('DONE_UNDOCKING in unloading context — delegating to _subtask_undocking')
            self._subtask_undocking()

    # =========================================================================
    # SENSOR STATUS METHODS
    # =========================================================================

    def _set_battery_status(self, robot_status: RobotStatus):
        """
        Populate battery fields in the outgoing RobotStatus.

        UNIT CONTRACT: RobotStatus.battery_level is published as a percentage
        in the 0 - 100 range. BatteryState.percentage is normalised here rather
        than by each consumer, because the raw BMS value may be 0.0 - 1.0 or
        0 - 100 depending on the driver and consumers cannot tell them apart.

        Estimates remaining runtime at the present discharge rate:
          remaining_ah / |current|
        remaining_ah is BatteryState.charge when the BMS provides it, otherwise
        capacity * percentage. current is smoothed with an EMA so a momentary
        draw spike does not make the estimate jump. BatteryState.current is
        negative while discharging (ROS convention), so its magnitude is used;
        while charging or idle (|current| ~ 0) the estimate is not published.
        """
        battery_pct = self._normalize_battery(self.battery_status.percentage)
        robot_status.battery_level = battery_pct

        placeholder = '00 hours 00 minutes remaining approx...'

        # Remaining charge in Ah
        if self.battery_status.charge > 0.0:
            remaining_ah = self.battery_status.charge
        elif self.battery_status.capacity > 0.0:
            remaining_ah = self.battery_status.capacity * (battery_pct / 100.0)
        else:
            robot_status.operation_hours_after_charging = placeholder
            return

        # EMA-smoothed discharge current magnitude (A)
        current_mag = abs(self.battery_status.current)
        if self._battery_current_ema is None:
            self._battery_current_ema = current_mag
        else:
            alpha = 0.1
            self._battery_current_ema = alpha * current_mag + (1.0 - alpha) * self._battery_current_ema

        if self._battery_current_ema < 0.01:
            # Charging or idle — remaining-runtime estimate is not meaningful
            robot_status.operation_hours_after_charging = placeholder
            return

        hours_remaining = remaining_ah / self._battery_current_ema
        robot_status.operation_hours_after_charging = self._format_time_remaining(hours_remaining)

    def _estimate_charge_eta(self, battery_pct: float) -> str | None:
        """
        Estimate time to reach battery_full_threshold while charging.

        Uses the BMS energy balance:
          remaining_ah / |charge_current|
        remaining_ah is the Ah gap between the full-threshold target and the
        present BatteryState.charge; if the BMS does not report charge, it is
        approximated from capacity and the percentage gap. The charge current
        is EMA-smoothed (alpha 0.1) so a momentary spike does not make the
        estimate jump. Linear in current — the real CV-phase taper near full
        is not modelled, so the estimate is optimistic close to the threshold.

        Returns an 'HH:MM:SS' string, or None when the BMS lacks the fields
        needed to estimate (no capacity, or charge current ~ 0).
        """
        capacity = self.battery_status.capacity
        if capacity <= 0.0:
            capacity = self.battery_status.design_capacity
        if capacity <= 0.0:
            return None

        target_ah = capacity * (self.battery_full_threshold / 100.0)
        if self.battery_status.charge > 0.0:
            remaining_ah = target_ah - self.battery_status.charge
        else:
            remaining_ah = capacity * (self.battery_full_threshold - battery_pct) / 100.0

        if remaining_ah <= 0.0:
            return '00:00:00'

        current_mag = abs(self.battery_status.current)
        if self._charge_current_ema is None:
            self._charge_current_ema = current_mag
        else:
            alpha = 0.1
            self._charge_current_ema = alpha * current_mag + (1.0 - alpha) * self._charge_current_ema

        if self._charge_current_ema < 0.01:
            return None

        eta_seconds = int((remaining_ah / self._charge_current_ema) * 3600)
        minutes, seconds = divmod(eta_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 99:
            return '99:59:59'
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    def _set_estop_status(self, robot_status: RobotStatus):
        """Map the emergency-stop signal to online_flag in the outgoing RobotStatus."""
        robot_status.online_flag = (
            OnlineFlagEnum.EMERGENCY_STOP.value if self.estop_status.data else OnlineFlagEnum.ONLINE.value
        )

    def _set_location_status(self, robot_status: RobotStatus):
        """Populate position and orientation fields from the latest TF pose."""
        if self.pose_status is not None:
            robot_status.topo_map_position = self.pose_status.pose.position
            robot_status.topo_map_orientation = self.pose_status.pose.orientation

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _wait_ready(self, key: str, delay: float) -> bool:
        """
        Non-blocking dwell. Replaces time.sleep() in the retry/dwell paths.

        First call for `key` arms a deadline `delay` seconds ahead on the ROS
        clock (respects use_sim_time). Returns False until that deadline passes,
        then clears the deadline and returns True once. The caller returns early
        while this is False and re-checks on the next timer tick.
        """
        now = self.get_clock().now()
        deadline = self._dwell_deadlines.get(key)
        if deadline is None:
            self._dwell_deadlines[key] = now + Duration(seconds=delay)
            return False
        if now >= deadline:
            del self._dwell_deadlines[key]
            return True
        return False

    def _is_robot_at_target(self):
        """
        Check if the robot is within 0.25m of the final waypoint.

        Used by _handle_navigation_retry to treat a nav abort as a success when
        the robot is physically close enough to the destination — Nav2 sometimes
        aborts a goal even when the robot has effectively arrived.

        Returns False if no subtask or pose data is available.
        """
        if not self.current_sub_task or self.pose_status is None:
            self.get_logger().debug('_is_robot_at_target: no subtask or pose — returning False')
            return False

        robot_position = self.pose_status.pose.position
        # Normalise waypoints in case the subtask data contains raw dicts
        waypoints_list = [WayPoint(**wp) if isinstance(wp, dict) else wp for wp in self.current_sub_task.data]
        target_wp = waypoints_list[-1]  # Final waypoint is the destination
        distance = self._calculate_distance(robot_position.x, robot_position.y, target_wp.x, target_wp.y)

        self.get_logger().debug(
            f'_is_robot_at_target | '
            f'robot=({robot_position.x:.3f}, {robot_position.y:.3f}) | '
            f'target=({target_wp.x:.3f}, {target_wp.y:.3f}) | '
            f'distance={distance:.3f}m threshold=0.25m'
        )
        return distance <= 0.25

    def _charge_target(self) -> float:
        """The level a charging task charges to: its CHARGING subtask's data_str {"charge_to": N} when the server
        sent one, else battery.full_threshold."""
        try:
            st = self.current_sub_task
            if st is not None and st.type == SubTask.CHARGING and st.data_str:
                value = json.loads(st.data_str).get('charge_to')
                if value is not None:
                    return float(value)
        except (ValueError, TypeError, AttributeError):
            pass
        return self.battery_full_threshold

    def _transition_status(self, new_status: RobotStatusEnum):
        """
        Safely transition to a new RobotStatusEnum value.

        No-ops if new_status equals current_status to avoid redundant log lines.
        Stores the previous status for debugging.
        """
        if self.current_status != new_status:
            self.previous_status = self.current_status
            self.current_status = new_status
            self.get_logger().info(f'Status: {self.previous_status.name} → {self.current_status.name}')

    def _normalize_battery(self, percentage: float) -> float:
        """
        Normalise battery percentage to the 0 - 100 range.

        BatteryState.percentage can be provided as 0.0 - 1.0 (ROS convention) or
        0 - 100 depending on the BMS driver. Multiplies by 100 if the value is <= 1.0.
        """
        return percentage * 100.0 if percentage <= 1.0 else percentage

    def _format_time_remaining(self, hours: float) -> str:
        """Convert a fractional hours value to a human-readable HH hours MM minutes string."""
        seconds = int(hours * 3600)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f'{hours:02} hours {minutes:02} minutes remaining approximately.'

    def _calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Return the Euclidean distance between two 2D points."""
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def main():
    # Teardown lives in finally: rclpy.spin() raises on Ctrl-C (KeyboardInterrupt)
    # or an external shutdown, so anything placed after it would never run.
    #
    # Not rclpy.spin(): under use_sim_time it never returns after SIGTERM/SIGINT. The signal shuts the context
    # down, but every timer runs on the ROS clock and /clock stops with the context, so the executor's wait has
    # nothing left to wake it (the node ignored `kill` and launch's SIGINT until SIGKILL). spin_once with a
    # wall-time timeout re-checks rclpy.ok() at least every 0.5 s.
    node = None
    executor = None
    try:
        rclpy.init()
        node = HuskyOperationsManager()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.5)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        # try_shutdown, not `if rclpy.ok(): rclpy.shutdown()`: after SIGTERM/SIGINT rclpy's own handler shuts the
        # context down concurrently and the second shutdown raised (exit code 1). Guarded for a failed init().
        try:
            rclpy.try_shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == '__main__':
    main()
