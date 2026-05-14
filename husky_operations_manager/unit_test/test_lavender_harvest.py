"""
test_lavender_harvest.py

Standalone integration-test node for lavender row harvesting using DriveClient
(camera-based detection) and NavigationActionClient.

Operational sequence:
  - Boot → startup undocking (if at dock) → _need_row_navigation=True
  - Each tick: _generate_next_task() produces the highest-priority task
      1. Battery low             → CHARGING_TASK  (route via row_end → dock)
      2. Load >= 100%            → UNLOADING_TASK (route via row_end → dock)
      3. All rows complete       → CHARGING_TASK  (final charge, then idle)
      4. Default                 → HARVESTING_TASK [MOVING, HARVESTING]

_subtask_moving handles both Nav2 (when _need_row_navigation=True) and
DriveClient (when _need_row_navigation=False) within a single MOVING subtask.

_subtask_harvesting is UNCHANGED from HuskyOperationsManager — 5s simulated
harvest timer, DESTINATION_REACHED → START_HARVESTING → HARVESTING → DONE_HARVESTING → JOB_DONE.

After every JOB_DONE the task type determines what comes next:
  HARVESTING_TASK → _need_row_navigation=False  (stay on row, resume DriveClient)
  UNLOADING_TASK  → restore return_row/side, _need_row_navigation=True, _via_start_pose=True
  CHARGING_TASK   → _need_row_navigation=True   (navigate back to current row start)
"""

import math
import time
from enum import IntEnum

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import BatteryState, Imu, NavSatFix
from std_msgs.msg import Bool
from status_interfaces.msg import (
    ImageDetectionPose,
    RobotStatus,
    SubTask,
    Task,
    UndockGoal,
    WayPoint,
)

from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.action_clients.drive_client_new import DriveClient
from husky_operations_manager.action_clients.navigation import NavigationActionClient
from husky_operations_manager.action_clients.reverse_drive_client import ReverseDriveClient
from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.dataclass import (
    DockingConfig,
    DockInstanceConfig,
    DockPluginConfig,
    DriveConfig,
)
from husky_operations_manager.docking_param_fetcher import DockingParamFetcher
from husky_operations_manager.enum import (
    DockingParamFetcherStatus,
    DriveStatus,
    NavigationStatus,
    OnlineFlagEnum,
    ReverseDriveStatus,
    RobotStatusEnum,
)
from husky_operations_manager.local_task_generator import TaskGenerator


class RowSide(IntEnum):
    A = 0
    B = 1

    def __str__(self):
        return self.name


class LavenderHarvestNode(Node):
    """
    Integration-test node for camera-guided lavender row harvesting.

    Mirrors HuskyOperationsManager architecture (same action clients, startup
    sequence, subtask handlers, and state machine) but generates tasks
    internally via TaskGenerator instead of receiving them from JobPublisher.
    """

    def __init__(self):
        super().__init__('lavender_harvest_node')

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f"Node namespace: {self.namespace}")

        self._declare_parameter()
        self._get_paramters()
        self._init_state_variables()
        self._init_sensor_data()
        self._init_subscriptions()

        self.robot_state_pub = self.create_publisher(
            RobotStatus, f'{self.namespace}/status/robot', 10)

        # Set to None until DockingParamFetcher completes
        self.docking_config:          DockingConfig | None      = None
        self.active_dock:             DockInstanceConfig | None = None
        self.active_plugin:           DockPluginConfig | None   = None
        self.reverse_drive_client:    ReverseDriveClient | None = None
        self.navigation:              NavigationActionClient | None = None
        self.docking_action_client:   DockingActionClient | None   = None
        self.undocking_action_client: UndockingActionClient | None = None
        self.drive_client:            DriveClient | None            = None

        self._CONFIG_POLL_TIMEOUT_SEC: float = 30.0
        self._param_fetcher       = DockingParamFetcher(self)
        self._param_fetcher.fetch()
        self._config_poll_start_time: float = self.get_clock().now().nanoseconds / 1e9
        self._config_poll_timer = self.create_timer(0.5, self._poll_docking_config)

    # =========================================================================
    # PARAMS AND VARIABLES INITIALIZATION
    # =========================================================================

    def _declare_parameter(self):
        """Declare all ROS2 parameters with default values."""
        self.declare_parameter('num_rows',              1)
        self.declare_parameter('no_detection_timeout',  10.0)
        self.declare_parameter('loading.increment',     20.0)

        # Global reference pose used when returning after unloading [x, y, theta]
        self.declare_parameter('start_pose', [0.0, -1.0, 0.0])

        # Row waypoints — indexed by row number, up to 10 rows
        for i in range(10):
            self.declare_parameter(f'row_{i}_side_a_start', [0.0, 0.0, 0.0])
            self.declare_parameter(f'row_{i}_side_a_end',   [0.0, 5.0, 0.0])
            self.declare_parameter(f'row_{i}_side_b_start', [0.5, 5.0, 3.14])
            self.declare_parameter(f'row_{i}_side_b_end',   [0.5, 0.0, 3.14])

        # DriveClient
        self.declare_parameter('drive.v_linear',           0.2)
        self.declare_parameter('drive.v_angular',          0.5)
        self.declare_parameter('drive.tf_polling_rate',    10.0)
        self.declare_parameter('drive.tolerance',          0.05)
        self.declare_parameter('drive.timeout',            30.0)
        self.declare_parameter('drive.tf_base_frame',      'arm_0_base_link')
        self.declare_parameter('drive.tf_detection_frame', 'arm_0_detections')

        # Navigation
        self.declare_parameter('navigation.max_retries', 3)
        self.declare_parameter('navigation.retry_delay', 5.0)

        # Docking
        self.declare_parameter('docking.max_retries', 2)
        self.declare_parameter('docking.retry_delay', 3.0)
        self.declare_parameter('docking.threshold',   0.25)

        # Battery
        self.declare_parameter('battery.low_threshold',  50.0)
        self.declare_parameter('battery.full_threshold', 99.0)

        # Timing
        self.declare_parameter('timing.timer_period',                 1.0)
        self.declare_parameter('timing.initial_position_check_delay', 2.0)
        self.declare_parameter('server.action_server_timeout',        5.0)

    def _get_paramters(self):
        """Read all declared parameters into instance variables."""
        self.num_rows             = int(self.get_parameter('num_rows').value)
        self.no_detection_timeout = float(self.get_parameter('no_detection_timeout').value)
        self.loading_increment    = float(self.get_parameter('loading.increment').value)

        def _pose(name) -> list[float]:
            return list(self.get_parameter(name).value)

        self.start_pose = _pose('start_pose')

        self.row_waypoints: list[dict] = []
        for i in range(self.num_rows):
            self.row_waypoints.append({
                'side_a_start': _pose(f'row_{i}_side_a_start'),
                'side_a_end':   _pose(f'row_{i}_side_a_end'),
                'side_b_start': _pose(f'row_{i}_side_b_start'),
                'side_b_end':   _pose(f'row_{i}_side_b_end'),
            })

        self.navigation_max_retries = int(self.get_parameter('navigation.max_retries').value)
        self.navigation_retry_delay = float(self.get_parameter('navigation.retry_delay').value)
        self.docking_max_retries    = int(self.get_parameter('docking.max_retries').value)
        self.docking_retry_delay    = float(self.get_parameter('docking.retry_delay').value)
        self.docking_threshold      = float(self.get_parameter('docking.threshold').value)
        self.battery_low_threshold  = float(self.get_parameter('battery.low_threshold').value)
        self.battery_full_threshold = float(self.get_parameter('battery.full_threshold').value)
        self.timing_timer_period                 = float(self.get_parameter('timing.timer_period').value)
        self.timing_initial_position_check_delay = float(self.get_parameter('timing.initial_position_check_delay').value)
        self.server_action_server_timeout        = float(self.get_parameter('server.action_server_timeout').value)

        self._drive_config = DriveConfig(
            base_frame         = 'base_link',
            v_linear           = float(self.get_parameter('drive.v_linear').value),
            v_angular          = float(self.get_parameter('drive.v_angular').value),
            tf_polling_rate    = float(self.get_parameter('drive.tf_polling_rate').value),
            tolerance          = float(self.get_parameter('drive.tolerance').value),
            timeout            = float(self.get_parameter('drive.timeout').value),
            tf_base_frame      = str(self.get_parameter('drive.tf_base_frame').value),
            tf_detection_frame = str(self.get_parameter('drive.tf_detection_frame').value),
        )

        self.get_logger().info(
            f"Parameters loaded | rows={self.num_rows} | "
            f"load_increment={self.loading_increment}% | "
            f"no_detection_timeout={self.no_detection_timeout}s | "
            f"battery_low={self.battery_low_threshold}%"
        )

    def _init_state_variables(self):
        """
        Initialise all state-tracking variables to their boot defaults.

        Grouped by concern:
          - Startup: boot sequence completion flags
          - Robot state: current and previous RobotStatusEnum
          - Task management: task, subtask, and index
          - Row traversal: row index, side, and navigation routing flags
          - Return context: row/side to return to after unloading
          - Detection: last detection timestamp and received flag
          - Retry counters: navigation and docking
          - Job timers: harvesting/unloading timer state
          - Undocking: stored subtask and task type context
          - Reverse drive: fallback active flag
        """
        # --- Startup ---
        self.is_initialized          = False
        self.is_at_docking_station   = False
        self.startup_undock_complete = False

        # --- Robot state ---
        self.current_status  = RobotStatusEnum.IDLE
        self.previous_status = RobotStatusEnum.IDLE

        # --- Task management ---
        self.current_task:          Task | None    = None
        self.current_sub_task:      SubTask | None = None
        self.current_sub_task_index: int           = 0
        self.last_handled_task_id:   int | None    = None
        self.last_handled_task_type: int | None    = None
        self.last_handled_subtask_type: int | None = None
        self._task_counter: int = 0

        self.current_node_id     = 0
        self.current_load_status = 0.0

        # --- Row traversal ---
        # _need_row_navigation=True  → JOB_START sends Nav2 goal first
        # _need_row_navigation=False → JOB_START skips Nav2, DriveClient starts at DESTINATION_REACHED
        self.current_row:          int     = 0
        self.current_side:         RowSide = RowSide.A
        self._need_row_navigation: bool    = True   # True at boot: navigate to row 0 start first
        self._all_rows_complete:   bool    = False
        self._harvest_complete:    bool    = False

        # --- Return context (set when unloading is triggered) ---
        # After unloading JOB_DONE, navigate back via start_pose to return_row/return_side start
        self.return_row:    int     = 0
        self.return_side:   RowSide = RowSide.A
        self._via_start_pose: bool  = False   # True → include start_pose waypoint in nav route

        # --- Detection ---
        self.last_detection_time: float | None = None
        self._detection_received: bool         = False

        # --- Retry counters ---
        self.navigation_retry_count = 0
        self.docking_retry_count    = 0

        # --- Job timers ---
        self.job_start_time = None
        self.job_duration   = 0.0

        # --- Undocking ---
        self.last_undocking_subtask:    SubTask | None = None
        self.undocking_after_task_type: int | None     = None

        # --- Reverse drive ---
        self.reverse_drive_active: bool = False

        self.get_logger().debug("State variables initialised")

    def _init_sensor_data(self):
        """Initialise sensor data containers with empty default messages."""
        self.battery_status = BatteryState()
        self.gps_status     = NavSatFix()
        self.pose_status    = PoseWithCovarianceStamped()
        self.imu_status     = Imu()
        self.estop_status   = Bool()

    def _init_subscriptions(self):
        """Create all ROS2 subscriptions."""
        self.battery_sub = self.create_subscription(
            BatteryState,
            f'{self.namespace}/platform/bms/state',
            lambda msg: setattr(self, 'battery_status', msg),
            qos_profile_sensor_data)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            f'{self.namespace}/ground_truth/pose',
            self._pose_callback,
            10)

        self.imu_sub = self.create_subscription(
            Imu,
            f'{self.namespace}/sensors/gps_0/imu',
            lambda msg: setattr(self, 'imu_status', msg),
            qos_profile_sensor_data)

        self.estop_sub = self.create_subscription(
            Bool,
            f'{self.namespace}/platform/emergency_stop',
            lambda msg: setattr(self, 'estop_status', msg),
            qos_profile_sensor_data)

        detection_topic = (
            f'{self.namespace}/manipulators/arm_0_detection/image_annotated/detection_pose'
        )
        self.detection_sub = self.create_subscription(
            ImageDetectionPose,
            detection_topic,
            self._detection_callback,
            10)

        self.get_logger().info(f"Subscribed to detection topic: {detection_topic}")
        self.get_logger().debug("Subscriptions initialised")

    # =========================================================================
    # DOCKING CONFIG POLL
    # =========================================================================

    def _poll_docking_config(self):
        """Poll DockingParamFetcher every 0.5s — mirrors HuskyOperationsManager."""
        status = self._param_fetcher.get_status()
        self.get_logger().debug(f"DockingParamFetcher poll | status={status.name}")

        if status == DockingParamFetcherStatus.DONE:
            self._config_poll_timer.cancel()
            self._on_docking_config_ready()
            return

        if status == DockingParamFetcherStatus.ERROR:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self._config_poll_start_time

            if elapsed >= self._CONFIG_POLL_TIMEOUT_SEC:
                self.get_logger().error(
                    f"DockingParamFetcher failed — "
                    f"docking_server unavailable after {self._CONFIG_POLL_TIMEOUT_SEC:.0f}s | "
                    f"elapsed={elapsed:.1f}s — shutting down"
                )
                self._config_poll_timer.cancel()
                self.destroy_node()
                rclpy.shutdown()
                return

            self.get_logger().warning(
                f"DockingParamFetcher ERROR — retrying | "
                f"elapsed={elapsed:.1f}s / {self._CONFIG_POLL_TIMEOUT_SEC:.0f}s"
            )
            self._param_fetcher.reset()
            self._param_fetcher.fetch()

    def _on_docking_config_ready(self):
        """
        Called once DockingParamFetcher reports DONE.

        Initialises action clients (require DockingConfig), DriveClient,
        TaskGenerator, and starts the main timer and initial position check.
        """
        self.docking_config = self._param_fetcher.get_config()
        self.active_dock    = self.docking_config.dock_configs[self.docking_config.docks[0]]
        self.active_plugin  = self.docking_config.plugin_configs[self.docking_config.dock_plugins[0]]

        self.get_logger().info(
            f"DockingConfig ready | "
            f"dock='{self.active_dock.instance_name}' | "
            f"plugin='{self.active_plugin.plugin_name}'"
        )

        self.navigation              = NavigationActionClient(self)
        self.docking_action_client   = DockingActionClient(self)
        self.undocking_action_client = UndockingActionClient(self)
        self.reverse_drive_client    = ReverseDriveClient(self, self.docking_config)
        self.drive_client            = DriveClient(self, self._drive_config)

        # TaskGenerator handles CHARGING and UNLOADING task construction including
        # dock parameters. No constructor args needed — dock params are hardcoded
        # fallbacks in the generator. build_route_to_dock() returns [] if farm
        # layout is not available (acceptable for early testing).
        self.task_generator = TaskGenerator()

        self.init_check_timer = self.create_timer(
            self.timing_initial_position_check_delay,
            self._initial_position_check_timer
        )
        self.timer = self.create_timer(self.timing_timer_period, self.timer_callback)

        self.get_logger().info("LavenderHarvestNode ready.")

    # =========================================================================
    # MAIN CALLBACK METHODS
    # =========================================================================

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
        robot_status.header.stamp    = self.get_clock().now().to_msg()
        robot_status.robot_namespace = self.namespace.replace('/', '')

        self._set_battery_status(robot_status)
        self._set_estop_status(robot_status)
        self._set_location_status(robot_status)

        self.get_logger().debug(
            f"Timer tick | status={self.current_status.name} | "
            f"row={self.current_row} side={self.current_side} | "
            f"load={self.current_load_status:.1f}% | "
            f"startup_done={self.startup_undock_complete} | "
            f"need_nav={self._need_row_navigation} | "
            f"battery={self._normalize_battery(self.battery_status.percentage):.1f}%"
        )

        if self.current_status in [RobotStatusEnum.ERROR, RobotStatusEnum.ABNORMAL]:
            self._handle_error_recovery()
        elif not self.startup_undock_complete:
            self._handle_startup_undocking(robot_status)
        else:
            self._handle_task_execution(robot_status)

        robot_status.status          = self.current_status.value
        robot_status.current_node_id = self.current_node_id
        robot_status.load_status     = self.current_load_status
        self.robot_state_pub.publish(robot_status)

    # =========================================================================
    # SUBSCRIPTION CALLBACKS
    # =========================================================================

    def _pose_callback(self, msg):
        """Store latest ground-truth pose for dock distance checks."""
        self.pose_status = msg

    def _detection_callback(self, msg: ImageDetectionPose):
        """
        ImageDetectionPose callback — event-driven DriveClient stop.

        On detection_valid=True:
          - Update last_detection_time (resets no-detection timeout)
          - If DriveClient is active, call drive_client.stop() which validates
            TF alignment before publishing zero velocity.

        On detection_valid=False: no action.
        """
        if not msg.detection_valid:
            return

        self.last_detection_time = self.get_clock().now().nanoseconds / 1e9
        self._detection_received = True

        self.get_logger().debug(
            f"Detection valid | "
            f"center=({msg.center.x:.3f}, {msg.center.y:.3f}, {msg.center.z:.3f})"
        )

        if self.drive_client is not None and self.drive_client.is_active():
            self.get_logger().info("Valid detection — calling drive_client.stop()")
            self.drive_client.stop()

    # =========================================================================
    # STARTUP AND INITIALIZATION
    # =========================================================================

    def _initial_position_check_timer(self):
        """Fires once after timing_initial_position_check_delay."""
        if self.pose_status is None:
            self.get_logger().warning("Waiting for pose data...")
            return
        self.init_check_timer.cancel()
        self.check_initial_position()

    def check_initial_position(self):
        """
        Determine whether the robot starts at a docking station.

        Sets startup_undock_complete=True if the robot is NOT at a dock so the
        node can skip the startup undocking sequence and proceed directly to tasks.
        """
        if self.is_initialized:
            return

        if not self.pose_status or not self.pose_status.pose:
            self.get_logger().warning("No pose data — retrying in 1s")
            time.sleep(1.0)
            self.check_initial_position()
            return

        charger_dock_pose = self.active_dock
        current_pos       = self.pose_status.pose.pose.position

        distance = self._calculate_distance(
            current_pos.x, current_pos.y,
            charger_dock_pose.dock_x, charger_dock_pose.dock_y)

        self.get_logger().info(
            f"Initial Robot position: ({current_pos.x:.3f}, {current_pos.y:.3f}), "
            f"Dock Name: '{charger_dock_pose.instance_name}' | "
            f"Distance to dock: {distance:.3f}m")

        if distance <= self.docking_threshold:
            self.is_at_docking_station = True
            self.get_logger().info("Robot at dock - will undock before tasks")
        else:
            self.is_at_docking_station   = False
            self.startup_undock_complete = True
            self.get_logger().info("Robot ready for tasks")

        self.is_initialized = True

    def _handle_startup_undocking(self, robot_status: RobotStatus):
        """Drive the startup undocking state machine — mirrors HuskyOperationsManager."""
        if not self.is_initialized or self.startup_undock_complete:
            return

        if self.docking_config is None:
            self.get_logger().warning("_handle_startup_undocking called but docking_config is None")
            return

        self.get_logger().debug(
            f"Startup undocking | current_status={self.current_status.name} | "
            f"reverse_drive_active={self.reverse_drive_active}"
        )

        if self.current_status == RobotStatusEnum.IDLE:
            self.get_logger().info("Starting startup undocking...")
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            robot_status.task = "Startup: Preparing to undock"

        elif self.current_status == RobotStatusEnum.START_UNDOCKING:
            robot_status.task      = "Startup: Undocking"
            dock_type              = self.active_dock.type
            staging_x_offset       = self.active_plugin.staging_x_offset
            v_linear               = self.docking_config.controller_v_linear_max
            max_undocking_time     = (abs(staging_x_offset) / max(v_linear, 0.01)) * 1.25

            self.get_logger().debug(
                f"Startup UndockGoal | dock_type='{dock_type}' | "
                f"max_undocking_time={max_undocking_time:.1f}s"
            )

            startup_subtask             = SubTask()
            startup_subtask.type        = SubTask.UNDOCKING
            startup_subtask.description = "Startup Undocking"
            startup_subtask.undock_goal = UndockGoal(
                dock_type=dock_type,
                max_undocking_time=max_undocking_time
            )
            self.last_undocking_subtask = startup_subtask
            self._subtask_undocking()

        else:
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

        Changes from HuskyOperationsManager:
          1. No task from JobPublisher — call _generate_next_task() when no current_task.
          2. Battery check done inline; clears current_task instead of going ERROR.
          3. _handle_task_start at JOB_DONE sets _need_row_navigation based on task type.
        """
        # CHANGE 1: Generate task internally when none is active
        if not self.current_task:
            self._generate_next_task()
            if not self.current_task:
                if self.current_status != RobotStatusEnum.IDLE:
                    self._transition_status(RobotStatusEnum.IDLE)
                return

        # CHANGE 2: Battery check — cancel active motion, clear task so
        # next tick generates a CHARGING_TASK (priority 1 in _generate_next_task)
        battery_pct = self._normalize_battery(self.battery_status.percentage)
        if (self.current_task.task_type != Task.CHARGING_TASK and
                battery_pct <= self.battery_low_threshold):
            self.get_logger().warning(
                f"Battery low: {battery_pct:.1f}% — "
                f"threshold={self.battery_low_threshold}% | "
                f"task_type={self.current_task.task_type} | "
                f"status={self.current_status.name}"
            )
            self._cancel_all_motion()
            self.current_task = None
            return

        self._process_action_clients(robot_status)
        self._update_current_subtask()

        robot_status.crop_type      = self.current_task.crop_type
        robot_status.target_node_id = self.current_task.target_node_id
        robot_status.task = (
            self.current_sub_task.description if self.current_sub_task
            else self.current_task.description
        )

        self.get_logger().debug(
            f"Task execution | status={self.current_status.name} | "
            f"task_id={self.current_task.task_id} task_type={self.current_task.task_type} | "
            f"subtask_index={self.current_sub_task_index} | "
            f"subtask_type={self.current_sub_task.type if self.current_sub_task else 'None'}"
        )

        if self.current_status in (RobotStatusEnum.IDLE, RobotStatusEnum.JOB_DONE):
            self._handle_task_start()
        else:
            self._execute_current_subtask()

    def _generate_next_task(self):
        """
        Select and build the highest-priority task.

        Priority order (highest to lowest):
          1. Battery low                → CHARGING_TASK via row_end route
          2. Load >= 100%               → UNLOADING_TASK via row_end route
          3. All rows complete          → CHARGING_TASK (final charge, then idle)
          4. Default                    → HARVESTING_TASK [MOVING, HARVESTING]
        """
        if self._harvest_complete:
            self.get_logger().info("All rows harvested and final charging done — staying IDLE.")
            return

        battery_pct = self._normalize_battery(self.battery_status.percentage)

        # Priority 1: Low battery — charge via row_end to avoid reversing over unvisited row
        if battery_pct <= self.battery_low_threshold and not self._all_rows_complete:
            self.get_logger().info(
                f"Battery low ({battery_pct:.1f}%) — generating CHARGING_TASK"
            )
            self._need_row_navigation = True
            waypoints = self._build_route_via_row_end()
            self.current_task = self.task_generator.generate_charging_task(waypoints)
            return

        # Priority 2: Full load — unload via row_end
        if self.current_load_status >= 100.0:
            self.get_logger().info(
                f"Load full ({self.current_load_status:.1f}%) — generating UNLOADING_TASK"
            )
            self.return_row  = self.current_row
            self.return_side = self.current_side
            self._need_row_navigation = True
            waypoints = self._build_route_via_row_end()
            self.current_task = self.task_generator.generate_unloading_task(waypoints)
            return

        # Priority 3: All rows complete — final charging
        if self._all_rows_complete:
            self.get_logger().info("All rows complete — generating final CHARGING_TASK")
            self._all_rows_complete  = False
            self._harvest_complete   = True
            self._need_row_navigation = True
            self.current_task = self.task_generator.generate_charging_task([])
            return

        # Default: drive row with DriveClient, harvest detected bush
        self._task_counter += 1
        nav_waypoints = self._get_nav_waypoints() if self._need_row_navigation else []
        self.current_task = self._build_harvest_task(self._task_counter, nav_waypoints)

        self.get_logger().info(
            f"Generated HARVESTING_TASK | "
            f"row={self.current_row} side={self.current_side} | "
            f"need_nav={self._need_row_navigation} | "
            f"task_id={self._task_counter}"
        )

    def _build_harvest_task(self, task_id: int, nav_waypoints: list) -> Task:
        """
        Build a [MOVING, HARVESTING] task for one harvest cycle.

        nav_waypoints is non-empty only when _need_row_navigation=True.
        When empty, _subtask_moving skips Nav2 and starts DriveClient directly.
        """
        move_sub              = SubTask()
        move_sub.sub_task_id  = 1
        move_sub.type         = SubTask.MOVING
        move_sub.description  = f'Row {self.current_row} Side {self.current_side} drive'
        move_sub.data         = nav_waypoints

        harvest_sub              = SubTask()
        harvest_sub.sub_task_id  = 2
        harvest_sub.type         = SubTask.HARVESTING
        harvest_sub.description  = 'Harvesting the bush'
        harvest_sub.data_str     = 'harvest data'

        return Task(
            task_id          = task_id,
            task_type        = Task.HARVESTING_TASK,
            assigned_robot_id = 4,
            description      = f'Harvest Row {self.current_row} Side {self.current_side}',
            crop_type        = 'lavender',
            sub_tasks        = [move_sub, harvest_sub],
        )

    def _get_nav_waypoints(self) -> list[WayPoint]:
        """
        Return Nav2 waypoints for the current row/side start.

        When _via_start_pose=True (return after unloading), prepends the
        start_pose as an intermediate waypoint so the robot exits the dock
        area cleanly before entering the row.
        """
        row = self.row_waypoints[self.current_row]
        key = 'side_a_start' if self.current_side == RowSide.A else 'side_b_start'
        target = row[key]  # [x, y, theta]

        target_wp         = WayPoint()
        target_wp.x       = target[0]
        target_wp.y       = target[1]
        target_wp.theta   = target[2]
        target_wp.node_id = self.current_row * 10 + int(self.current_side)

        if self._via_start_pose:
            self._via_start_pose = False
            start_wp         = WayPoint()
            start_wp.x       = self.start_pose[0]
            start_wp.y       = self.start_pose[1]
            start_wp.theta   = self.start_pose[2]
            start_wp.node_id = -1
            return [start_wp, target_wp]

        return [target_wp]

    def _build_route_via_row_end(self) -> list[WayPoint]:
        """
        Return a waypoint list routed from the current row_end to the dock staging area.

        Passes through the row end so the robot exits cleanly rather than
        reversing back over un-harvested crop.
        """
        row = self.row_waypoints[self.current_row]
        key = 'side_a_end' if self.current_side == RowSide.A else 'side_b_end'
        end = row[key]  # [x, y, theta]
        return self.task_generator.build_route_to_dock(end[0], end[1])

    def _advance_row(self):
        """
        Advance to the next row or side after no-detection timeout.

        Side A done → Side B, _need_row_navigation=True
        Side B done, rows remaining → next row Side A, _need_row_navigation=True
        Side B done, last row → _all_rows_complete=True
        """
        if self.current_side == RowSide.A:
            self.get_logger().info(
                f"Row {self.current_row} Side A complete → advancing to Side B"
            )
            self.current_side         = RowSide.B
            self._need_row_navigation = True

        elif self.current_row < self.num_rows - 1:
            self.get_logger().info(
                f"Row {self.current_row} Side B complete → advancing to Row {self.current_row + 1} Side A"
            )
            self.current_row         += 1
            self.current_side         = RowSide.A
            self._need_row_navigation = True

        else:
            self.get_logger().info(
                f"Row {self.current_row} Side B complete — all rows done"
            )
            self._all_rows_complete = True

    def _process_action_clients(self, robot_status: RobotStatus):
        nav_status    = self.navigation.get_navigation_status()
        dock_status   = self.docking_action_client.get_status()
        undock_status = self.undocking_action_client.get_status()

        self.get_logger().debug(
            f"Action clients | nav={nav_status.name} | "
            f"dock={dock_status.name} | "
            f"undock={undock_status.name} | "
            f"reverse_drive_active={self.reverse_drive_active}"
        )

        if nav_status != NavigationStatus.IDLE:
            self._handle_navigation(robot_status)
        elif dock_status != RobotStatusEnum.IDLE:
            self._handle_docking(robot_status)
        elif undock_status != RobotStatusEnum.IDLE:
            self._handle_undocking(robot_status)
        elif self.reverse_drive_active:
            self._handle_reverse_drive(robot_status)

    def _update_current_subtask(self):
        """Refresh current_sub_task from sub_tasks list using current_sub_task_index."""
        if not self.current_task:
            return

        if isinstance(self.current_task.sub_tasks, list):
            if self.current_sub_task_index < len(self.current_task.sub_tasks):
                self.current_sub_task = self.current_task.sub_tasks[self.current_sub_task_index]
            else:
                self.current_sub_task = None

        self.get_logger().debug(
            f"Current subtask | index={self.current_sub_task_index} | "
            f"type={self.current_sub_task.type if self.current_sub_task else 'None'} | "
            f"desc='{self.current_sub_task.description if self.current_sub_task else 'None'}'"
        )

    def _handle_task_start(self):
        """
        Initialise a new task or handle task completion at JOB_DONE.

        New task (task_id or task_type changed):
          - Cache task identifiers, clear undocking state, → JOB_START

        CHANGE 3: At JOB_DONE, set _need_row_navigation based on completed task type:
          HARVESTING_TASK → False  (resume same row with DriveClient)
          UNLOADING_TASK  → True + restore return context + _via_start_pose=True
          CHARGING_TASK   → True   (navigate back to row after charging)
        """
        if not self.current_task:
            return

        if (self.current_task.task_id != self.last_handled_task_id or
                self.current_task.task_type != self.last_handled_task_type):
            self.get_logger().info(
                f"Starting Task: {self.current_task.description} | "
                f"ID: {self.current_task.task_id}"
            )
            self.get_logger().debug(
                f"Task start | task_type={self.current_task.task_type} | "
                f"last_task_id={self.last_handled_task_id} | "
                f"num_subtasks={len(self.current_task.sub_tasks) if isinstance(self.current_task.sub_tasks, list) else 0}"
            )
            self.last_handled_task_id      = self.current_task.task_id
            self.last_handled_task_type    = self.current_task.task_type
            self.last_handled_subtask_type = None
            self.last_undocking_subtask    = None
            self.undocking_after_task_type = None
            self.current_sub_task_index    = 0
            self._transition_status(RobotStatusEnum.JOB_START)

        elif self.current_status == RobotStatusEnum.JOB_DONE:
            task_type = self.current_task.task_type

            if task_type == Task.HARVESTING_TASK:
                # Stay on same row — DriveClient resumes without Nav2
                self._need_row_navigation = False
                self.get_logger().debug(
                    f"HARVESTING JOB_DONE — resuming row {self.current_row} side {self.current_side}"
                )
            elif task_type == Task.UNLOADING_TASK:
                # Return to interrupted row/side via start_pose
                self.current_row      = self.return_row
                self.current_side     = self.return_side
                self._need_row_navigation = True
                self._via_start_pose  = True
                self.get_logger().info(
                    f"UNLOADING JOB_DONE — returning to row {self.return_row} "
                    f"side {self.return_side} via start_pose"
                )
            elif task_type == Task.CHARGING_TASK:
                # Navigate back to the current row after charging
                self._need_row_navigation = True
                self.get_logger().debug(
                    f"CHARGING JOB_DONE — navigating to row {self.current_row} "
                    f"side {self.current_side}"
                )

            self._transition_status(RobotStatusEnum.IDLE)
            self.current_task = None

    def _execute_current_subtask(self):
        """Route to the correct subtask handler based on current_sub_task.type."""
        if not isinstance(self.current_sub_task, SubTask):
            self.get_logger().debug(
                f"_execute_current_subtask called but no valid subtask | "
                f"status={self.current_status.name}"
            )
            return

        if self.current_sub_task.type != self.last_handled_subtask_type:
            self.get_logger().info(f"Executing: {self.current_sub_task.description}")
            self.last_handled_subtask_type = self.current_sub_task.type

        task_handler_map = {
            SubTask.MOVING:     self._subtask_moving,
            SubTask.HARVESTING: self._subtask_harvesting,
            SubTask.DOCKING:    self._subtask_docking,
            SubTask.CHARGING:   self._subtask_charging,
            SubTask.UNLOADING:  self._subtask_unloading,
        }

        handler = task_handler_map.get(self.current_sub_task.type)
        if handler:
            handler()
        else:
            self.get_logger().warning(
                f"Unknown subtask type: {self.current_sub_task.type} — "
                f"no handler registered"
            )

    # =========================================================================
    # ACTION CLIENT HANDLERS — mirrors HuskyOperationsManager
    # =========================================================================

    def _handle_navigation(self, robot_status: RobotStatus):
        """Monitor NavigationActionClient and update RobotStatus."""
        nav_status = self.navigation.get_navigation_status()
        wpf_status = self.navigation.get_current_status()

        if wpf_status:
            robot_status.task            = wpf_status.task
            robot_status.current_node_id = wpf_status.current_node_id
            robot_status.target_node_id  = wpf_status.target_node_id
        else:
            robot_status.task            = self.current_task.description if self.current_task else ""
            robot_status.current_node_id = self.current_node_id
            robot_status.target_node_id  = self.current_task.target_node_id if self.current_task else -1

        self.get_logger().debug(
            f"Navigation handler | nav_status={nav_status.name} | "
            f"retry_count={self.navigation_retry_count}/{self.navigation_max_retries}"
        )

        if nav_status == NavigationStatus.ACTIVE:
            self._transition_status(RobotStatusEnum.MOVING)

        elif nav_status == NavigationStatus.SUCCEEDED:
            self.get_logger().info("Navigation complete")
            self.navigation_retry_count = 0
            self.navigation.reset()
            self.current_node_id = wpf_status.target_node_id if wpf_status else self.current_node_id
            self._transition_status(RobotStatusEnum.DESTINATION_REACHED)

        elif nav_status in [NavigationStatus.ABORTED, NavigationStatus.ERROR]:
            self._handle_navigation_retry()

        elif nav_status == NavigationStatus.CANCELED:
            self.get_logger().info("Navigation canceled")
            self.navigation.reset()
            self.navigation_retry_count = 0
            self._transition_status(RobotStatusEnum.IDLE)

    def _handle_navigation_retry(self):
        """Retry NavigateThroughPoses up to navigation_max_retries."""
        self.get_logger().warning(
            f"Navigation failed | retry {self.navigation_retry_count + 1}/{self.navigation_max_retries}"
        )

        if self._is_robot_at_target():
            self.get_logger().info("Robot already at target — treating navigation as complete")
            wpf_status = self.navigation.get_current_status()
            self.navigation_retry_count = 0
            self.navigation.reset()
            self.current_node_id = wpf_status.target_node_id if wpf_status else self.current_node_id
            self._transition_status(RobotStatusEnum.DESTINATION_REACHED)
            return

        if self.navigation_retry_count < self.navigation_max_retries:
            self.navigation_retry_count += 1
            self.get_logger().info(
                f"Retrying navigation in {self.navigation_retry_delay:.1f}s | "
                f"attempt {self.navigation_retry_count}/{self.navigation_max_retries}"
            )
            time.sleep(self.navigation_retry_delay)
            self._retry_navigation()
        else:
            self.get_logger().error(
                f"Navigation failed after {self.navigation_max_retries} retries"
            )
            self.navigation.reset()
            self._transition_status(RobotStatusEnum.ERROR)
            self.navigation_retry_count = 0

    def _retry_navigation(self):
        """Re-send the NavigateThroughPoses goal for the current task."""
        if not self.current_task:
            self.get_logger().error("Navigation retry failed — no current task")
            return

        nav_status = self.navigation.get_navigation_status()
        if nav_status in [NavigationStatus.ACTIVE, NavigationStatus.SENDING]:
            self.get_logger().warning(
                f"Navigation still active ({nav_status.name}) — skipping retry until complete"
            )
            return

        self.navigation.reset()
        time.sleep(self.navigation_retry_delay)

        if self.navigation.send_goal(self.current_task):
            self._transition_status(RobotStatusEnum.MOVING)
        else:
            self.get_logger().error("Navigation retry send_goal failed")
            self._transition_status(RobotStatusEnum.ERROR)

    def _handle_docking(self, robot_status: RobotStatus):
        """Monitor DockingActionClient and update RobotStatus."""
        status   = self.docking_action_client.get_status()
        feedback = self.docking_action_client.get_feedback()

        if feedback:
            robot_status.task            = feedback.task
            robot_status.current_node_id = self.current_node_id
        else:
            robot_status.task = "Docking in progress"

        self.get_logger().debug(
            f"Docking handler | status={status.name} | "
            f"retry_count={self.docking_retry_count}/{self.docking_max_retries}"
        )

        if status == RobotStatusEnum.DOCKING:
            self._transition_status(RobotStatusEnum.DOCKING)

        elif status == RobotStatusEnum.DONE_DOCKING:
            self.get_logger().info("Docking complete")
            self._transition_status(RobotStatusEnum.DONE_DOCKING)
            self.docking_action_client.reset()
            self.docking_retry_count = 0

        elif status == RobotStatusEnum.ERROR:
            self._handle_docking_retry()

    def _handle_docking_retry(self):
        """Retry dock_robot goal up to docking_max_retries, then go to ERROR."""
        dock_id = (self.current_sub_task.dock_goal.dock_id
                   if self.current_sub_task and self.current_sub_task.dock_goal else 'unknown')
        self.get_logger().error(
            f"Docking failed | retry {self.docking_retry_count + 1}/{self.docking_max_retries} | "
            f"dock_id='{dock_id}'"
        )

        if self.docking_retry_count < self.docking_max_retries:
            self.docking_retry_count += 1
            time.sleep(self.docking_retry_delay)
            self._retry_docking()
        else:
            self.get_logger().error(
                f"Docking failed after {self.docking_max_retries} retries — transitioning to ERROR"
            )
            self.docking_action_client.reset()
            self._transition_status(RobotStatusEnum.ERROR)
            self.docking_retry_count = 0

    def _retry_docking(self):
        """Re-send the dock_robot goal using the current subtask's DockGoal."""
        if not self.current_sub_task:
            self.get_logger().error("Docking retry failed — no current subtask")
            return
        if self.docking_action_client.send_docking_goal(self.current_sub_task):
            self._transition_status(RobotStatusEnum.DOCKING)
        else:
            self.get_logger().error("Docking retry send_goal failed")
            self._transition_status(RobotStatusEnum.ERROR)

    def _handle_undocking(self, robot_status: RobotStatus):
        """
        Monitor UndockingActionClient and update RobotStatus.

        Context-aware DONE_UNDOCKING handling:
          startup context (startup_undock_complete=False):
            Set startup_undock_complete=True, clear task IDs, → IDLE
          task context (startup_undock_complete=True):
            Stay at DONE_UNDOCKING — _subtask_charging or _subtask_unloading
            will call _subtask_undocking on the next tick → JOB_DONE.
        """
        status   = self.undocking_action_client.get_status()
        feedback = self.undocking_action_client.get_feedback()

        if feedback:
            robot_status.task            = feedback.task
            robot_status.current_node_id = self.current_node_id
        else:
            robot_status.task = "Undocking in progress"

        self.get_logger().debug(
            f"Undocking handler | status={status.name} | "
            f"startup_undock_complete={self.startup_undock_complete}"
        )

        if status == RobotStatusEnum.UNDOCKING:
            self._transition_status(RobotStatusEnum.UNDOCKING)

        elif status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info("Undocking complete")
            self._transition_status(RobotStatusEnum.DONE_UNDOCKING)
            self.undocking_action_client.reset()

            if not self.startup_undock_complete:
                self.get_logger().debug(
                    "Undocking DONE — startup context: setting startup_undock_complete=True → IDLE"
                )
                self.startup_undock_complete = True
                self.last_handled_task_id    = None
                self.last_handled_task_type  = None
                self._transition_status(RobotStatusEnum.IDLE)
                self.get_logger().info("Robot ready for tasks")
            else:
                self.get_logger().debug(
                    "Undocking DONE — task context: staying at DONE_UNDOCKING for _subtask_undocking"
                )

        elif status == RobotStatusEnum.ERROR:
            self._handle_undocking_retry()

    def _handle_undocking_retry(self):
        """Start ReverseDriveClient as undocking fallback."""
        self.undocking_action_client.reset()
        self.get_logger().warning(
            f"Undocking failed — starting reverse drive to staging pose | "
            f"dock='{self.active_dock.instance_name if self.active_dock else 'unknown'}'"
        )

        if self.reverse_drive_client.drive_to_staging():
            self.reverse_drive_active = True
            self._transition_status(RobotStatusEnum.UNDOCKING)
        else:
            self.get_logger().error(
                f"ReverseDriveClient refused to start — transitioning to ERROR"
            )
            self._transition_status(RobotStatusEnum.ERROR)

    def _handle_reverse_drive(self, robot_status: RobotStatus):
        """Monitor ReverseDriveClient — context-aware, mirrors HuskyOperationsManager."""
        robot_status.task = "Reverse drive to staging pose"
        status = self.reverse_drive_client.get_status()

        self.get_logger().debug(
            f"Reverse drive handler | status={status.name} | "
            f"startup_undock_complete={self.startup_undock_complete}"
        )

        if status == ReverseDriveStatus.REVERSING:
            self._transition_status(RobotStatusEnum.UNDOCKING)

        elif status == ReverseDriveStatus.DONE:
            self.get_logger().info("Reverse drive complete — undocking done")
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()

            if not self.startup_undock_complete:
                self.startup_undock_complete = True
                self.last_handled_task_id    = None
                self.last_handled_task_type  = None
                self._transition_status(RobotStatusEnum.DONE_UNDOCKING)
                self._transition_status(RobotStatusEnum.IDLE)
                self.get_logger().info("Robot ready for tasks")
            else:
                self._transition_status(RobotStatusEnum.DONE_UNDOCKING)

        elif status == ReverseDriveStatus.ERROR:
            self.get_logger().error("Reverse drive failed — transitioning to ERROR")
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()
            self._transition_status(RobotStatusEnum.ERROR)

        elif status == ReverseDriveStatus.CANCELED:
            self.get_logger().warning("Reverse drive canceled — transitioning to IDLE")
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()
            self._transition_status(RobotStatusEnum.IDLE)

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    def _handle_error_recovery(self):
        """Cancel active navigation and reset to IDLE — mirrors HuskyOperationsManager."""
        if self.current_status not in [RobotStatusEnum.ERROR, RobotStatusEnum.ABNORMAL]:
            return

        self.get_logger().warning(
            f"Error recovery | status={self.current_status.name} | "
            f"row={self.current_row} side={self.current_side}"
        )

        nav_status = self.navigation.get_navigation_status()
        if nav_status in [NavigationStatus.ACTIVE, NavigationStatus.ACCEPTED]:
            try:
                self.navigation.cancel_goal()
                time.sleep(self.navigation_retry_delay)
            except Exception as e:
                self.get_logger().warning(f"Navigation cancel raised exception: {e}")

        self.current_sub_task       = None
        self.current_sub_task_index = 0
        self.get_logger().debug("Error recovery — reset subtask state, transitioning to IDLE")
        self._transition_status(RobotStatusEnum.IDLE)

    # =========================================================================
    # SUBTASK HANDLERS
    # =========================================================================

    def _subtask_moving(self):
        """
        Handle the MOVING subtask — unified handler for Nav2 navigation and
        DriveClient row traversal.

        When _need_row_navigation=True (new row, cross-to-side, post-charge/unload):
          JOB_START → START_MOVING → send Nav2 goal → MOVING
          (nav result handled by _handle_navigation via _process_action_clients)
          DESTINATION_REACHED → set _need_row_navigation=False + branch by task type:
            HARVESTING_TASK → drive_client.forward() → MOVING
            CHARGING/UNLOADING → advance subtask index to DOCKING, stay DESTINATION_REACHED

        When _need_row_navigation=False (resume same row after harvest):
          JOB_START → DESTINATION_REACHED directly (skip Nav2)
          DESTINATION_REACHED → drive_client.forward() → MOVING

        During MOVING (DriveClient active):
          no_detection_timeout → _advance_row(), current_task=None, → IDLE
          DriveStatus.IDLE (detection stopped + aligned) → advance subtask to HARVESTING,
                                                           → DESTINATION_REACHED
        """
        if not self.current_task:
            return

        self.get_logger().debug(
            f"_subtask_moving | status={self.current_status.name} | "
            f"need_nav={self._need_row_navigation} | "
            f"row={self.current_row} side={self.current_side}"
        )

        if self.current_status == RobotStatusEnum.JOB_START:
            if self._need_row_navigation:
                self._transition_status(RobotStatusEnum.START_MOVING)
            else:
                # DriveClient path — skip Nav2, start driving immediately
                self._transition_status(RobotStatusEnum.DESTINATION_REACHED)

        elif self.current_status == RobotStatusEnum.START_MOVING:
            if self.navigation.is_navigation_active():
                self.get_logger().warning(
                    f"Navigation send skipped — already active | "
                    f"nav_status={self.navigation.get_navigation_status().name}"
                )
                return
            self.get_logger().info(
                f"Starting navigation to row {self.current_row} side {self.current_side}"
            )
            if self.navigation.send_goal(self.current_task):
                self._transition_status(RobotStatusEnum.MOVING)
            else:
                self.get_logger().error("Failed to send navigation goal")
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            # Nav2 phase complete (or was skipped) — decide next action by task type
            self._need_row_navigation = False

            if self.current_task.task_type == Task.HARVESTING_TASK:
                # Row traversal — start DriveClient
                self.get_logger().info(
                    f"Starting DriveClient for row {self.current_row} side {self.current_side}"
                )
                self.last_detection_time = self.get_clock().now().nanoseconds / 1e9
                self._detection_received = False
                self.drive_client.forward()
                self._transition_status(RobotStatusEnum.MOVING)
            else:
                # CHARGING_TASK / UNLOADING_TASK — Nav2 got us to the dock staging area.
                # Advance subtask index so _subtask_docking picks up on the next tick.
                self.get_logger().info(
                    f"Nav2 complete for {Task.CHARGING_TASK if self.current_task.task_type == Task.CHARGING_TASK else 'UNLOADING'} task — "
                    "advancing to DOCKING subtask"
                )
                self.current_sub_task_index += 1
                # Status stays DESTINATION_REACHED — _subtask_docking entry condition

        elif self.current_status == RobotStatusEnum.MOVING:
            # For Nav2 path: _handle_navigation handles all transitions — no-op here
            if self._need_row_navigation:
                return

            # DriveClient path: check timeout and drive status
            drive_status = self.drive_client.get_status()

            if drive_status in (DriveStatus.FORWARD, DriveStatus.REVERSE):
                if self.last_detection_time is not None:
                    elapsed = (
                        self.get_clock().now().nanoseconds / 1e9
                        - self.last_detection_time
                    )
                    if elapsed >= self.no_detection_timeout:
                        self.get_logger().info(
                            f"No detection for {elapsed:.1f}s "
                            f"(timeout={self.no_detection_timeout}s) — row end assumed"
                        )
                        self.drive_client.cancel()
                        self._advance_row()
                        self.current_task = None
                        self._transition_status(RobotStatusEnum.IDLE)

            elif drive_status == DriveStatus.IDLE:
                # Detection stopped and aligned — advance to HARVESTING subtask
                self.get_logger().info(
                    "DriveClient stopped (detection aligned) — advancing to HARVESTING subtask"
                )
                self.current_sub_task_index += 1
                self._transition_status(RobotStatusEnum.DESTINATION_REACHED)

            elif drive_status == DriveStatus.CANCELED:
                # Triggered by cancel() in the timeout path above — transition handled there
                pass

            else:
                self.get_logger().error(
                    f"Unexpected DriveClient status: {drive_status.name} — transitioning to ERROR"
                )
                self._transition_status(RobotStatusEnum.ERROR)

    def _subtask_docking(self):
        """
        Handle the DOCKING subtask.

        State progression:
          DESTINATION_REACHED → START_DOCKING → 1s delay → send dock_robot → DOCKING
          DONE_DOCKING        → advance current_sub_task_index to next subtask
                                (CHARGING or UNLOADING); status stays DONE_DOCKING
                                so the next handler's entry condition is satisfied.
          (docking result handled by _handle_docking via _process_action_clients)
        """
        dock_id = (self.current_sub_task.dock_goal.dock_id
                   if self.current_sub_task and self.current_sub_task.dock_goal else 'None')
        self.get_logger().debug(
            f"_subtask_docking | status={self.current_status.name} | dock_id='{dock_id}'"
        )

        if self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            self._transition_status(RobotStatusEnum.START_DOCKING)

        elif self.current_status == RobotStatusEnum.START_DOCKING:
            self.get_logger().info("Started docking")
            time.sleep(1.0)
            if self.docking_action_client.send_docking_goal(self.current_sub_task):
                self._transition_status(RobotStatusEnum.DOCKING)
            else:
                self.get_logger().error(
                    f"Failed to send docking goal | "
                    f"subtask='{self.current_sub_task.description if self.current_sub_task else 'None'}'"
                )
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DONE_DOCKING:
            # Docking complete — advance to CHARGING or UNLOADING subtask.
            # Status stays DONE_DOCKING: that is the entry condition for both
            # _subtask_charging and _subtask_unloading.
            self.get_logger().info(
                "Docking done — advancing subtask index to CHARGING/UNLOADING"
            )
            self.docking_retry_count    = 0
            self.current_sub_task_index += 1
            # Do NOT change status here — let the next subtask handler pick it up

    def _subtask_harvesting(self):
        """
        Handle the HARVESTING subtask.

        NOTE: This method is UNCHANGED from HuskyOperationsManager (test_husky_ops_navigation.py).
        It simulates harvesting with a 5s timer. The arm fires the actual cutting cycle
        in production; this timer is a stand-in for testing.

        State progression:
          DESTINATION_REACHED → START_HARVESTING → HARVESTING (5s) → DONE_HARVESTING → JOB_DONE
        """
        self.get_logger().debug(f"_subtask_harvesting | status={self.current_status.name} | ")

        if self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            self._transition_status(RobotStatusEnum.START_HARVESTING)

        elif self.current_status == RobotStatusEnum.START_HARVESTING:
            self._transition_status(RobotStatusEnum.HARVESTING)
            self.get_logger().info("Harvesting started")

            # NOTE: Start simulated harvest timer (replace with hardware signal)
            self.job_start_time = self.get_clock().now()
            self.job_duration   = 5.0  # seconds

        elif self.current_status == RobotStatusEnum.HARVESTING:
            if self.job_start_time:
                elapsed = (self.get_clock().now() - self.job_start_time).nanoseconds / 1e9
            if elapsed >= self.job_duration:
                self.get_logger().info("Harvesting complete (simulated)")
                self.get_logger().debug(
                    f"Harvesting timer done | elapsed={elapsed:.2f}s duration={self.job_duration:.1f}s"
                )
                self._transition_status(RobotStatusEnum.DONE_HARVESTING)
                self.job_start_time = None

        elif self.current_status == RobotStatusEnum.DONE_HARVESTING:
            new_load = min(self.current_load_status + self.loading_increment, 100.0)
            self.get_logger().debug(
                f"Load update | {self.current_load_status:.1f}% → {new_load:.1f}% "
                f"(+{self.loading_increment:.1f}%)"
            )
            self.current_load_status = new_load
            self.get_logger().info(f"Load status: {self.current_load_status:.1f}%")
            self._transition_status(RobotStatusEnum.JOB_DONE)

    def _subtask_charging(self):
        """
        Handle the CHARGING subtask.

        State progression:
          DONE_DOCKING   → START_CHARGING → CHARGING
          CHARGING       → poll battery until >= battery_full_threshold → DONE_CHARGING
          DONE_CHARGING  → store last_undocking_subtask, internally trigger undocking
          DONE_UNDOCKING → delegate to _subtask_undocking → JOB_DONE

        UNDOCKING is triggered internally — not a received subtask for CHARGING_TASK.
        """
        battery_pct = self._normalize_battery(self.battery_status.percentage)
        self.get_logger().debug(
            f"_subtask_charging | status={self.current_status.name} | "
            f"battery={battery_pct:.1f}% | "
            f"full_threshold={self.battery_full_threshold}%"
        )

        if self.current_status == RobotStatusEnum.DONE_DOCKING:
            self._transition_status(RobotStatusEnum.START_CHARGING)

        elif self.current_status == RobotStatusEnum.START_CHARGING:
            self._transition_status(RobotStatusEnum.CHARGING)
            self.get_logger().info("Charging started")

        elif self.current_status == RobotStatusEnum.CHARGING:
            self.get_logger().info(
                f"Battery charging: {battery_pct:.1f}%", throttle_duration_sec=10.0)
            if battery_pct >= self.battery_full_threshold:
                self.get_logger().info(f"Battery charged: {battery_pct:.1f}%")
                self._transition_status(RobotStatusEnum.DONE_CHARGING)

        elif self.current_status == RobotStatusEnum.DONE_CHARGING:
            self.get_logger().debug(
                "DONE_CHARGING — storing last_undocking_subtask and triggering undocking"
            )
            self.last_undocking_subtask    = self.current_sub_task
            self.undocking_after_task_type = Task.CHARGING_TASK
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            self._subtask_undocking()

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().debug(
                "DONE_UNDOCKING in charging context — delegating to _subtask_undocking"
            )
            self._subtask_undocking()

    def _subtask_unloading(self):
        """
        Handle the UNLOADING subtask.

        State progression:
          DONE_DOCKING   → START_UNLOADING → UNLOADING (4s timer) → DONE_UNLOADING
          DONE_UNLOADING → reset load, store last_undocking_subtask, trigger undocking
          DONE_UNDOCKING → delegate to _subtask_undocking → JOB_DONE

        current_load_status is reset to 0.0 when unloading completes.
        """
        self.get_logger().debug(
            f"_subtask_unloading | status={self.current_status.name} | "
            f"current_load={self.current_load_status:.1f}%"
        )

        if self.current_status == RobotStatusEnum.DONE_DOCKING:
            self._transition_status(RobotStatusEnum.START_UNLOADING)

        elif self.current_status == RobotStatusEnum.START_UNLOADING:
            self._transition_status(RobotStatusEnum.UNLOADING)
            self.get_logger().info("Unloading started")
            self.job_start_time = self.get_clock().now()
            self.job_duration   = 4.0  # seconds

        elif self.current_status == RobotStatusEnum.UNLOADING:
            if self.job_start_time:
                elapsed = (self.get_clock().now() - self.job_start_time).nanoseconds / 1e9
                if elapsed >= self.job_duration:
                    self.get_logger().info("Unloading complete (simulated)")
                    self.get_logger().debug(
                        f"Unloading timer done | elapsed={elapsed:.2f}s duration={self.job_duration:.1f}s"
                    )
                    self._transition_status(RobotStatusEnum.DONE_UNLOADING)
                    self.job_start_time = None

        elif self.current_status == RobotStatusEnum.DONE_UNLOADING:
            self.get_logger().debug(
                "DONE_UNLOADING — storing last_undocking_subtask and triggering undocking"
            )
            self.last_undocking_subtask    = self.current_sub_task
            self.undocking_after_task_type = Task.UNLOADING_TASK
            self.current_load_status       = 0.0
            self.get_logger().info("Unloading done, starting undocking")
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            self._subtask_undocking()

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().debug(
                "DONE_UNDOCKING in unloading context — delegating to _subtask_undocking"
            )
            self._subtask_undocking()

    def _subtask_undocking(self):
        """
        Handle the UNDOCKING subtask.

        Called in two ways:
          1. Directly by _subtask_charging/_subtask_unloading when their tasks complete.
          2. Via _handle_startup_undocking during the startup sequence.

        UndockGoal source: last_undocking_subtask (stored by callers above).

        State progression:
          START_UNDOCKING → send undock_robot → UNDOCKING
          DONE_UNDOCKING  → clear stored state → JOB_DONE
          (undocking result handled by _handle_undocking via _process_action_clients)
        """
        self.get_logger().debug(
            f"_subtask_undocking | status={self.current_status.name} | "
            f"last_undocking_subtask={'set' if self.last_undocking_subtask else 'None'} | "
            f"undocking_after_task_type={self.undocking_after_task_type}"
        )

        if self.current_status == RobotStatusEnum.START_UNDOCKING:
            undock_subtask = self.current_sub_task if self.current_sub_task else self.last_undocking_subtask
            dock_type = (undock_subtask.undock_goal.dock_type
                         if undock_subtask and undock_subtask.undock_goal else 'None')

            self.get_logger().debug(
                f"Undocking subtask | "
                f"source={'current_sub_task' if self.current_sub_task else 'last_undocking_subtask'} | "
                f"dock_type='{dock_type}'"
            )

            if self.undocking_action_client.send_undocking_goal(undock_subtask):
                self._transition_status(RobotStatusEnum.UNDOCKING)
            else:
                self.get_logger().error(
                    f"Failed to send undocking goal | "
                    f"undocking_after_task_type={self.undocking_after_task_type}"
                )
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info("Undocking done")
            self.last_undocking_subtask    = None
            self.undocking_after_task_type = None
            self._transition_status(RobotStatusEnum.JOB_DONE)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _cancel_all_motion(self):
        """Cancel DriveClient and navigation without transitioning status."""
        if self.drive_client and self.drive_client.is_active():
            self.drive_client.cancel()
        if self.navigation:
            nav_status = self.navigation.get_navigation_status()
            if nav_status in [NavigationStatus.ACTIVE, NavigationStatus.ACCEPTED]:
                try:
                    self.navigation.cancel_goal()
                except Exception as e:
                    self.get_logger().warning(f"Navigation cancel exception: {e}")

    def _is_robot_at_target(self):
        """
        Check if the robot is within 0.25m of the final waypoint.

        Used by _handle_navigation_retry to treat a nav abort as a success when
        the robot is physically close enough to the destination.

        Returns False if no subtask or pose data is available.
        """
        if not (self.current_sub_task and self.pose_status):
            return False

        robot_position = self.pose_status.pose.pose.position
        waypoints_list = [
            WayPoint(**wp) if isinstance(wp, dict) else wp
            for wp in self.current_sub_task.data
        ]
        if not waypoints_list:
            return False

        target_wp = waypoints_list[-1]
        distance  = self._calculate_distance(
            robot_position.x, robot_position.y, target_wp.x, target_wp.y)

        self.get_logger().debug(
            f"_is_robot_at_target | "
            f"robot=({robot_position.x:.3f}, {robot_position.y:.3f}) | "
            f"target=({target_wp.x:.3f}, {target_wp.y:.3f}) | "
            f"distance={distance:.3f}m threshold=0.25m"
        )
        return distance <= 0.25

    def _transition_status(self, new_status: RobotStatusEnum):
        """Safely transition to a new RobotStatusEnum, no-op on repeated same status."""
        if self.current_status != new_status:
            self.previous_status = self.current_status
            self.current_status  = new_status
            self.get_logger().info(
                f"Status: {self.previous_status.name} → {self.current_status.name}")

    # =========================================================================
    # SENSOR STATUS METHODS — mirrors HuskyOperationsManager
    # =========================================================================

    def _set_battery_status(self, robot_status: RobotStatus):
        """Populate battery fields in the outgoing RobotStatus."""
        robot_status.battery_level = self.battery_status.percentage
        if self.battery_status.capacity > 0.0 and self.battery_status.current > 0.0:
            battery_pct    = self._normalize_battery(self.battery_status.percentage)
            time_remaining = (self.battery_status.capacity * (battery_pct / 100.0) /
                              self.battery_status.current)
            robot_status.operation_hours_after_charging = self._format_time_remaining(time_remaining)
        else:
            robot_status.operation_hours_after_charging = "00 hours 00 minutes remaining approx..."

    def _set_estop_status(self, robot_status: RobotStatus):
        """Map emergency stop signal to online_flag in the outgoing RobotStatus."""
        robot_status.online_flag = (
            self.estop_status.data if self.estop_status.data else OnlineFlagEnum.ONLINE.value)

    def _set_location_status(self, robot_status: RobotStatus):
        """Populate position and orientation fields from the latest ground-truth pose."""
        if self.pose_status and self.pose_status.pose:
            robot_status.topo_map_position    = self.pose_status.pose.pose.position
            robot_status.topo_map_orientation = self.pose_status.pose.pose.orientation

    def _normalize_battery(self, percentage: float) -> float:
        """Normalise battery percentage to 0–100 range."""
        return percentage * 100.0 if percentage <= 1.0 else percentage

    def _format_time_remaining(self, hours: float) -> str:
        """Convert fractional hours to HH hours MM minutes string."""
        seconds          = int(hours * 3600)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes   = divmod(minutes, 60)
        return f"{hours:02} hours {minutes:02} minutes remaining approximately."

    def _calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Return the Euclidean distance between two 2D points."""
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def main(args=None):
    rclpy.init(args=args)
    node = LavenderHarvestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
