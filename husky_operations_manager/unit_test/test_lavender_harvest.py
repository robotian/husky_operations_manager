"""
test_lavender_harvest.py

Standalone integration test node for lavender row harvesting using DriveClient
and NavigationActionClient. Mirrors the architecture of HuskyOperationsManager
but replaces JobPublisher task management with a self-contained row traversal
state machine.

Operational sequence per row:
  1. Navigate (NavigationActionClient) to row side_a_start
  2. Drive (DriveClient) forward along Side A — detect and harvest bushes
  3. Navigate to row side_a_end → cross to side_b_start
  4. Drive (DriveClient) forward along Side B — detect and harvest bushes
  5. Navigate to row side_b_end → move to next row
  6. If load_status >= 100 at any point → interrupt → navigate to unloading dock
     → dock → simulate unload → undock → return to last known bush pose via start_pose

Safety:
  - E-stop monitoring: cancels all motion immediately → EMERGENCY_STOP
  - Battery low: same mechanism as HuskyOperationsManager → ERROR
  - Navigation failure: retry logic same as HuskyOperationsManager
  - DriveClient alignment timeout: log warning → proceed with harvest
  - Lost TF (arm_0_detections): → ERROR + stop robot
  - No detection timeout: assume row end → navigate to next waypoint
"""

import math
import time

import rclpy
import rclpy.time
import rclpy.duration
import tf2_ros
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformException

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import BatteryState, NavSatFix, Imu
from std_msgs.msg import Bool
from status_interfaces.msg import RobotStatus, Task, SubTask, UndockGoal, WayPoint, ImageDetectionPose

from husky_operations_manager.enum import (
    OnlineFlagEnum,
    RobotStatusEnum,
    NavigationStatus,
    ReverseDriveStatus,
    DockingParamFetcherStatus,
    DriveStatus,
)
from husky_operations_manager.dataclass import (
    DockingConfig, DockInstanceConfig, DockPluginConfig, DriveConfig
)
from husky_operations_manager.docking_param_fetcher import DockingParamFetcher
from husky_operations_manager.action_clients.reverse_drive_client import ReverseDriveClient
from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.action_clients.navigation import NavigationActionClient
from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.action_clients.drive_client_new import DriveClient


# ---------------------------------------------------------------------------
# Internal task phase enum — drives the high-level state machine
# ---------------------------------------------------------------------------

from enum import Enum, auto

class HarvestPhase(Enum):
    """High-level phase of the harvest operation."""
    INIT                    = auto()  # Startup / docking config fetch
    NAVIGATE_TO_ROW_START   = auto()  # NavigationActionClient → side_x_start
    DRIVE_ROW               = auto()  # DriveClient forward along row
    NAVIGATE_TO_ROW_END     = auto()  # NavigationActionClient → side_x_end
    NAVIGATE_TO_NEXT_SIDE   = auto()  # NavigationActionClient → side_x_start (other side)
    NAVIGATE_TO_UNLOAD_DOCK = auto()  # NavigationActionClient → active_dock pose
    DOCKING_FOR_UNLOAD      = auto()  # DockingActionClient
    UNLOADING               = auto()  # Simulated unload timer
    UNDOCKING_AFTER_UNLOAD  = auto()  # UndockingActionClient
    NAVIGATE_RETURN         = auto()  # NavigationActionClient → start_pose → last_bush_pose
    COMPLETE                = auto()  # All rows done


class RowSide(Enum):
    A = "A"
    B = "B"


# ---------------------------------------------------------------------------
# Waypoint helpers
# ---------------------------------------------------------------------------

def _make_waypoint(x: float, y: float, theta: float, node_id: int = 0) -> WayPoint:
    """Build a WayPoint message from x, y, theta."""
    wp = WayPoint()
    wp.node_id = node_id
    wp.x       = float(x)
    wp.y       = float(y)
    wp.theta   = float(theta)
    return wp


def _make_task_with_subtask( subtask: SubTask, task_id: int, task_type: int, description: str ) -> Task:
    """Build a Task wrapper around a single SubTask."""
    task = Task()
    task.task_id = task_id
    task.task_type = task_type
    task.description = description
    task.sub_tasks = [subtask]
    return task


class LocalTaskGenerator:
    """Lightweight in-node TaskGenerator replica for test execution."""

    @staticmethod
    def generate_moving_task(
        task_id: int,
        description: str,
        waypoints: list[WayPoint],
        task_type: int = Task.HARVESTING_TASK
        ) -> Task:
        move_sub_task = SubTask()
        move_sub_task.sub_task_id = 1
        move_sub_task.type = SubTask.MOVING
        move_sub_task.description = description
        move_sub_task.data = waypoints
        return _make_task_with_subtask(move_sub_task, task_id, task_type, description)

    @staticmethod
    def generate_docking_task(
        task_id: int,
        description: str,
        dock_goal,
        task_type: int = Task.UNLOADING_TASK,
    ) -> Task:
        docking_sub_task = SubTask()
        docking_sub_task.sub_task_id = 1
        docking_sub_task.type = SubTask.DOCKING
        docking_sub_task.description = description
        docking_sub_task.dock_goal = dock_goal
        return _make_task_with_subtask(docking_sub_task, task_id, task_type, description)

    @staticmethod
    def generate_unloading_task(
        task_id: int,
        description: str,
        undock_goal,
        task_type: int = Task.UNLOADING_TASK,
    ) -> Task:
        unloading_sub_task = SubTask()
        unloading_sub_task.sub_task_id = 1
        unloading_sub_task.type = SubTask.UNLOADING
        unloading_sub_task.description = description
        unloading_sub_task.undock_goal = undock_goal
        return _make_task_with_subtask(unloading_sub_task, task_id, task_type, description)

    @staticmethod
    def generate_harvesting_task(
        task_id: int,
        description: str = "Harvesting row",
    ) -> Task:
        harvesting_sub_task = SubTask()
        harvesting_sub_task.sub_task_id = 1
        harvesting_sub_task.type = SubTask.HARVESTING
        harvesting_sub_task.description = description
        return _make_task_with_subtask(harvesting_sub_task, task_id, Task.HARVESTING_TASK, description)


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class LavenderHarvestTestNode(Node):
    """
    Integration test node for lavender row harvesting.

    Uses NavigationActionClient to reach row waypoints and DriveClient
    to traverse rows detecting and harvesting lavender bushes.
    Mirrors HuskyOperationsManager architecture for safe portability.
    """

    def __init__(self):
        super().__init__('lavender_harvest_test_node')

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f"Node namespace: {self.namespace}")

        self._declare_parameters()
        self._get_parameters()
        self.task_generator = LocalTaskGenerator()
        self._init_state_variables()
        self._init_sensor_data()
        self._init_subscriptions()

        # RobotStatus publisher — mirrors HuskyOpsManager
        self.robot_state_pub = self.create_publisher(
            RobotStatus, f'{self.namespace}/status/robot', 10)

        # Action clients — set to None until DockingParamFetcher completes
        self.docking_config:          DockingConfig | None     = None
        self.active_dock:             DockInstanceConfig | None = None
        self.active_plugin:           DockPluginConfig | None   = None
        self.reverse_drive_client:    ReverseDriveClient | None = None
        self.navigation:              NavigationActionClient | None = None
        self.docking_action_client:   DockingActionClient | None   = None
        self.undocking_action_client: UndockingActionClient | None = None
        self.drive_client:            DriveClient | None            = None

        # TF for last known bush pose lookup
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # DockingParamFetcher — 30s timeout
        self._CONFIG_POLL_TIMEOUT_SEC: float = 30.0
        self._param_fetcher       = DockingParamFetcher(self)
        self._param_fetcher.fetch()
        self._config_poll_start   = self.get_clock().now().nanoseconds / 1e9
        self._config_poll_timer   = self.create_timer(0.5, self._poll_docking_config)

    # =========================================================================
    # PARAMETER DECLARATION AND LOADING
    # =========================================================================

    def _declare_parameters(self):
        """Declare all ROS2 parameters — sourced from YAML at launch."""
        # General
        self.declare_parameter('num_rows',              1)
        self.declare_parameter('no_detection_timeout',  10.0)
        self.declare_parameter('harvest_duration',      5.0)
        self.declare_parameter('unload_duration',       4.0)
        self.declare_parameter('loading.increment',     20.0)

        # Global poses [x, y, theta]
        self.declare_parameter('start_pose',           [0.0, -1.0, 0.0])

        # Row waypoints — flat structure, indexed by row number
        # Supports up to 10 rows; unused rows are ignored at runtime
        for i in range(10):
            self.declare_parameter(f'row_{i}_side_a_start', [0.0, 0.0, 0.0])
            self.declare_parameter(f'row_{i}_side_a_end',   [0.0, 5.0, 0.0])
            self.declare_parameter(f'row_{i}_side_b_start', [0.5, 5.0, 3.14])
            self.declare_parameter(f'row_{i}_side_b_end',   [0.5, 0.0, 3.14])

        # DriveClient
        self.declare_parameter('drive_v_linear',             0.2)
        self.declare_parameter('drive_v_angular',            0.5)
        self.declare_parameter('drive_tf_polling_rate',      10.0)
        self.declare_parameter('drive_alignment_tolerance',  0.05)
        self.declare_parameter('drive_alignment_timeout',    30.0)
        self.declare_parameter('drive_tf_target_frame',      'map')
        self.declare_parameter('drive_tf_base_frame',        'arm_0_base_link')
        self.declare_parameter('drive_tf_detection_frame',   'arm_0_detections')

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
        self.declare_parameter('timing.timer_period',                   1.0)
        self.declare_parameter('timing.initial_position_check_delay',   2.0)
        self.declare_parameter('server.action_server_timeout',          5.0)

    def _get_parameters(self):
        """Read all declared parameters into instance variables."""
        self.num_rows             = int(self.get_parameter('num_rows').value)
        self.no_detection_timeout = float(self.get_parameter('no_detection_timeout').value)
        self.harvest_duration     = float(self.get_parameter('harvest_duration').value)
        self.unload_duration      = float(self.get_parameter('unload_duration').value)
        self.loading_increment    = float(self.get_parameter('loading.increment').value)

        def _pose(name) -> list[float]:
            return list(self.get_parameter(name).value)

        self.start_pose          = _pose('start_pose')

        # Load all row waypoints into a list of dicts
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
        self.timing_timer_period    = float(self.get_parameter('timing.timer_period').value)
        self.timing_initial_check_delay = float(
            self.get_parameter('timing.initial_position_check_delay').value)
        self.server_action_server_timeout = float(
            self.get_parameter('server.action_server_timeout').value)

        # DriveConfig
        self._drive_config = DriveConfig(
            base_frame            = 'base_link',
            fixed_frame           = 'map',
            v_linear              = float(self.get_parameter('drive_v_linear').value),
            v_angular             = float(self.get_parameter('drive_v_angular').value),
            tf_polling_rate       = float(self.get_parameter('drive_tf_polling_rate').value),
            alignment_tolerance   = float(self.get_parameter('drive_alignment_tolerance').value),
            alignment_timeout     = float(self.get_parameter('drive_alignment_timeout').value),
            tf_target_frame       = str(self.get_parameter('drive_tf_target_frame').value),
            tf_base_frame         = str(self.get_parameter('drive_tf_base_frame').value),
            tf_detection_frame    = str(self.get_parameter('drive_tf_detection_frame').value),
        )

        self.get_logger().info(
            f"Parameters loaded | rows={self.num_rows} | "
            f"harvest={self.harvest_duration}s unload={self.unload_duration}s | "
            f"load_increment={self.loading_increment}% | "
            f"no_detection_timeout={self.no_detection_timeout}s"
        )

    # =========================================================================
    # STATE VARIABLE INITIALISATION
    # =========================================================================

    def _init_state_variables(self):
        """Initialise all state-tracking variables to boot defaults."""

        # --- Startup ---
        self.is_initialized          = False
        self.is_at_docking_station   = False
        self.startup_undock_complete = False

        # --- Robot state ---
        self.current_status  = RobotStatusEnum.IDLE
        self.previous_status = RobotStatusEnum.IDLE

        # --- Harvest phase ---
        self.current_phase: str = "INIT"
        self.current_row:     int          = 0
        self.current_side:    RowSide      = RowSide.A

        # --- Load ---
        self.current_load_status: float = 0.0

        # --- Last known bush pose (arm_0_base_link in map frame at interrupt) ---
        self.last_bush_pose: list[float] | None = None  # [x, y, theta]

        # --- Unloading return context ---
        # Stores which row/side/pose to return to after unloading completes
        self.return_row:  int          = 0
        self.return_side: RowSide      = RowSide.A

        # --- Retry counters ---
        self.navigation_retry_count = 0
        self.docking_retry_count    = 0

        # --- Job simulation timers ---
        self.job_start_time = None
        self.job_duration   = 0.0

        # --- Undocking ---
        self.last_undocking_subtask:    SubTask | None = None
        self.undocking_after_task_type: int | None     = None

        # --- Reverse drive ---
        self.reverse_drive_active: bool = False

        # --- Task/SubTask context (aligned with HuskyOperationsManager pattern) ---
        self.current_task: Task | None = None
        self.current_sub_task: SubTask | None = None
        self.current_sub_task_index: int = 0
        self.last_handled_task_id: int = -1
        self.last_handled_task_type: int = -1
        self.last_handled_subtask_type: int | None = None

        # --- No detection timeout ---
        self.last_detection_time: float | None = None
        self._detection_received: bool         = False

        # --- E-stop state ---
        self._estop_active: bool = False

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
            f'{self.namespace}/ssensors/gps_0/imu',
            lambda msg: setattr(self, 'imu_status', msg),
            qos_profile_sensor_data)

        self.estop_sub = self.create_subscription(
            Bool,
            f'{self.namespace}/platform/emergency_stop',
            self._estop_callback,
            qos_profile_sensor_data)

        # ImageDetectionPose — owns subscription, calls drive_client.stop() on valid detection
        detection_topic = (
            f'{self.namespace}/manipulators/arm_0_detection'
            f'/image_annotated/detection_pose'
        )
        self.detection_sub = self.create_subscription(
            ImageDetectionPose,
            detection_topic,
            self._detection_callback,
            10)

        self.get_logger().info(f"Subscribed to detection topic: {detection_topic}")
        self.get_logger().debug("Subscriptions initialised")

    # =========================================================================
    # DOCKING CONFIG FETCHER
    # =========================================================================

    def _poll_docking_config(self):
        """Poll DockingParamFetcher every 0.5s. Mirrors HuskyOperationsManager."""
        status = self._param_fetcher.get_status()
        self.get_logger().debug(f"DockingParamFetcher poll | status={status.name}")

        if status == DockingParamFetcherStatus.DONE:
            self._config_poll_timer.cancel()
            self._on_docking_config_ready()
            return

        if status == DockingParamFetcherStatus.ERROR:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self._config_poll_start

            if elapsed >= self._CONFIG_POLL_TIMEOUT_SEC:
                self.get_logger().error(
                    f"DockingParamFetcher failed after {self._CONFIG_POLL_TIMEOUT_SEC:.0f}s "
                    f"— shutting down")
                self._config_poll_timer.cancel()
                self.destroy_node()
                rclpy.shutdown()
                return

            self.get_logger().warning(
                f"DockingParamFetcher ERROR — retrying | "
                f"elapsed={elapsed:.1f}s / {self._CONFIG_POLL_TIMEOUT_SEC:.0f}s")
            self._param_fetcher.reset()
            self._param_fetcher.fetch()

    def _on_docking_config_ready(self):
        """Initialise all action clients once DockingConfig is available."""
        self.docking_config = self._param_fetcher.get_config()
        self.active_dock    = self.docking_config.dock_configs[self.docking_config.docks[0]]
        self.active_plugin  = self.docking_config.plugin_configs[self.docking_config.dock_plugins[0]]

        self.get_logger().info(
            f"DockingConfig ready | dock='{self.active_dock.instance_name}' | "
            f"plugin='{self.active_plugin.plugin_name}'")

        self.navigation              = NavigationActionClient(self)
        self.docking_action_client   = DockingActionClient(self)
        self.undocking_action_client = UndockingActionClient(self)
        self.reverse_drive_client    = ReverseDriveClient(self, self.docking_config)
        self.drive_client            = DriveClient(self, self._drive_config)

        self.init_check_timer = self.create_timer(
            self.timing_initial_check_delay,
            self._initial_position_check_timer)

        self.timer = self.create_timer(self.timing_timer_period, self.timer_callback)
        self.get_logger().info("LavenderHarvestTestNode ready.")

    # =========================================================================
    # SUBSCRIPTION CALLBACKS
    # =========================================================================

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        """Store latest ground-truth pose."""
        self.pose_status = msg

    def _estop_callback(self, msg: Bool):
        """
        E-stop safety handler.
        On rising edge (estop activated): cancel all motion immediately.
        """
        was_active       = self._estop_active
        self._estop_active = bool(msg.data)
        self.estop_status  = msg

        if self._estop_active and not was_active:
            self.get_logger().error(
                "EMERGENCY STOP activated — cancelling all motion immediately.")
            self._emergency_stop_all()

    def _detection_callback(self, msg: ImageDetectionPose):
        """
        ImageDetectionPose callback — owned by this node, mirrors HuskyOpsManager.

        On detection_valid=True:
          - Update last detection timestamp (for no-detection timeout)
          - If robot is in DRIVE_ROW phase and DriveClient is active → call drive_client.stop()
        On detection_valid=False:
          - No action on DriveClient
        """
        if not msg.detection_valid:
            return

        self.last_detection_time = self.get_clock().now().nanoseconds / 1e9
        self._detection_received = True

        self.get_logger().debug(
            f"Detection valid | "
            f"center=({msg.center.x:.3f}, {msg.center.y:.3f}, {msg.center.z:.3f}) | "
            f"phase={self.current_phase}")

        # Only call stop during DRIVE_ROW phase when DriveClient is active
        if (self.current_phase == "DRIVE_ROW" and
                self.drive_client is not None and
                self.drive_client.is_active()):
            self.get_logger().info(
                "Valid detection received during DRIVE_ROW — calling drive_client.stop()")
            self.drive_client.stop()

    # =========================================================================
    # MAIN TIMER CALLBACK
    # =========================================================================

    def timer_callback(self):
        """
        Main control loop — fires at timing_timer_period (default 1Hz).

        Priority order:
          1. E-stop — already handled in _estop_callback (event-driven)
          2. ERROR/ABNORMAL → _handle_error_recovery
          3. Startup undocking not done → _handle_startup_undocking
          4. Normal operation → _handle_harvest_execution
        """
        robot_status = RobotStatus()
        robot_status.header.stamp    = self.get_clock().now().to_msg()
        robot_status.robot_namespace = self.namespace.replace('/', '')

        self._set_battery_status(robot_status)
        self._set_estop_status(robot_status)
        self._set_location_status(robot_status)

        self.get_logger().debug(
            f"Timer tick | status={self.current_status.name} | "
            f"phase={self.current_phase} | "
            f"row={self.current_row} side={self.current_side.value} | "
            f"load={self.current_load_status:.1f}% | "
            f"startup_done={self.startup_undock_complete} | "
            f"battery={self._normalize_battery(self.battery_status.percentage):.1f}%")

        if self.current_status in [RobotStatusEnum.EMERGENCY_STOP]:
            robot_status.status      = self.current_status.value
            robot_status.load_status = self.current_load_status
            self.robot_state_pub.publish(robot_status)
            return

        if self.current_status in [RobotStatusEnum.ERROR, RobotStatusEnum.ABNORMAL]:
            self._handle_error_recovery()
        elif not self.startup_undock_complete:
            self._handle_startup_undocking(robot_status)
        else:
            self._handle_harvest_execution(robot_status)

        robot_status.status      = self.current_status.value
        robot_status.load_status = self.current_load_status
        self.robot_state_pub.publish(robot_status)

    # =========================================================================
    # STARTUP SEQUENCE — mirrors HuskyOperationsManager
    # =========================================================================

    def _initial_position_check_timer(self):
        """Fires once after initial_position_check_delay."""
        if self.pose_status is None:
            self.get_logger().warning("Waiting for pose data...")
            return
        self.init_check_timer.cancel()
        self.check_initial_position()

    def check_initial_position(self):
        """Determine whether the robot starts at the docking station."""
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
            f"Initial position | robot=({current_pos.x:.3f}, {current_pos.y:.3f}) | "
            f"dock='{charger_dock_pose.instance_name}' | distance={distance:.3f}m")

        if distance <= self.docking_threshold:
            self.is_at_docking_station = True
            self.get_logger().info("Robot at dock — will undock before harvest")
        else:
            self.is_at_docking_station   = False
            self.startup_undock_complete = True
            self.current_phase           = "NAVIGATE_TO_ROW_START"
            self.get_logger().info("Robot not at dock — ready for harvest")

        self.is_initialized = True

    def _handle_startup_undocking(self, robot_status: RobotStatus):
        """Drive startup undocking state machine — mirrors HuskyOperationsManager."""
        if not self.is_initialized or self.startup_undock_complete:
            return

        if self.docking_config is None:
            self.get_logger().warning("Startup undocking called but docking_config is None")
            return

        if self.current_status == RobotStatusEnum.IDLE:
            self.get_logger().info("Starting startup undocking...")
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            robot_status.task = "Startup: Preparing to undock"

        elif self.current_status == RobotStatusEnum.START_UNDOCKING:
            robot_status.task  = "Startup: Undocking"
            dock_type          = self.active_dock.type
            staging_x_offset   = self.active_plugin.staging_x_offset
            v_linear           = self.docking_config.controller_v_linear_max
            max_undocking_time = (abs(staging_x_offset) / max(v_linear, 0.01)) * 1.25

            startup_subtask             = SubTask()
            startup_subtask.type        = SubTask.UNDOCKING
            startup_subtask.description = "Startup Undocking"
            startup_subtask.undock_goal = UndockGoal(
                dock_type=dock_type,
                max_undocking_time=max_undocking_time)
            self.last_undocking_subtask = startup_subtask
            self._subtask_undocking()

        else:
            if self.reverse_drive_active:
                self._handle_reverse_drive(robot_status)
            else:
                self._handle_undocking(robot_status)

    # =========================================================================
    # HARVEST EXECUTION — main phase router
    # =========================================================================

    def _handle_harvest_execution(self, robot_status: RobotStatus):
        """
        Route to the correct handler based on current HarvestPhase.

        Checks battery low before routing — same mechanism as HuskyOperationsManager.
        """
        battery_pct = self._normalize_battery(self.battery_status.percentage)

        # Battery low check — cancel current motion and go to ERROR
        if battery_pct <= self.battery_low_threshold:
            self.get_logger().warning(
                f"Battery low: {battery_pct:.1f}% — interrupting harvest")
            self._cancel_all_motion()
            self._transition_status(RobotStatusEnum.ERROR)
            return

        # Process active navigation/docking/undocking/reverse drive clients
        # before phase routing so async results are processed on same tick
        self._process_action_clients(robot_status)

        robot_status.task = (
            f"Phase: {self.current_phase} | "
            f"Row: {self.current_row} | Side: {self.current_side.value}"
        )

        if self.current_phase == "UNDOCKING_AFTER_UNLOAD":
            self._phase_undocking_after_unload(robot_status)
            return
        if self.current_phase == "COMPLETE":
            self._phase_complete(robot_status)
            return

        self._prepare_phase_task_if_needed()
        self._update_current_subtask()
        self._handle_task_start()
        self._execute_current_subtask()

    def _prepare_phase_task_if_needed(self):
        """Generate in-node Task/SubTask matching the current phase."""
        if self.current_task is not None:
            return

        if self.current_phase == "DRIVE_ROW":
            self.current_task = self.task_generator.generate_harvesting_task(
                task_id=2000 + self.current_row,
                description=f"Harvest Row {self.current_row} Side {self.current_side.value}",
            )
            return

        def _set_nav_task(task_id: int, desc: str, poses: list[list[float]]):
            waypoints = [_make_waypoint(*pose) for pose in poses]
            self.current_task = self.task_generator.generate_moving_task(task_id, desc, waypoints)

        row = self.row_waypoints[self.current_row]
        if self.current_phase == "NAVIGATE_TO_ROW_START":
            key = 'side_a_start' if self.current_side == RowSide.A else 'side_b_start'
            _set_nav_task(
                self.current_row,
                f"Navigate to Row {self.current_row} Side {self.current_side.value} Start",
                [row[key]],
            )
        elif self.current_phase == "NAVIGATE_TO_ROW_END":
            key = 'side_a_end' if self.current_side == RowSide.A else 'side_b_end'
            _set_nav_task(
                self.current_row + 100,
                f"Navigate to Row {self.current_row} Side {self.current_side.value} End",
                [row[key]],
            )
        elif self.current_phase == "NAVIGATE_TO_NEXT_SIDE":
            _set_nav_task(
                self.current_row + 200,
                f"Cross to Row {self.current_row} Side B",
                [row['side_b_start']],
            )
        elif self.current_phase == "NAVIGATE_TO_UNLOAD_DOCK":
            _set_nav_task(
                999,
                "Navigate to Unloading Dock",
                [[self.active_dock.dock_x, self.active_dock.dock_y, self.active_dock.dock_theta]],
            )
        elif self.current_phase == "NAVIGATE_RETURN":
            poses = [self.start_pose]
            if self.last_bush_pose:
                poses.append(self.last_bush_pose)
            else:
                row = self.row_waypoints[self.return_row]
                key = 'side_a_start' if self.return_side == RowSide.A else 'side_b_start'
                poses.append(row[key])
            _set_nav_task(998, "Return to last bush pose via start pose", poses)
        elif self.current_phase == "DOCKING_FOR_UNLOAD":
            from status_interfaces.msg import DockGoal
            dock_goal = DockGoal(
                use_dock_id=True,
                dock_id=self.active_dock.id,
                navigate_to_staging_pose=True,
            )
            self.current_task = self.task_generator.generate_docking_task(
                task_id=996,
                description="Dock for unloading",
                dock_goal=dock_goal,
            )
        elif self.current_phase == "UNLOADING":
            dock_type = self.active_dock.type
            staging_x_offset = self.active_plugin.staging_x_offset
            v_linear = self.docking_config.controller_v_linear_max
            max_undocking_time = (abs(staging_x_offset) / max(v_linear, 0.01)) * 1.25
            undock_goal = UndockGoal(
                dock_type=dock_type,
                max_undocking_time=max_undocking_time,
            )
            self.current_task = self.task_generator.generate_unloading_task(
                task_id=997,
                description="Unload at unloading station",
                undock_goal=undock_goal,
            )

    def _update_current_subtask(self):
        """Match husky-ops subtask indexing behavior."""
        if not self.current_task:
            self.current_sub_task = None
            return
        if isinstance(self.current_task.sub_tasks, list):
            if self.current_sub_task_index < len(self.current_task.sub_tasks):
                self.current_sub_task = self.current_task.sub_tasks[self.current_sub_task_index]
            else:
                self.current_sub_task = None

    def _handle_task_start(self):
        """Match husky-ops task start behavior for generated tasks."""
        if not self.current_task:
            return
        if (
            self.current_task.task_id != self.last_handled_task_id
            or self.current_task.task_type != self.last_handled_task_type
        ):
            self.last_handled_task_id = self.current_task.task_id
            self.last_handled_task_type = self.current_task.task_type
            self.last_handled_subtask_type = None
            self.current_sub_task_index = 0
            if self.current_status == RobotStatusEnum.IDLE:
                self._transition_status(RobotStatusEnum.JOB_START)

    def _execute_current_subtask(self):
        """Route by SubTask type like husky-ops lifecycle."""
        if not isinstance(self.current_sub_task, SubTask):
            return
        if self.current_sub_task.type != self.last_handled_subtask_type:
            self.get_logger().info(f"Executing: {self.current_sub_task.description}")
            self.last_handled_subtask_type = self.current_sub_task.type
        handler = {
            SubTask.MOVING: self._subtask_moving,
            SubTask.HARVESTING: self._subtask_harvesting,
            SubTask.DOCKING: self._subtask_docking,
            SubTask.UNLOADING: self._subtask_unloading,
        }.get(self.current_sub_task.type)
        if handler:
            handler()

    def _subtask_moving(self):
        if self.current_phase == "NAVIGATE_TO_ROW_START":
            self._phase_navigate_to_row_start(RobotStatus())
        elif self.current_phase == "NAVIGATE_TO_ROW_END":
            self._phase_navigate_to_row_end(RobotStatus())
        elif self.current_phase == "NAVIGATE_TO_NEXT_SIDE":
            self._phase_navigate_to_next_side(RobotStatus())
        elif self.current_phase == "NAVIGATE_TO_UNLOAD_DOCK":
            self._phase_navigate_to_unload_dock(RobotStatus())
        elif self.current_phase == "NAVIGATE_RETURN":
            self._phase_navigate_return(RobotStatus())

    def _subtask_harvesting(self):
        self._phase_drive_row(RobotStatus())

    def _subtask_docking(self):
        self._phase_docking_for_unload(RobotStatus())

    def _subtask_unloading(self):
        self._phase_unloading(RobotStatus())

    def _clear_current_task_context(self):
        self.current_task = None
        self.current_sub_task = None
        self.current_sub_task_index = 0
        self.last_handled_subtask_type = None

    # =========================================================================
    # PHASE HANDLERS
    # =========================================================================

    def _phase_navigate_to_row_start(self, robot_status: RobotStatus):
        """
        Navigate to the start of the current row side using NavigationActionClient.

        State: IDLE → JOB_START → START_MOVING → MOVING → DESTINATION_REACHED
        """
        if self.current_status in (RobotStatusEnum.IDLE, RobotStatusEnum.JOB_START):
            self._transition_status(RobotStatusEnum.START_MOVING)

        elif self.current_status == RobotStatusEnum.START_MOVING:
            if self.navigation.is_navigation_active():
                return
            if self.navigation.send_goal(self.current_task):
                self._transition_status(RobotStatusEnum.MOVING)
            else:
                self.get_logger().error("Failed to send navigation goal to row start")
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            # Arrived at row start — reset detection state and start DriveClient
            self.last_detection_time = None
            self._detection_received = False
            self.current_phase       = "DRIVE_ROW"
            self._clear_current_task_context()
            self._transition_status(RobotStatusEnum.IDLE)
            self.get_logger().info(
                f"Arrived at Row {self.current_row} Side {self.current_side.value} start "
                f"— starting DriveClient")

    def _phase_drive_row(self, robot_status: RobotStatus):
        """
        Drive along the row using DriveClient.

        DriveClient.forward() starts moving and TF polling.
        Detection callback calls drive_client.stop() on detection_valid=True.
        DriveClient validates pose alignment before stopping.

        Monitors:
          - DriveStatus.IDLE after stop → DESTINATION_REACHED → START_HARVESTING
          - DriveStatus.ERROR → log + transition to ERROR
          - No detection timeout → assume row end → NAVIGATE_TO_ROW_END
          - Load >= 100 → interrupt → NAVIGATE_TO_UNLOAD_DOCK
        """
        if self.drive_client is None:
            return

        drive_status = self.drive_client.get_status()

        # Map DriveStatus to RobotStatusEnum
        if drive_status in (DriveStatus.FORWARD, DriveStatus.REVERSE):
            self._transition_status(RobotStatusEnum.MOVING)
        elif drive_status in (DriveStatus.CANCELED, DriveStatus.ERROR):
            self.get_logger().error(
                f"DriveClient error during row traversal | status={drive_status.name}")
            self._transition_status(RobotStatusEnum.ERROR)
            return

        # Start DriveClient if IDLE and not yet moving
        if (self.current_status == RobotStatusEnum.IDLE and
                drive_status == DriveStatus.IDLE):
            self.get_logger().info(
                f"Starting DriveClient for Row {self.current_row} "
                f"Side {self.current_side.value}")
            self.last_detection_time = self.get_clock().now().nanoseconds / 1e9
            self.drive_client.forward()
            return

        # No-detection timeout check — assume row end
        if self.last_detection_time is not None:
            elapsed = (self.get_clock().now().nanoseconds / 1e9
                       - self.last_detection_time)
            if elapsed >= self.no_detection_timeout:
                self.get_logger().info(
                    f"No detection for {elapsed:.1f}s "
                    f"(timeout={self.no_detection_timeout}s) — assuming row end")
                self.drive_client.cancel()
                self.current_phase = "NAVIGATE_TO_ROW_END"
                self._transition_status(RobotStatusEnum.IDLE)
                return

        # DriveClient stopped (aligned) — check load then harvest
        if (drive_status == DriveStatus.IDLE and
                self.current_status == RobotStatusEnum.MOVING):

            # Load interrupt check
            if self.current_load_status >= 100.0:
                self.get_logger().info(
                    "Load status >= 100% — interrupting harvest for unloading")
                self._capture_last_bush_pose()
                self.return_row  = self.current_row
                self.return_side = self.current_side
                self.current_phase = "NAVIGATE_TO_UNLOAD_DOCK"
                self._transition_status(RobotStatusEnum.IDLE)
                return

            # Proceed to harvest
            self._transition_status(RobotStatusEnum.DESTINATION_REACHED)

        elif self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            self._transition_status(RobotStatusEnum.START_HARVESTING)

        elif self.current_status == RobotStatusEnum.START_HARVESTING:
            self._transition_status(RobotStatusEnum.HARVESTING)
            self.get_logger().info("Harvesting started (simulated)")
            self.job_start_time = self.get_clock().now()
            self.job_duration   = self.harvest_duration

        elif self.current_status == RobotStatusEnum.HARVESTING:
            if self.job_start_time:
                elapsed = (self.get_clock().now() - self.job_start_time).nanoseconds / 1e9
                if elapsed >= self.job_duration:
                    self.get_logger().info("Harvesting complete (simulated)")
                    self._transition_status(RobotStatusEnum.DONE_HARVESTING)
                    self.job_start_time = None

        elif self.current_status == RobotStatusEnum.DONE_HARVESTING:
            new_load = min(self.current_load_status + self.loading_increment, 100.0)
            self.get_logger().info(
                f"Load update: {self.current_load_status:.1f}% → {new_load:.1f}%")
            self.current_load_status = new_load

            # Load interrupt — save pose and go unload
            if self.current_load_status >= 100.0:
                self.get_logger().info(
                    "Load reached 100% after harvest — navigating to unload dock")
                self._capture_last_bush_pose()
                self.return_row    = self.current_row
                self.return_side   = self.current_side
                self.current_phase = "NAVIGATE_TO_UNLOAD_DOCK"
                self._transition_status(RobotStatusEnum.IDLE)
            else:
                # Resume DriveClient forward for next bush
                self.get_logger().info("Resuming DriveClient for next bush")
                self.last_detection_time = self.get_clock().now().nanoseconds / 1e9
                self._detection_received = False
                self._transition_status(RobotStatusEnum.IDLE)
                self.drive_client.forward()

    def _phase_navigate_to_row_end(self, robot_status: RobotStatus):
        """Navigate to the end of the current row side."""
        if self.current_status in (RobotStatusEnum.IDLE, RobotStatusEnum.JOB_START):
            self._transition_status(RobotStatusEnum.START_MOVING)

        elif self.current_status == RobotStatusEnum.START_MOVING:
            if self.navigation.is_navigation_active():
                return
            if self.navigation.send_goal(self.current_task):
                self._transition_status(RobotStatusEnum.MOVING)
            else:
                self.get_logger().error("Failed to send navigation goal to row end")
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            # Decide next phase based on which side just completed
            if self.current_side == RowSide.A:
                # Side A done — cross to Side B start
                self.current_phase = "NAVIGATE_TO_NEXT_SIDE"
                self.current_side  = RowSide.B
            else:
                # Side B done — move to next row or complete
                self.current_row += 1
                self.current_side  = RowSide.A
                if self.current_row >= self.num_rows:
                    self.current_phase = "COMPLETE"
                else:
                    self.current_phase = "NAVIGATE_TO_ROW_START"
            self._clear_current_task_context()
            self._transition_status(RobotStatusEnum.IDLE)

    def _phase_navigate_to_next_side(self, robot_status: RobotStatus):
        """Navigate from side A end to side B start (crossing the row)."""
        if self.current_status in (RobotStatusEnum.IDLE, RobotStatusEnum.JOB_START):
            self._transition_status(RobotStatusEnum.START_MOVING)

        elif self.current_status == RobotStatusEnum.START_MOVING:
            if self.navigation.is_navigation_active():
                return
            if self.navigation.send_goal(self.current_task):
                self._transition_status(RobotStatusEnum.MOVING)
            else:
                self.get_logger().error("Failed to send navigation goal to Side B start")
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            self.get_logger().info(
                f"Arrived at Row {self.current_row} Side B — starting DriveClient")
            self.current_phase = "DRIVE_ROW"
            self._clear_current_task_context()
            self._transition_status(RobotStatusEnum.IDLE)

    def _phase_navigate_to_unload_dock(self, robot_status: RobotStatus):
        """Navigate to the active dock pose before docking."""
        if self.current_status in (RobotStatusEnum.IDLE, RobotStatusEnum.JOB_START):
            self._transition_status(RobotStatusEnum.START_MOVING)

        elif self.current_status == RobotStatusEnum.START_MOVING:
            if self.navigation.is_navigation_active():
                return
            if self.navigation.send_goal(self.current_task):
                self._transition_status(RobotStatusEnum.MOVING)
            else:
                self.get_logger().error("Failed to send navigation goal to unload dock")
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            self.current_phase = "DOCKING_FOR_UNLOAD"
            self._clear_current_task_context()
            self._transition_status(RobotStatusEnum.IDLE)

    def _phase_docking_for_unload(self, robot_status: RobotStatus):
        """
        Dock at the unloading station.
        State: IDLE → START_DOCKING → DOCKING → DONE_DOCKING
        """
        dock_status = self.docking_action_client.get_status()

        if self.current_status == RobotStatusEnum.IDLE:
            self._transition_status(RobotStatusEnum.START_DOCKING)

        elif self.current_status == RobotStatusEnum.START_DOCKING:
            time.sleep(1.0)  # Allow robot to fully stop before docking
            # Primary path: use received/generated current_sub_task.dock_goal.
            # Fallback construction is handled in _handle_docking_retry only.
            if self.current_sub_task and self.current_sub_task.dock_goal:
                dock_subtask = self.current_sub_task
            else:
                self.get_logger().error("Docking subtask missing dock_goal")
                self._transition_status(RobotStatusEnum.ERROR)
                return

            if self.docking_action_client.send_docking_goal(dock_subtask):
                self._transition_status(RobotStatusEnum.DOCKING)
            else:
                self.get_logger().error("Failed to send docking goal")
                self._transition_status(RobotStatusEnum.ERROR)

        elif dock_status == RobotStatusEnum.DOCKING:
            self._transition_status(RobotStatusEnum.DOCKING)

        elif dock_status == RobotStatusEnum.DONE_DOCKING:
            self.get_logger().info("Docking complete — starting unload")
            self.docking_action_client.reset()
            self.docking_retry_count = 0
            self.current_phase       = "UNLOADING"
            self._clear_current_task_context()
            self._transition_status(RobotStatusEnum.DONE_DOCKING)

        elif dock_status == RobotStatusEnum.ERROR:
            self._handle_docking_retry()

    def _phase_unloading(self, robot_status: RobotStatus):
        """
        Simulate unloading — mirrors HuskyOperationsManager._subtask_unloading().
        State: DONE_DOCKING → START_UNLOADING → UNLOADING → DONE_UNLOADING
        """
        if self.current_status == RobotStatusEnum.DONE_DOCKING:
            self._transition_status(RobotStatusEnum.START_UNLOADING)

        elif self.current_status == RobotStatusEnum.START_UNLOADING:
            self._transition_status(RobotStatusEnum.UNLOADING)
            self.get_logger().info("Unloading started (simulated)")
            self.job_start_time = self.get_clock().now()
            self.job_duration   = self.unload_duration

        elif self.current_status == RobotStatusEnum.UNLOADING:
            if self.job_start_time:
                elapsed = (self.get_clock().now() - self.job_start_time).nanoseconds / 1e9
                if elapsed >= self.job_duration:
                    self.get_logger().info("Unloading complete (simulated)")
                    self.current_load_status = 0.0
                    self._transition_status(RobotStatusEnum.DONE_UNLOADING)
                    self.job_start_time = None

        elif self.current_status == RobotStatusEnum.DONE_UNLOADING:
            # Trigger undocking
            self.get_logger().info("Unloading done — triggering undocking")
            if not (self.current_sub_task and self.current_sub_task.undock_goal):
                self.get_logger().error("Unloading subtask missing undock_goal")
                self._transition_status(RobotStatusEnum.ERROR)
                return
            undock_subtask = SubTask()
            undock_subtask.type = SubTask.UNDOCKING
            undock_subtask.description = "Undock after unloading"
            undock_subtask.undock_goal = self.current_sub_task.undock_goal

            self.last_undocking_subtask    = undock_subtask
            self.undocking_after_task_type = Task.UNLOADING_TASK
            self.current_phase             = "UNDOCKING_AFTER_UNLOAD"
            self._clear_current_task_context()
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            self._subtask_undocking()

    def _phase_undocking_after_unload(self, robot_status: RobotStatus):
        """Monitor undocking after unloading — delegates to _handle_undocking."""
        if self.reverse_drive_active:
            self._handle_reverse_drive(robot_status)
        else:
            self._handle_undocking(robot_status)

        # When undocking completes → navigate back to last bush pose
        if self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info(
                "Undocking complete — navigating back to last bush pose")
            self.current_phase = "NAVIGATE_RETURN"
            self._transition_status(RobotStatusEnum.IDLE)

    def _phase_navigate_return(self, robot_status: RobotStatus):
        """
        Navigate back to last known bush pose via start_pose.

        Single NavigationActionClient goal:
          active_dock_pose → start_pose → last_bush_pose
        """
        if self.current_status in (RobotStatusEnum.IDLE, RobotStatusEnum.JOB_START):
            self._transition_status(RobotStatusEnum.START_MOVING)

        elif self.current_status == RobotStatusEnum.START_MOVING:
            if self.navigation.is_navigation_active():
                return
            if self.navigation.send_goal(self.current_task):
                self._transition_status(RobotStatusEnum.MOVING)
            else:
                self.get_logger().error("Failed to send return navigation goal")
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            # Resume harvest from the side we were interrupted on
            self.get_logger().info(
                f"Returned to last bush pose — resuming "
                f"Row {self.return_row} Side {self.return_side.value}")
            self.current_row   = self.return_row
            self.current_side  = self.return_side
            self.current_phase = "DRIVE_ROW"
            self.last_bush_pose = None
            self._clear_current_task_context()
            self._transition_status(RobotStatusEnum.IDLE)

    def _phase_complete(self, robot_status: RobotStatus):
        """All rows harvested — log completion and remain IDLE."""
        if self.current_status != RobotStatusEnum.IDLE:
            self.get_logger().info(
                f"All {self.num_rows} row(s) harvested. "
                f"Total load delivered: {self.current_load_status:.1f}%")
            self._transition_status(RobotStatusEnum.IDLE)

    # =========================================================================
    # ACTION CLIENT HANDLERS — mirrors HuskyOperationsManager
    # =========================================================================

    def _process_action_clients(self, robot_status: RobotStatus):
        """Process active action clients each timer tick."""
        nav_status    = self.navigation.get_navigation_status()
        dock_status   = self.docking_action_client.get_status()
        undock_status = self.undocking_action_client.get_status()

        if nav_status != NavigationStatus.IDLE:
            self._handle_navigation(robot_status)
        elif dock_status != RobotStatusEnum.IDLE:
            self._handle_docking(robot_status)
        elif undock_status != RobotStatusEnum.IDLE:
            self._handle_undocking(robot_status)
        elif self.reverse_drive_active:
            self._handle_reverse_drive(robot_status)

    def _handle_navigation(self, robot_status: RobotStatus):
        """Monitor NavigationActionClient — mirrors HuskyOperationsManager."""
        nav_status = self.navigation.get_navigation_status()
        wpf_status = self.navigation.get_current_status()

        if wpf_status:
            robot_status.task            = wpf_status.task
            robot_status.current_node_id = wpf_status.current_node_id
            robot_status.target_node_id  = wpf_status.target_node_id

        if nav_status == NavigationStatus.ACTIVE:
            self._transition_status(RobotStatusEnum.MOVING)

        elif nav_status == NavigationStatus.SUCCEEDED:
            self.get_logger().info("Navigation complete")
            self.navigation_retry_count = 0
            self.navigation.reset()
            self._transition_status(RobotStatusEnum.DESTINATION_REACHED)

        elif nav_status in [NavigationStatus.ABORTED, NavigationStatus.ERROR]:
            self._handle_navigation_retry()

        elif nav_status == NavigationStatus.CANCELED:
            self.get_logger().info("Navigation canceled")
            self.navigation.reset()
            self.navigation_retry_count = 0
            self._transition_status(RobotStatusEnum.IDLE)

    def _handle_navigation_retry(self):
        """Retry navigation up to max_retries — mirrors HuskyOperationsManager."""
        self.get_logger().warning(
            f"Navigation failed | retry {self.navigation_retry_count + 1}/"
            f"{self.navigation_max_retries}")

        if self._is_robot_at_target():
            self.get_logger().info(
                "Robot already at target — treating navigation as complete")
            self.navigation_retry_count = 0
            self.navigation.reset()
            self._transition_status(RobotStatusEnum.DESTINATION_REACHED)
            return

        if self.navigation_retry_count < self.navigation_max_retries:
            self.navigation_retry_count += 1
            time.sleep(self.navigation_retry_delay)
            self._retry_navigation()
        else:
            self.get_logger().error(
                f"Navigation failed after {self.navigation_max_retries} retries")
            self.navigation.reset()
            self._transition_status(RobotStatusEnum.ERROR)
            self.navigation_retry_count = 0

    def _retry_navigation(self):
        """Re-send the current navigation goal."""
        if not self.current_task:
            self.get_logger().error("Navigation retry failed — no current task")
            return

        nav_status = self.navigation.get_navigation_status()
        if nav_status in [NavigationStatus.ACTIVE, NavigationStatus.SENDING]:
            self.get_logger().warning(
                f"Navigation still active ({nav_status.name}) — skipping retry")
            return

        self.navigation.reset()
        time.sleep(self.navigation_retry_delay)

        if self.navigation.send_goal(self.current_task):
            self._transition_status(RobotStatusEnum.MOVING)
        else:
            self.get_logger().error("Navigation retry send_goal failed")
            self._transition_status(RobotStatusEnum.ERROR)

    def _handle_docking(self, robot_status: RobotStatus):
        """Monitor DockingActionClient."""
        status   = self.docking_action_client.get_status()
        feedback = self.docking_action_client.get_feedback()

        if feedback:
            robot_status.task = feedback.task

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
        """Retry docking up to max_retries."""
        if self.docking_retry_count < self.docking_max_retries:
            self.docking_retry_count += 1
            self.get_logger().warning(
                f"Retrying docking | attempt {self.docking_retry_count}/"
                f"{self.docking_max_retries}")
            time.sleep(self.docking_retry_delay)
            from status_interfaces.msg import DockGoal
            dock_subtask             = SubTask()
            dock_subtask.type        = SubTask.DOCKING
            dock_subtask.description = "Dock retry"
            dock_subtask.dock_goal   = DockGoal(
                use_dock_id=True,
                dock_id=self.active_dock.id,
                navigate_to_staging_pose=True)
            if self.docking_action_client.send_docking_goal(dock_subtask):
                self._transition_status(RobotStatusEnum.DOCKING)
            else:
                self._transition_status(RobotStatusEnum.ERROR)
        else:
            self.get_logger().error(
                f"Docking failed after {self.docking_max_retries} retries")
            self.docking_action_client.reset()
            self._transition_status(RobotStatusEnum.ERROR)
            self.docking_retry_count = 0

    def _handle_undocking(self, robot_status: RobotStatus):
        """Monitor UndockingActionClient — mirrors HuskyOperationsManager."""
        status   = self.undocking_action_client.get_status()
        feedback = self.undocking_action_client.get_feedback()

        if feedback:
            robot_status.task = feedback.task

        if status == RobotStatusEnum.UNDOCKING:
            self._transition_status(RobotStatusEnum.UNDOCKING)

        elif status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info("Undocking complete")
            self._transition_status(RobotStatusEnum.DONE_UNDOCKING)
            self.undocking_action_client.reset()

            if not self.startup_undock_complete:
                self.startup_undock_complete = True
                self.current_phase           = "NAVIGATE_TO_ROW_START"
                self._transition_status(RobotStatusEnum.IDLE)
                self.get_logger().info("Startup undocking done — ready for harvest")

        elif status == RobotStatusEnum.ERROR:
            self._handle_undocking_retry()

    def _handle_undocking_retry(self):
        """Start ReverseDriveClient as fallback — mirrors HuskyOperationsManager."""
        self.undocking_action_client.reset()
        self.get_logger().warning("Undocking failed — starting reverse drive fallback")

        if self.reverse_drive_client.drive_to_staging():
            self.reverse_drive_active = True
            self._transition_status(RobotStatusEnum.UNDOCKING)
        else:
            self.get_logger().error(
                "ReverseDriveClient refused to start — transitioning to ERROR")
            self._transition_status(RobotStatusEnum.ERROR)

    def _handle_reverse_drive(self, robot_status: RobotStatus):
        """Monitor ReverseDriveClient — mirrors HuskyOperationsManager."""
        robot_status.task = "Reverse drive to staging pose"
        status = self.reverse_drive_client.get_status()

        if status == ReverseDriveStatus.REVERSING:
            self._transition_status(RobotStatusEnum.UNDOCKING)

        elif status == ReverseDriveStatus.DONE:
            self.get_logger().info("Reverse drive complete — undocking done")
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()

            if not self.startup_undock_complete:
                self.startup_undock_complete = True
                self.current_phase           = "NAVIGATE_TO_ROW_START"
                self._transition_status(RobotStatusEnum.DONE_UNDOCKING)
                self._transition_status(RobotStatusEnum.IDLE)
                self.get_logger().info("Robot ready for harvest")
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

    def _subtask_undocking(self):
        """Send undocking goal — mirrors HuskyOperationsManager._subtask_undocking()."""
        if self.current_status == RobotStatusEnum.START_UNDOCKING:
            undock_subtask = self.last_undocking_subtask
            if self.undocking_action_client.send_undocking_goal(undock_subtask):
                self._transition_status(RobotStatusEnum.UNDOCKING)
            else:
                self.get_logger().error("Failed to send undocking goal")
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            self.last_undocking_subtask    = None
            self.undocking_after_task_type = None

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    def _handle_error_recovery(self):
        """Cancel active navigation and reset to IDLE — mirrors HuskyOperationsManager."""
        if self.current_status not in [RobotStatusEnum.ERROR, RobotStatusEnum.ABNORMAL]:
            return

        self.get_logger().warning(
            f"Error recovery | status={self.current_status.name} | "
            f"phase={self.current_phase}")

        nav_status = self.navigation.get_navigation_status()
        if nav_status in [NavigationStatus.ACTIVE, NavigationStatus.ACCEPTED]:
            try:
                self.navigation.cancel_goal()
                time.sleep(self.navigation_retry_delay)
            except Exception as e:
                self.get_logger().warning(f"Navigation cancel raised exception: {e}")

        self._cancel_all_motion()
        self._transition_status(RobotStatusEnum.IDLE)

    def _emergency_stop_all(self):
        """Immediately cancel all motion — called on E-stop rising edge."""
        if self.drive_client:
            self.drive_client.cancel()
        if self.navigation:
            try:
                self.navigation.cancel_goal()
            except Exception as e:
                self.get_logger().warning(f"Navigation cancel during e-stop: {e}")
        self._transition_status(RobotStatusEnum.EMERGENCY_STOP)

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

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _capture_last_bush_pose(self):
        """
        Capture arm_0_base_link pose in map frame as last known bush location.
        Called at the moment of load interrupt.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                'map',
                self._drive_config.tf_base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            x     = tf.transform.translation.x
            y     = tf.transform.translation.y
            # Extract yaw from quaternion
            q     = tf.transform.rotation
            siny  = 2.0 * (q.w * q.z + q.x * q.y)
            cosy  = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            theta = math.atan2(siny, cosy)
            self.last_bush_pose = [x, y, theta]
            self.get_logger().info(
                f"Last bush pose captured: x={x:.3f} y={y:.3f} theta={theta:.3f}")
        except TransformException as e:
            self.get_logger().error(
                f"Failed to capture last bush pose: {e} — will use row start as fallback")
            self.last_bush_pose = None

    def _is_robot_at_target(self) -> bool:
        """Check if robot is within 0.25m of the final navigation waypoint."""
        if not (self.current_task and self.pose_status):
            return False

        robot_pos = self.pose_status.pose.pose.position
        subtasks  = self.current_task.sub_tasks
        if not subtasks or not subtasks[0].data:
            return False

        target_wp = subtasks[0].data[-1]
        distance  = self._calculate_distance(
            robot_pos.x, robot_pos.y, target_wp.x, target_wp.y)

        self.get_logger().debug(
            f"_is_robot_at_target | "
            f"robot=({robot_pos.x:.3f}, {robot_pos.y:.3f}) | "
            f"target=({target_wp.x:.3f}, {target_wp.y:.3f}) | "
            f"distance={distance:.3f}m")

        return distance <= 0.25

    def _transition_status(self, new_status: RobotStatusEnum):
        """Safely transition to a new RobotStatusEnum value."""
        if self.current_status != new_status:
            self.previous_status = self.current_status
            self.current_status  = new_status
            self.get_logger().info(
                f"Status: {self.previous_status.name} → {self.current_status.name}")

    # =========================================================================
    # SENSOR STATUS METHODS — mirrors HuskyOperationsManager
    # =========================================================================

    def _set_battery_status(self, robot_status: RobotStatus):
        robot_status.battery_level = self.battery_status.percentage
        if self.battery_status.capacity > 0.0 and self.battery_status.current > 0.0:
            battery_pct    = self._normalize_battery(self.battery_status.percentage)
            time_remaining = (self.battery_status.capacity * (battery_pct / 100.0) /
                              self.battery_status.current)
            robot_status.operation_hours_after_charging = self._format_time_remaining(
                time_remaining)
        else:
            robot_status.operation_hours_after_charging = "00 hours 00 minutes remaining approx..."

    def _set_estop_status(self, robot_status: RobotStatus):
        robot_status.online_flag = (
            self.estop_status.data if self.estop_status.data
            else OnlineFlagEnum.ONLINE.value)

    def _set_location_status(self, robot_status: RobotStatus):
        if self.pose_status and self.pose_status.pose:
            robot_status.topo_map_position    = self.pose_status.pose.pose.position
            robot_status.topo_map_orientation = self.pose_status.pose.pose.orientation

    def _normalize_battery(self, percentage: float) -> float:
        return percentage * 100.0 if percentage <= 1.0 else percentage

    def _format_time_remaining(self, hours: float) -> str:
        seconds          = int(hours * 3600)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes   = divmod(minutes, 60)
        return f"{hours:02} hours {minutes:02} minutes remaining approximately."

    def _calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = LavenderHarvestTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()