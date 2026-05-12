import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState, Imu, NavSatFix
from std_msgs.msg import Bool

from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.action_clients.navigation import NavigationActionClient
from husky_operations_manager.action_clients.reverse_drive_client import ReverseDriveClient
from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.dataclass import DockingConfig, DockInstanceConfig, DockPluginConfig, DriveConfig
from husky_operations_manager.docking_param_fetcher import DockingParamFetcher
from husky_operations_manager.enum import (
    DockingParamFetcherStatus,
    NavigationStatus,
    OnlineFlagEnum,
    ReverseDriveStatus,
    RobotStatusEnum,
)
from status_interfaces.msg import ImageDetectionPose, RobotStatus, SubTask, Task, UndockGoal, WayPoint

DOCKING_CONFIG_POLL_TIMEOUT_SEC = 30.0

class HuskyOperationsManager(Node):
    def __init__(self):
        super().__init__("husky_operations_manager")

        self.namespace = self.get_namespace().rstrip("/")
        self.get_logger().info(f"Node namespace: {self.namespace}")

        self._declare_parameters()
        self._get_parameters()
        self._get_farm_row_parameters()
        self._init_state_variables()
        self._init_sensor_data()
        self._init_subscriptions()

        self.robot_state_pub = self.create_publisher(RobotStatus, f"{self.namespace}/status/robot", 10)

        self._param_fetcher = DockingParamFetcher(self)
        self._param_fetcher.fetch()
        self._config_poll_start_time: float = self.get_clock().now().nanoseconds / 1e9
        self._config_poll_timer = self.create_timer(0.5, self._poll_docking_config)

    def _on_docking_config_ready(self):
        self.docking_config = self._param_fetcher.get_config()

        self.active_dock = self.docking_config.dock_configs[self.docking_config.docks[0]]
        self.active_plugin = self.docking_config.plugin_configs[self.docking_config.dock_plugins[0]]

        self.get_logger().info(
            f"DockingConfig ready | dock='{self.active_dock.instance_name}' | plugin='{self.active_plugin.plugin_name}'"
        )
        self.get_logger().debug(
            f"DockingConfig detail | "
            f"dock_type='{self.active_dock.type}' | "
            f"staging_x_offset={self.active_plugin.staging_x_offset} | "
            f"v_linear_min={self.docking_config.controller_v_linear_min} | "
            f"dock_backwards={self.docking_config.dock_backwards}"
        )

        self.navigation = NavigationActionClient(self)
        self.docking_action_client = DockingActionClient(self)
        self.undocking_action_client = UndockingActionClient(self)
        self.reverse_drive_client = ReverseDriveClient(self, self.docking_config)

        self.init_timer = self.create_timer(self.timing_initial_check_delay, self._initial_position_checker)
        self.timer = self.create_timer(self.timing_timer_period, self.timer_callback)

        self.get_logger().info("Husky Operations Manager ready.")

    def _poll_docking_config(self):
        status = self._param_fetcher.get_status()
        self.get_logger().debug(f"DockingParamFetcher poll | status={status.name}")

        if status == DockingParamFetcherStatus.DONE:
            self._config_poll_timer.cancel()
            self._on_docking_config_ready()
            return

        if status == DockingParamFetcherStatus.ERROR:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self._config_poll_start_time

            if elapsed >= DOCKING_CONFIG_POLL_TIMEOUT_SEC:
                self.get_logger().error(
                    f"DockingParamFetcher failed — "
                    f"docking_server unavailable after {DOCKING_CONFIG_POLL_TIMEOUT_SEC:.0f}s | "
                    f"elapsed={elapsed:.1f}s — shutting down"
                )
                self._config_poll_timer.cancel()
                self.destroy_node()
                rclpy.shutdown()
                return

            self.get_logger().warning(
                f"DockingParamFetcher ERROR — retrying | elapsed={elapsed:.1f}s / {DOCKING_CONFIG_POLL_TIMEOUT_SEC:.0f}s"
            )
            self._param_fetcher.reset()
            self._param_fetcher.fetch()

    # =========================================================================
    # PARAMS AND VARIABLES INITIALIZATION
    # =========================================================================

    def _declare_parameters(self):
        """Declare all ROS2 parameters — sourced from YAML at launch."""
        # General
        self.declare_parameter("num_rows", 1)
        self.declare_parameter("no_detection_timeout", 10.0)
        self.declare_parameter("harvest_duration", 5.0)
        self.declare_parameter("unload_duration", 5.0)
        self.declare_parameter("loading.increment", 20.0)

        # DriveClient
        self.declare_parameter("drive.tf_polling_rate", 10.0)
        self.declare_parameter("drive.tf_target_frame", "map")
        self.declare_parameter("drive.tf_base_frame", "arm_0_base_link")
        self.declare_parameter("drive.tf_detection_frame", "arm_0_detections")
        self.declare_parameter("drive.tolerance", 0.05)
        self.declare_parameter("drive.timeout", 30.0)
        self.declare_parameter("drive.v_linear", 0.2)
        self.declare_parameter("drive.v_angular", 0.5)

        # Navigation
        self.declare_parameter("navigation.max_retries", 3)
        self.declare_parameter("navigation.retry_delay", 5.0)

        # Docking
        self.declare_parameter("docking.max_retries", 2)
        self.declare_parameter("docking.retry_delay", 3.0)
        self.declare_parameter("docking.threshold", 0.25)

        # Battery
        self.declare_parameter("battery.low_threshold", 50.0)
        self.declare_parameter("battery.full_threshold", 99.0)

        # Timing
        self.declare_parameter("timing.timer_period", 1.0)
        self.declare_parameter("timing.initial_position_check_delay", 2.0)

    def _get_parameters(self):
        """Read all declared parameters into instance variables."""
        self.num_rows = int(self.get_parameter("num_rows").value)
        self.no_detection_timeout = float(self.get_parameter("no_detection_timeout").value)
        self.harvest_duration = float(self.get_parameter("harvest_duration").value)
        self.unload_duration = float(self.get_parameter("unload_duration").value)
        self.loading_increment = float(self.get_parameter("loading.increment").value)

        self.navigation_max_retries = int(self.get_parameter("navigation.max_retries").value)
        self.navigation_retry_delay = float(self.get_parameter("navigation.retry_delay").value)
        self.docking_max_retries = int(self.get_parameter("docking.max_retries").value)
        self.docking_retry_delay = float(self.get_parameter("docking.retry_delay").value)
        self.docking_threshold = float(self.get_parameter("docking.threshold").value)
        self.battery_low_threshold = float(self.get_parameter("battery.low_threshold").value)
        self.battery_full_threshold = float(self.get_parameter("battery.full_threshold").value)
        self.timing_timer_period = float(self.get_parameter("timing.timer_period").value)
        self.timing_initial_check_delay = float(self.get_parameter("timing.initial_position_check_delay").value)

        # DriveConfig
        self._drive_config = DriveConfig(
            base_frame            = 'base_link',
            tolerance   = float(self.get_parameter('drive.tolerance').value),
            timeout     = float(self.get_parameter('drive.timeout').value),
            tf_polling_rate       = float(self.get_parameter('drive.tf_polling_rate').value),
            tf_target_frame       = str(self.get_parameter('drive.tf_target_frame').value),
            tf_base_frame         = str(self.get_parameter('drive.tf_base_frame').value),
            tf_detection_frame    = str(self.get_parameter('drive.tf_detection_frame').value),
            v_linear              = float(self.get_parameter('drive.v_linear').value),
            v_angular             = float(self.get_parameter('drive.v_angular').value),            
        )

        self.get_logger().info(
            f"Parameters loaded | rows={self.num_rows} | "
            f"harvest_duration={self.harvest_duration}s unload_duration={self.unload_duration}s | "
            f"load_increment={self.loading_increment}% | "
            f"no_detection_timeout={self.no_detection_timeout}s"
        )

    def _get_farm_row_parameters(self):
        # Row waypoints — flat structure, indexed by row number
        # Supports up to 10 rows; unused rows are ignored at runtime
        for i in range(self.num_rows):
            self.declare_parameter(f'row_{i}_side_a_start', [0.0, 0.0, 0.0])
            self.declare_parameter(f'row_{i}_side_a_end',   [0.0, 5.0, 0.0])
            self.declare_parameter(f'row_{i}_side_b_start', [0.5, 5.0, 3.14])
            self.declare_parameter(f'row_{i}_side_b_end',   [0.5, 0.0, 3.14])

        # Load all row waypoints into a list of dicts
        self.row_waypoints: list[dict] = []
        for i in range(self.num_rows):
            self.row_waypoints.append({
                'side_a_start': float(self.get_parameter(f'row_{i}_side_a_start').value),
                'side_a_end':   float(self.get_parameter(f'row_{i}_side_a_end').value),
                'side_b_start': float(self.get_parameter(f'row_{i}_side_b_start').value),
                'side_b_end':   float(self.get_parameter(f'row_{i}_side_b_end').value),
            })

    def _init_state_variables(self):
        # --- Startup ---
        self.is_initialized = False
        self.is_at_docking_station = False
        self.startup_undock_complete = False

        # --- Robot state ---
        self.current_status = RobotStatusEnum.IDLE
        self.previous_status = RobotStatusEnum.IDLE

        # --- Task management ---
        self.current_task: Task | None = None
        self.current_sub_task: SubTask | None = None
        self.current_sub_task_index = 0
        self.last_handled_task_id: int | None = None
        self.last_handled_task_type: int | None = None
        self.last_handled_subtask_type: int | None = None
        self.current_node_id = 0
        self.current_load_status = 0.0

        # --- Retry counters ---
        self.navigation_retry_count = 0
        self.docking_retry_count = 0

        # --- Job duration (used for unloading/charging timers) ---
        self.job_start_time = None
        self.job_duration = 0.0

        # --- Undocking ---
        self.last_undocking_subtask: SubTask | None = None
        self.undocking_after_task_type: int | None = None
        self.reverse_drive_active: bool = False

        self.docking_config: DockingConfig | None = None
        self.active_dock: DockInstanceConfig | None = None
        self.active_plugin: DockPluginConfig | None = None
        self.reverse_drive_client: ReverseDriveClient | None = None

        self.get_logger().debug("State variables initialised")

    def _init_sensor_data(self):
        self.battery_status = BatteryState()
        self.task = Task()
        self.gps_status = NavSatFix()
        self.pose_status = PoseWithCovarianceStamped()
        self.imu_status = Imu()
        self.estop_status = Bool()
        self.image_detection_pose = ImageDetectionPose()

    def _init_subscriptions(self):
        # Battery state — used for low-battery detection and charge completion checks
        self.battery_sub = self.create_subscription(
            BatteryState,
            f"{self.namespace}/platform/bms/state",
            lambda msg: setattr(self, "battery_status", msg),
            qos_profile_sensor_data,
        )

        # Ground-truth pose — used for initial dock distance check and navigation target check
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, f"{self.namespace}/ground_truth/pose", self._pose_callback, 10
        )

        # IMU — stored but not currently used in state machine logic
        self.imu_sub = self.create_subscription(
            Imu,
            f"{self.namespace}/ssensors/gps_0/imu",
            lambda msg: setattr(self, "imu_status", msg),
            qos_profile_sensor_data,
        )

        # Emergency stop — mapped to online_flag in published RobotStatus
        self.estop_sub = self.create_subscription(
            Bool,
            f"{self.namespace}/platform/emergency_stop",
            lambda msg: setattr(self, "estop_status", msg),
            qos_profile_sensor_data,
        )

        # ImageDetectionPose — owns subscription, calls drive_client.stop() on valid detection
        detection_topic = f"{self.namespace}/manipulators/arm_0_detection/image_annotated/detection_pose"
        self.detection_sub = self.create_subscription(
            ImageDetectionPose,
            detection_topic,
            self._detection_callback,
            10)

        self.get_logger().info(f"Subscribed to detection topic: {detection_topic}")
        self.get_logger().debug("Subscriptions initialised")

    # =========================================================================
    # SUBSCRIPTION CALLBACK
    # =========================================================================

    def _pose_callback(self, msg):
        """Store latest ground-truth pose for dock distance checks and target validation."""
        self.pose_status = msg
    
    # =========================================================================
    # MAIN CALLBACK METHODS
    # =========================================================================

    def _task_callback(self, msg: Task):
        subtasks_summary = [(st.sub_task_id, st.type, st.description) for st in msg.sub_tasks]
        self.get_logger().debug(
            f"Received task | ID: {msg.task_id} | Type: {msg.task_type} | "
            f"Target Node: {msg.target_node_id} | SubTasks: {subtasks_summary}"
        )

        is_new_task = (
            msg.task_id != self.last_handled_task_id
            or self.current_task
            and msg.task_type != self.current_task.task_type
        )

        is_new_subtask = False
        if isinstance(msg.sub_tasks, list) and len(msg.sub_tasks) > 0:
            first_subtask = msg.sub_tasks[0]
            if isinstance(first_subtask, SubTask):
                is_new_subtask = first_subtask.type != self.last_handled_subtask_type

        self.get_logger().debug(
            f"Task callback | id={msg.task_id} type={msg.task_type} | "
            f"is_new_task={is_new_task} is_new_subtask={is_new_subtask} | "
            f"last_task_id={self.last_handled_task_id} "
            f"last_subtask_type={self.last_handled_subtask_type}"
        )

        if is_new_task:
            self.get_logger().info(f"New Task: {msg.description} (ID: {msg.task_id})")
            self.current_sub_task_index = 0
            self.last_handled_subtask_type = None
            # self.current_load_status = msg.crop_load
        elif is_new_subtask:
            self.get_logger().info(f"New Subtask for task ID: {msg.task_id}")
            self.current_sub_task_index = 0
            # If the node completed the previous subtask and is waiting at JOB_DONE,
            # receiving a new subtask means the job publisher wants us to continue
            if self.current_status == RobotStatusEnum.JOB_DONE:
                self._transition_status(RobotStatusEnum.IDLE)

        # On Node start verify the load with server
        if self.last_handled_task_id is None:
            self.current_load_status = self.task.crop_load

        self.task = msg

    def timer_callback(self):
        robot_status = RobotStatus()
        robot_status.header.stamp = self.get_clock().now().to_msg()
        robot_status.robot_namespace = self.namespace.replace(r"/", "")

        # Update sensor fields before routing so all paths see fresh data
        self._set_battery_status(robot_status)
        self._set_estop_status(robot_status)
        self._set_location_status(robot_status)

        self.get_logger().debug(
            f"Timer tick | status={self.current_status.name} | "
            f"startup_undock_complete={self.startup_undock_complete} | "
            f"reverse_drive_active={self.reverse_drive_active} | "
            f"battery={self._normalize_battery(self.battery_status.percentage):.1f}%"
        )

        # ERROR and ABNORMAL are handled before startup check so a fault during
        # startup undocking is also covered by _handle_error_recovery
        if self.current_status in [RobotStatusEnum.ERROR, RobotStatusEnum.ABNORMAL]:
            self._handle_error_recovery()
        elif not self.startup_undock_complete:
            self._handle_startup_undocking(robot_status)
        else:
            self._handle_task_execution(robot_status)

        # Write final state into the outgoing message
        robot_status.status = self.current_status.value
        robot_status.current_node_id = self.current_node_id
        robot_status.load_status = self.current_load_status
        self.robot_state_pub.publish(robot_status)
    
    # =========================================================================
    # STARTUP AND INITIALIZATION
    # =========================================================================

    def _initial_position_checker(self):
        """
        Fires once after timing_initial_position_check_delay.

        Waits until pose data is available, then cancels itself and runs
        check_initial_position. If pose is still None the timer continues
        firing until it arrives.
        """
        if self.pose_status is None:
            self.get_logger().warning("Waiting for pose data...")
            return
        self.init_timer.cancel()
        self.check_initial_position()

    def check_initial_position(self):
        """
        Determine whether the robot starts at a docking station.

        With a single dock in DockingConfig, active_dock is used directly.
        With multiple docks, the nearest one by Euclidean distance is used.

        Sets startup_undock_complete=True if the robot is NOT at a dock, so the
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
        current_pos = self.pose_status.pose.pose.position

        distance = self._calculate_distance(
            current_pos.x, current_pos.y, charger_dock_pose.dock_x, charger_dock_pose.dock_y
        )

        self.get_logger().info(
            f"Initial Robot position: ({current_pos.x:.3f}, {current_pos.y:.3f}), "
            f"Dock Name: '{charger_dock_pose.instance_name}' | "
            f"Distance to dock: {distance:.3f}m"
        )

        self.get_logger().debug(
            f"Dock position: ({charger_dock_pose.dock_x:.3f}, {charger_dock_pose.dock_y:.3f}) | "
            f"docking_threshold={self.docking_threshold}m | "
            f"num_docks={len(self.docking_config.dock_configs)}"
        )

        # TODO: validate the if condition so that undocking can occur when the robot
        # pose shows robot is anywhere between dock pose and staging pose along x-axis
        # of robot.
        if distance <= self.docking_threshold:
            self.is_at_docking_station = True
            self.get_logger().info("Robot at dock - will undock before tasks")
        else:
            self.is_at_docking_station = False
            self.startup_undock_complete = True
            self.get_logger().info("Robot ready for tasks")

        self.is_initialized = True

    def _handle_startup_undocking(self, robot_status: RobotStatus):
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
            robot_status.task = "Startup: Undocking"
            dock_type = self.active_dock.type
            staging_x_offset = self.active_plugin.staging_x_offset
            v_linear = self.docking_config.controller_v_linear_max
            max_undocking_time = (abs(staging_x_offset) / max(v_linear, 0.01)) * 1.25

            self.get_logger().debug(
                f"Startup UndockGoal | dock_type='{dock_type}' | "
                f"staging_x_offset={staging_x_offset} | "
                f"v_linear_min={v_linear} | "
                f"max_undocking_time={max_undocking_time:.1f}s"
            )

            startup_subtask = SubTask()
            startup_subtask.type = SubTask.UNDOCKING
            startup_subtask.description = "Startup Undocking"
            startup_subtask.undock_goal = UndockGoal(dock_type=dock_type, max_undocking_time=max_undocking_time)

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
        if not self.task or not self.task.description or not self.task.job_schedule:
            if self.current_status != RobotStatusEnum.IDLE:
                self._transition_status(RobotStatusEnum.IDLE)
            return

        self.current_task = self.task

        if self._check_and_handle_low_battery():
            return

        self._process_action_clients(robot_status)
        self._update_current_subtask()

        robot_status.crop_type = self.current_task.crop_type
        robot_status.target_node_id = self.current_task.target_node_id
        robot_status.task = (
            self.current_sub_task.description if self.current_sub_task else self.current_task.description
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

    def _check_and_handle_low_battery(self) -> bool:
        if not self.task or not self.current_task:
            return False

        battery_pct = self._normalize_battery(self.battery_status.percentage)

        self.get_logger().debug(
            f"Battery check | pct={battery_pct:.1f}% | "
            f"low_threshold={self.battery_low_threshold}% | "
            f"task_type={self.task.task_type}"
        )

        if self.task.task_type != Task.CHARGING_TASK and battery_pct <= self.battery_low_threshold:
            self.get_logger().warning(
                f"Battery low: {battery_pct:.1f}% — threshold={self.battery_low_threshold}% | "
                f"task_type={self.task.task_type} | status={self.current_status.name}"
            )

            if self.current_status in [RobotStatusEnum.START_MOVING, RobotStatusEnum.MOVING]:
                self.get_logger().warning("Cancelling navigation due to low battery")
                self.navigation.cancel_goal()
            self._transition_status(RobotStatusEnum.ERROR)
            return True

        if self.current_status == RobotStatusEnum.ERROR and self.current_task.task_type == Task.CHARGING_TASK:
            self.get_logger().info(f"Recovering from low battery — CHARGING_TASK received | battery={battery_pct:.1f}%")
            self._transition_status(RobotStatusEnum.IDLE)

        return False

    def _process_action_clients(self, robot_status: RobotStatus):
        nav_status = self.navigation.get_navigation_status()
        dock_status = self.docking_action_client.get_status()
        undock_status = self.undocking_action_client.get_status()

        self.get_logger().debug(
            f"Action clients | nav={nav_status.name} | "
            f"dock={dock_status.name} | "
            f"undock={undock_status.name} | "
            f"reverse_drive_active={self.reverse_drive_active} | "
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
        if not self.current_task:
            return

        if (
            self.current_task.task_id != self.last_handled_task_id
            or self.current_task.task_type != self.last_handled_task_type
        ):
            self.get_logger().info(
                f"Starting Task: {self.current_task.description} | "
                f"ID: {self.current_task.task_id} | "
                f"Current Node: {self.current_node_id} | "
                f"Target Node: {self.current_task.target_node_id}"
            )
            self.get_logger().debug(
                f"Task start | task_type={self.current_task.task_type} | "
                f"last_task_id={self.last_handled_task_id} | "
                f"last_task_type={self.last_handled_task_type} | "
                f"num_subtasks={len(self.current_task.sub_tasks) if isinstance(self.current_task.sub_tasks, list) else 0}"
            )

            self.last_handled_task_id = self.current_task.task_id
            self.last_handled_task_type = self.current_task.task_type
            self.last_handled_subtask_type = None
            self.last_undocking_subtask = None
            self.undocking_after_task_type = None
            self._transition_status(RobotStatusEnum.JOB_START)

        elif self.current_status == RobotStatusEnum.JOB_DONE:
            self.get_logger().debug(
                f"JOB_DONE — same task still publishing | task_id={self.current_task.task_id} — transitioning to IDLE"
            )
            self._transition_status(RobotStatusEnum.IDLE)
            self.current_task = None

    def _execute_current_subtask(self):
        if not isinstance(self.current_sub_task, SubTask):
            self.get_logger().debug(
                f"_execute_current_subtask called but no valid subtask | status={self.current_status.name}"
            )
            return
        
        if self.current_sub_task.type != self.last_handled_subtask_type:
            self.get_logger().info(f"Executing: {self.current_sub_task.description}")
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
            self.get_logger().warning(f"Unknown subtask type: {self.current_sub_task.type} — no handler registered")

    # =========================================================================
    # ACTION CLIENT HANDLERS
    # =========================================================================

    def _handle_navigation(self, robot_status: RobotStatus):
        nav_status = self.navigation.get_navigation_status()
        wpf_status = self.navigation.get_current_status()

        # Prefer live waypoint-following status for the published task field
        if wpf_status:
            robot_status.task = wpf_status.task
            robot_status.current_node_id = wpf_status.current_node_id
            robot_status.target_node_id = wpf_status.target_node_id
        else:
            robot_status.task = self.current_task.description if self.current_task else ""
            robot_status.current_node_id = self.current_node_id
            robot_status.target_node_id = self.current_task.target_node_id if self.current_task else -1

        self.get_logger().debug(
            f"Navigation handler | nav_status={nav_status.name} | "
            f"current_node={self.current_node_id} | "
            f"target_node={robot_status.target_node_id} | "
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
            self.get_logger().debug(
                f"Navigation aborted/error | nav_status={nav_status.name} | retry_count={self.navigation_retry_count}"
            )
            self._handle_navigation_retry()

        elif nav_status == NavigationStatus.CANCELED:
            self.get_logger().info("Navigation canceled")
            self.navigation.reset()
            self.navigation_retry_count = 0
            self._transition_status(RobotStatusEnum.IDLE)

    def _handle_navigation_retry(self):
        self.get_logger().warning(
            f"Navigation failed | retry {self.navigation_retry_count + 1}/{self.navigation_max_retries} | "
            f"current_node={self.current_node_id}"
        )

        # If robot is physically at the target despite nav failure, treat as success
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
                f"Navigation failed after {self.navigation_max_retries} retries | "
                f"current_node={self.current_node_id} | "
                f"target_node={self.current_task.target_node_id if self.current_task else 'None'}"
            )
            self.navigation.reset()
            self._transition_status(RobotStatusEnum.ERROR)
            self.navigation_retry_count = 0

    def _retry_navigation(self):
        if not self.current_task:
            self.get_logger().error("Navigation retry failed — no current task")
            return

        nav_status = self.navigation.get_navigation_status()
        self.get_logger().debug(
            f"Retry navigation | nav_status={nav_status.name} | "
            f"attempt={self.navigation_retry_count}/{self.navigation_max_retries}"
        )

        # Do not send a new goal if the previous one has not terminated yet
        if nav_status in [NavigationStatus.ACTIVE, NavigationStatus.SENDING]:
            self.get_logger().warning(f"Navigation still active ({nav_status.name}) — skipping retry until complete")
            return

        self.navigation.reset()
        time.sleep(self.navigation_retry_delay)

        if self.navigation.send_goal(self.current_task):
            self._transition_status(RobotStatusEnum.MOVING)
        else:
            self.get_logger().error(
                f"Navigation retry send_goal failed | "
                f"attempt={self.navigation_retry_count}/{self.navigation_max_retries}"
            )
            self._transition_status(RobotStatusEnum.ERROR)

    def _handle_docking(self, robot_status: RobotStatus):
        status = self.docking_action_client.get_status()
        feedback = self.docking_action_client.get_feedback()

        if feedback:
            robot_status.task = feedback.task
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
            self.get_logger().debug(
                f"Docking error — triggering retry | retry_count={self.docking_retry_count}/{self.docking_max_retries}"
            )
            self._handle_docking_retry()

    def _handle_docking_retry(self):
        dock_id = (
            self.current_sub_task.dock_goal.dock_id
            if self.current_sub_task and self.current_sub_task.dock_goal
            else "unknown"
        )
        self.get_logger().error(
            f"Docking failed | retry {self.docking_retry_count + 1}/{self.docking_max_retries} | dock_id='{dock_id}'"
        )

        if self.docking_retry_count < self.docking_max_retries:
            self.docking_retry_count += 1
            self.get_logger().info(
                f"Retrying docking in {self.docking_retry_delay:.1f}s | "
                f"attempt {self.docking_retry_count}/{self.docking_max_retries}"
            )
            time.sleep(self.docking_retry_delay)
            self._retry_docking()
        else:
            self.get_logger().error(f"Docking failed after {self.docking_max_retries} retries — transitioning to ERROR")
            self.docking_action_client.reset()
            self._transition_status(RobotStatusEnum.ERROR)
            self.docking_retry_count = 0

    def _retry_docking(self):
        if not self.current_sub_task:
            self.get_logger().error("Docking retry failed — no current subtask")
            return
        self.get_logger().debug(
            f"Retry docking | attempt={self.docking_retry_count}/{self.docking_max_retries} | "
            f"subtask='{self.current_sub_task.description}'"
        )
        if self.docking_action_client.send_docking_goal(self.current_sub_task):
            self._transition_status(RobotStatusEnum.DOCKING)
        else:
            self.get_logger().error(
                f"Docking retry send_goal failed | attempt={self.docking_retry_count}/{self.docking_max_retries}"
            )
            self._transition_status(RobotStatusEnum.ERROR)

    def _handle_undocking(self, robot_status: RobotStatus):
        status = self.undocking_action_client.get_status()
        feedback = self.undocking_action_client.get_feedback()

        if feedback:
            robot_status.task = feedback.task
            robot_status.current_node_id = self.current_node_id
        else:
            robot_status.task = "Undocking in progress"

        self.get_logger().debug(
            f"Undocking handler | status={status.name} | startup_undock_complete={self.startup_undock_complete}"
        )

        if status == RobotStatusEnum.UNDOCKING:
            self._transition_status(RobotStatusEnum.UNDOCKING)

        elif status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info("Undocking complete")
            self._transition_status(RobotStatusEnum.DONE_UNDOCKING)
            self.undocking_action_client.reset()

            if not self.startup_undock_complete:
                # Startup context: robot has left the dock, ready for tasks
                self.get_logger().debug("Undocking DONE — startup context: setting startup_undock_complete=True → IDLE")
                self.startup_undock_complete = True
                # Clear task IDs so the first real task is treated as new
                self.last_handled_task_id = None
                self.last_handled_task_type = None
                self._transition_status(RobotStatusEnum.IDLE)
                self.get_logger().info("Robot ready for tasks")
            else:
                # Task context: stay at DONE_UNDOCKING — _subtask_undocking
                # will pick this up on the next tick and transition to JOB_DONE
                self.get_logger().debug(
                    "Undocking DONE — task context: staying at DONE_UNDOCKING for _subtask_undocking"
                )

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().debug(
                f"Undocking error — triggering reverse drive fallback | "
                f"startup_undock_complete={self.startup_undock_complete}"
            )
            self._handle_undocking_retry()

    def _handle_undocking_retry(self):
        self.undocking_action_client.reset()
        self.get_logger().warning(
            f"Undocking failed — starting reverse drive to staging pose | "
            f"dock='{self.active_dock.instance_name if self.active_dock else 'unknown'}'"
        )

        if self.reverse_drive_client.drive_to_staging():
            # drive_to_staging returns True when REVERSING begins successfully
            self.reverse_drive_active = True
            self.get_logger().debug(f"ReverseDriveClient started | reverse_drive_active={self.reverse_drive_active}")
            self._transition_status(RobotStatusEnum.UNDOCKING)
        else:
            # Returns False when dock_backwards=True or already reversing
            self.get_logger().error(
                f"ReverseDriveClient refused to start — "
                f"dock_backwards={self.docking_config.dock_backwards if self.docking_config else 'unknown'}"
            )
            self._transition_status(RobotStatusEnum.ERROR)

    def _handle_reverse_drive(self, robot_status: RobotStatus):
        robot_status.task = "Reverse drive to staging pose"
        status = self.reverse_drive_client.get_status()

        self.get_logger().debug(
            f"Reverse drive handler | status={status.name} | startup_undock_complete={self.startup_undock_complete}"
        )

        if status == ReverseDriveStatus.REVERSING:
            self._transition_status(RobotStatusEnum.UNDOCKING)

        elif status == ReverseDriveStatus.DONE:
            self.get_logger().info("Reverse drive complete — undocking done")
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()

            if not self.startup_undock_complete:
                # Startup context: robot reached staging pose, ready for tasks
                self.get_logger().debug(
                    "Reverse drive DONE — startup context: setting startup_undock_complete=True → IDLE"
                )
                self.startup_undock_complete = True
                self.last_handled_task_id = None
                self.last_handled_task_type = None
                self._transition_status(RobotStatusEnum.DONE_UNDOCKING)
                self._transition_status(RobotStatusEnum.IDLE)

                self.get_logger().info("Robot ready for tasks")
            else:
                # Task context: stay at DONE_UNDOCKING for _subtask_undocking
                self.get_logger().debug(
                    "Reverse drive DONE — task context: staying at DONE_UNDOCKING for _subtask_undocking"
                )
                self._transition_status(RobotStatusEnum.DONE_UNDOCKING)

        elif status == ReverseDriveStatus.ERROR:
            self.get_logger().error(
                f"Reverse drive failed — transitioning to ERROR | "
                f"startup_undock_complete={self.startup_undock_complete}"
            )
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()
            self._transition_status(RobotStatusEnum.ERROR)

        elif status == ReverseDriveStatus.CANCELED:
            self.get_logger().warning(
                f"Reverse drive canceled — transitioning to IDLE | "
                f"startup_undock_complete={self.startup_undock_complete}"
            )
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()
            self._transition_status(RobotStatusEnum.IDLE)

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    def _handle_error_recovery(self):
        if self.current_status not in [RobotStatusEnum.ERROR, RobotStatusEnum.ABNORMAL]:
            return

        self.get_logger().warning(
            f"Error recovery | status={self.current_status.name} | "
            f"task_id={self.last_handled_task_id} | "
            f"subtask_type={self.last_handled_subtask_type}"
        )

        nav_status = self.navigation.get_navigation_status()
        self.get_logger().debug(f"Error recovery | nav_status={nav_status.name}")

        if nav_status in [NavigationStatus.ACTIVE, NavigationStatus.ACCEPTED]:
            self.get_logger().info("Cancelling active navigation during error recovery")
            try:
                self.navigation.cancel_goal()
                # Brief wait to allow Nav2 to process the cancellation before resetting
                time.sleep(self.navigation_retry_delay)
            except Exception as e:
                self.get_logger().warning(f"Navigation cancel raised exception: {e}")

        # Reset subtask state so the next task starts clean
        self.current_sub_task = None
        self.current_sub_task_index = 0
        self.get_logger().debug("Error recovery — reset subtask state, transitioning to IDLE")
        self._transition_status(RobotStatusEnum.IDLE)

    # =========================================================================
    # SUBTASK HANDLERS
    # =========================================================================

    def _subtask_moving(self):
        if not self.current_task:
            return

        self.get_logger().debug(
            f"_subtask_moving | status={self.current_status.name} | "
            f"current_node={self.current_node_id} | "
            f"target_node={self.current_task.target_node_id}"
        )

        if self.current_status == RobotStatusEnum.JOB_START:
            self._transition_status(RobotStatusEnum.START_MOVING)

        elif self.current_status == RobotStatusEnum.START_MOVING:
            # Guard: do not send a second goal if the previous one is still being accepted
            if self.navigation.is_navigation_active():
                self.get_logger().warning(
                    f"Navigation send skipped — already active | "
                    f"nav_status={self.navigation.get_navigation_status().name}"
                )
                return
            self.get_logger().info(f"Starting navigation: {self.current_node_id} → {self.current_task.target_node_id}")
            if self.navigation.send_goal(self.current_task):
                self._transition_status(RobotStatusEnum.MOVING)
            else:
                self.get_logger().error(
                    f"Failed to send navigation goal | "
                    f"task_id={self.current_task.task_id} | "
                    f"target_node={self.current_task.target_node_id}"
                )
                self._transition_status(RobotStatusEnum.ERROR)

    def _subtask_docking(self):
        dock_id = (
            self.current_sub_task.dock_goal.dock_id
            if self.current_sub_task and self.current_sub_task.dock_goal
            else "None"
        )
        self.get_logger().debug(f"_subtask_docking | status={self.current_status.name} | dock_id='{dock_id}'")

        if self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            self._transition_status(RobotStatusEnum.START_DOCKING)

        elif self.current_status == RobotStatusEnum.START_DOCKING:
            self.get_logger().info("Started docking")
            # Brief pause to allow the robot to fully stop before docking starts
            time.sleep(1.0)
            if self.docking_action_client.send_docking_goal(self.current_sub_task):
                self._transition_status(RobotStatusEnum.DOCKING)
            else:
                self.get_logger().error(
                    f"Failed to send docking goal | "
                    f"subtask='{self.current_sub_task.description if self.current_sub_task else 'None'}'"
                )
                self._transition_status(RobotStatusEnum.ERROR)

    def _subtask_undocking(self):
        self.get_logger().debug(
            f"_subtask_undocking | status={self.current_status.name} | "
            f"last_undocking_subtask={'set' if self.last_undocking_subtask else 'None'} | "
            f"undocking_after_task_type={self.undocking_after_task_type} | "
        )

        if self.current_status == RobotStatusEnum.START_UNDOCKING:
            undock_subtask = self.current_sub_task if self.current_sub_task else self.last_undocking_subtask
            dock_type = (
                undock_subtask.undock_goal.dock_type if undock_subtask and undock_subtask.undock_goal else "None"
            )

            self.get_logger().debug(
                f"Undocking subtask | "
                f"source={'current_sub_task' if self.current_sub_task else 'last_undocking_subtask'} | "
                f"dock_type='{dock_type}'"
            )

            if self.undocking_action_client.send_undocking_goal(undock_subtask):
                self._transition_status(RobotStatusEnum.UNDOCKING)
            else:
                self.get_logger().error(
                    f"Failed to send undocking goal | undocking_after_task_type={self.undocking_after_task_type}"
                )
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            # Undocking succeeded — clean up stored state and mark task complete
            self.get_logger().info("Undocking done")
            self.get_logger().debug(
                f"Undocking done — clearing state | undocking_after_task_type={self.undocking_after_task_type}"
            )
            self.last_undocking_subtask = None
            self.undocking_after_task_type = None
            self._transition_status(RobotStatusEnum.JOB_DONE)

    def _subtask_harvesting(self):
        self.get_logger().debug(f"_subtask_harvesting | status={self.current_status.name} | ")

        if self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            self._transition_status(RobotStatusEnum.START_HARVESTING)

        elif self.current_status == RobotStatusEnum.START_HARVESTING:
            self._transition_status(RobotStatusEnum.HARVESTING)
            self.get_logger().info("Harvesting started")

            # NOTE: Start simulated harvest timer (replace with hardware signal)
            self.job_start_time = self.get_clock().now()
            self.job_duration = 5.0  # seconds

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
                f"Load update | {self.current_load_status:.1f}% → {new_load:.1f}% (+{self.loading_increment:.1f}%)"
            )
            self.current_load_status = new_load
            self.get_logger().info(f"Load status: {self.current_load_status:.1f}%")
            self._transition_status(RobotStatusEnum.JOB_DONE)

    def _subtask_charging(self):
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
            # Throttled to avoid flooding logs at 1Hz during the charging wait
            self.get_logger().info(f"Battery charging: {battery_pct:.1f}%", throttle_duration_sec=10.0)
            if battery_pct >= self.battery_full_threshold:
                self.get_logger().info(f"Battery charged: {battery_pct:.1f}%")
                self._transition_status(RobotStatusEnum.DONE_CHARGING)

        elif self.current_status == RobotStatusEnum.DONE_CHARGING:
            # Store the charging subtask so _subtask_undocking can retrieve its UndockGoal.
            # The STOW gate inside _subtask_undocking will fire if arm is not already stowed.
            self.get_logger().debug("DONE_CHARGING — storing last_undocking_subtask and triggering undocking")
            self.last_undocking_subtask = self.current_sub_task
            self.undocking_after_task_type = Task.CHARGING_TASK
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            # Call immediately — next tick would call this handler again on CHARGING subtask
            # but current_status is now START_UNDOCKING so _subtask_undocking handles it
            self._subtask_undocking()

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            # _handle_undocking set status to DONE_UNDOCKING (task context).
            # Delegate to _subtask_undocking to transition to JOB_DONE.
            self.get_logger().debug("DONE_UNDOCKING in charging context — delegating to _subtask_undocking")
            self._subtask_undocking()

    def _subtask_unloading(self):
        self.get_logger().debug(
            f"_subtask_unloading | status={self.current_status.name} | current_load={self.current_load_status:.1f}%"
        )

        if self.current_status == RobotStatusEnum.DONE_DOCKING:
            self._transition_status(RobotStatusEnum.START_UNLOADING)

        elif self.current_status == RobotStatusEnum.START_UNLOADING:
            self._transition_status(RobotStatusEnum.UNLOADING)
            self.get_logger().info("Unloading started")
            # Start simulated unload timer (replace with hardware signal)
            self.job_start_time = self.get_clock().now()
            self.job_duration = 4.0  # seconds

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
            # Reset load to 0 and store undocking subtask before triggering undocking.
            # The STOW gate inside _subtask_undocking will fire if arm is not already stowed.
            self.get_logger().debug("DONE_UNLOADING — storing last_undocking_subtask and triggering undocking")
            self.last_undocking_subtask = self.current_sub_task
            self.undocking_after_task_type = Task.UNLOADING_TASK
            self.current_load_status = 0.0
            self.get_logger().info("Unloading done, starting undocking")
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            self._subtask_undocking()

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            # _handle_undocking set status to DONE_UNDOCKING (task context).
            # Delegate to _subtask_undocking to transition to JOB_DONE.
            self.get_logger().debug("DONE_UNDOCKING in unloading context — delegating to _subtask_undocking")
            self._subtask_undocking()

    # =========================================================================
    # SENSOR STATUS METHODS
    # =========================================================================

    def _set_battery_status(self, robot_status: RobotStatus):
        """
        Populate battery fields in the outgoing RobotStatus.

        Estimates remaining operation time from capacity, current draw, and
        battery percentage. Falls back to a zero-string if BMS data is unavailable.
        """
        robot_status.battery_level = self.battery_status.percentage
        if self.battery_status.capacity > 0.0 and self.battery_status.current > 0.0:
            battery_pct = self._normalize_battery(self.battery_status.percentage)
            # Remaining time (hours) = remaining capacity (Ah) / current draw (A)
            time_remaining = self.battery_status.capacity * (battery_pct / 100.0) / self.battery_status.current
            robot_status.operation_hours_after_charging = self._format_time_remaining(time_remaining)
        else:
            robot_status.operation_hours_after_charging = "00 hours 00 minutes remaining approx..."

    def _set_estop_status(self, robot_status: RobotStatus):
        """Map emergency stop signal to online_flag in the outgoing RobotStatus."""
        robot_status.online_flag = self.estop_status.data if self.estop_status.data else OnlineFlagEnum.ONLINE.value

    def _set_location_status(self, robot_status: RobotStatus):
        """Populate position and orientation fields from the latest ground-truth pose."""
        if self.pose_status and self.pose_status.pose:
            robot_status.topo_map_position = self.pose_status.pose.pose.position
            robot_status.topo_map_orientation = self.pose_status.pose.pose.orientation

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _is_robot_at_target(self):
        """
        Check if the robot is within 0.25m of the final waypoint.

        Used by _handle_navigation_retry to treat a nav abort as a success when
        the robot is physically close enough to the destination — Nav2 sometimes
        aborts a goal even when the robot has effectively arrived.

        Returns False if no subtask or pose data is available.
        """
        if not (self.current_sub_task and self.pose_status):
            self.get_logger().debug("_is_robot_at_target: no subtask or pose — returning False")
            return False

        robot_position = self.pose_status.pose.pose.position
        # Normalise waypoints in case the subtask data contains raw dicts
        waypoints_list = [WayPoint(**wp) if isinstance(wp, dict) else wp for wp in self.current_sub_task.data]
        target_wp = waypoints_list[-1]  # Final waypoint is the destination
        distance = self._calculate_distance(robot_position.x, robot_position.y, target_wp.x, target_wp.y)

        self.get_logger().debug(
            f"_is_robot_at_target | "
            f"robot=({robot_position.x:.3f}, {robot_position.y:.3f}) | "
            f"target=({target_wp.x:.3f}, {target_wp.y:.3f}) | "
            f"distance={distance:.3f}m threshold=0.25m"
        )
        return distance <= 0.25

    def _transition_status(self, new_status: RobotStatusEnum):
        """
        Safely transition to a new RobotStatusEnum value.

        No-ops if new_status equals current_status to avoid redundant log lines.
        Stores the previous status for debugging.
        """
        if self.current_status != new_status:
            self.previous_status = self.current_status
            self.current_status = new_status
            self.get_logger().info(f"Status: {self.previous_status.name} → {self.current_status.name}")

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
        return f"{hours:02} hours {minutes:02} minutes remaining approximately."

    def _calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Return the Euclidean distance between two 2D points."""
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def main():
    rclpy.init()
    node = HuskyOperationsManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
