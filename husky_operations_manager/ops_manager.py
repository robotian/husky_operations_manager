import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import BatteryState, NavSatFix, Imu
from std_msgs.msg import Bool
from status_interfaces.msg import RobotStatus, Task, SubTask, UndockGoal, WayPoint

from husky_operations_manager.enum import OnlineFlagEnum, RobotStatusEnum, NavigationStatus, ReverseDriveStatus
from husky_operations_manager.dataclass import DockingConfig, DockInstanceConfig, DockPluginConfig
from husky_operations_manager.docking_param_fetcher import DockingParamFetcher
from husky_operations_manager.action_clients.reverse_drive_client import ReverseDriveClient
from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.action_clients.navigation import NavigationActionClient
from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.action_clients.manipulator import ManipulatorTaskActionClient, ArmCommand


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
        self.get_logger().info(f"Node namespace: {self.namespace}")

        self._declare_parameter()
        self._get_paramters()
        self._init_state_variables()
        self._init_sensor_data()
        self._init_subscriptions()

        self.robot_state_pub = self.create_publisher(
            RobotStatus, f'{self.namespace}/status/robot', 10)

        # These are set to None here and populated once DockingParamFetcher completes.
        # All action clients depend on DockingConfig so they cannot be created until
        # _on_docking_config_ready fires.
        self.docking_config: DockingConfig | None = None
        self.active_dock:   DockInstanceConfig | None = None
        self.active_plugin: DockPluginConfig | None = None
        self.reverse_drive_client: ReverseDriveClient | None = None

        # Fetch all params from docking_server asynchronously.
        # _poll_docking_config checks readiness every 0.5s and fires
        # _on_docking_config_ready once the config is built.
        self._param_fetcher = DockingParamFetcher(self)
        self._param_fetcher.fetch()
        self._config_poll_timer = self.create_timer(0.5, self._poll_docking_config)

    def _on_docking_config_ready(self):
        """
        Called once DockingParamFetcher reports DONE.

        Resolves active_dock and active_plugin from index 0 of the respective lists.
        Initialises all action clients (they require DockingConfig at construction)
        and starts the main 1Hz timer and the initial position check timer.

        TODO: Update active_dock and active_plugin values based on the assigned dock
        to each robot in multi-robot setup
        """
        self.docking_config = self._param_fetcher.get_config()

        # Index 0 is used as the single global dock and plugin throughout the node.
        # Multi-dock support would require changing these to per-task lookups.
        self.active_dock   = self.docking_config.dock_configs[self.docking_config.docks[0]]
        self.active_plugin = self.docking_config.plugin_configs[self.docking_config.dock_plugins[0]]

        self.get_logger().info(
            f"DockingConfig ready | "
            f"dock='{self.active_dock.instance_name}' | "
            f"plugin='{self.active_plugin.plugin_name}'"
        )
        self.get_logger().debug(
            f"DockingConfig detail | "
            f"dock_type='{self.active_dock.type}' | "
            f"staging_x_offset={self.active_plugin.staging_x_offset} | "
            f"v_linear_min={self.docking_config.controller_v_linear_min} | "
            f"dock_backwards={self.docking_config.dock_backwards}"
        )

        # Initialise action clients now that DockingConfig is available
        self.navigation              = NavigationActionClient(self)
        self.docking_action_client   = DockingActionClient(self)
        self.undocking_action_client = UndockingActionClient(self)
        self.manipulator_client      = ManipulatorTaskActionClient(self)
        # ReverseDriveClient is the fallback when undock_robot action fails.
        # It drives the robot in reverse using TF closed-loop feedback.
        self.reverse_drive_client    = ReverseDriveClient(self, self.docking_config)

        # Delay the initial position check to allow the pose subscription to
        # receive its first message before comparing against the dock position.
        self.init_check_timer = self.create_timer(
            self.timing_initial_position_check_delay,
            self._initial_position_check_timer
        )
        self.timer = self.create_timer(self.timing_timer_period, self.timer_callback)

        self.get_logger().info("Husky Operations Manager ready.")

    def _poll_docking_config(self):
        """
        Polls DockingParamFetcher every 0.5s.

        Cancels itself on DONE or ERROR. On DONE, fires _on_docking_config_ready
        which initialises all clients and starts the main timers.

        TODO: Need a better mechanism to handle on ERROR status as this node needs to run
        independent of docking config
        """
        from husky_operations_manager.enum import DockingParamFetcherStatus
        status = self._param_fetcher.get_status()
        self.get_logger().debug(f"DockingParamFetcher poll | status={status.name}")

        if status == DockingParamFetcherStatus.DONE:
            self._config_poll_timer.cancel()
            self._on_docking_config_ready()

        elif status == DockingParamFetcherStatus.ERROR:
            # Fatal — node cannot operate without docking config
            self._config_poll_timer.cancel()
            self.get_logger().error("DockingParamFetcher failed — node cannot start")

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
        self.declare_parameter('battery.low_threshold', 50.0)
        self.declare_parameter('battery.full_threshold', 99.0)
        self.declare_parameter('loading.increment', 20.0)
        self.declare_parameter('timing.timer_period', 1.0)
        self.declare_parameter('timing.initial_position_check_delay', 2.0)
        self.declare_parameter('server.action_server_timeout', 5.0)

    def _get_paramters(self):
        """Read all declared parameters into instance variables."""
        self.navigation_max_retries = int(self.get_parameter('navigation.max_retries').value)
        self.navigation_retry_delay = float(self.get_parameter('navigation.retry_delay').value)
        self.docking_max_retries    = int(self.get_parameter('docking.max_retries').value)
        self.docking_retry_delay    = float(self.get_parameter('docking.retry_delay').value)
        self.docking_threshold      = float(self.get_parameter('docking.threshold').value)
        self.battery_low_threshold  = float(self.get_parameter('battery.low_threshold').value)
        self.battery_full_threshold = float(self.get_parameter('battery.full_threshold').value)
        self.loading_increment      = float(self.get_parameter('loading.increment').value)
        self.timing_timer_period                 = float(self.get_parameter('timing.timer_period').value)
        self.timing_initial_position_check_delay = float(self.get_parameter('timing.initial_position_check_delay').value)
        self.server_action_server_timeout        = float(self.get_parameter('server.action_server_timeout').value)

        self.get_logger().debug(f"Parameters loaded | "
            f"nav_retries={self.navigation_max_retries} nav_delay={self.navigation_retry_delay}s | "
            f"dock_retries={self.docking_max_retries} dock_delay={self.docking_retry_delay}s "
            f"dock_threshold={self.docking_threshold}m | "
            f"battery_low={self.battery_low_threshold}% battery_full={self.battery_full_threshold}% | "
            f"load_increment={self.loading_increment}% | "
            f"timer={self.timing_timer_period}s"
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
        self.is_initialized          = False   # True after check_initial_position runs
        self.is_at_docking_station   = False   # True if robot started within docking_threshold
        self.startup_undock_complete = False   # True once startup undocking path is done
                                               # Also used to distinguish startup vs task
                                               # context in _handle_undocking and
                                               # _handle_reverse_drive

        # --- Robot state ---
        self.current_status  = RobotStatusEnum.IDLE
        self.previous_status = RobotStatusEnum.IDLE

        # --- Task management ---
        self.current_task: Task | None        = None
        self.current_sub_task: SubTask | None = None
        # Index into current_task.sub_tasks list; incremented externally by JobPublisher
        # publishing the next subtask. Reset to 0 on each new task.
        self.current_sub_task_index           = 0

        # Used to detect new tasks/subtasks arriving on the /status/task topic
        self.last_handled_task_id: int | None      = None
        self.last_handled_task_type: int | None    = None
        self.last_handled_subtask_type: int | None = None

        # Current topological map node the robot occupies; updated on DESTINATION_REACHED
        self.current_node_id     = 0
        # Load level 0-100%; incremented by loading_increment per harvest cycle,
        # reset to 0 after successful unloading
        self.current_load_status = 0.0

        # --- Retry counters ---
        self.navigation_retry_count = 0   # Reset to 0 on success or after max reached
        self.docking_retry_count    = 0   # Reset to 0 on success or after max reached

        # --- Job duration (used for unloading/charging timers) ---
        self.job_start_time = None   # rclpy clock time when current job phase started
        self.job_duration   = 0.0    # Duration in seconds for current phase

        # --- Undocking ---
        # Stores the charging or unloading SubTask so _subtask_undocking can send the
        # correct UndockGoal. UNDOCKING is not in the received subtask list for
        # CHARGING_TASK or UNLOADING_TASK — it is triggered internally.
        self.last_undocking_subtask: SubTask | None = None
        # Tracks which task type triggered undocking (CHARGING_TASK or UNLOADING_TASK)
        self.undocking_after_task_type: int | None  = None

        # --- Reverse drive ---
        # Set True when ReverseDriveClient.drive_to_staging() starts.
        # Set False in _handle_reverse_drive when DONE, ERROR, or CANCELED.
        # Used as the priority-4 check in _process_action_clients.
        self.reverse_drive_active: bool = False

        # --- Arm state ---
        # Tracks the last successfully confirmed arm command so STOW can be
        # verified before undocking and READY before harvesting without issuing
        # redundant goals when the arm is already in the correct configuration.
        # Boot assumption: arm is in STOW at startup (safe default).
        self.last_confirmed_arm_command: str = ArmCommand.GO_STOW
        # True while waiting for a STOW goal to complete via _handle_manipulator.
        # Gates _subtask_undocking from proceeding until arm is safe.
        self.arm_stow_pending: bool  = False
        # True while waiting for a READY goal to complete via _handle_manipulator.
        # Gates _subtask_harvesting from advancing past DESTINATION_REACHED.
        self.arm_ready_pending: bool = False

        self.get_logger().debug("State variables initialised")

    def _init_sensor_data(self):
        """Initialise sensor data containers with empty default messages."""
        self.battery_status = BatteryState()
        self.task           = Task()
        self.gps_status     = NavSatFix()
        self.pose_status    = PoseWithCovarianceStamped()
        self.imu_status     = Imu()
        self.estop_status   = Bool()

    def _init_subscriptions(self):
        """Create all ROS2 subscriptions."""
        # Battery state — used for low-battery detection and charge completion checks
        self.battery_sub = self.create_subscription(
            BatteryState,
            f'{self.namespace}/platform/bms/state',
            lambda msg: setattr(self, 'battery_status', msg),
            qos_profile_sensor_data)

        # Ground-truth pose — used for initial dock distance check and navigation target check
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            f"{self.namespace}/ground_truth/pose",
            self._pose_callback,
            10)

        # IMU — stored but not currently used in state machine logic
        self.imu_sub = self.create_subscription(
            Imu,
            f'{self.namespace}/ssensors/gps_0/imu',
            lambda msg: setattr(self, 'imu_status', msg),
            qos_profile_sensor_data)

        # Emergency stop — mapped to online_flag in published RobotStatus
        self.estop_sub = self.create_subscription(
            Bool,
            f'{self.namespace}/platform/emergency_stop',
            lambda msg: setattr(self, 'estop_status', msg),
            qos_profile_sensor_data)

        # Task topic — published by JobPublisher with HARVESTING, CHARGING, or UNLOADING tasks
        self.task_sub = self.create_subscription(
            Task,
            f'{self.namespace}/status/task',
            self._task_callback,
            10)

        self.get_logger().debug("Subscriptions initialised")

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
        subtasks_summary = [(st.sub_task_id, st.type, st.description) for st in msg.sub_tasks]
        self.get_logger().debug(
            f"Received task | ID: {msg.task_id} | Type: {msg.task_type} | "
            f"Target Node: {msg.target_node_id} | SubTasks: {subtasks_summary}")

        # A new task is detected when the task_id changes OR when task_type changes
        # on the same id (e.g. fallback from HARVESTING to CHARGING on low battery)
        is_new_task = (msg.task_id != self.last_handled_task_id or
                       self.current_task and msg.task_type != self.current_task.task_type)

        # A new subtask is detected when the first subtask type in the message differs
        # from the last one we processed — JobPublisher advances the subtask index
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
            self.current_sub_task_index    = 0
            self.last_handled_subtask_type = None
        elif is_new_subtask:
            self.get_logger().info(f"New Subtask for task ID: {msg.task_id}")
            self.current_sub_task_index = 0
            # If the node completed the previous subtask and is waiting at JOB_DONE,
            # receiving a new subtask means the job publisher wants us to continue
            if self.current_status == RobotStatusEnum.JOB_DONE:
                self._transition_status(RobotStatusEnum.IDLE)

        self.task = msg

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
        robot_status.robot_namespace = self.namespace.replace(r'/', '')

        # Update sensor fields before routing so all paths see fresh data
        self._set_battery_status(robot_status)
        self._set_estop_status(robot_status)
        self._set_location_status(robot_status)

        self.get_logger().debug(
            f"Timer tick | status={self.current_status.name} | "
            f"startup_undock_complete={self.startup_undock_complete} | "
            f"reverse_drive_active={self.reverse_drive_active} | "
            f"arm_stow_pending={self.arm_stow_pending} | "
            f"arm_ready_pending={self.arm_ready_pending} | "
            f"last_confirmed_arm='{self.last_confirmed_arm_command}' | "
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
        robot_status.status          = self.current_status.value
        robot_status.current_node_id = self.current_node_id
        robot_status.load_status     = self.current_load_status
        self.robot_state_pub.publish(robot_status)

    # =========================================================================
    # SUBSCRIPTION CALLBACK
    # =========================================================================

    def _pose_callback(self, msg):
        """Store latest ground-truth pose for dock distance checks and target validation."""
        self.pose_status = msg

    # =========================================================================
    # STARTUP AND INITIALIZATION
    # =========================================================================

    def _initial_position_check_timer(self):
        """
        Fires once after timing_initial_position_check_delay.

        Waits until pose data is available, then cancels itself and runs
        check_initial_position. If pose is still None the timer continues
        firing until it arrives.
        """
        if self.pose_status is None:
            self.get_logger().warning("Waiting for pose data...")
            return
        self.init_check_timer.cancel()
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

        #  # TODO: Implement this logic once new station is added for unloading.
        #  # Find nearest dock — use index 0 if only one dock, else find closest
        # dock_configs = self.docking_config.dock_configs
        # if len(dock_configs) == 1:
        #     charger_dock_pose = self.active_dock
        # else:
        #     charger_dock_pose = min(
        #         dock_configs.values(),
        #         key=lambda d: self._calculate_distance(
        #             current_pos.x, current_pos.y, d.dock_x, d.dock_y)
        #     )

        charger_dock_pose = self.active_dock
        current_pos = self.pose_status.pose.pose.position

        distance = self._calculate_distance(
            current_pos.x, current_pos.y,
            charger_dock_pose.dock_x, charger_dock_pose.dock_y)

        self.get_logger().info(
            f"Initial Robot position: ({current_pos.x:.3f}, {current_pos.y:.3f}), "
            f"Dock Name: '{charger_dock_pose.instance_name}' | "
            f"Distance to dock: {distance:.3f}m")

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
        """
        Drive the startup undocking state machine.

        Called from timer_callback while startup_undock_complete is False.

        Tick 1 (IDLE):            transition to START_UNDOCKING
        Tick 2 (START_UNDOCKING): compute UndockGoal from DockingConfig,
                                  STOW check runs inside _subtask_undocking,
                                  send to UndockingActionClient, transition to UNDOCKING
        Tick 3+ (else):           hand off to _handle_undocking or _handle_reverse_drive
                                  depending on reverse_drive_active flag
        """
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

            # Compute max_undocking_time from docking config rather than hardcoding.
            # Formula: distance to staging pose / minimum speed * safety factor of 2
            dock_type          = self.active_dock.type
            staging_x_offset   = self.active_plugin.staging_x_offset or 0.7
            v_linear_min       = self.docking_config.controller_v_linear_min or 0.01
            max_undocking_time = (abs(staging_x_offset) / max(v_linear_min, 0.01)) * 2.0

            self.get_logger().debug(
                f"Startup UndockGoal | dock_type='{dock_type}' | "
                f"staging_x_offset={staging_x_offset} | "
                f"v_linear_min={v_linear_min} | "
                f"max_undocking_time={max_undocking_time:.1f}s"
            )

            startup_subtask = SubTask()
            startup_subtask.type        = SubTask.UNDOCKING
            startup_subtask.description = "Startup Undocking"
            startup_subtask.undock_goal = UndockGoal(
                dock_type=dock_type,
                max_undocking_time=max_undocking_time
            )

            # Store so the STOW gate inside _subtask_undocking can reference it
            self.last_undocking_subtask = startup_subtask
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

    def _check_and_handle_low_battery(self) -> bool:
        """
        Check battery and interrupt task if level is critical.

        Returns True if the caller should abort the current tick (battery is low
        or recovery is in progress). Returns False to continue normally.

        Low battery during a non-CHARGING_TASK:
          - Cancel navigation if active
          - Transition to ERROR so JobPublisher sends a CHARGING_TASK

        Low battery recovery:
          - When in ERROR and a CHARGING_TASK arrives, transition back to IDLE
            so the task can be executed without waiting for _handle_error_recovery
        """
        if not self.task or not self.current_task:
            return False

        battery_pct = self._normalize_battery(self.battery_status.percentage)

        self.get_logger().debug(
            f"Battery check | pct={battery_pct:.1f}% | "
            f"low_threshold={self.battery_low_threshold}% | "
            f"task_type={self.task.task_type}"
        )

        if (self.task.task_type != Task.CHARGING_TASK and
                battery_pct <= self.battery_low_threshold):
            self.get_logger().warning(
                f"Battery low: {battery_pct:.1f}% — threshold={self.battery_low_threshold}% | "
                f"task_type={self.task.task_type} | status={self.current_status.name}"
            )
            # Cancel in-progress navigation to allow robot to receive charging task
            if self.current_status in [RobotStatusEnum.START_MOVING, RobotStatusEnum.MOVING]:
                self.get_logger().warning("Cancelling navigation due to low battery")
                self.navigation.cancel_goal()
            self._transition_status(RobotStatusEnum.ERROR)
            return True

        # Bypass _handle_error_recovery for low-battery recovery — a CHARGING_TASK
        # arriving while in ERROR means JobPublisher has handled the situation
        if (self.current_status == RobotStatusEnum.ERROR and
                self.current_task.task_type == Task.CHARGING_TASK):
            self.get_logger().info(
                f"Recovering from low battery — CHARGING_TASK received | "
                f"battery={battery_pct:.1f}%"
            )
            self._transition_status(RobotStatusEnum.IDLE)

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
        nav_status    = self.navigation.get_navigation_status()
        dock_status   = self.docking_action_client.get_status()
        undock_status = self.undocking_action_client.get_status()

        self.get_logger().debug(
            f"Action clients | nav={nav_status.name} | "
            f"dock={dock_status.name} | "
            f"undock={undock_status.name} | "
            f"reverse_drive_active={self.reverse_drive_active} | "
            f"arm_stow_pending={self.arm_stow_pending} | "
            f"arm_ready_pending={self.arm_ready_pending}"
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
            self.current_status == RobotStatusEnum.HARVESTING and
            not self.arm_stow_pending and
            not self.arm_ready_pending
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
            f"Current subtask | index={self.current_sub_task_index} | "
            f"type={self.current_sub_task.type if self.current_sub_task else 'None'} | "
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

        if (self.current_task.task_id != self.last_handled_task_id or
                self.current_task.task_type != self.last_handled_task_type):
            self.get_logger().info(
                f"Starting Task: {self.current_task.description} | "
                f"ID: {self.current_task.task_id} | "
                f"Current Node: {self.current_node_id} | "
                f"Target Node: {self.current_task.target_node_id}")
            self.get_logger().debug(
                f"Task start | task_type={self.current_task.task_type} | "
                f"last_task_id={self.last_handled_task_id} | "
                f"last_task_type={self.last_handled_task_type} | "
                f"num_subtasks={len(self.current_task.sub_tasks) if isinstance(self.current_task.sub_tasks, list) else 0}"
            )

            # Cache to prevent re-triggering JOB_START on repeated publishes
            self.last_handled_task_id      = self.current_task.task_id
            self.last_handled_task_type    = self.current_task.task_type
            self.last_handled_subtask_type = None
            # Clear any stale undocking state from a previous task
            self.last_undocking_subtask    = None
            self.undocking_after_task_type = None
            self._transition_status(RobotStatusEnum.JOB_START)

        elif self.current_status == RobotStatusEnum.JOB_DONE:
            self.get_logger().debug(
                f"JOB_DONE — same task still publishing | "
                f"task_id={self.current_task.task_id} — transitioning to IDLE"
            )
            self._transition_status(RobotStatusEnum.IDLE)
            self.current_task = None

    def _execute_current_subtask(self):
        """
        Route to the correct subtask handler based on current_sub_task.type.
        """
        if not isinstance(self.current_sub_task, SubTask):
            self.get_logger().debug(
                f"_execute_current_subtask called but no valid subtask | "
                f"status={self.current_status.name}"
            )
            return

        # Log only on subtask type change to avoid flooding at 1Hz
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
            robot_status.task            = wpf_status.task
            robot_status.current_node_id = wpf_status.current_node_id
            robot_status.target_node_id  = wpf_status.target_node_id
        else:
            robot_status.task            = self.current_task.description if self.current_task else ""
            robot_status.current_node_id = self.current_node_id
            robot_status.target_node_id  = self.current_task.target_node_id if self.current_task else -1

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
            # Advance tracked node to the waypoint just reached
            self.current_node_id = wpf_status.target_node_id if wpf_status else self.current_node_id
            self._transition_status(RobotStatusEnum.DESTINATION_REACHED)

        elif nav_status in [NavigationStatus.ABORTED, NavigationStatus.ERROR]:
            self.get_logger().debug(
                f"Navigation aborted/error | nav_status={nav_status.name} | "
                f"retry_count={self.navigation_retry_count}"
            )
            self._handle_navigation_retry()

        elif nav_status == NavigationStatus.CANCELED:
            self.get_logger().info("Navigation canceled")
            self.navigation.reset()
            self.navigation_retry_count = 0
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
        """
        Re-send the NavigateThroughPoses goal for the current task.

        Guards against sending a new goal when the previous one is still
        active (ACTIVE or SENDING) — can happen if the retry delay is too short.
        """
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
            self.get_logger().warning(
                f"Navigation still active ({nav_status.name}) — skipping retry until complete"
            )
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
        """
        Monitor DockingActionClient and update RobotStatus.

        DOCKING      → ensure status is DOCKING
        DONE_DOCKING → reset client, → DONE_DOCKING (next subtask picks up)
        ERROR        → retry via _handle_docking_retry
        """
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
            self.get_logger().debug(
                f"Docking error — triggering retry | "
                f"retry_count={self.docking_retry_count}/{self.docking_max_retries}"
            )
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
            self.get_logger().info(
                f"Retrying docking in {self.docking_retry_delay:.1f}s | "
                f"attempt {self.docking_retry_count}/{self.docking_max_retries}"
            )
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
        self.get_logger().debug(
            f"Retry docking | attempt={self.docking_retry_count}/{self.docking_max_retries} | "
            f"subtask='{self.current_sub_task.description}'"
        )
        if self.docking_action_client.send_docking_goal(self.current_sub_task):
            self._transition_status(RobotStatusEnum.DOCKING)
        else:
            self.get_logger().error(
                f"Docking retry send_goal failed | "
                f"attempt={self.docking_retry_count}/{self.docking_max_retries}"
            )
            self._transition_status(RobotStatusEnum.ERROR)

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
                # Startup context: robot has left the dock, ready for tasks
                self.get_logger().debug(
                    "Undocking DONE — startup context: setting startup_undock_complete=True → IDLE"
                )
                self.startup_undock_complete = True
                # Clear task IDs so the first real task is treated as new
                self.last_handled_task_id   = None
                self.last_handled_task_type = None
                self._transition_status(RobotStatusEnum.IDLE)
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
        """
        Undocking action failed — start ReverseDriveClient as fallback.

        The undock_robot action is ABORTED (status 6, mapped to ERROR=94).
        ReverseDriveClient drives the robot in reverse using TF closed-loop
        feedback to reach the pre-computed staging pose.

        Safety: If dock_backwards=True in DockingConfig, reversing is unsafe
        and the node transitions to ERROR for manual recovery.
        """
        self.undocking_action_client.reset()
        self.get_logger().warning(
            f"Undocking failed — starting reverse drive to staging pose | "
            f"dock='{self.active_dock.instance_name if self.active_dock else 'unknown'}'"
        )

        if self.reverse_drive_client.drive_to_staging():
            # drive_to_staging returns True when REVERSING begins successfully
            self.reverse_drive_active = True
            self.get_logger().debug(
                f"ReverseDriveClient started | reverse_drive_active={self.reverse_drive_active}"
            )
            self._transition_status(RobotStatusEnum.UNDOCKING)
        else:
            # Returns False when dock_backwards=True or already reversing
            self.get_logger().error(
                f"ReverseDriveClient refused to start — "
                f"dock_backwards={self.docking_config.dock_backwards if self.docking_config else 'unknown'}"
            )
            self._transition_status(RobotStatusEnum.ERROR)

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
                # Startup context: robot reached staging pose, ready for tasks
                self.get_logger().debug(
                    "Reverse drive DONE — startup context: setting startup_undock_complete=True → IDLE"
                )
                self.startup_undock_complete = True
                self.last_handled_task_id   = None
                self.last_handled_task_type = None
                self._transition_status(RobotStatusEnum.DONE_UNDOCKING)
                self._transition_status(RobotStatusEnum.IDLE)
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

    def _handle_manipulator(self, robot_status: RobotStatus):
        """
        Monitor ManipulatorTaskActionClient for pending STOW, READY, or START_HARVEST goals.

        Called from _process_action_clients every tick while any arm operation is pending.

        Context-aware completion handling:
          STOW completion:
            - START_UNDOCKING context → arm is safe, re-enter _subtask_undocking to proceed
            - DONE_HARVESTING context → arm is safe, _subtask_harvesting will advance to
                                        JOB_DONE on the next tick
          READY completion:
            - DESTINATION_REACHED context → arm is prepared, _subtask_harvesting will
                                            advance to START_HARVESTING on the next tick
          START_HARVEST completion (DONE_HARVESTING result from manipulator):
            - HARVESTING context → harvest cycle complete, transition to DONE_HARVESTING
                                   so _subtask_harvesting can begin STOW on next tick

        ERROR in any arm command transitions the node to ERROR — proceeding with an
        arm in an unknown configuration is unsafe regardless of context.
        """
        arm_status = self.manipulator_client.get_status()

        self.get_logger().debug(
            f"Manipulator handler | arm_status={arm_status.name} | "
            f"current_status={self.current_status.name} | "
            f"arm_stow_pending={self.arm_stow_pending} | "
            f"arm_ready_pending={self.arm_ready_pending} | "
            f"last_confirmed_arm='{self.last_confirmed_arm_command}'"
        )

        # ----------------------------------------------------------------
        # STOW completion
        # ----------------------------------------------------------------
        if self.arm_stow_pending and arm_status == RobotStatusEnum.DONE_HARVESTING:
            self.get_logger().info(
                f"Arm STOW confirmed | context={self.current_status.name}"
            )
            self.arm_stow_pending = False
            self.last_confirmed_arm_command = ArmCommand.GO_STOW
            self.manipulator_client.reset()

            if self.current_status == RobotStatusEnum.START_UNDOCKING:
                # Undocking flow was gated on STOW — arm is now safe, proceed
                self.get_logger().debug(
                    "STOW confirmed in START_UNDOCKING context — re-entering _subtask_undocking"
                )
                self._subtask_undocking()
            elif self.current_status == RobotStatusEnum.DONE_HARVESTING:
                # Harvest flow was gated on STOW — _subtask_harvesting will see
                # last_confirmed_arm_command == GO_STOW and advance to JOB_DONE
                # on the next timer tick. Nothing to call here.
                self.get_logger().debug(
                    "STOW confirmed in DONE_HARVESTING context — "
                    "_subtask_harvesting will advance to JOB_DONE on next tick"
                )
            else:
                self.get_logger().warning(
                    f"STOW confirmed in unexpected context={self.current_status.name} — "
                    "no action taken"
                )

        # ----------------------------------------------------------------
        # READY completion
        # ----------------------------------------------------------------
        elif self.arm_ready_pending and arm_status == RobotStatusEnum.DONE_HARVESTING:
            self.get_logger().info(
                f"Arm READY confirmed | context={self.current_status.name}"
            )
            self.arm_ready_pending = False
            self.last_confirmed_arm_command = ArmCommand.GO_READY
            self.manipulator_client.reset()

            if self.current_status == RobotStatusEnum.DESTINATION_REACHED:
                # Harvest flow was gated on READY — _subtask_harvesting will see
                # last_confirmed_arm_command == GO_READY and advance to START_HARVESTING
                # on the next timer tick. Nothing to call here.
                self.get_logger().debug(
                    "READY confirmed in DESTINATION_REACHED context — "
                    "_subtask_harvesting will advance to START_HARVESTING on next tick"
                )
            else:
                self.get_logger().warning(
                    f"READY confirmed in unexpected context={self.current_status.name} — "
                    "no action taken"
                )

        # ----------------------------------------------------------------
        # START_HARVEST completion
        # ----------------------------------------------------------------
        elif (not self.arm_stow_pending and
              not self.arm_ready_pending and
              arm_status == RobotStatusEnum.DONE_HARVESTING and
              self.current_status == RobotStatusEnum.HARVESTING):
            self.get_logger().info("Harvest goal complete — transitioning to DONE_HARVESTING")
            self.manipulator_client.reset()
            self._transition_status(RobotStatusEnum.DONE_HARVESTING)

        # ----------------------------------------------------------------
        # ERROR — any arm command failure
        # ----------------------------------------------------------------
        elif arm_status == RobotStatusEnum.ERROR:
            self.get_logger().error(
                f"Arm command failed | context={self.current_status.name} | "
                f"stow_pending={self.arm_stow_pending} | "
                f"ready_pending={self.arm_ready_pending} | "
                f"last_confirmed='{self.last_confirmed_arm_command}'"
            )
            self.arm_stow_pending  = False
            self.arm_ready_pending = False
            self.manipulator_client.reset()
            self._transition_status(RobotStatusEnum.ERROR)

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    def _handle_error_recovery(self):
        """
        Attempt to recover from ERROR or ABNORMAL status.

        Cancels any active navigation goal (with a brief wait for cancellation
        to propagate), resets subtask state, and transitions to IDLE.

        Note: Low-battery recovery bypasses this method — it is handled directly
        in _check_and_handle_low_battery when a CHARGING_TASK arrives.
        """
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

        # Clear any pending arm flags so the next task starts with a clean slate
        self.arm_stow_pending  = False
        self.arm_ready_pending = False
        self.manipulator_client.reset()

        # Reset subtask state so the next task starts clean
        self.current_sub_task       = None
        self.current_sub_task_index = 0
        self.get_logger().debug("Error recovery — reset subtask state, transitioning to IDLE")
        self._transition_status(RobotStatusEnum.IDLE)

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
        """
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
            self.get_logger().info(
                f"Starting navigation: {self.current_node_id} → "
                f"{self.current_task.target_node_id}")
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
        """
        Handle the DOCKING subtask.

        State progression:
          DESTINATION_REACHED → START_DOCKING → 1s delay → send dock_robot → DOCKING
          (docking result handled by _handle_docking via _process_action_clients)

        The 1s delay gives Nav2 time to decelerate fully before the docking
        controller starts, reducing the chance of a dock alignment failure.
        """
        dock_id = (self.current_sub_task.dock_goal.dock_id
                   if self.current_sub_task and self.current_sub_task.dock_goal else 'None')
        self.get_logger().debug(
            f"_subtask_docking | status={self.current_status.name} | "
            f"dock_id='{dock_id}'"
        )

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
            f"_subtask_undocking | status={self.current_status.name} | "
            f"last_undocking_subtask={'set' if self.last_undocking_subtask else 'None'} | "
            f"undocking_after_task_type={self.undocking_after_task_type} | "
            f"last_confirmed_arm='{self.last_confirmed_arm_command}' | "
            f"arm_stow_pending={self.arm_stow_pending}"
        )

        if self.current_status == RobotStatusEnum.START_UNDOCKING:
            # ---- Arm safety gate ----
            # Arm must be confirmed STOW before undocking begins.
            # If not already stowed, dispatch STOW now and hold here.
            # _handle_manipulator will re-enter this method once confirmed.
            if self.last_confirmed_arm_command != ArmCommand.GO_STOW:
                if not self.arm_stow_pending:
                    self.get_logger().info(
                        f"Arm not in STOW (last='{self.last_confirmed_arm_command}') — "
                        "sending STOW before undocking"
                    )
                    undock_ref = self.current_sub_task or self.last_undocking_subtask
                    if self.manipulator_client.send_stow_goal(undock_ref):
                        self.arm_stow_pending = True
                    else:
                        self.get_logger().error(
                            "Failed to send STOW goal before undocking — transitioning to ERROR"
                        )
                        self._transition_status(RobotStatusEnum.ERROR)
                else:
                    self.get_logger().debug(
                        "Arm STOW already in progress — waiting before undocking"
                    )
                return  # Hold at START_UNDOCKING until arm_stow_pending clears

            # ---- Arm confirmed STOW — proceed with undocking ----
            self.get_logger().info("Starting undocking — arm confirmed STOW")

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
            # Undocking succeeded — clean up stored state and mark task complete
            self.get_logger().info("Undocking done")
            self.get_logger().debug(
                f"Undocking done — clearing state | "
                f"undocking_after_task_type={self.undocking_after_task_type}"
            )
            self.last_undocking_subtask    = None
            self.undocking_after_task_type = None
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
            f"_subtask_harvesting | status={self.current_status.name} | "
            f"arm_ready_pending={self.arm_ready_pending} | "
            f"arm_stow_pending={self.arm_stow_pending} | "
            f"last_confirmed_arm='{self.last_confirmed_arm_command}'"
        )

        if self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            # Ensure arm is in READY before harvesting begins.
            # Hold here until _handle_manipulator confirms READY.
            if self.last_confirmed_arm_command != ArmCommand.GO_READY:
                if not self.arm_ready_pending:
                    self.get_logger().info(
                        f"Arm not in READY (last='{self.last_confirmed_arm_command}') — "
                        "sending READY before harvesting"
                    )
                    if self.manipulator_client.send_ready_goal(self.current_sub_task):
                        self.arm_ready_pending = True
                    else:
                        self.get_logger().error(
                            "Failed to send READY goal before harvesting — transitioning to ERROR"
                        )
                        self._transition_status(RobotStatusEnum.ERROR)
                else:
                    self.get_logger().debug("Arm READY already in progress — waiting")
                return  # Hold at DESTINATION_REACHED until arm_ready_pending clears

            # Arm confirmed in READY — proceed to harvesting
            self._transition_status(RobotStatusEnum.START_HARVESTING)

        elif self.current_status == RobotStatusEnum.START_HARVESTING:
            # Guard: do not send a second goal if manipulator is already active
            if self.manipulator_client.get_status() == RobotStatusEnum.HARVESTING:
                self.get_logger().warning(
                    "Harvest goal send skipped — manipulator already active"
                )
                return
            self.get_logger().info("Sending harvest goal to manipulator")
            if self.manipulator_client.send_harvesting_goal(self.current_sub_task):
                self._transition_status(RobotStatusEnum.HARVESTING)
            else:
                self.get_logger().error(
                    "Failed to send harvest goal — transitioning to ERROR"
                )
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.HARVESTING:
            # Manipulator result is handled by _handle_manipulator via _process_action_clients.
            # When the goal succeeds, ManipulatorTaskActionClient sets its internal status to
            # DONE_HARVESTING, which _handle_manipulator detects and transitions the node
            # to DONE_HARVESTING. Nothing to do here — just wait.
            self.get_logger().debug(
                "Harvesting in progress — waiting for manipulator result"
            )

        elif self.current_status == RobotStatusEnum.DONE_HARVESTING:
            # Update load before checking arm — load must be recorded regardless of
            # whether STOW has been sent yet. Guard with arm_stow_pending to prevent
            # re-incrementing on subsequent ticks while waiting for STOW confirmation.
            if not self.arm_stow_pending and self.last_confirmed_arm_command != ArmCommand.GO_STOW:
                new_load = min(self.current_load_status + self.loading_increment, 100.0)
                self.get_logger().debug(
                    f"Load update | {self.current_load_status:.1f}% → {new_load:.1f}% "
                    f"(+{self.loading_increment:.1f}%)"
                )
                self.current_load_status = new_load
                self.get_logger().info(f"Load status: {self.current_load_status:.1f}%")

                # Always stow the arm after harvesting before transitioning to JOB_DONE.
                # Hold here until _handle_manipulator confirms STOW.
                self.get_logger().info("Harvesting done — sending arm to STOW")
                if self.manipulator_client.send_stow_goal(self.current_sub_task):
                    self.arm_stow_pending = True
                else:
                    self.get_logger().error(
                        "Failed to send STOW goal after harvesting — transitioning to ERROR"
                    )
                    self._transition_status(RobotStatusEnum.ERROR)
                return  # Hold until STOW confirmed

            # STOW confirmed — advance to JOB_DONE
            if self.last_confirmed_arm_command == ArmCommand.GO_STOW and not self.arm_stow_pending:
                self.get_logger().info("Arm stowed after harvest — transitioning to JOB_DONE")
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
            self.get_logger().info(
                f"Battery charging: {battery_pct:.1f}%", throttle_duration_sec=10.0)
            if battery_pct >= self.battery_full_threshold:
                self.get_logger().info(f"Battery charged: {battery_pct:.1f}%")
                self._transition_status(RobotStatusEnum.DONE_CHARGING)

        elif self.current_status == RobotStatusEnum.DONE_CHARGING:
            # Store the charging subtask so _subtask_undocking can retrieve its UndockGoal.
            # The STOW gate inside _subtask_undocking will fire if arm is not already stowed.
            self.get_logger().debug(
                "DONE_CHARGING — storing last_undocking_subtask and triggering undocking"
            )
            self.last_undocking_subtask    = self.current_sub_task
            self.undocking_after_task_type = Task.CHARGING_TASK
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            # Call immediately — next tick would call this handler again on CHARGING subtask
            # but current_status is now START_UNDOCKING so _subtask_undocking handles it
            self._subtask_undocking()

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            # _handle_undocking set status to DONE_UNDOCKING (task context).
            # Delegate to _subtask_undocking to transition to JOB_DONE.
            self.get_logger().debug(
                "DONE_UNDOCKING in charging context — delegating to _subtask_undocking"
            )
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
            f"_subtask_unloading | status={self.current_status.name} | "
            f"current_load={self.current_load_status:.1f}%"
        )

        if self.current_status == RobotStatusEnum.DONE_DOCKING:
            self._transition_status(RobotStatusEnum.START_UNLOADING)

        elif self.current_status == RobotStatusEnum.START_UNLOADING:
            self._transition_status(RobotStatusEnum.UNLOADING)
            self.get_logger().info("Unloading started")
            # Start simulated unload timer (replace with hardware signal)
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
            # Reset load to 0 and store undocking subtask before triggering undocking.
            # The STOW gate inside _subtask_undocking will fire if arm is not already stowed.
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
            # _handle_undocking set status to DONE_UNDOCKING (task context).
            # Delegate to _subtask_undocking to transition to JOB_DONE.
            self.get_logger().debug(
                "DONE_UNDOCKING in unloading context — delegating to _subtask_undocking"
            )
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
            self.get_logger().debug(
                "_is_robot_at_target: no subtask or pose — returning False"
            )
            return False

        robot_position = self.pose_status.pose.pose.position
        # Normalise waypoints in case the subtask data contains raw dicts
        waypoints_list = [
            WayPoint(**wp) if isinstance(wp, dict) else wp
            for wp in self.current_sub_task.data
        ]
        target_wp = waypoints_list[-1]  # Final waypoint is the destination
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
        """
        Safely transition to a new RobotStatusEnum value.

        No-ops if new_status equals current_status to avoid redundant log lines.
        Stores the previous status for debugging.
        """
        if self.current_status != new_status:
            self.previous_status = self.current_status
            self.current_status  = new_status
            self.get_logger().info(
                f"Status: {self.previous_status.name} → {self.current_status.name}")

    def _normalize_battery(self, percentage: float) -> float:
        """
        Normalise battery percentage to the 0 - 100 range.

        BatteryState.percentage can be provided as 0.0 - 1.0 (ROS convention) or
        0 - 100 depending on the BMS driver. Multiplies by 100 if the value is <= 1.0.
        """
        return percentage * 100.0 if percentage <= 1.0 else percentage

    def _format_time_remaining(self, hours: float) -> str:
        """Convert a fractional hours value to a human-readable HH hours MM minutes string."""
        seconds          = int(hours * 3600)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes   = divmod(minutes, 60)
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


if __name__ == '__main__':
    main()