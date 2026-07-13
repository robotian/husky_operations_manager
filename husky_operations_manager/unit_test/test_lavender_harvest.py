#!/usr/bin/env python3
"""
LavenderHarvestNode — combined NavigateThroughPoses + DriveClient harvesting node.

Startup sequence:  nearest-dock selection → startup undocking (with arm STOW gate)
                   → send task trigger to server
Task execution:    dual-mode navigation
                     - Nav2 (NavigateThroughPoses) for row start / dock area
                     - DriveClient for camera-guided row traversal
Arm control:       STOW/READY gating via ManipulatorTaskActionClient
                   (identical pattern to husky_operations_manager.py)
Unloading:         UnloaderActionClient  END → delay → HOME
Task generation:   publishes trigger to server; server creates task and stores in DB
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import BatteryState, Imu, NavSatFix
from std_msgs.msg import Bool, String
from status_interfaces.msg import (
    ImageDetectionPose,
    RobotStatus,
    SubTask,
    Task,
    UndockGoal,
    WayPoint,
)
from status_interfaces.action import OperateUnloader

from husky_operations_manager.robot_enums import (
    DriveStatus,
    NavigationStatus,
    OnlineFlagEnum,
    ReverseDriveStatus,
    RobotStatusEnum,
)
from husky_operations_manager.types import (
    DockInstanceConfig,
    DockPose,
    DriveConfig,
    ReverseDriveConfig,
)
from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.action_clients.drive import DriveClient
from husky_operations_manager.action_clients.manipulator import (
    ArmCommand,
    ManipulatorTaskActionClient,
)
from husky_operations_manager.action_clients.navigation import NavigationActionClient
from husky_operations_manager.action_clients.reverse_drive import ReverseDriveClient
from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.action_clients.unloader import UnloaderActionClient


class LavenderHarvestNode(Node):
    """
    Combined NavigateThroughPoses + DriveClient node for lavender row harvesting.

    Startup: nearest-dock selection → startup undocking → task trigger to server
    Task:    Nav2 to row start then DriveClient for bush-by-bush traversal
    Arm:     STOW/READY gating around every harvest cycle and undocking
    Unload:  UnloaderActionClient  END → delay → HOME
    """

    def __init__(self):
        super().__init__('test_lavender_harvest')

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'Node namespace: {self.namespace}')

        self._declare_parameter()
        self._get_paramters()
        self._init_state_variables()
        self._init_sensor_data()
        self._init_subscriptions()

        # Publishers
        self.robot_state_pub = self.create_publisher(RobotStatus, f'{self.namespace}/status/robot', 10)
        self.task_trigger_pub = self.create_publisher(String, f'{self.namespace}/{self.task_trigger_topic}', 10)

        # Action clients that do NOT require active_dock — built here
        self.navigation = NavigationActionClient(self)
        self.docking_action_client = DockingActionClient(self)
        self.undocking_action_client = UndockingActionClient(self)
        self.manipulator_client = ManipulatorTaskActionClient(self)
        self.unloader_action_client = UnloaderActionClient(self)

        # DriveClient built from YAML drive.* params
        self.drive_client = DriveClient(
            self,
            DriveConfig(
                detection_topic=self.drive_detection_topic,
                odom_topic=self.drive_odom_topic,
                base_frame=self.drive_base_frame,
                cmd_vel_rate=self.drive_cmd_vel_rate,
                ex_tolerance=self.drive_ex_tolerance,
                stop_lookahead=self.drive_stop_lookahead,
                ex_coast_gate=self.drive_ex_coast_gate,
                ex_angular_gate=self.drive_ex_angular_gate,
                k_rho=self.drive_k_rho,
                v_linear_min=self.drive_v_linear_min,
                v_linear_max=self.drive_v_linear_max,
                v_angular_max=self.drive_v_angular_max,
                departure_clearance=self.drive_departure_clearance,
                no_detection_distance=self.drive_no_detection_distance,
                ang_tol=self.drive_ang_tol,
                k_v_p=self.drive_k_v_p,
                k_v_d=self.drive_k_v_d,
                k_omega_p=self.drive_k_omega_p,
                k_omega_d=self.drive_k_omega_d,
                k_beta_p=self.drive_k_beta_p,
                k_beta_d=self.drive_k_beta_d,
                a_max=self.drive_a_max,
                alpha_max=self.drive_alpha_max,
                backward_distance_threshold=self.drive_backward_distance_threshold,
                cam_tx=self.drive_cam_tx,
                cam_ty=self.drive_cam_ty,
                bushrow_theta=self.drive_bushrow_theta,
                arm_tx_offset=self.drive_arm_tx_offset,
            ),
        )

        # active_dock and reverse_drive_client are resolved in _check_initial_position
        self.active_dock: DockInstanceConfig | None = None
        self.reverse_drive_client: ReverseDriveClient | None = None

        self.init_check_timer = self.create_timer(
            self.timing_initial_position_check_delay,
            self._initial_position_check_timer,
        )
        self.timer = self.create_timer(self.timing_timer_period, self.timer_callback)

        self.get_logger().info('LavenderHarvestNode initialised.')

    # =========================================================================
    # PARAMETERS
    # =========================================================================

    def _declare_parameter(self):
        """Declare all ROS2 parameters with defaults. Values are overridden by YAML."""
        self.declare_parameter('navigation.max_retries', 3)
        self.declare_parameter('navigation.retry_delay', 5.0)
        self.declare_parameter('docking.max_retries', 2)
        self.declare_parameter('docking.retry_delay', 3.0)
        self.declare_parameter('docking.threshold', 0.25)
        self.declare_parameter('battery.low_threshold', 50.0)
        self.declare_parameter('battery.full_threshold', 99.0)
        self.declare_parameter('loading.increment', 20.0)
        self.declare_parameter('unloading.home_delay_s', 2.0)
        self.declare_parameter('timing.timer_period', 1.0)
        self.declare_parameter('timing.initial_position_check_delay', 2.0)

        # Dock instances
        self.declare_parameter('docks.names', ['main_dock'])
        dock_names = list(self.get_parameter('docks.names').value)
        for name in dock_names:
            self.declare_parameter(f'docks.{name}.type', 'simple_charging_dock')
            self.declare_parameter(f'docks.{name}.frame', 'map')
            self.declare_parameter(f'docks.{name}.pose', [0.0, 0.0, 0.0])

        # Plugin (staging offsets)
        self.declare_parameter('plugin.name', 'simple_charging_dock')
        self.declare_parameter('plugin.staging_x_offset', -1.5)
        self.declare_parameter('plugin.staging_yaw_offset', 0.0)

        # Controller (reverse drive motion params)
        self.declare_parameter('controller.base_frame', 'base_link')
        self.declare_parameter('controller.controller_frequency', 50.0)
        self.declare_parameter('controller.v_linear_min', 0.15)
        self.declare_parameter('controller.v_linear_max', 0.15)
        self.declare_parameter('controller.v_angular_max', 0.25)

        # Undocking tolerances
        self.declare_parameter('undocking.linear_tolerance', 0.05)
        self.declare_parameter('undocking.angular_tolerance', 0.1)
        self.declare_parameter('undocking.dock_backwards', False)

        # DriveClient
        self.declare_parameter('drive.detection_topic', 'manipulators/arm_0_detection/image_annotated/detection_pose')
        self.declare_parameter('drive.odom_topic', 'platform/odom')
        self.declare_parameter('drive.base_frame', 'base_link')
        self.declare_parameter('drive.cmd_vel_rate', 10.0)
        self.declare_parameter('drive.v_linear_min', 0.05)
        self.declare_parameter('drive.v_linear_max', 0.2)
        self.declare_parameter('drive.v_angular_max', 0.5)
        self.declare_parameter('drive.k_rho', 1.0)
        self.declare_parameter('drive.ex_tolerance', 0.05)
        self.declare_parameter('drive.stop_lookahead', 0.05)
        self.declare_parameter('drive.ex_coast_gate', 0.1)
        self.declare_parameter('drive.ex_angular_gate', 0.05)
        self.declare_parameter('drive.departure_clearance', 0.3)
        self.declare_parameter('drive.no_detection_distance', 0.5)
        # --- PD target-pose controller (drive.py) ---
        self.declare_parameter('drive.ang_tol', 0.05)
        self.declare_parameter('drive.k_v_p', 0.2)
        self.declare_parameter('drive.k_v_d', 0.2)
        self.declare_parameter('drive.k_omega_p', 0.4)
        self.declare_parameter('drive.k_omega_d', 0.1)
        self.declare_parameter('drive.k_beta_p', 1.0)
        self.declare_parameter('drive.k_beta_d', 1.0)
        self.declare_parameter('drive.a_max', 0.05)
        self.declare_parameter('drive.alpha_max', 0.3)
        self.declare_parameter('drive.backward_distance_threshold', 1.0)
        # --- Camera mount / row geometry (drive.py) ---
        self.declare_parameter('drive.cam_tx', -0.239)
        self.declare_parameter('drive.cam_ty', -0.500)
        self.declare_parameter('drive.bushrow_theta', 0.0)
        self.declare_parameter('drive.arm_tx_offset', 0.214)

        # Task trigger
        self.declare_parameter('task.trigger_topic', 'job/trigger')

        # Subscription topics (relative to robot namespace)
        self.declare_parameter('topics.battery', 'platform/bms/state')
        self.declare_parameter('topics.pose', 'ground_truth/pose')
        self.declare_parameter('topics.imu', 'sensors/gps_0/imu')
        self.declare_parameter('topics.estop', 'platform/emergency_stop')
        self.declare_parameter('topics.task', 'status/task')
        self.declare_parameter('topics.detection', 'manipulators/arm_0_detection/image_annotated/detection_pose')

    def _get_paramters(self):
        """Read all declared parameters into instance variables."""
        self.navigation_max_retries = int(self.get_parameter('navigation.max_retries').value)
        self.navigation_retry_delay = float(self.get_parameter('navigation.retry_delay').value)
        self.docking_max_retries = int(self.get_parameter('docking.max_retries').value)
        self.docking_retry_delay = float(self.get_parameter('docking.retry_delay').value)
        self.docking_threshold = float(self.get_parameter('docking.threshold').value)
        self.battery_low_threshold = float(self.get_parameter('battery.low_threshold').value)
        self.battery_full_threshold = float(self.get_parameter('battery.full_threshold').value)
        self.loading_increment = float(self.get_parameter('loading.increment').value)
        self.unloading_home_delay_s = float(self.get_parameter('unloading.home_delay_s').value)
        self.timing_timer_period = float(self.get_parameter('timing.timer_period').value)
        self.timing_initial_position_check_delay = float(
            self.get_parameter('timing.initial_position_check_delay').value
        )

        # Build dock configs dict — used for nearest-dock selection in _check_initial_position
        dock_names = list(self.get_parameter('docks.names').value)
        self.dock_configs: dict[str, DockInstanceConfig] = {}
        for name in dock_names:
            pose = list(self.get_parameter(f'docks.{name}.pose').value)
            self.dock_configs[name] = DockInstanceConfig(
                instance_name=name,
                type=str(self.get_parameter(f'docks.{name}.type').value),
                frame=str(self.get_parameter(f'docks.{name}.frame').value),
                pose=DockPose(x=float(pose[0]), y=float(pose[1]), theta=float(pose[2])),
            )

        self.plugin_name = str(self.get_parameter('plugin.name').value)
        self.staging_x_offset = float(self.get_parameter('plugin.staging_x_offset').value)
        self.staging_yaw_offset = float(self.get_parameter('plugin.staging_yaw_offset').value)

        self.base_frame = str(self.get_parameter('controller.base_frame').value)
        self.controller_frequency = float(self.get_parameter('controller.controller_frequency').value)
        self.v_linear_min = float(self.get_parameter('controller.v_linear_min').value)
        self.v_linear_max = float(self.get_parameter('controller.v_linear_max').value)
        self.v_angular_max = float(self.get_parameter('controller.v_angular_max').value)

        self.linear_tolerance = float(self.get_parameter('undocking.linear_tolerance').value)
        self.angular_tolerance = float(self.get_parameter('undocking.angular_tolerance').value)
        self.dock_backwards = bool(self.get_parameter('undocking.dock_backwards').value)

        self.drive_detection_topic = str(self.get_parameter('drive.detection_topic').value)
        self.drive_odom_topic = str(self.get_parameter('drive.odom_topic').value)
        self.drive_base_frame = str(self.get_parameter('drive.base_frame').value)
        self.drive_cmd_vel_rate = float(self.get_parameter('drive.cmd_vel_rate').value)
        self.drive_v_linear_min = float(self.get_parameter('drive.v_linear_min').value)
        self.drive_v_linear_max = float(self.get_parameter('drive.v_linear_max').value)
        self.drive_v_angular_max = float(self.get_parameter('drive.v_angular_max').value)
        self.drive_k_rho = float(self.get_parameter('drive.k_rho').value)
        self.drive_ex_tolerance = float(self.get_parameter('drive.ex_tolerance').value)
        self.drive_stop_lookahead = float(self.get_parameter('drive.stop_lookahead').value)
        self.drive_ex_coast_gate = float(self.get_parameter('drive.ex_coast_gate').value)
        self.drive_ex_angular_gate = float(self.get_parameter('drive.ex_angular_gate').value)
        self.drive_departure_clearance = float(self.get_parameter('drive.departure_clearance').value)
        self.drive_no_detection_distance = float(self.get_parameter('drive.no_detection_distance').value)
        # --- PD target-pose controller (drive.py) ---
        self.drive_ang_tol = float(self.get_parameter('drive.ang_tol').value)
        self.drive_k_v_p = float(self.get_parameter('drive.k_v_p').value)
        self.drive_k_v_d = float(self.get_parameter('drive.k_v_d').value)
        self.drive_k_omega_p = float(self.get_parameter('drive.k_omega_p').value)
        self.drive_k_omega_d = float(self.get_parameter('drive.k_omega_d').value)
        self.drive_k_beta_p = float(self.get_parameter('drive.k_beta_p').value)
        self.drive_k_beta_d = float(self.get_parameter('drive.k_beta_d').value)
        self.drive_a_max = float(self.get_parameter('drive.a_max').value)
        self.drive_alpha_max = float(self.get_parameter('drive.alpha_max').value)
        self.drive_backward_distance_threshold = float(self.get_parameter('drive.backward_distance_threshold').value)
        # --- Camera mount / row geometry (drive.py) ---
        self.drive_cam_tx = float(self.get_parameter('drive.cam_tx').value)
        self.drive_cam_ty = float(self.get_parameter('drive.cam_ty').value)
        self.drive_bushrow_theta = float(self.get_parameter('drive.bushrow_theta').value)
        self.drive_arm_tx_offset = float(self.get_parameter('drive.arm_tx_offset').value)

        self.task_trigger_topic = str(self.get_parameter('task.trigger_topic').value)

        self.topic_battery = str(self.get_parameter('topics.battery').value)
        self.topic_pose = str(self.get_parameter('topics.pose').value)
        self.topic_imu = str(self.get_parameter('topics.imu').value)
        self.topic_estop = str(self.get_parameter('topics.estop').value)
        self.topic_task = str(self.get_parameter('topics.task').value)
        self.topic_detection = str(self.get_parameter('topics.detection').value)

        self.get_logger().debug(
            f'Parameters loaded | '
            f'nav_retries={self.navigation_max_retries} nav_delay={self.navigation_retry_delay}s | '
            f'dock_retries={self.docking_max_retries} dock_delay={self.docking_retry_delay}s '
            f'dock_threshold={self.docking_threshold}m | '
            f'battery_low={self.battery_low_threshold}% battery_full={self.battery_full_threshold}% | '
            f'load_increment={self.loading_increment}% | '
            f'timer={self.timing_timer_period}s'
        )

    def _init_state_variables(self):
        """Initialise all state-tracking variables to their boot defaults."""
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

        # --- Undocking ---
        self.last_undocking_subtask: SubTask | None = None
        self.undocking_after_task_type: int | None = None

        # --- Reverse drive ---
        self.reverse_drive_active: bool = False

        # --- Navigation routing ---
        # True  = next HARVESTING MOVING sends Nav2 goal first
        # False = skip Nav2, go straight to DriveClient at DESTINATION_REACHED
        self._need_row_navigation: bool = True

        # --- DriveClient detection tracking ---
        # Reset False on each scan()/resume(); True when a valid detection arrives.
        # Distinguishes STOPPED-at-bush from STOPPED-by-no-detection-timeout.
        self._detection_received: bool = False
        self.last_detection_time: float | None = None

        # --- Unloader phase ---
        # False = awaiting AT_END; True = AT_END done, now commanding HOME.
        self._unloader_at_end: bool = False

        # --- Arm state (mirrors husky_operations_manager.py) ---
        # Boot assumption: arm configuration unknown — STOW gate fires on first undock.
        self.last_confirmed_arm_command: str = ArmCommand.UNKNOWN
        # True while waiting for a STOW goal to complete via _handle_manipulator.
        self.arm_stow_pending: bool = False
        # True while waiting for a READY goal to complete via _handle_manipulator.
        self.arm_ready_pending: bool = False

        self.get_logger().debug('State variables initialised')

    def _init_sensor_data(self):
        """Initialise sensor data containers with empty default messages."""
        self.battery_status = BatteryState()
        self.task = Task()
        self.gps_status = NavSatFix()
        self.pose_status = PoseWithCovarianceStamped()
        self.imu_status = Imu()
        self.estop_status = Bool()

    def _init_subscriptions(self):
        """Create all ROS2 subscriptions using YAML-configured topic names."""
        self.battery_sub = self.create_subscription(
            BatteryState,
            f'{self.namespace}/{self.topic_battery}',
            lambda msg: setattr(self, 'battery_status', msg),
            qos_profile_sensor_data,
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            f'{self.namespace}/{self.topic_pose}',
            self._pose_callback,
            10,
        )
        self.imu_sub = self.create_subscription(
            Imu,
            f'{self.namespace}/{self.topic_imu}',
            lambda msg: setattr(self, 'imu_status', msg),
            qos_profile_sensor_data,
        )
        self.estop_sub = self.create_subscription(
            Bool,
            f'{self.namespace}/{self.topic_estop}',
            lambda msg: setattr(self, 'estop_status', msg),
            qos_profile_sensor_data,
        )
        self.task_sub = self.create_subscription(
            Task,
            f'{self.namespace}/{self.topic_task}',
            self._task_callback,
            10,
        )
        self.detection_sub = self.create_subscription(
            ImageDetectionPose,
            f'{self.namespace}/{self.topic_detection}',
            self._detection_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().debug('Subscriptions initialised')

    # =========================================================================
    # SUBSCRIPTION CALLBACKS
    # =========================================================================

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        """Store latest ground-truth pose for dock distance checks."""
        self.pose_status = msg

    def _detection_callback(self, msg: ImageDetectionPose):
        """Set _detection_received on valid detection — used by _subtask_moving."""
        if msg.detection_valid:
            self._detection_received = True
            self.last_detection_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().debug(f'Detection | center=({msg.center.x:.3f}, {msg.center.y:.3f})')

    def _task_callback(self, msg: Task):
        """
        Handle incoming Task messages from JobPublisher.

        Detects new task (task_id or task_type changed) or new subtask (first subtask
        type differs from last processed). Repeated identical publishes are no-ops.
        """
        subtasks_summary = [(st.sub_task_id, st.type, st.description) for st in msg.sub_tasks]
        self.get_logger().debug(
            f'Received task | ID: {msg.task_id} | Type: {msg.task_type} | '
            f'Target Node: {msg.target_node_id} | SubTasks: {subtasks_summary}'
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
            f'Task callback | id={msg.task_id} type={msg.task_type} | '
            f'is_new_task={is_new_task} is_new_subtask={is_new_subtask} | '
            f'last_task_id={self.last_handled_task_id} '
            f'last_subtask_type={self.last_handled_subtask_type}'
        )

        if is_new_task:
            self.get_logger().info(f'New Task: {msg.description} (ID: {msg.task_id})')
            self.current_sub_task_index = 0
            self.last_handled_subtask_type = None
        elif is_new_subtask:
            self.get_logger().info(f'New Subtask for task ID: {msg.task_id}')
            self.current_sub_task_index = 0
            if self.current_status == RobotStatusEnum.JOB_DONE:
                self._transition_status(RobotStatusEnum.IDLE)

        if self.last_handled_task_id is None:
            self.current_load_status = self.task.crop_load

        self.task = msg

    # =========================================================================
    # TASK TRIGGER
    # =========================================================================

    def _send_task_trigger(self):
        """Publish a trigger to the server to generate and store the next task."""
        msg = String()
        msg.data = self.namespace
        self.task_trigger_pub.publish(msg)
        self.get_logger().info(f'Task trigger published to {self.task_trigger_topic}')

    # =========================================================================
    # MAIN TIMER CALLBACK
    # =========================================================================

    def timer_callback(self):
        """
        1 Hz control loop.

        Routes to error recovery, startup undocking, or task execution.
        Publishes RobotStatus at the end of every tick.
        """
        robot_status = RobotStatus()
        robot_status.header.stamp = self.get_clock().now().to_msg()
        robot_status.robot_namespace = self.namespace.replace('/', '')

        self._set_battery_status(robot_status)
        self._set_estop_status(robot_status)
        self._set_location_status(robot_status)

        self.get_logger().debug(
            f'Tick | status={self.current_status.name} | '
            f'startup_undock_complete={self.startup_undock_complete} | '
            f'reverse_drive_active={self.reverse_drive_active}'
        )

        if self.current_status in [RobotStatusEnum.ERROR, RobotStatusEnum.ABNORMAL]:
            self._handle_error_recovery()
        elif not self.startup_undock_complete:
            self._handle_startup_undocking(robot_status)
        else:
            self._handle_task_execution(robot_status)

        robot_status.status = self.current_status.value
        robot_status.current_node_id = self.current_node_id
        robot_status.load_status = self.current_load_status
        self.robot_state_pub.publish(robot_status)

    # =========================================================================
    # STARTUP — INITIAL POSITION CHECK
    # =========================================================================

    def _initial_position_check_timer(self):
        """Fire once after initial_position_check_delay; wait for pose then check."""
        if self.pose_status is None:
            self.get_logger().warning('Waiting for pose data...')
            return
        self.init_check_timer.cancel()
        self._check_initial_position()

    def _check_initial_position(self):
        """
        Find nearest dock by Euclidean distance, build ReverseDriveClient,
        and decide whether startup undocking is needed.
        """
        if self.is_initialized:
            return

        if not self.pose_status or not self.pose_status.pose:
            self.get_logger().warning('No pose data — retrying in 1s')
            time.sleep(1.0)
            self._check_initial_position()
            return

        pos = self.pose_status.pose.pose.position

        # Nearest dock by Euclidean distance (not dock_names[0])
        nearest_dock = min(
            self.dock_configs.values(),
            key=lambda d: math.sqrt((pos.x - d.pose.x) ** 2 + (pos.y - d.pose.y) ** 2),
        )
        dist = math.sqrt((pos.x - nearest_dock.pose.x) ** 2 + (pos.y - nearest_dock.pose.y) ** 2)

        self.active_dock = nearest_dock

        # Build ReverseDriveClient with nearest dock at index 0
        ordered_names = [nearest_dock.instance_name] + [n for n in self.dock_configs if n != nearest_dock.instance_name]
        self.reverse_drive_client = ReverseDriveClient(
            self,
            ReverseDriveConfig(
                dock_names=ordered_names,
                dock_configs=self.dock_configs,
                plugin_name=self.plugin_name,
                staging_x_offset=self.staging_x_offset,
                staging_yaw_offset=self.staging_yaw_offset,
                base_frame=self.base_frame,
                controller_frequency=self.controller_frequency,
                v_linear_min=self.v_linear_min,
                v_angular_max=self.v_angular_max,
                linear_tolerance=self.linear_tolerance,
                angular_tolerance=self.angular_tolerance,
                dock_backwards=self.dock_backwards,
            ),
        )

        self.get_logger().info(
            f'Position check | robot=({pos.x:.3f}, {pos.y:.3f}) | '
            f"nearest='{nearest_dock.instance_name}' "
            f'({nearest_dock.pose.x:.3f}, {nearest_dock.pose.y:.3f}) | '
            f'dist={dist:.3f}m | threshold={self.docking_threshold}m'
        )

        if dist <= self.docking_threshold:
            self.is_at_docking_station = True
            self.startup_undock_complete = False
            self.get_logger().info('Robot AT dock — startup undocking required')
        else:
            self.is_at_docking_station = False
            self.startup_undock_complete = True
            self.get_logger().info('Robot NOT at dock — skipping startup undocking')
            self._send_task_trigger()

        self.is_initialized = True

    # =========================================================================
    # STARTUP — UNDOCKING STATE MACHINE
    # =========================================================================

    def _handle_startup_undocking(self, robot_status: RobotStatus):
        """
        Drive startup undocking state machine (called while startup_undock_complete=False).

        Tick 1 (IDLE):            → START_UNDOCKING
        Tick 2 (START_UNDOCKING): call _handle_manipulator if arm_stow_pending,
                                  then call _subtask_undocking (which has STOW gate)
        Tick 3+ (else):           poll undocking or reverse drive
        """
        if not self.is_initialized or self.startup_undock_complete:
            return

        self.get_logger().debug(
            f'Startup undocking | current_status={self.current_status.name} | '
            f'reverse_drive_active={self.reverse_drive_active}'
        )

        if self.current_status == RobotStatusEnum.IDLE:
            self.get_logger().info('Starting startup undocking...')
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            robot_status.task = 'Startup: Preparing to undock'

        elif self.current_status == RobotStatusEnum.START_UNDOCKING:
            robot_status.task = 'Startup: Undocking'

            dock_type = self.active_dock.type
            staging_x_offset = self.staging_x_offset
            v_linear = self.v_linear_max
            max_undocking_time = (abs(staging_x_offset) / max(v_linear, 0.01)) * 1.25

            self.get_logger().debug(
                f"Startup UndockGoal | dock_type='{dock_type}' | "
                f'staging_x_offset={staging_x_offset} | '
                f'max_undocking_time={max_undocking_time:.1f}s'
            )

            startup_subtask = SubTask()
            startup_subtask.type = SubTask.UNDOCKING
            startup_subtask.description = 'Startup Undocking'
            startup_subtask.undock_goal = UndockGoal(
                dock_type=dock_type,
                max_undocking_time=max_undocking_time,
            )
            self.last_undocking_subtask = startup_subtask

            # Poll manipulator during startup so the STOW gate can be cleared
            if self.arm_stow_pending:
                self._handle_manipulator(robot_status)

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
        Main task loop — called from timer_callback when startup is complete and
        status is not ERROR/ABNORMAL.

        Each tick:
          1. Validate task — go IDLE if absent
          2. Battery check — interrupt on low battery
          3. Process action clients (nav/dock/undock/reverse-drive/arm)
          4. Refresh current_sub_task
          5. Route: IDLE/JOB_DONE → _handle_task_start; else → _execute_current_subtask
        """
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
        Interrupt task on low battery. Returns True when caller should skip the tick.

        Non-CHARGING task with low battery → cancel navigation → ERROR
        ERROR with CHARGING_TASK → clear error, IDLE (bypass _handle_error_recovery)
        """
        if not self.task or not self.current_task:
            return False

        battery_pct = self._normalize_battery(self.battery_status.percentage)

        if self.task.task_type != Task.CHARGING_TASK and battery_pct <= self.battery_low_threshold:
            self.get_logger().warning(
                f'Battery low: {battery_pct:.1f}% — '
                f'threshold={self.battery_low_threshold}% | '
                f'task_type={self.task.task_type} | status={self.current_status.name}'
            )
            if self.current_status in [RobotStatusEnum.START_MOVING, RobotStatusEnum.MOVING]:
                self.navigation.cancel_goal()
            self._transition_status(RobotStatusEnum.ERROR)
            return True

        if self.current_status == RobotStatusEnum.ERROR and self.current_task.task_type == Task.CHARGING_TASK:
            self.get_logger().info(f'Recovering from low battery — CHARGING_TASK received | battery={battery_pct:.1f}%')
            self._transition_status(RobotStatusEnum.IDLE)

        return False

    def _process_action_clients(self, robot_status: RobotStatus):
        """
        Poll all active action clients each tick.

        Priority: navigation → docking → undocking → reverse drive
        Arm manipulator is polled independently (runs in parallel with motion).
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

        # Arm polled independently — its completion unblocks harvesting and undocking flows
        arm_harvest_active = (
            self.current_status == RobotStatusEnum.HARVESTING
            and not self.arm_stow_pending
            and not self.arm_ready_pending
        )
        if self.arm_stow_pending or self.arm_ready_pending or arm_harvest_active:
            self._handle_manipulator(robot_status)

    def _update_current_subtask(self):
        """Refresh current_sub_task from sub_tasks[current_sub_task_index]."""
        if not self.current_task:
            return

        if isinstance(self.current_task.sub_tasks, list):
            if self.current_sub_task_index < len(self.current_task.sub_tasks):
                self.current_sub_task = self.current_task.sub_tasks[self.current_sub_task_index]
            else:
                self.current_sub_task = None

        self.get_logger().debug(
            f'Current subtask | index={self.current_sub_task_index} | '
            f'type={self.current_sub_task.type if self.current_sub_task else "None"} | '
            f"desc='{self.current_sub_task.description if self.current_sub_task else 'None'}'"
        )

    def _handle_task_start(self):
        """
        Initialise a new task or clear a completed one.

        New task (id/type changed):
          - Update _need_row_navigation based on last task type
          - Cache new task identifiers, clear undocking state, → JOB_START
          - Send task trigger after CHARGING/UNLOADING task completes (return to row)

        Same task, JOB_DONE:
          - If HARVESTING_TASK: _need_row_navigation = False (resume row on next subtask)
          - → IDLE, clear current_task, send trigger for next task
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
                f'last_task_type={self.last_handled_task_type}'
            )

            # Update _need_row_navigation: returning from dock requires Nav2 back to row
            if self.last_handled_task_type in (Task.CHARGING_TASK, Task.UNLOADING_TASK):
                self._need_row_navigation = True
            # HARVESTING → HARVESTING: keep current value (set by _subtask_moving)

            self.last_handled_task_id = self.current_task.task_id
            self.last_handled_task_type = self.current_task.task_type
            self.last_handled_subtask_type = None
            self.last_undocking_subtask = None
            self.undocking_after_task_type = None
            self._transition_status(RobotStatusEnum.JOB_START)

        elif self.current_status == RobotStatusEnum.JOB_DONE:
            # Same task still publishing — consecutive subtask (e.g. next bush)
            if self.current_task.task_type == Task.HARVESTING_TASK:
                # Next subtask continues the row — DriveClient resumes, no Nav2
                self._need_row_navigation = False
            elif self.current_task.task_type in (Task.CHARGING_TASK, Task.UNLOADING_TASK):
                self._need_row_navigation = True

            self.get_logger().debug(
                f'JOB_DONE — same task still publishing | task_id={self.current_task.task_id} — transitioning to IDLE'
            )
            self._transition_status(RobotStatusEnum.IDLE)
            self.current_task = None
            self._send_task_trigger()

    def _execute_current_subtask(self):
        """Route to the correct subtask handler based on current_sub_task.type."""
        if not isinstance(self.current_sub_task, SubTask):
            self.get_logger().debug(f'_execute_current_subtask — no valid subtask | status={self.current_status.name}')
            return

        if self.current_sub_task.type != self.last_handled_subtask_type:
            self.get_logger().info(f'Executing: {self.current_sub_task.description}')
            self.last_handled_subtask_type = self.current_sub_task.type

        task_handler_map = {
            SubTask.MOVING: self._subtask_moving,
            SubTask.HARVESTING: self._subtask_harvesting,
            SubTask.DOCKING: self._subtask_docking,
            SubTask.CHARGING: self._subtask_charging,
            SubTask.UNLOADING: self._subtask_unloading,
            SubTask.UNDOCKING: self._subtask_undocking,
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
        Monitor NavigationActionClient.

        ACTIVE    → MOVING
        SUCCEEDED → reset, update node, → DESTINATION_REACHED
        ABORTED/ERROR → retry logic
        CANCELED  → reset, → IDLE
        """
        nav_status = self.navigation.get_navigation_status()
        wpf_status = self.navigation.get_current_status()

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
            f'retry_count={self.navigation_retry_count}/{self.navigation_max_retries}'
        )

        if nav_status == NavigationStatus.ACTIVE:
            self._transition_status(RobotStatusEnum.MOVING)

        elif nav_status == NavigationStatus.SUCCEEDED:
            self.get_logger().info('Navigation complete')
            self.navigation_retry_count = 0
            self.navigation.reset()
            self.current_node_id = wpf_status.target_node_id if wpf_status else self.current_node_id
            self._transition_status(RobotStatusEnum.DESTINATION_REACHED)

        elif nav_status in [NavigationStatus.ABORTED, NavigationStatus.ERROR]:
            self._handle_navigation_retry()

        elif nav_status == NavigationStatus.CANCELED:
            self.get_logger().info('Navigation canceled')
            self.navigation.reset()
            self.navigation_retry_count = 0
            self._transition_status(RobotStatusEnum.IDLE)

    def _handle_navigation_retry(self):
        """Retry NavigateThroughPoses up to navigation_max_retries, then ERROR."""
        self.get_logger().warning(
            f'Navigation failed | retry {self.navigation_retry_count + 1}/{self.navigation_max_retries}'
        )

        if self._is_robot_at_target():
            self.get_logger().info('Robot already at target — treating navigation as complete')
            wpf_status = self.navigation.get_current_status()
            self.navigation_retry_count = 0
            self.navigation.reset()
            self.current_node_id = wpf_status.target_node_id if wpf_status else self.current_node_id
            self._transition_status(RobotStatusEnum.DESTINATION_REACHED)
            return

        if self.navigation_retry_count < self.navigation_max_retries:
            self.navigation_retry_count += 1
            self.get_logger().info(
                f'Retrying navigation in {self.navigation_retry_delay:.1f}s | '
                f'attempt {self.navigation_retry_count}/{self.navigation_max_retries}'
            )
            time.sleep(self.navigation_retry_delay)
            self._retry_navigation()
        else:
            self.get_logger().error(f'Navigation failed after {self.navigation_max_retries} retries')
            self.navigation.reset()
            self._transition_status(RobotStatusEnum.ERROR)
            self.navigation_retry_count = 0

    def _retry_navigation(self):
        """Re-send the NavigateThroughPoses goal for the current task."""
        if not self.current_task:
            self.get_logger().error('Navigation retry failed — no current task')
            return

        nav_status = self.navigation.get_navigation_status()
        if nav_status in [NavigationStatus.ACTIVE, NavigationStatus.SENDING]:
            self.get_logger().warning(f'Navigation still active ({nav_status.name}) — skipping retry')
            return

        self.navigation.reset()
        time.sleep(self.navigation_retry_delay)

        if self.navigation.send_goal(self.current_task):
            self._transition_status(RobotStatusEnum.MOVING)
        else:
            self.get_logger().error('Navigation retry send_goal failed')
            self._transition_status(RobotStatusEnum.ERROR)

    def _handle_docking(self, robot_status: RobotStatus):
        """
        Monitor DockingActionClient.

        DOCKING      → DOCKING
        DONE_DOCKING → reset, → DONE_DOCKING
        ERROR        → retry
        """
        status = self.docking_action_client.get_status()
        feedback = self.docking_action_client.get_feedback()

        if feedback:
            robot_status.task = feedback.task
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
            self._transition_status(RobotStatusEnum.DONE_DOCKING)
            self.docking_action_client.reset()
            self.docking_retry_count = 0

        elif status == RobotStatusEnum.ERROR:
            self._handle_docking_retry()

    def _handle_docking_retry(self):
        """Retry dock_robot goal up to docking_max_retries, then ERROR."""
        self.get_logger().error(f'Docking failed | retry {self.docking_retry_count + 1}/{self.docking_max_retries}')

        if self.docking_retry_count < self.docking_max_retries:
            self.docking_retry_count += 1
            self.get_logger().info(
                f'Retrying docking in {self.docking_retry_delay:.1f}s | '
                f'attempt {self.docking_retry_count}/{self.docking_max_retries}'
            )
            time.sleep(self.docking_retry_delay)
            self._retry_docking()
        else:
            self.get_logger().error(f'Docking failed after {self.docking_max_retries} retries — ERROR')
            self.docking_action_client.reset()
            self._transition_status(RobotStatusEnum.ERROR)
            self.docking_retry_count = 0

    def _retry_docking(self):
        """Re-send the dock_robot goal using the current subtask's DockGoal."""
        if not self.current_sub_task:
            self.get_logger().error('Docking retry failed — no current subtask')
            return

        if self.docking_action_client.send_docking_goal(self.current_sub_task):
            self._transition_status(RobotStatusEnum.DOCKING)
        else:
            self.get_logger().error('Docking retry send_goal failed')
            self._transition_status(RobotStatusEnum.ERROR)

    def _handle_undocking(self, robot_status: RobotStatus):
        """
        Monitor UndockingActionClient.

        Context-aware DONE_UNDOCKING:
          startup context → startup_undock_complete=True, clear task IDs, → IDLE,
                            send task trigger (request first task)
          task context    → stay at DONE_UNDOCKING for _subtask_undocking → JOB_DONE
        """
        status = self.undocking_action_client.get_status()
        feedback = self.undocking_action_client.get_feedback()

        if feedback:
            robot_status.task = feedback.task
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
                self.get_logger().debug('Undocking DONE — startup context: → IDLE, sending task trigger')
                self.startup_undock_complete = True
                self.last_handled_task_id = None
                self.last_handled_task_type = None
                self._transition_status(RobotStatusEnum.IDLE)
                self.get_logger().info('Startup undocking complete — requesting first task')
                self._send_task_trigger()
            else:
                self.get_logger().debug(
                    'Undocking DONE — task context: staying at DONE_UNDOCKING for _subtask_undocking'
                )

        elif status == RobotStatusEnum.ERROR:
            self._handle_undocking_retry()

    def _handle_undocking_retry(self):
        """Undocking action failed — start ReverseDriveClient as fallback."""
        self.undocking_action_client.reset()
        self.get_logger().warning(
            f'Undocking failed — starting reverse drive | '
            f"dock='{self.active_dock.instance_name if self.active_dock else 'unknown'}'"
        )

        if self.reverse_drive_client and self.reverse_drive_client.drive_to_staging():
            self.reverse_drive_active = True
            self._transition_status(RobotStatusEnum.UNDOCKING)
        else:
            self.get_logger().error(f'ReverseDriveClient refused to start — dock_backwards={self.dock_backwards}')
            self._transition_status(RobotStatusEnum.ERROR)

    def _handle_reverse_drive(self, robot_status: RobotStatus):
        """
        Monitor ReverseDriveClient (fallback after undocking failure).

        Same context-aware logic as _handle_undocking for the DONE branch.
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
                self.get_logger().debug('Reverse drive DONE — startup context: → IDLE, sending task trigger')
                self.startup_undock_complete = True
                self.last_handled_task_id = None
                self.last_handled_task_type = None
                self._transition_status(RobotStatusEnum.DONE_UNDOCKING)
                self._transition_status(RobotStatusEnum.IDLE)
                self.get_logger().info('Startup undocking complete — requesting first task')
                self._send_task_trigger()
            else:
                self.get_logger().debug('Reverse drive DONE — task context: DONE_UNDOCKING for _subtask_undocking')
                self._transition_status(RobotStatusEnum.DONE_UNDOCKING)

        elif status == ReverseDriveStatus.ERROR:
            self.get_logger().error('Reverse drive failed — ERROR')
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()
            self._transition_status(RobotStatusEnum.ERROR)

        elif status == ReverseDriveStatus.CANCELED:
            self.get_logger().warning('Reverse drive canceled — IDLE')
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()
            self._transition_status(RobotStatusEnum.IDLE)

    def _handle_manipulator(self, robot_status: RobotStatus):
        """
        Monitor ManipulatorTaskActionClient for STOW, READY, or START_HARVEST goals.

        Verbatim from husky_operations_manager.py:1226-1337.

        Context-aware completion:
          STOW confirmed in START_UNDOCKING   → re-enter _subtask_undocking
          STOW confirmed in DONE_HARVESTING   → no-op (_subtask_harvesting picks up next tick)
          READY confirmed in DESTINATION_REACHED → no-op (_subtask_harvesting picks up next tick)
          Harvest complete (HARVESTING, no flags) → DONE_HARVESTING
          ERROR → clear flags, reset, → ERROR
        """
        arm_status = self.manipulator_client.get_status()

        self.get_logger().debug(
            f'Manipulator handler | arm_status={arm_status.name} | '
            f'current_status={self.current_status.name} | '
            f'arm_stow_pending={self.arm_stow_pending} | '
            f'arm_ready_pending={self.arm_ready_pending} | '
            f"last_confirmed_arm='{self.last_confirmed_arm_command}'"
        )

        # ---- STOW completion ----
        if self.arm_stow_pending and arm_status == RobotStatusEnum.DONE_HARVESTING:
            self.get_logger().info(f'Arm STOW confirmed | context={self.current_status.name}')
            self.arm_stow_pending = False
            self.last_confirmed_arm_command = ArmCommand.GO_STOW
            self.manipulator_client.reset()

            if self.current_status == RobotStatusEnum.START_UNDOCKING:
                self.get_logger().debug('STOW confirmed in START_UNDOCKING — re-entering _subtask_undocking')
                self._subtask_undocking()
            elif self.current_status == RobotStatusEnum.DONE_HARVESTING:
                self.get_logger().debug(
                    'STOW confirmed in DONE_HARVESTING — _subtask_harvesting will advance to JOB_DONE on next tick'
                )
            else:
                self.get_logger().warning(
                    f'STOW confirmed in unexpected context={self.current_status.name} — no action taken'
                )

        # ---- READY completion ----
        elif self.arm_ready_pending and arm_status == RobotStatusEnum.DONE_HARVESTING:
            self.get_logger().info(f'Arm READY confirmed | context={self.current_status.name}')
            self.arm_ready_pending = False
            self.last_confirmed_arm_command = ArmCommand.GO_READY
            self.manipulator_client.reset()

            if self.current_status == RobotStatusEnum.DESTINATION_REACHED:
                self.get_logger().debug(
                    'READY confirmed in DESTINATION_REACHED — '
                    '_subtask_harvesting will advance to START_HARVESTING on next tick'
                )
            else:
                self.get_logger().warning(
                    f'READY confirmed in unexpected context={self.current_status.name} — no action taken'
                )

        # ---- Harvest completion (START_HARVEST goal succeeded) ----
        elif (
            not self.arm_stow_pending
            and not self.arm_ready_pending
            and arm_status == RobotStatusEnum.DONE_HARVESTING
            and self.current_status == RobotStatusEnum.HARVESTING
        ):
            self.get_logger().info('Harvest goal complete — transitioning to DONE_HARVESTING')
            self.manipulator_client.reset()
            self._transition_status(RobotStatusEnum.DONE_HARVESTING)

        # ---- ERROR ----
        elif arm_status == RobotStatusEnum.ERROR:
            self.get_logger().error(
                f'Arm command failed | context={self.current_status.name} | '
                f'stow_pending={self.arm_stow_pending} | '
                f'ready_pending={self.arm_ready_pending}'
            )
            self.arm_stow_pending = False
            self.arm_ready_pending = False
            self.manipulator_client.reset()
            self._transition_status(RobotStatusEnum.ERROR)

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    def _handle_error_recovery(self):
        """Cancel navigation and arm goals, reset subtask state, → IDLE."""
        if self.current_status not in [RobotStatusEnum.ERROR, RobotStatusEnum.ABNORMAL]:
            return

        self.get_logger().warning(
            f'Error recovery | status={self.current_status.name} | '
            f'task_id={self.last_handled_task_id} | '
            f'subtask_type={self.last_handled_subtask_type}'
        )

        nav_status = self.navigation.get_navigation_status()
        if nav_status in [NavigationStatus.ACTIVE, NavigationStatus.ACCEPTED]:
            self.get_logger().info('Cancelling active navigation during error recovery')
            try:
                self.navigation.cancel_goal()
                time.sleep(self.navigation_retry_delay)
            except Exception as e:
                self.get_logger().warning(f'Navigation cancel raised exception: {e}')

        self.arm_stow_pending = False
        self.arm_ready_pending = False
        self.manipulator_client.reset()

        self.current_sub_task = None
        self.current_sub_task_index = 0
        self.get_logger().debug('Error recovery — reset subtask state, → IDLE')
        self._transition_status(RobotStatusEnum.IDLE)

    def _cancel_all_motion(self):
        """Cancel navigation, DriveClient, and manipulator. Clear arm pending flags."""
        nav_status = self.navigation.get_navigation_status()
        if nav_status in [NavigationStatus.ACTIVE, NavigationStatus.ACCEPTED]:
            self.navigation.cancel_goal()

        if self.drive_client.is_active():
            self.drive_client.cancel()

        if self.manipulator_client.get_status() == RobotStatusEnum.HARVESTING:
            self.manipulator_client.cancel_goal()

        self.arm_stow_pending = False
        self.arm_ready_pending = False

    # =========================================================================
    # SUBTASK HANDLERS
    # =========================================================================

    def _subtask_moving(self):
        """
        Dual-mode navigation subtask.

        JOB_START:
          HARVESTING and _need_row_navigation=False → DESTINATION_REACHED (skip Nav2)
          Otherwise → START_MOVING (Nav2 path)

        DESTINATION_REACHED:
          HARVESTING_TASK → start DriveClient, → MOVING
          CHARGING/UNLOADING → task interruption: cancel, trigger server, → IDLE

        MOVING (DriveClient active):
          STOPPED + detection → advance subtask, → DESTINATION_REACHED
          STOPPED + no detection → row end; _need_row_navigation=True, trigger, → JOB_DONE
          CANCELED/ERROR → ERROR
        """
        if not self.current_task:
            return

        task_type = self.current_task.task_type

        self.get_logger().debug(
            f'_subtask_moving | status={self.current_status.name} | '
            f'task_type={task_type} | '
            f'_need_row_navigation={self._need_row_navigation}'
        )

        if self.current_status == RobotStatusEnum.JOB_START:
            if task_type == Task.HARVESTING_TASK and not self._need_row_navigation:
                # Skip Nav2 — DriveClient resumes the current row
                self.get_logger().info(
                    'HARVESTING + _need_row_navigation=False — skipping Nav2, proceeding directly to DriveClient'
                )
                self._transition_status(RobotStatusEnum.DESTINATION_REACHED)
            else:
                self._transition_status(RobotStatusEnum.START_MOVING)

        elif self.current_status == RobotStatusEnum.START_MOVING:
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
                self.get_logger().error(f'Failed to send navigation goal | task_id={self.current_task.task_id}')
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            if task_type == Task.HARVESTING_TASK:
                # --- Start DriveClient for row traversal ---
                self._need_row_navigation = False
                drive_status = self.drive_client.get_status()
                if drive_status == DriveStatus.STOPPED:
                    self.get_logger().info('Resuming DriveClient from STOPPED (depart from bush)')
                    self.drive_client.resume()
                else:
                    self.get_logger().info('Starting DriveClient scan (fresh row start)')
                    self.drive_client.scan()
                self._detection_received = False
                self.last_detection_time = self.get_clock().now().nanoseconds / 1e9
                self._transition_status(RobotStatusEnum.MOVING)

            else:
                # --- CHARGING or UNLOADING: task interruption ---
                self.get_logger().info(
                    f'DESTINATION_REACHED for task_type={task_type} (CHARGING/UNLOADING) — task interruption'
                )
                self._cancel_all_motion()
                self._send_task_trigger()
                # Reset so server's response is treated as a fresh task
                self.last_handled_task_id = None
                self.last_handled_task_type = None
                self.current_task = None
                self.current_sub_task = None
                self.current_sub_task_index = 0
                self._transition_status(RobotStatusEnum.IDLE)

        elif self.current_status == RobotStatusEnum.MOVING:
            # DriveClient is active — check its status
            drive_status = self.drive_client.get_status()

            self.get_logger().debug(
                f'DriveClient | drive_status={drive_status.name} | _detection_received={self._detection_received}'
            )

            if drive_status in (DriveStatus.SCANNING, DriveStatus.DEPARTING):
                pass  # Still moving — wait

            elif drive_status == DriveStatus.STOPPED:
                if self._detection_received:
                    # Bush detected — advance to next subtask (HARVESTING)
                    self.get_logger().info('Drive stopped at bush — advancing to HARVESTING subtask')
                    self.current_sub_task_index += 1
                    self._transition_status(RobotStatusEnum.DESTINATION_REACHED)
                else:
                    # No detection — row end
                    self.get_logger().info('Drive stopped without detection — row end, requesting next task')
                    self._need_row_navigation = True
                    self.drive_client.reset()
                    self._send_task_trigger()
                    self._transition_status(RobotStatusEnum.JOB_DONE)

            elif drive_status in (DriveStatus.CANCELED, DriveStatus.ERROR):
                self.get_logger().error(f'DriveClient error/canceled | drive_status={drive_status.name}')
                self._transition_status(RobotStatusEnum.ERROR)

    def _subtask_docking(self):
        """
        Handle the DOCKING subtask.

        DESTINATION_REACHED → START_DOCKING → 1s delay → send dock_robot → DOCKING
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
            self.get_logger().info('Started docking')
            time.sleep(1.0)
            if self.docking_action_client.send_docking_goal(self.current_sub_task):
                self._transition_status(RobotStatusEnum.DOCKING)
            else:
                self.get_logger().error(
                    f'Failed to send docking goal | '
                    f"subtask='{self.current_sub_task.description if self.current_sub_task else 'None'}'"
                )
                self._transition_status(RobotStatusEnum.ERROR)

    def _subtask_undocking(self):
        """
        Handle the UNDOCKING subtask (called from charging, unloading, startup, or standalone).

        Arm safety gate at START_UNDOCKING:
          Arm must be confirmed STOW before the undocking goal is sent.
          If not: send STOW goal, arm_stow_pending=True, return.
          _handle_manipulator re-enters this method once STOW is confirmed.

        START_UNDOCKING → (STOW gate) → send undock_robot → UNDOCKING
        DONE_UNDOCKING  → clear state → JOB_DONE
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
            if self.last_confirmed_arm_command != ArmCommand.GO_STOW:
                if not self.arm_stow_pending:
                    self.get_logger().info(
                        f"Arm not in STOW (last='{self.last_confirmed_arm_command}') — sending STOW before undocking"
                    )
                    undock_ref = self.current_sub_task or self.last_undocking_subtask
                    if self.manipulator_client.send_stow_goal(undock_ref):
                        self.arm_stow_pending = True
                    else:
                        self.get_logger().error('Failed to send STOW goal before undocking — ERROR')
                        self._transition_status(RobotStatusEnum.ERROR)
                else:
                    self.get_logger().debug('Arm STOW already in progress — waiting before undocking')
                return  # Hold at START_UNDOCKING until STOW confirmed

            # ---- Arm confirmed STOW — proceed ----
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

            if self.undocking_action_client.send_undocking_goal(undock_subtask):
                self._transition_status(RobotStatusEnum.UNDOCKING)
            else:
                self.get_logger().error(
                    f'Failed to send undocking goal | undocking_after_task_type={self.undocking_after_task_type}'
                )
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info('Undocking done')
            self.last_undocking_subtask = None
            self.undocking_after_task_type = None
            self._transition_status(RobotStatusEnum.JOB_DONE)

    def _subtask_harvesting(self):
        """
        Handle the HARVESTING subtask using ManipulatorTaskActionClient.

        DESTINATION_REACHED → READY gate → START_HARVESTING → HARVESTING
          → (via _handle_manipulator) → DONE_HARVESTING
          → STOW gate → JOB_DONE

        Arm gates mirror husky_operations_manager.py exactly.
        """
        self.get_logger().debug(
            f'_subtask_harvesting | status={self.current_status.name} | '
            f'arm_ready_pending={self.arm_ready_pending} | '
            f'arm_stow_pending={self.arm_stow_pending} | '
            f"last_confirmed_arm='{self.last_confirmed_arm_command}'"
        )

        if self.current_status == RobotStatusEnum.DESTINATION_REACHED:
            # ---- READY gate ----
            if self.last_confirmed_arm_command != ArmCommand.GO_READY:
                if not self.arm_ready_pending:
                    self.get_logger().info(
                        f"Arm not in READY (last='{self.last_confirmed_arm_command}') — sending READY before harvesting"
                    )
                    if self.manipulator_client.send_ready_goal(self.current_sub_task):
                        self.arm_ready_pending = True
                    else:
                        self.get_logger().error('Failed to send READY goal before harvesting — ERROR')
                        self._transition_status(RobotStatusEnum.ERROR)
                else:
                    self.get_logger().debug('Arm READY already in progress — waiting')
                return  # Hold at DESTINATION_REACHED until READY confirmed

            # Arm confirmed READY — proceed to harvesting
            self._transition_status(RobotStatusEnum.START_HARVESTING)

        elif self.current_status == RobotStatusEnum.START_HARVESTING:
            if self.manipulator_client.get_status() == RobotStatusEnum.HARVESTING:
                self.get_logger().warning('Harvest goal send skipped — manipulator already active')
                return
            self.get_logger().info('Sending harvest goal to manipulator')
            if self.manipulator_client.send_harvesting_goal(self.current_sub_task):
                self._transition_status(RobotStatusEnum.HARVESTING)
            else:
                self.get_logger().error('Failed to send harvest goal — ERROR')
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.HARVESTING:
            # Result handled asynchronously by _handle_manipulator → DONE_HARVESTING
            self.get_logger().debug('Harvesting in progress — waiting for manipulator result')

        elif self.current_status == RobotStatusEnum.DONE_HARVESTING:
            # ---- STOW gate (guarded with arm_stow_pending to prevent re-increment) ----
            if not self.arm_stow_pending and self.last_confirmed_arm_command != ArmCommand.GO_STOW:
                new_load = min(self.current_load_status + self.loading_increment, 100.0)
                self.get_logger().debug(
                    f'Load update | {self.current_load_status:.1f}% → {new_load:.1f}% (+{self.loading_increment:.1f}%)'
                )
                self.current_load_status = new_load
                self.get_logger().info(f'Load status: {self.current_load_status:.1f}%')

                self.get_logger().info('Harvesting done — sending arm to STOW')
                if self.manipulator_client.send_stow_goal(self.current_sub_task):
                    self.arm_stow_pending = True
                else:
                    self.get_logger().error('Failed to send STOW goal after harvesting — ERROR')
                    self._transition_status(RobotStatusEnum.ERROR)
                return  # Hold until STOW confirmed

            # STOW confirmed — advance to JOB_DONE
            if self.last_confirmed_arm_command == ArmCommand.GO_STOW and not self.arm_stow_pending:
                self.get_logger().info('Arm stowed after harvest — transitioning to JOB_DONE')
                self._transition_status(RobotStatusEnum.JOB_DONE)

    def _subtask_charging(self):
        """
        Handle the CHARGING subtask.

        DONE_DOCKING → START_CHARGING → CHARGING (poll battery) → DONE_CHARGING
        → store undocking subtask → START_UNDOCKING → _subtask_undocking → JOB_DONE
        """
        battery_pct = self._normalize_battery(self.battery_status.percentage)
        self.get_logger().debug(
            f'_subtask_charging | status={self.current_status.name} | '
            f'battery={battery_pct:.1f}% | full_threshold={self.battery_full_threshold}%'
        )

        if self.current_status == RobotStatusEnum.DONE_DOCKING:
            self._transition_status(RobotStatusEnum.START_CHARGING)

        elif self.current_status == RobotStatusEnum.START_CHARGING:
            self._transition_status(RobotStatusEnum.CHARGING)
            self.get_logger().info('Charging started')

        elif self.current_status == RobotStatusEnum.CHARGING:
            self.get_logger().info(f'Battery charging: {battery_pct:.1f}%', throttle_duration_sec=10.0)
            if battery_pct >= self.battery_full_threshold:
                self.get_logger().info(f'Battery charged: {battery_pct:.1f}%')
                self._transition_status(RobotStatusEnum.DONE_CHARGING)

        elif self.current_status == RobotStatusEnum.DONE_CHARGING:
            self.get_logger().debug('DONE_CHARGING — storing last_undocking_subtask and triggering undocking')
            self.last_undocking_subtask = self.current_sub_task
            self.undocking_after_task_type = Task.CHARGING_TASK
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            self._subtask_undocking()

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().debug('DONE_UNDOCKING in charging context — delegating to _subtask_undocking')
            self._subtask_undocking()

    def _subtask_unloading(self):
        """
        Handle the UNLOADING subtask via UnloaderActionClient.

        DONE_DOCKING  → START_UNLOADING → send_goal(END) → UNLOADING
        UNLOADING     → poll unloader status:
                         DONE_UNLOADING + not _unloader_at_end
                           → _unloader_at_end=True, delay, send_goal(HOME)
                         DONE_UNLOADING + _unloader_at_end
                           → current_load_status=0, → DONE_UNLOADING
        DONE_UNLOADING → store undocking subtask → START_UNDOCKING → JOB_DONE
        DONE_UNDOCKING → delegate to _subtask_undocking → JOB_DONE
        """
        self.get_logger().debug(
            f'_subtask_unloading | status={self.current_status.name} | '
            f'current_load={self.current_load_status:.1f}% | '
            f'_unloader_at_end={self._unloader_at_end}'
        )

        if self.current_status == RobotStatusEnum.DONE_DOCKING:
            self._transition_status(RobotStatusEnum.START_UNLOADING)

        elif self.current_status == RobotStatusEnum.START_UNLOADING:
            self.get_logger().info('Sending unloader END goal')
            self._unloader_at_end = False
            if self.unloader_action_client.send_goal(OperateUnloader.Goal.END):
                self._transition_status(RobotStatusEnum.UNLOADING)
            else:
                self.get_logger().error('Failed to send unloader END goal — ERROR')
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.UNLOADING:
            unloader_status = self.unloader_action_client.get_status()

            if unloader_status == RobotStatusEnum.DONE_UNLOADING:
                if not self._unloader_at_end:
                    self.get_logger().info(f'Unloader AT_END — delaying {self.unloading_home_delay_s:.1f}s then HOME')
                    self._unloader_at_end = True
                    time.sleep(self.unloading_home_delay_s)
                    if not self.unloader_action_client.send_goal(OperateUnloader.Goal.HOME):
                        self.get_logger().error('Failed to send unloader HOME goal — ERROR')
                        self._transition_status(RobotStatusEnum.ERROR)
                    # Status will return to UNLOADING on next tick (client resets internally)
                else:
                    # HOME confirmed — carriage returned
                    self.get_logger().info('Unloader AT_HOME — unloading complete')
                    self.current_load_status = 0.0
                    self.unloader_action_client.reset()
                    self._transition_status(RobotStatusEnum.DONE_UNLOADING)

            elif unloader_status == RobotStatusEnum.ERROR:
                self.get_logger().error('Unloader reported ERROR — transitioning to ERROR')
                self.unloader_action_client.reset()
                self._transition_status(RobotStatusEnum.ERROR)

        elif self.current_status == RobotStatusEnum.DONE_UNLOADING:
            self.get_logger().debug('DONE_UNLOADING — storing last_undocking_subtask and triggering undocking')
            self.last_undocking_subtask = self.current_sub_task
            self.undocking_after_task_type = Task.UNLOADING_TASK
            self.get_logger().info('Unloading done, starting undocking')
            self._transition_status(RobotStatusEnum.START_UNDOCKING)
            self._subtask_undocking()

        elif self.current_status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().debug('DONE_UNDOCKING in unloading context — delegating to _subtask_undocking')
            self._subtask_undocking()

    # =========================================================================
    # SENSOR STATUS HELPERS
    # =========================================================================

    def _set_battery_status(self, robot_status: RobotStatus):
        robot_status.battery_level = self.battery_status.percentage
        if self.battery_status.capacity > 0.0 and self.battery_status.current > 0.0:
            battery_pct = self._normalize_battery(self.battery_status.percentage)
            time_remaining = self.battery_status.capacity * (battery_pct / 100.0) / self.battery_status.current
            robot_status.operation_hours_after_charging = self._format_time_remaining(time_remaining)
        else:
            robot_status.operation_hours_after_charging = '00 hours 00 minutes remaining approx...'

    def _set_estop_status(self, robot_status: RobotStatus):
        robot_status.online_flag = self.estop_status.data if self.estop_status.data else OnlineFlagEnum.ONLINE.value

    def _set_location_status(self, robot_status: RobotStatus):
        if self.pose_status and self.pose_status.pose:
            robot_status.topo_map_position = self.pose_status.pose.pose.position
            robot_status.topo_map_orientation = self.pose_status.pose.pose.orientation

    # =========================================================================
    # UTILITY
    # =========================================================================

    def _transition_status(self, new_status: RobotStatusEnum):
        """No-op if already in new_status; otherwise log and update."""
        if self.current_status != new_status:
            self.previous_status = self.current_status
            self.current_status = new_status
            self.get_logger().info(f'Status: {self.previous_status.name} → {self.current_status.name}')

    def _normalize_battery(self, percentage: float) -> float:
        """Normalise 0-1 BMS percentage to 0-100."""
        return percentage * 100.0 if percentage <= 1.0 else percentage

    def _format_time_remaining(self, hours: float) -> str:
        seconds = int(hours * 3600)
        minutes, seconds = divmod(seconds, 60)
        hours_i, minutes = divmod(minutes, 60)
        return f'{hours_i:02} hours {minutes:02} minutes remaining approximately.'

    def _calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _is_robot_at_target(self) -> bool:
        """Return True if the robot is within 0.25 m of the final subtask waypoint."""
        if not (self.current_sub_task and self.pose_status):
            return False

        robot_pos = self.pose_status.pose.pose.position
        waypoints_list = [WayPoint(**wp) if isinstance(wp, dict) else wp for wp in self.current_sub_task.data]
        if not waypoints_list:
            return False
        target_wp = waypoints_list[-1]
        distance = self._calculate_distance(robot_pos.x, robot_pos.y, target_wp.x, target_wp.y)

        self.get_logger().debug(
            f'_is_robot_at_target | '
            f'robot=({robot_pos.x:.3f}, {robot_pos.y:.3f}) | '
            f'target=({target_wp.x:.3f}, {target_wp.y:.3f}) | '
            f'distance={distance:.3f}m threshold=0.25m'
        )
        return distance <= 0.25


# =============================================================================
# ENTRY POINT
# =============================================================================


def main(args=None):
    rclpy.init(args=args)
    node = LavenderHarvestNode()
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
