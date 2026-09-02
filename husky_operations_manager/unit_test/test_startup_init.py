"""
Isolated test node for the HuskyOperationsManager startup-init sequence.

Dock poses and motion parameters are hardcoded below — no YAML file, no
DockingParamFetcher, no docking_server service call.

Startup sequence:
  1. Subscribe to ground-truth pose
  2. Find nearest dock by Euclidean distance → decide whether startup undocking is needed
  3. If at dock: run IDLE → START_UNDOCKING → UNDOCKING → DONE_UNDOCKING
     (ReverseDriveClient fallback if undocking action fails)
  4. Shut down cleanly

Usage:
    ros2 run husky_operations_manager test_startup \
        --ros-args -r __ns:=/a300_00036 -r /tf:=tf -r /tf_static:=tf_static
"""

import math
import time

import rclpy
import tf2_ros
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import TransformException

from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.action_clients.navigation import NavigationActionClient
from husky_operations_manager.action_clients.reverse_drive import ReverseDriveClient
from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.robot_enums import ReverseDriveStatus, RobotStatusEnum
from husky_operations_manager.types import DockInstanceConfig, DockPose, ReverseDriveConfig
from status_interfaces.msg import SubTask, UndockGoal

# =============================================================================
# GLOBAL CONFIG — edit here instead of a YAML file
# =============================================================================

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

DOCKING_THRESHOLD = 1.0  # metres — distance to consider robot "at dock"
TIMER_PERIOD = 1.0  # seconds
INIT_CHECK_DELAY = 2.0  # seconds


# =============================================================================
# Node
# =============================================================================


class StartupInitTestNode(Node):
    def __init__(self):
        super().__init__('test_startup_init_node')

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'StartupInitTestNode | namespace={self.namespace}')

        self._init_state()

        # Dock poses are declared in their own frame (`DockInstanceConfig.frame`),
        # which need not be the odometry frame — TF closes the gap.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.pose_status: Odometry | None = None
        self._pose_sub = self.create_subscription(
            Odometry,
            f'{self.namespace}/platform/odom/filtered',
            lambda msg: setattr(self, 'pose_status', msg),
            10,
        )

        # active_dock is set after nearest-dock selection.
        self.active_dock: DockInstanceConfig | None = None

        self.navigation = NavigationActionClient(self)
        self.docking_action_client = DockingActionClient(self)
        self.undocking_action_client = UndockingActionClient(self)

        # Dock identity is per-run inside the client — drive_to_staging(dock_name)
        # picks one and reset() clears it — so nothing here waits on the
        # nearest-dock decision. Built before spin starts rather than from a
        # timer callback, where it would add two publishers and a 50 Hz timer
        # to a running executor.
        self.reverse_drive_client = ReverseDriveClient(self, ReverseDriveConfig(**MOTION_CONFIG))

        self.get_logger().info(
            f'Clients ready | docks={list(DOCK_CONFIGS.keys())} | '
            f'staging_x_offset={MOTION_CONFIG["staging_x_offset"]} | '
            f'dock_backwards={MOTION_CONFIG["dock_backwards"]}'
        )

        self._init_check_timer = self.create_timer(INIT_CHECK_DELAY, self._initial_position_check_timer)
        self._main_timer = self.create_timer(TIMER_PERIOD, self._timer_callback)

    # =========================================================================
    # STATE
    # =========================================================================

    def _init_state(self):
        self.is_initialized: bool = False
        self.is_at_docking_station: bool = False
        self.startup_undock_complete: bool = False
        self.reverse_drive_active: bool = False
        self.current_status: RobotStatusEnum = RobotStatusEnum.IDLE
        self.previous_status: RobotStatusEnum = RobotStatusEnum.IDLE
        self.last_undocking_subtask: SubTask | None = None
        # Wall-clock bound on the undocking action. None outside that phase.
        self._undock_deadline: float | None = None
        # Set once a cancel has been requested, so the next tick can finish and
        # the executor gets one period to actually transmit the cancel.
        self._abort_pending: bool = False
        # _finish() tears the node down from inside a callback, so anything
        # running after it must not touch a timer or handle again.
        self._shutdown_requested: bool = False

    # =========================================================================
    # PHASE 1 — INITIAL POSITION CHECK
    # =========================================================================

    def _initial_position_check_timer(self):
        if self.pose_status is None or not self.pose_status.pose:
            self.get_logger().warning('Waiting for pose data...', throttle_duration_sec=2.0)
            return
        if not self._check_initial_position():
            return  # TF not ready yet — retry on the next tick
        if self._shutdown_requested:
            return  # _check_initial_position already finished — handle is gone
        self._init_check_timer.cancel()

    def _robot_xy_in_frame(self, target_frame: str) -> tuple[float, float] | None:
        """
        Robot position expressed in `target_frame`.

        Read straight off the odometry when it is already stamped in that frame,
        transformed through TF otherwise. Returns None while TF cannot resolve
        the pair, so the caller can retry instead of comparing across frames.
        """
        pose = self.pose_status.pose.pose
        pose_frame = self.pose_status.header.frame_id.lstrip('/')
        target = target_frame.lstrip('/')

        if pose_frame == target:
            return pose.position.x, pose.position.y

        try:
            # Time() asks for the latest available transform. The odometry stamp
            # would be rejected as an extrapolation while map->odom is still
            # being filled in at boot.
            tf = self._tf_buffer.lookup_transform(target, pose_frame, Time(), timeout=Duration(seconds=0.2))
        except TransformException as e:
            self.get_logger().warning(
                f"TF lookup '{pose_frame}' -> '{target}' failed: {e}", throttle_duration_sec=2.0
            )
            return None

        transformed = do_transform_pose(pose, tf)
        return transformed.position.x, transformed.position.y

    def _check_initial_position(self) -> bool:
        """Pick the nearest dock. False means TF was not ready — call again later."""
        if self.is_initialized:
            return True

        # Docks may not all declare the same frame, so resolve the robot once
        # per distinct frame and compare each dock inside its own frame.
        # Every dock must resolve before deciding: picking a "nearest" dock from
        # a partial set would silently select the wrong one at boot.
        robot_xy_by_frame: dict[str, tuple[float, float]] = {}
        candidates: list[tuple[DockInstanceConfig, float]] = []
        for dock in DOCK_CONFIGS.values():
            frame = dock.frame.lstrip('/')
            if frame not in robot_xy_by_frame:
                robot_xy = self._robot_xy_in_frame(frame)
                if robot_xy is None:
                    self.get_logger().warning(
                        f"Dock '{dock.instance_name}' frame '{frame}' unresolved — deferring position check",
                        throttle_duration_sec=2.0,
                    )
                    return False
                robot_xy_by_frame[frame] = robot_xy
            robot_x, robot_y = robot_xy_by_frame[frame]
            candidates.append((dock, math.hypot(robot_x - dock.pose.x, robot_y - dock.pose.y)))

        nearest_dock, dist = min(candidates, key=lambda candidate: candidate[1])
        robot_x, robot_y = robot_xy_by_frame[nearest_dock.frame.lstrip('/')]

        self.active_dock = nearest_dock

        dock = self.active_dock
        self.get_logger().info(
            f"Position check | robot=({robot_x:.3f}, {robot_y:.3f}) in '{dock.frame}' "
            f"(odom stamped '{self.pose_status.header.frame_id}') | "
            f"nearest='{dock.instance_name}' ({dock.pose.x:.3f}, {dock.pose.y:.3f}) | "
            f'dist={dist:.3f}m | threshold={DOCKING_THRESHOLD}m'
        )

        if dist <= DOCKING_THRESHOLD:
            self.is_at_docking_station = True
            self.startup_undock_complete = False
            self.get_logger().info('Robot AT dock — startup undocking required')
        else:
            self.is_at_docking_station = False
            self.startup_undock_complete = True
            self.get_logger().info('Robot NOT at dock — no startup undocking needed')
            self._finish('Startup init complete — robot ready for tasks')

        self.is_initialized = True
        return True

    # =========================================================================
    # PHASE 2 — STARTUP UNDOCKING STATE MACHINE (1 Hz)
    # =========================================================================

    def _timer_callback(self):
        if not self.is_initialized or self.startup_undock_complete:
            return

        self.get_logger().debug(
            f'Tick | status={self.current_status.name} | reverse_drive_active={self.reverse_drive_active}'
        )

        if self._abort_pending:
            # Cancel was requested last tick and has had a period to go out.
            self._finish('Startup undocking FAILED — undock deadline exceeded')
            return

        if self.current_status == RobotStatusEnum.IDLE:
            self.get_logger().info('Starting startup undocking')
            self._transition(RobotStatusEnum.START_UNDOCKING)

        elif self.current_status == RobotStatusEnum.START_UNDOCKING:
            self._prepare_and_send_undocking_goal()

        else:
            if self._undock_deadline is not None and time.monotonic() >= self._undock_deadline:
                self._abort_undocking()
                return
            if self.reverse_drive_active:
                self._handle_reverse_drive()
            else:
                self._handle_undocking()

    def _prepare_and_send_undocking_goal(self):
        staging_x_offset = MOTION_CONFIG['staging_x_offset']
        v_max = max(abs(staging_x_offset) / 30.0, 0.01)
        max_undocking_time = (abs(staging_x_offset) / v_max) * 1.25

        subtask = SubTask()
        subtask.type = SubTask.UNDOCKING
        subtask.description = 'Startup Undocking'
        subtask.undock_goal = UndockGoal(
            dock_type=self.active_dock.type,
            max_undocking_time=max_undocking_time,
        )
        self.last_undocking_subtask = subtask

        self.get_logger().info(
            f"Sending undocking goal | dock_type='{self.active_dock.type}' | "
            f'max_undocking_time={max_undocking_time:.1f}s'
        )
        if self.undocking_action_client.send_undocking_goal(subtask):
            # Local abort covering a server that never answers at all: without
            # it get_status() sits at UNDOCKING and this timer ticks forever.
            # 2x the goal budget, so the server's own failure reports first
            # whenever it is alive.
            self._undock_deadline = time.monotonic() + max_undocking_time * 2.0
            self._transition(RobotStatusEnum.UNDOCKING)
        else:
            self.get_logger().error('Failed to send undocking goal')
            self._transition(RobotStatusEnum.ERROR)

    # =========================================================================
    # ACTION CLIENT MONITORS
    # =========================================================================

    def _handle_undocking(self):
        status = self.undocking_action_client.get_status()
        self.get_logger().info(f'Undocking: {status.name}')

        if status == RobotStatusEnum.UNDOCKING:
            self._transition(RobotStatusEnum.UNDOCKING)

        elif status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info('Undocking complete')
            self._undock_deadline = None
            self.undocking_action_client.reset()
            self._transition(RobotStatusEnum.DONE_UNDOCKING)
            self._on_startup_undock_done()

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().warning('Undocking failed — starting reverse drive fallback')
            # ReverseDriveClient bounds itself (config-derived timeout, see its
            # _timeout), so the action deadline is dropped rather than carried
            # over — two timers on one phase would race.
            self._undock_deadline = None
            self.undocking_action_client.reset()
            if self.reverse_drive_client.drive_to_staging(self.active_dock.instance_name):
                self.reverse_drive_active = True
                self._transition(RobotStatusEnum.UNDOCKING)
            else:
                self._finish('Startup undocking FAILED — reverse drive refused (dock_backwards?)')

    def _handle_reverse_drive(self):
        status = self.reverse_drive_client.get_status()
        self.get_logger().info(f'Reverse drive: {status.name}')

        if status == ReverseDriveStatus.REVERSING:
            self._transition(RobotStatusEnum.UNDOCKING)

        elif status == ReverseDriveStatus.DONE:
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()
            self._transition(RobotStatusEnum.DONE_UNDOCKING)
            self._on_startup_undock_done()

        elif status in (ReverseDriveStatus.ERROR, ReverseDriveStatus.CANCELED):
            self.reverse_drive_active = False
            self.reverse_drive_client.reset()
            self._finish(f'Startup undocking FAILED — reverse drive {status.name}')

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _abort_undocking(self):
        """
        Give up on an undocking action that blew its deadline.

        cancel_goal() before reset(): reset() only clears this client's local
        state, so without the cancel the server keeps its goal and keeps driving
        the robot after this node stops watching.
        """
        self.get_logger().error('Undock deadline exceeded — canceling goal')
        self._undock_deadline = None
        self.undocking_action_client.cancel_goal()
        self.undocking_action_client.reset()
        self._transition(RobotStatusEnum.ERROR)
        # Finish on the next tick so cancel_goal_async has a period to transmit.
        self._abort_pending = True

    def _on_startup_undock_done(self):
        self.startup_undock_complete = True
        self._transition(RobotStatusEnum.IDLE)
        self._finish('Startup undocking complete — robot ready for tasks')

    def _transition(self, new_status: RobotStatusEnum):
        if self.current_status != new_status:
            self.previous_status = self.current_status
            self.current_status = new_status
            self.get_logger().info(f'Status: {self.previous_status.name} → {self.current_status.name}')

    def _odom_pose_summary(self) -> str:
        """
        Last odometry sample, exactly as stamped — no TF applied.

        Logged at shutdown so the run ends with the raw pose next to the
        map-frame value the position check reported. A mismatch between the two
        is the map->odom offset, which is what to read when a dock distance
        looks wrong.
        """
        if self.pose_status is None:
            return 'Odom pose: none received'

        pose = self.pose_status.pose.pose
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (
            f"Odom pose ('{self.pose_status.header.frame_id}'): "
            f'({pose.position.x:.3f}, {pose.position.y:.3f}, {math.degrees(yaw):.1f}deg)'
        )

    def _finish(self, message: str):
        if self._shutdown_requested:
            return  # already torn down — a second pass would hit dead handles
        self._shutdown_requested = True
        self.get_logger().info(f'\n{"═" * 52}\n  {message}\n  {self._odom_pose_summary()}\n{"═" * 52}')
        if hasattr(self, '_main_timer'):
            self._main_timer.cancel()
        if hasattr(self, '_init_check_timer'):
            self._init_check_timer.cancel()
        self.destroy_node()
        rclpy.shutdown()


# =============================================================================
# ENTRY POINT
# =============================================================================


def main(args=None):
    rclpy.init(args=args)
    node = StartupInitTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted')
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
