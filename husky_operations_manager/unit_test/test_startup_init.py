#!/usr/bin/env python3
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
    ros2 run husky_operations_manager test_startup_init \
        --ros-args -r __ns:=/a200_0284
"""

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped
from status_interfaces.msg import SubTask, UndockGoal

from husky_operations_manager.robot_enums import ReverseDriveStatus, RobotStatusEnum
from husky_operations_manager.types import DockInstanceConfig, DockPose, ReverseDriveConfig
from husky_operations_manager.action_clients.navigation import NavigationActionClient
from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.action_clients.reverse_drive import ReverseDriveClient


# =============================================================================
# GLOBAL CONFIG — edit here instead of a YAML file
# =============================================================================

DOCK_CONFIGS: dict[str, DockInstanceConfig] = {
    'husky_charger': DockInstanceConfig(
        instance_name='husky_charger',
        type='simple_charging_dock',
        frame='map',
        pose=DockPose(x=0.8, y=-1.945, theta=0.0),
    ),
    'unloading_station': DockInstanceConfig(
        instance_name='unloading_station',
        type='simple_charging_dock',
        frame='map',
        pose=DockPose(x=0.85, y=1.60, theta=1.571),
    ),
}

MOTION_CONFIG = dict(
    dock_configs=DOCK_CONFIGS,
    plugin_name='simple_charging_dock',
    staging_x_offset=-1.5,
    staging_yaw_offset=0.0,
    base_frame='base_link',
    controller_frequency=50.0,
    v_linear_min=0.15,
    v_angular_max=0.25,
    linear_tolerance=0.05,
    angular_tolerance=0.1,
    dock_backwards=False,
)

DOCKING_THRESHOLD = 0.25  # metres — distance to consider robot "at dock"
TIMER_PERIOD      = 1.0   # seconds
INIT_CHECK_DELAY  = 2.0   # seconds


# =============================================================================
# Node
# =============================================================================

class StartupInitTestNode(Node):

    def __init__(self):
        super().__init__('test_startup_init_node')

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f"StartupInitTestNode | namespace={self.namespace}")

        self._init_state()

        self.pose_status: PoseWithCovarianceStamped | None = None
        self._pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            f'{self.namespace}/ground_truth/pose',
            lambda msg: setattr(self, 'pose_status', msg),
            10,
        )

        # active_dock and reverse_drive_client are set after nearest-dock selection.
        self.active_dock:          DockInstanceConfig | None = None
        self.reverse_drive_client: ReverseDriveClient | None = None

        self.navigation              = NavigationActionClient(self)
        self.docking_action_client   = DockingActionClient(self)
        self.undocking_action_client = UndockingActionClient(self)

        self.get_logger().info(
            f"Clients ready | docks={list(DOCK_CONFIGS.keys())} | "
            f"staging_x_offset={MOTION_CONFIG['staging_x_offset']} | "
            f"dock_backwards={MOTION_CONFIG['dock_backwards']}"
        )

        self._init_check_timer = self.create_timer(
            INIT_CHECK_DELAY, self._initial_position_check_timer
        )
        self._main_timer = self.create_timer(TIMER_PERIOD, self._timer_callback)

    # =========================================================================
    # STATE
    # =========================================================================

    def _init_state(self):
        self.is_initialized:          bool = False
        self.is_at_docking_station:   bool = False
        self.startup_undock_complete: bool = False
        self.reverse_drive_active:    bool = False
        self.current_status:   RobotStatusEnum = RobotStatusEnum.IDLE
        self.previous_status:  RobotStatusEnum = RobotStatusEnum.IDLE
        self.last_undocking_subtask: SubTask | None = None

    # =========================================================================
    # PHASE 1 — INITIAL POSITION CHECK
    # =========================================================================

    def _initial_position_check_timer(self):
        if self.pose_status is None:
            self.get_logger().warning("Waiting for pose data...")
            return
        self._init_check_timer.cancel()
        self._check_initial_position()

    def _check_initial_position(self):
        if self.is_initialized:
            return

        if not self.pose_status or not self.pose_status.pose:
            self.get_logger().warning("No pose — retrying in 1 s")
            time.sleep(1.0)
            self._check_initial_position()
            return

        pos = self.pose_status.pose.pose.position

        # Find nearest dock by Euclidean distance.
        nearest_dock = min(
            DOCK_CONFIGS.values(),
            key=lambda d: math.sqrt((pos.x - d.pose.x) ** 2 + (pos.y - d.pose.y) ** 2),
        )
        dist = math.sqrt((pos.x - nearest_dock.pose.x) ** 2 + (pos.y - nearest_dock.pose.y) ** 2)

        self.active_dock = nearest_dock

        # Build ReverseDriveConfig with nearest dock at index 0.
        ordered = [nearest_dock.instance_name] + [
            n for n in DOCK_CONFIGS if n != nearest_dock.instance_name
        ]
        self.reverse_drive_client = ReverseDriveClient(self, ReverseDriveConfig(
            dock_names=ordered,
            **MOTION_CONFIG,
        ))

        dock = self.active_dock
        self.get_logger().info(
            f"Position check | robot=({pos.x:.3f}, {pos.y:.3f}) | "
            f"nearest='{dock.instance_name}' ({dock.pose.x:.3f}, {dock.pose.y:.3f}) | "
            f"dist={dist:.3f}m | threshold={DOCKING_THRESHOLD}m"
        )

        if dist <= DOCKING_THRESHOLD:
            self.is_at_docking_station   = True
            self.startup_undock_complete = False
            self.get_logger().info("Robot AT dock — startup undocking required")
        else:
            self.is_at_docking_station   = False
            self.startup_undock_complete = True
            self.get_logger().info("Robot NOT at dock — no startup undocking needed")
            self._finish("Startup init complete — robot ready for tasks")

        self.is_initialized = True

    # =========================================================================
    # PHASE 2 — STARTUP UNDOCKING STATE MACHINE (1 Hz)
    # =========================================================================

    def _timer_callback(self):
        if not self.is_initialized or self.startup_undock_complete:
            return

        self.get_logger().debug(
            f"Tick | status={self.current_status.name} | "
            f"reverse_drive_active={self.reverse_drive_active}"
        )

        if self.current_status == RobotStatusEnum.IDLE:
            self.get_logger().info("Starting startup undocking")
            self._transition(RobotStatusEnum.START_UNDOCKING)

        elif self.current_status == RobotStatusEnum.START_UNDOCKING:
            self._prepare_and_send_undocking_goal()

        else:
            if self.reverse_drive_active:
                self._handle_reverse_drive()
            else:
                self._handle_undocking()

    def _prepare_and_send_undocking_goal(self):
        staging_x_offset   = MOTION_CONFIG['staging_x_offset']
        v_max              = max(abs(staging_x_offset) / 30.0, 0.01)
        max_undocking_time = (abs(staging_x_offset) / v_max) * 1.25

        subtask             = SubTask()
        subtask.type        = SubTask.UNDOCKING
        subtask.description = "Startup Undocking"
        subtask.undock_goal = UndockGoal(
            dock_type=self.active_dock.type,
            max_undocking_time=max_undocking_time,
        )
        self.last_undocking_subtask = subtask

        self.get_logger().info(
            f"Sending undocking goal | dock_type='{self.active_dock.type}' | "
            f"max_undocking_time={max_undocking_time:.1f}s"
        )
        if self.undocking_action_client.send_undocking_goal(subtask):
            self._transition(RobotStatusEnum.UNDOCKING)
        else:
            self.get_logger().error("Failed to send undocking goal")
            self._transition(RobotStatusEnum.ERROR)

    # =========================================================================
    # ACTION CLIENT MONITORS
    # =========================================================================

    def _handle_undocking(self):
        status = self.undocking_action_client.get_status()
        self.get_logger().info(f"Undocking: {status.name}")

        if status == RobotStatusEnum.UNDOCKING:
            self._transition(RobotStatusEnum.UNDOCKING)

        elif status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info("Undocking complete")
            self.undocking_action_client.reset()
            self._transition(RobotStatusEnum.DONE_UNDOCKING)
            self._on_startup_undock_done()

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().warning("Undocking failed — starting reverse drive fallback")
            self.undocking_action_client.reset()
            if self.reverse_drive_client.drive_to_staging():
                self.reverse_drive_active = True
                self._transition(RobotStatusEnum.UNDOCKING)
            else:
                self._finish("Startup undocking FAILED — reverse drive refused (dock_backwards?)")

    def _handle_reverse_drive(self):
        status = self.reverse_drive_client.get_status()
        self.get_logger().info(f"Reverse drive: {status.name}")

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
            self._finish(f"Startup undocking FAILED — reverse drive {status.name}")

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _on_startup_undock_done(self):
        self.startup_undock_complete = True
        self._transition(RobotStatusEnum.IDLE)
        self._finish("Startup undocking complete — robot ready for tasks")

    def _transition(self, new_status: RobotStatusEnum):
        if self.current_status != new_status:
            self.previous_status = self.current_status
            self.current_status  = new_status
            self.get_logger().info(
                f"Status: {self.previous_status.name} → {self.current_status.name}"
            )

    def _finish(self, message: str):
        self.get_logger().info(f"\n{'═' * 52}\n  {message}\n{'═' * 52}")
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
        node.get_logger().info("Interrupted")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
