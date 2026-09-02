#!/usr/bin/env python3
"""
Test node for UndockingActionClient and ReverseDriveClient.

Dock poses and motion parameters are hardcoded below. Set the active dock
via the 'active_dock' ROS2 parameter at launch:

    ros2 run husky_operations_manager test_undocking_client \
        --ros-args -p active_dock:=unloading_station \
        -r __ns:=/husky_0

Valid values for active_dock: husky_charger, unloading_station

Exercises the full undocking sequence:
  1. Send an undock_robot goal via UndockingActionClient
  2. If the undocking action fails (ERROR), fall back to ReverseDriveClient
     which drives the robot in reverse using TF closed-loop feedback.
"""

import rclpy
from rclpy.node import Node

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
        pose=DockPose(x=-6.001, y=2.194, theta=0.0)
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

DEFAULT_ACTIVE_DOCK = 'husky_charger'
DOCK_TYPE           = 'simple_charging_dock'
MAX_UNDOCKING_TIME  = 30.0   # seconds
TIMER_PERIOD        = 1.0    # seconds


# =============================================================================
# NODE
# =============================================================================

class TestUndockingNode(Node):
    """
    Test harness for UndockingActionClient and ReverseDriveClient.

    State machine (driven by a 1 Hz timer):

      IDLE
        └─► UNDOCKING          Send undock_robot goal via UndockingActionClient
              ├─► DONE_UNDOCKING  Undocking succeeded → log success → IDLE
              └─► ERROR           Undocking failed → start ReverseDriveClient fallback
                    ├─► REVERSING   Robot is reversing to staging pose
                    ├─► DONE        Reverse drive succeeded → log success → IDLE
                    └─► ERROR       Reverse drive also failed → log failure → IDLE
    """

    _PHASE_IDLE      = "IDLE"
    _PHASE_UNDOCKING = "UNDOCKING"
    _PHASE_REVERSE   = "REVERSE_DRIVE"
    _PHASE_DONE      = "DONE"

    def __init__(self):
        super().__init__('test_undocking_client')

        self.declare_parameter('active_dock', DEFAULT_ACTIVE_DOCK)
        active_dock = str(self.get_parameter('active_dock').value)

        if active_dock not in DOCK_CONFIGS:
            self.get_logger().error(
                f"Unknown active_dock='{active_dock}'. "
                f"Valid options: {list(DOCK_CONFIGS.keys())}. Shutting down."
            )
            raise SystemExit(1)

        config = ReverseDriveConfig(dock_names=[active_dock], **MOTION_CONFIG)

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(
            f"TestUndockingNode starting | namespace={self.namespace} | "
            f"active_dock='{active_dock}' | "
            f"dock_type='{DOCK_TYPE}' | max_undocking_time={MAX_UNDOCKING_TIME}s"
        )

        self._phase          = self._PHASE_IDLE
        self._reverse_active = False
        self._test_complete  = False

        self.undocking_client     = UndockingActionClient(self)
        self.reverse_drive_client = ReverseDriveClient(self, config)

        self._main_timer = self.create_timer(TIMER_PERIOD, self._timer_callback)
        self.get_logger().info("Main timer started — beginning undocking test")

    # =========================================================================
    # MAIN CONTROL LOOP
    # =========================================================================

    def _timer_callback(self):
        if self._test_complete:
            return

        self.get_logger().debug(f"Timer tick | phase={self._phase}")

        if self._phase == self._PHASE_IDLE:
            self._send_undocking_goal()
        elif self._phase == self._PHASE_UNDOCKING:
            self._poll_undocking()
        elif self._phase == self._PHASE_REVERSE:
            self._poll_reverse_drive()
        elif self._phase == self._PHASE_DONE:
            self._finish_test()

    # =========================================================================
    # PHASE: IDLE → UNDOCKING
    # =========================================================================

    def _send_undocking_goal(self):
        undock_goal = UndockGoal(
            dock_type=DOCK_TYPE,
            max_undocking_time=MAX_UNDOCKING_TIME,
        )

        subtask             = SubTask()
        subtask.type        = SubTask.UNDOCKING
        subtask.description = "Test Undocking"
        subtask.undock_goal = undock_goal

        self.get_logger().info(
            f"Sending undocking goal | dock_type='{DOCK_TYPE}' | "
            f"max_undocking_time={MAX_UNDOCKING_TIME:.1f}s"
        )

        if self.undocking_client.send_undocking_goal(subtask):
            self.get_logger().info("✓ Undocking goal sent — monitoring UndockingActionClient")
            self._phase = self._PHASE_UNDOCKING
        else:
            self.get_logger().error("✗ Failed to send undocking goal")
            self._phase = self._PHASE_DONE

    # =========================================================================
    # PHASE: UNDOCKING
    # =========================================================================

    def _poll_undocking(self):
        status   = self.undocking_client.get_status()
        feedback = self.undocking_client.get_feedback()

        self.get_logger().info(f"Undocking status: {status.name}")

        if feedback:
            self.get_logger().debug(
                f"Feedback | task='{feedback.task}' | "
                f"location='{feedback.docking_location}' | "
                f"msg='{feedback.feedback_message}'"
            )

        if status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info("✓ Undocking SUCCEEDED via UndockingActionClient")
            self.undocking_client.reset()
            self._phase = self._PHASE_DONE

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().warning(
                "✗ UndockingActionClient reported ERROR — "
                "attempting ReverseDriveClient fallback"
            )
            self.undocking_client.reset()
            self._start_reverse_drive()

        elif status in (RobotStatusEnum.IDLE, RobotStatusEnum.START_UNDOCKING):
            self.get_logger().debug(f"Waiting for undocking to begin | status={status.name}")

    # =========================================================================
    # PHASE: REVERSE_DRIVE (fallback)
    # =========================================================================

    def _start_reverse_drive(self):
        self.get_logger().info("Starting ReverseDriveClient fallback...")

        if self.reverse_drive_client.drive_to_staging():
            self.get_logger().info("✓ ReverseDriveClient started — monitoring reverse drive")
            self._reverse_active = True
            self._phase          = self._PHASE_REVERSE
        else:
            self.get_logger().error(
                f"✗ ReverseDriveClient refused to start "
                f"(dock_backwards={MOTION_CONFIG['dock_backwards']})"
            )
            self._phase = self._PHASE_DONE

    def _poll_reverse_drive(self):
        status = self.reverse_drive_client.get_status()
        self.get_logger().info(f"Reverse drive status: {status.name}")

        if status == ReverseDriveStatus.DONE:
            self.get_logger().info("✓ Reverse drive SUCCEEDED — robot reached staging pose")
            self._reverse_active = False
            self.reverse_drive_client.reset()
            self._phase = self._PHASE_DONE

        elif status == ReverseDriveStatus.ERROR:
            self.get_logger().error("✗ Reverse drive FAILED — both undocking paths exhausted")
            self._reverse_active = False
            self.reverse_drive_client.reset()
            self._phase = self._PHASE_DONE

        elif status == ReverseDriveStatus.CANCELED:
            self.get_logger().warning("Reverse drive was CANCELED")
            self._reverse_active = False
            self.reverse_drive_client.reset()
            self._phase = self._PHASE_DONE

        elif status == ReverseDriveStatus.REVERSING:
            self.get_logger().debug("Reverse drive in progress...")

    # =========================================================================
    # PHASE: DONE
    # =========================================================================

    def _finish_test(self):
        self._test_complete = True
        self._main_timer.cancel()
        self.get_logger().info(
            "══════════════════════════════════════\n"
            "  Undocking test sequence complete.\n"
            "  Shutting down node.\n"
            "══════════════════════════════════════"
        )
        self.cancel_active_goals()
        self.destroy_node()
        rclpy.shutdown()

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def cancel_active_goals(self):
        if self._phase == self._PHASE_UNDOCKING:
            self.get_logger().info("Cancelling active undocking goal...")
            self.undocking_client.cancel_goal()

        if self._reverse_active:
            self.get_logger().info("Cancelling active reverse drive...")
            self.reverse_drive_client.reset()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = TestUndockingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user — shutting down")
        node.cancel_active_goals()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
