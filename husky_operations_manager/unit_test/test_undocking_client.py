#!/usr/bin/env python3
"""
Test node for UndockingActionClient and ReverseDriveClient.

Exercises the full undocking sequence:
  1. Send an undock_robot goal via UndockingActionClient
  2. If the undocking action fails (ERROR), fall back to ReverseDriveClient
     which drives the robot in reverse using TF closed-loop feedback.

Motion / dock configuration is read synchronously from `motion.*` ROS params
(same layout as husky_operations_manager's config.yaml).

Usage:
    ros2 run husky_operations_manager test_undocking_client \
        --ros-args --params-file config/config.yaml \
        -r __ns:=/husky_0
"""

import rclpy
from rclpy.node import Node
from status_interfaces.msg import SubTask, UndockGoal

from husky_operations_manager.enum import RobotStatusEnum, ReverseDriveStatus
from husky_operations_manager.dataclass import (
    DockInstanceConfig,
    DockPose,
    ReverseDriveConfig,
)
from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.action_clients.reverse_drive_client import ReverseDriveClient


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

    # Internal phase labels used by the timer state machine
    _PHASE_IDLE        = "IDLE"
    _PHASE_UNDOCKING   = "UNDOCKING"
    _PHASE_REVERSE     = "REVERSE_DRIVE"
    _PHASE_DONE        = "DONE"

    def __init__(self):
        super().__init__('test_undocking_client')

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f"TestUndockingNode starting | namespace={self.namespace}")

        self._declare_parameters()
        self._read_parameters()

        # Phase tracking
        self._phase             = self._PHASE_IDLE
        self._reverse_active    = False
        self._test_complete     = False

        # Clients — depend on motion_config
        self.undocking_client     = UndockingActionClient(self)
        self.reverse_drive_client = ReverseDriveClient(self, self.motion_config)

        self._main_timer = self.create_timer(self._timer_period, self._timer_callback)
        self.get_logger().info(
            f"Main timer started — beginning undocking test | "
            f"active_dock='{self._active_dock_name}'"
        )

    # =========================================================================
    # PARAMETER DECLARATION / READ
    # =========================================================================

    def _declare_parameters(self):
        """Declare all parameters with safe defaults."""
        self.declare_parameter('undocking.dock_type',          'simple_charging_dock')
        self.declare_parameter('undocking.max_undocking_time', 30.0)
        self.declare_parameter('timing.timer_period',          1.0)

        # Motion / dock config — per-dock leaves declared in _build_motion_config
        self.declare_parameter('motion.dock_names', ['husky_charger'])
        self.declare_parameter('motion.active_dock', '')
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

    def _read_parameters(self):
        """Read declared parameters into instance variables."""
        self._dock_type           = str(self.get_parameter('undocking.dock_type').value)
        self._max_undocking_time  = float(self.get_parameter('undocking.max_undocking_time').value)
        self._timer_period        = float(self.get_parameter('timing.timer_period').value)

        self.motion_config = self._build_motion_config()

        active = str(self.get_parameter('motion.active_dock').value)
        names  = list(self.motion_config.dock_configs.keys())
        self._active_dock_name = active if active in self.motion_config.dock_configs else names[0]

        self.get_logger().info(
            f"Parameters | dock_type='{self._dock_type}' | "
            f"max_undocking_time={self._max_undocking_time}s | "
            f"timer_period={self._timer_period}s | "
            f"docks={names} | active_dock='{self._active_dock_name}'"
        )

    def _build_motion_config(self) -> ReverseDriveConfig:
        """Build a ReverseDriveConfig from the `motion.*` ROS params."""
        m = 'motion'
        dock_names = list(self.get_parameter(f'{m}.dock_names').value)

        dock_configs: dict[str, DockInstanceConfig] = {}
        for name in dock_names:
            p = f'{m}.dock_configs.{name}'
            self.declare_parameter(f'{p}.type', 'simple_charging_dock')
            self.declare_parameter(f'{p}.frame', 'map')
            self.declare_parameter(f'{p}.pose', [0.0, 0.0, 0.0])
            pose = list(self.get_parameter(f'{p}.pose').value)
            dock_configs[name] = DockInstanceConfig(
                instance_name=name,
                type=str(self.get_parameter(f'{p}.type').value),
                frame=str(self.get_parameter(f'{p}.frame').value),
                pose=DockPose(x=float(pose[0]), y=float(pose[1]), theta=float(pose[2])),
            )

        return ReverseDriveConfig(
            dock_configs=dock_configs,
            staging_x_offset=float(self.get_parameter(f'{m}.staging_x_offset').value),
            staging_yaw_offset=float(self.get_parameter(f'{m}.staging_yaw_offset').value),
            base_frame=str(self.get_parameter(f'{m}.base_frame').value),
            controller_frequency=float(self.get_parameter(f'{m}.controller_frequency').value),
            v_linear_min=float(self.get_parameter(f'{m}.v_linear_min').value),
            v_angular_max=float(self.get_parameter(f'{m}.v_angular_max').value),
            linear_tolerance=float(self.get_parameter(f'{m}.linear_tolerance').value),
            angular_tolerance=float(self.get_parameter(f'{m}.angular_tolerance').value),
            dock_backwards=bool(self.get_parameter(f'{m}.dock_backwards').value),
        )

    # =========================================================================
    # MAIN CONTROL LOOP
    # =========================================================================

    def _timer_callback(self):
        """
        1 Hz state machine that drives the undocking test sequence.

        Phase transitions:
          IDLE        → send undocking goal → UNDOCKING
          UNDOCKING   → poll UndockingActionClient
                          DONE_UNDOCKING → DONE (success)
                          ERROR          → start ReverseDriveClient → REVERSE_DRIVE
          REVERSE_DRIVE → poll ReverseDriveClient
                          DONE  → DONE (success via fallback)
                          ERROR → DONE (both paths failed)
          DONE        → cancel timer, log result
        """
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
        """Build a SubTask with UndockGoal from motion_config and send it."""
        dock = self.motion_config.dock_configs.get(self._active_dock_name)
        dock_type = dock.type if dock and dock.type else self._dock_type

        staging_x_offset   = self.motion_config.staging_x_offset
        v_linear_min       = self.motion_config.v_linear_min
        max_undocking_time = (abs(staging_x_offset) / max(v_linear_min, 0.01)) * 1.25

        self.get_logger().debug(
            f"max_undocking_time from config | staging_x_offset={staging_x_offset} | "
            f"v_linear_min={v_linear_min} | result={max_undocking_time:.1f}s"
        )

        undock_goal = UndockGoal(
            dock_type=dock_type,
            max_undocking_time=max_undocking_time
        )

        subtask = SubTask()
        subtask.type        = SubTask.UNDOCKING
        subtask.description = "Test Undocking"
        subtask.undock_goal = undock_goal

        self.get_logger().info(
            f"Sending undocking goal | dock_type='{dock_type}' | "
            f"max_undocking_time={max_undocking_time:.1f}s"
        )

        if self.undocking_client.send_undocking_goal(subtask):
            self.get_logger().info("Undocking goal sent — monitoring UndockingActionClient")
            self._phase = self._PHASE_UNDOCKING
        else:
            self.get_logger().error("Failed to send undocking goal")
            self._phase = self._PHASE_DONE

    # =========================================================================
    # PHASE: UNDOCKING
    # =========================================================================

    def _poll_undocking(self):
        """Poll UndockingActionClient and advance phase on terminal status."""
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
            self.get_logger().info("Undocking SUCCEEDED via UndockingActionClient")
            self.undocking_client.reset()
            self._phase = self._PHASE_DONE

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().warning(
                "UndockingActionClient reported ERROR — "
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
        """Activate ReverseDriveClient as fallback when the undocking action fails."""
        self.get_logger().info("Starting ReverseDriveClient fallback...")

        if self.reverse_drive_client.drive_to_staging(self._active_dock_name):
            self.get_logger().info("ReverseDriveClient started — monitoring reverse drive")
            self._reverse_active = True
            self._phase          = self._PHASE_REVERSE
        else:
            self.get_logger().error(
                "ReverseDriveClient refused to start "
                f"(dock='{self._active_dock_name}' "
                f"dock_backwards={self.motion_config.dock_backwards})"
            )
            self._phase = self._PHASE_DONE

    def _poll_reverse_drive(self):
        """Poll ReverseDriveClient and advance phase on terminal status."""
        status = self.reverse_drive_client.get_status()
        self.get_logger().info(f"Reverse drive status: {status.name}")

        if status == ReverseDriveStatus.DONE:
            self.get_logger().info("Reverse drive SUCCEEDED — robot reached staging pose")
            self._reverse_active = False
            self.reverse_drive_client.reset()
            self._phase = self._PHASE_DONE

        elif status == ReverseDriveStatus.ERROR:
            self.get_logger().error("Reverse drive FAILED — both undocking paths exhausted")
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
        """Cancel the main timer, clean up any active goals, and shut the node down."""
        self._test_complete = True
        self._main_timer.cancel()
        self.get_logger().info(
            "======================================\n"
            "  Undocking test sequence complete.\n"
            "  Shutting down node.\n"
            "======================================"
        )
        self.cancel_active_goals()
        self.destroy_node()
        rclpy.shutdown()

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def cancel_active_goals(self):
        """Cancel any in-flight goals before shutdown."""
        if self.undocking_client and self._phase == self._PHASE_UNDOCKING:
            self.get_logger().info("Cancelling active undocking goal...")
            self.undocking_client.cancel_goal()

        if self.reverse_drive_client and self._reverse_active:
            self.get_logger().info("Cancelling active reverse drive...")
            self.reverse_drive_client.reset()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = TestUndockingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user — shutting down")
        node.cancel_active_goals()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
