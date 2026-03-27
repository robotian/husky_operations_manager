#!/usr/bin/env python3
"""
Test node for UndockingActionClient and ReverseDriveClient.

Exercises the full undocking sequence:
  1. Send an undock_robot goal via UndockingActionClient
  2. If the undocking action fails (ERROR), fall back to ReverseDriveClient
     which drives the robot in reverse using TF closed-loop feedback.

Usage:
    ros2 run husky_operations_manager test_undocking_client \
        --ros-args --params-file config/test_undocking_client.yaml \
        -r __ns:=/husky_0
"""

import rclpy
from rclpy.node import Node
from status_interfaces.msg import SubTask, UndockGoal

from husky_operations_manager.enum import RobotStatusEnum, ReverseDriveStatus
from husky_operations_manager.dataclass import DockingConfig, DockInstanceConfig, DockPluginConfig
from husky_operations_manager.docking_param_fetcher import DockingParamFetcher
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

        # Populated once DockingParamFetcher finishes
        self.docking_config: DockingConfig | None       = None
        self.active_dock:    DockInstanceConfig | None  = None
        self.active_plugin:  DockPluginConfig | None    = None

        self.undocking_client: UndockingActionClient | None = None
        self.reverse_drive_client: ReverseDriveClient | None  = None

        # Fetch docking params from docking_server before building clients
        self._param_fetcher = DockingParamFetcher(self)
        self._param_fetcher.fetch()
        self._config_poll_timer = self.create_timer(0.5, self._poll_docking_config)

    # =========================================================================
    # PARAMETER DECLARATION / READ
    # =========================================================================

    def _declare_parameters(self):
        """Declare all parameters with safe defaults."""
        self.declare_parameter('undocking.dock_type',          'simple_charging_dock')
        self.declare_parameter('undocking.max_undocking_time', 30.0)
        self.declare_parameter('timing.timer_period',          1.0)

    def _read_parameters(self):
        """Read declared parameters into instance variables."""
        self._dock_type           = str(self.get_parameter('undocking.dock_type').value)
        self._max_undocking_time  = float(self.get_parameter('undocking.max_undocking_time').value)
        self._timer_period        = float(self.get_parameter('timing.timer_period').value)

        self.get_logger().info(
            f"Parameters | dock_type='{self._dock_type}' | "
            f"max_undocking_time={self._max_undocking_time}s | "
            f"timer_period={self._timer_period}s"
        )

    # =========================================================================
    # DOCKING CONFIG POLLING
    # =========================================================================

    def _poll_docking_config(self):
        """
        Poll DockingParamFetcher every 0.5 s.

        Cancels itself on DONE or ERROR. On DONE, fires _on_docking_config_ready
        which constructs the action clients and starts the main timer.
        """
        from husky_operations_manager.enum import DockingParamFetcherStatus
        status = self._param_fetcher.get_status()
        self.get_logger().debug(f"DockingParamFetcher poll | status={status.name}")

        if status == DockingParamFetcherStatus.DONE:
            self._config_poll_timer.cancel()
            self._on_docking_config_ready()

        elif status == DockingParamFetcherStatus.ERROR:
            self._config_poll_timer.cancel()
            self.get_logger().error(
                "DockingParamFetcher failed — cannot build ReverseDriveClient. "
                "Running with UndockingActionClient only."
            )
            self._build_clients_without_config()

    def _on_docking_config_ready(self):
        """Build action clients with full DockingConfig and start main timer."""
        self.docking_config = self._param_fetcher.get_config()
        self.active_dock    = self.docking_config.dock_configs[self.docking_config.docks[0]]
        self.active_plugin  = self.docking_config.plugin_configs[self.docking_config.dock_plugins[0]]

        self.get_logger().info(
            f"DockingConfig ready | "
            f"dock='{self.active_dock.instance_name}' | "
            f"plugin='{self.active_plugin.plugin_name}' | "
            f"staging_x_offset={self.active_plugin.staging_x_offset} | "
            f"dock_backwards={self.docking_config.dock_backwards}"
        )

        self.undocking_client     = UndockingActionClient(self)
        self.reverse_drive_client = ReverseDriveClient(self, self.docking_config)

        self._start_main_timer()

    def _build_clients_without_config(self):
        """
        Build only UndockingActionClient when DockingParamFetcher fails.

        ReverseDriveClient cannot be created without DockingConfig, so the
        fallback path will not be available in this run.
        """
        self.undocking_client = UndockingActionClient(self)
        self.get_logger().warning(
            "ReverseDriveClient NOT available (no DockingConfig). "
            "Only UndockingActionClient will be used."
        )
        self._start_main_timer()

    def _start_main_timer(self):
        """Create the 1 Hz main control timer and kick off the undocking sequence."""
        self._main_timer = self.create_timer(self._timer_period, self._timer_callback)
        self.get_logger().info("Main timer started — beginning undocking test")

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
        """Build a SubTask with UndockGoal from params and send to UndockingActionClient."""
        if self.undocking_client is None:
            self.get_logger().error("UndockingActionClient not yet initialised — waiting")
            return

        # Use dock_type from DockingConfig if available, otherwise fall back to param
        dock_type = (
            self.active_dock.type
            if self.active_dock and self.active_dock.type
            else self._dock_type
        )

        # Compute max_undocking_time from DockingConfig when possible
        if self.active_plugin and self.docking_config:
            staging_x_offset  = self.active_plugin.staging_x_offset or 0.7
            v_linear_min      = self.docking_config.controller_v_linear_min or 0.01
            max_undocking_time = (abs(staging_x_offset) / max(v_linear_min, 0.01)) * 2.0
            self.get_logger().debug(
                f"max_undocking_time computed from config | "
                f"staging_x_offset={staging_x_offset} | "
                f"v_linear_min={v_linear_min} | "
                f"result={max_undocking_time:.1f}s"
            )
        else:
            max_undocking_time = self._max_undocking_time

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

        success = self.undocking_client.send_undocking_goal(subtask)

        if success:
            self.get_logger().info("✓ Undocking goal sent — monitoring UndockingActionClient")
            self._phase = self._PHASE_UNDOCKING
        else:
            self.get_logger().error("✗ Failed to send undocking goal")
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
            # Still waiting for the action server to respond
            self.get_logger().debug(f"Waiting for undocking to begin | status={status.name}")

    # =========================================================================
    # PHASE: REVERSE_DRIVE (fallback)
    # =========================================================================

    def _start_reverse_drive(self):
        """
        Activate ReverseDriveClient as fallback when undocking action fails.

        Falls through to DONE immediately if ReverseDriveClient is unavailable
        (e.g. DockingConfig was not fetched successfully).
        """
        if self.reverse_drive_client is None:
            self.get_logger().error(
                "ReverseDriveClient not available — cannot attempt reverse drive fallback"
            )
            self._phase = self._PHASE_DONE
            return

        self.get_logger().info("Starting ReverseDriveClient fallback...")

        if self.reverse_drive_client.drive_to_staging():
            self.get_logger().info("✓ ReverseDriveClient started — monitoring reverse drive")
            self._reverse_active = True
            self._phase          = self._PHASE_REVERSE
        else:
            self.get_logger().error(
                "✗ ReverseDriveClient refused to start "
                f"(dock_backwards={self.docking_config.dock_backwards if self.docking_config else 'unknown'})"
            )
            self._phase = self._PHASE_DONE

    def _poll_reverse_drive(self):
        """Poll ReverseDriveClient and advance phase on terminal status."""
        if self.reverse_drive_client is None:
            self._phase = self._PHASE_DONE
            return

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
        """
        Cancel the main timer, clean up any active goals, and shut the node down.

        Called irrespective of outcome (success, error, or fallback failure) so
        the process always exits cleanly without requiring Ctrl-C.
        """
        self._test_complete = True
        self._main_timer.cancel()
        self.get_logger().info(
            "══════════════════════════════════════\n"
            "  Undocking test sequence complete.\n"
            "  Shutting down node.\n"
            "══════════════════════════════════════"
        )
        # Cancel any goals that may still be in-flight before tearing down
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