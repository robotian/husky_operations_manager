#!/usr/bin/env python3
"""
Simple test node for DockingActionClient

This node provides a minimal test harness to verify the docking action client
works correctly. It simulates the role of status.py for testing purposes.
The node shuts itself down once docking reaches a terminal state.

Usage:
    ros2 run husky_operations_manager test_docking_client.py
"""

import rclpy
from rclpy.node import Node
from status_interfaces.msg import SubTask, DockGoal

from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.enum import RobotStatusEnum

# Terminal states that trigger node shutdown
TERMINAL_STATES = {
    RobotStatusEnum.DONE_DOCKING,
    RobotStatusEnum.IDLE,
    RobotStatusEnum.ERROR,
}


class TestDockingNode(Node):
    """Simple test node for DockingActionClient"""

    def __init__(self):
        super().__init__('test_docking_client')

        self.get_logger().info('Test Docking Client Node Started')

        # Initialize docking action client
        self.docking_client = DockingActionClient(self)

        self._last_status = None
        self._shutting_down = False

        # Status monitoring timer
        self.status_timer = self.create_timer(1.0, self._status_callback)

        # Send goal on startup
        self._send_docking_goal()

    # ------------------------------------------------------------------
    # Goal Management
    # ------------------------------------------------------------------

    def _send_docking_goal(self):
        """Send a test docking goal."""
        self.get_logger().info('START DOCKING REQUEST RECEIVED')        

        # Build DockGoal
        dock_goal = DockGoal()
        dock_goal.use_dock_id = True
        dock_goal.dock_id = "husky_charger"
        dock_goal.navigate_to_staging_pose = True

        # Build SubTask
        sub_task = SubTask()
        sub_task.type = SubTask.DOCKING
        sub_task.description = "Test Docking Operation"
        sub_task.dock_goal = dock_goal

        self.get_logger().info('Sending docking goal...')
        success = self.docking_client.send_docking_goal(sub_task)

        if success:
            self.get_logger().info('✓ Goal sent successfully')
        else:
            self.get_logger().error('✗ Failed to send goal')
            self._shutdown()

    # ------------------------------------------------------------------
    # Status Monitoring
    # ------------------------------------------------------------------

    def _status_callback(self):
        """Periodic status monitoring. Triggers shutdown on terminal state."""
        if self._shutting_down:
            return

        status = self.docking_client.get_status()

        if self._last_status != status:
            self.get_logger().info(f'Status: {status.name:20s}')

            feedback = self.docking_client.get_feedback()
            if feedback:
                self.get_logger().info(
                    f'Dock: {feedback.docking_location} | '
                    f'Feedback: {feedback.feedback_message} | '
                    f'Time: {feedback.docking_time:.1f}s | '
                    f'Retries: {feedback.num_retries}'
                )

            self._last_status = status

        # Only shut down on terminal state after the goal has been accepted
        if status in TERMINAL_STATES and status != RobotStatusEnum.IDLE:
            self.get_logger().info(
                f'Docking reached terminal state: {status.name} — shutting down'
            )
            self._shutdown()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self):
        """Cancel timers and signal the executor to stop."""
        if self._shutting_down:
            return
        self._shutting_down = True
        self.status_timer.cancel()
        raise SystemExit


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    node = TestDockingNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info('Shutting down...')
    finally:
        node.get_logger().info('Test complete — shutting down')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()