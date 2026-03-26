#!/usr/bin/env python3
"""
Simple test node for UndockingActionClient

This node provides a minimal test harness to verify the undocking action client
works correctly. It simulates the role of status.py for testing purposes.

Usage:
    ros2 run husky_operations_manager test_undocking_client.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from status_interfaces.msg import SubTask, UndockGoal

from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.enum import RobotStatusEnum

# Terminal states that trigger node shutdown
TERMINAL_STATES = {
    RobotStatusEnum.DONE_UNDOCKING,
    RobotStatusEnum.IDLE,
    RobotStatusEnum.ERROR,
}


class TestUndockingNode(Node):
    """Simple test node for UndockingActionClient"""

    def __init__(self):
        super().__init__('test_undocking_client')

        self.get_logger().info('Test Undocking Client Node Started')

        # Initialize undocking action client
        self.undocking_client = UndockingActionClient(self)

        self._last_status = None
        self._shutting_down = False

        # Status monitoring timer
        self.status_timer = self.create_timer(1.0, self._status_callback)

        # Send goal on startup
        self.start_undocking_goal()

    # ------------------------------------------------------------------
    # Goal Management
    # ------------------------------------------------------------------

    def start_undocking_goal(self):
        """Send a test undocking goal."""
        self.get_logger().info('START UNDOCKING REQUEST RECEIVED')

        # Build UndockGoal
        undock_goal = UndockGoal()
        undock_goal.dock_type = "simple_charging_dock"
        undock_goal.max_undocking_time = 30.0

        # Build SubTask
        sub_task = SubTask()
        sub_task.type = SubTask.UNDOCKING
        sub_task.description = "Test Undocking Operation"
        sub_task.undock_goal = undock_goal

        self.get_logger().info('Sending undocking goal...')
        success = self.undocking_client.send_undocking_goal(sub_task)

        if success:
            self.get_logger().info('✓ Goal sent successfully')
        else:
            self.get_logger().error('✗ Failed to send goal')

    # ------------------------------------------------------------------
    # Status Monitoring
    # ------------------------------------------------------------------

    def _status_callback(self):
        """Periodic status monitoring."""
        if self._shutting_down:
            return

        status = self.undocking_client.get_status()

        if self._last_status != status:
            self.get_logger().info(f'Status: {status.name:20s}')

            feedback = self.undocking_client.get_feedback()
            if feedback:
                self.get_logger().info(
                    f'Dock Type: {feedback.docking_location} | '
                    f'Feedback: {feedback.feedback_message} | '
                    f'Time: {feedback.docking_time:.1f}s',
                )

            self._last_status = status
        
        # Only shut down on terminal state after the goal has been accepted
        if status in TERMINAL_STATES:
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

    node = TestUndockingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.get_logger().info('Test complete — shutting down')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
