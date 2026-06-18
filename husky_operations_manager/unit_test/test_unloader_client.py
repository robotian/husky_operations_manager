#!/usr/bin/env python3
"""
Test node for UnloaderActionClient.

Sends a single HOME or END goal to the unloader action server and monitors
progress until a terminal state is reached, then shuts down.

Usage:
    ros2 run husky_operations_manager test_unloader_client --ros-args -p target:=home
    ros2 run husky_operations_manager test_unloader_client --ros-args -p target:=end
"""

import rclpy
from rclpy.node import Node

from status_interfaces.action import OperateUnloader
from husky_operations_manager.action_clients.unloader import UnloaderActionClient
from husky_operations_manager.robot_enums import RobotStatusEnum

_TARGET_MAP = {
    'home': OperateUnloader.Goal.HOME,
    'end': OperateUnloader.Goal.END,
}

TERMINAL_STATES = {
    RobotStatusEnum.DONE_UNLOADING,
    RobotStatusEnum.ERROR,
}


class TestUnloaderNode(Node):
    """Test node for UnloaderActionClient."""

    def __init__(self):
        super().__init__('test_unloader_client')

        self.declare_parameter('target', 'home')
        target_str = self.get_parameter('target').value.lower()

        if target_str not in _TARGET_MAP:
            self.get_logger().error(
                f'Invalid target "{target_str}". Use home or end.'
            )
            raise SystemExit(1)

        self._target_position = _TARGET_MAP[target_str]
        self._last_status = None
        self._shutting_down = False

        self.unloader_client = UnloaderActionClient(self)

        self.status_timer = self.create_timer(1.0, self._status_callback)

        self.get_logger().info(f'Sending unloader goal: {target_str.upper()}')
        success = self.unloader_client.send_goal(self._target_position)

        if not success:
            self.get_logger().error('Failed to send unloader goal — shutting down.')
            self._shutdown()

    # ------------------------------------------------------------------
    # Status Monitoring
    # ------------------------------------------------------------------

    def _status_callback(self):
        if self._shutting_down:
            return

        status = self.unloader_client.get_status()

        if status != self._last_status:
            self.get_logger().info(f'Status: {status.name}')

            feedback = self.unloader_client.get_feedback()
            if feedback:
                self.get_logger().info(
                    f'  target={feedback.target} | '
                    f'progress={feedback.progress_percent:.1f}% | '
                    f'steps={feedback.step_count} | '
                    f'home_limit={int(feedback.at_home_limit)} | '
                    f'end_limit={int(feedback.at_end_limit)} | '
                    f'msg={feedback.feedback_message}'
                )

            self._last_status = status

        if status in TERMINAL_STATES:
            outcome = 'SUCCESS' if status == RobotStatusEnum.DONE_UNLOADING else 'FAILED'
            self.get_logger().info(
                f'Unloader reached terminal state: {status.name} [{outcome}] — shutting down'
            )
            self._shutdown()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        self.status_timer.cancel()
        raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = TestUnloaderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.get_logger().info('Test complete — shutting down')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
