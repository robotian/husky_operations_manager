#!/usr/bin/env python3
"""
Simple test node for NavigationActionClient

This node provides a minimal test harness to verify the navigation action client
works correctly. It simulates the role of status.py for testing purposes.
The node shuts itself down once navigation reaches a terminal state.

Usage:
    ros2 run husky_operations_manager test_navigation_client
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from status_interfaces.msg import Task, SubTask, WayPoint, RobotStatus

from husky_operations_manager.action_clients.navigation import NavigationActionClient
from husky_operations_manager.enum import NavigationStatus

# Terminal states that trigger node shutdown
TERMINAL_STATES = {
    NavigationStatus.SUCCEEDED,
    NavigationStatus.FAILED,
    NavigationStatus.ABORTED,
    NavigationStatus.CANCELED,
    NavigationStatus.REJECTED,
    NavigationStatus.ERROR,
}


class TestNavigationNode(Node):
    """Simple test node for NavigationActionClient"""

    def __init__(self):
        super().__init__('test_navigation_client')

        self.get_logger().info('Test Navigation Client Node Started')

        # Initialize navigation action client
        self.navigation_client = NavigationActionClient(self)

        # Publish a mock robot status so the action client has a starting position
        self.robot_status_pub = self.create_publisher(
            RobotStatus,
            f'{self.get_namespace().rstrip("/")}/status/robot',
            10
        )

        self._last_status = None
        self._goal_sent = False
        self._shutting_down = False

        # Status monitoring timer — starts after goal is sent
        self.status_timer = self.create_timer(1.0, self._status_callback)

    # ------------------------------------------------------------------
    # Goal Management
    # ------------------------------------------------------------------

    def publish_robot_status_and_wait(self, executor: SingleThreadedExecutor):
        """Publish robot status and spin until the action client has received it."""
        robot_status = RobotStatus()
        robot_status.topo_map_position.x = 0.0
        robot_status.topo_map_position.y = 0.0

        self.get_logger().info('Publishing robot status and waiting for receipt...')

        # Keep publishing and spinning until the action client subscriber fires
        while rclpy.ok() and self.navigation_client.robot_status is None:
            self.robot_status_pub.publish(robot_status)
            executor.spin_once(timeout_sec=0.1)

        self.get_logger().info('Robot status received by action client')

    def send_goal(self, executor: SingleThreadedExecutor):
        """Publish robot status (blocking until received), then send goal once."""
        self.publish_robot_status_and_wait(executor)

        self.get_logger().info('Sending navigation goal...')

        # Build test waypoints
        wp1 = WayPoint()
        wp1.node_id = 1
        wp1.x = 0.0
        wp1.y = 1.0
        wp1.theta = 0.0

        wp2 = WayPoint()
        wp2.node_id = 2
        wp2.x = 1.0
        wp2.y = -1.0
        wp2.theta = 1.5708  # ~90 degrees

        wp3 = WayPoint()
        wp3.node_id = 3
        wp3.x = -1.0
        wp3.y = 2.0
        wp3.theta = 0.0

        # Build SubTask — SubTask.MOVING = SubTaskEnum.MOVING = 1
        sub_task = SubTask()
        sub_task.type = SubTask.MOVING
        sub_task.description = "Test Navigation Operation"
        sub_task.data = [wp1, wp2, wp3]

        task = Task()
        task.sub_tasks = [sub_task]

        success = self.navigation_client.send_goal(task)
        self._goal_sent = True

        if success:
            self.get_logger().info('✓ Goal sent successfully')
        else:
            self.get_logger().error('✗ Failed to send goal — check action server')
            self._shutting_down = True

    # ------------------------------------------------------------------
    # Status Monitoring
    # ------------------------------------------------------------------

    def _status_callback(self):
        """Periodic status monitoring. Sets shutdown flag on terminal state."""
        if self._shutting_down or not self._goal_sent:
            return

        status = self.navigation_client.get_navigation_status()
        is_active = self.navigation_client.is_navigation_active()

        if self._last_status != status:
            active_str = "ACTIVE" if is_active else "IDLE"
            self.get_logger().info(f'Status: {status.name:20s} | {active_str}')

            wpf_status = self.navigation_client.get_current_status()
            if wpf_status:
                self.get_logger().info(
                    f'  Current Node: {wpf_status.current_node_id} → '
                    f'Target Node: {wpf_status.target_node_id}'
                )

            self._last_status = status

        if status in TERMINAL_STATES:
            self.get_logger().info(
                f'Navigation reached terminal state: {status.name} — shutting down'
            )
            self._shutting_down = True

    def is_done(self) -> bool:
        """Return True when the node is ready to shut down."""
        return self._shutting_down


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    node = TestNavigationNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        # Send goal once — blocks until robot_status is received, then fires
        node.send_goal(executor)

        # Spin until navigation completes
        while rclpy.ok() and not node.is_done():
            executor.spin_once(timeout_sec=0.1)

    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Test complete — shutting down')
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()