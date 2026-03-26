"""
Navigation Action Client using NavigateThroughPoses for robot navigation.

This module provides a client interface for sending navigation goals to Nav2's
NavigateThroughPoses action server and tracking navigation status.
"""
import math
from transforms3d import euler

import action_msgs.srv
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.task import Future
# from tf_transformations import quaternion_from_euler

from geometry_msgs.msg import PoseStamped, Pose
from nav2_msgs.action import NavigateThroughPoses
from status_interfaces.msg import Task, SubTask, WayPoint, RobotStatus

from husky_operations_manager.enum import NavigationStatus
from husky_operations_manager.dataclass import WPFStatus


class NavigationActionClient:
    """
    Action client for robot navigation using NavigateThroughPoses.

    Handles sending navigation goals, processing feedback, and tracking
    navigation status. Retry logic should be handled by the calling node.
    """

    def __init__(self, node: Node) -> None:
        """
        Initialize the navigation action client.

        Args:
            node: ROS2 node instance for creating action client
        """
        self.logger = RcutilsLogger(self.__class__.__name__)
        self.node = node

        # Robot and task state
        self.robot_status: RobotStatus | None = None
        self.task: Task | None = None
        self.sub_task: SubTask | None = None
        self.waypoints_list: list[WayPoint] = []

        # Goal management
        self.goal_handle: ClientGoalHandle | None = None
        self.navigation_status: NavigationStatus = NavigationStatus.IDLE

        # Waypoint tracking
        self.last_waypoint_index: int = 0

        # Get namespace for topic names
        self.namespace = self.node.get_namespace().rstrip('/')

        # Subscribe to robot status
        self.status_sub = self.node.create_subscription(
            RobotStatus,
            f'{self.namespace}/status/robot',
            self._robot_status_callback,
            10
        )

        # Create action client
        self.client = ActionClient(
            self.node,
            NavigateThroughPoses,
            f'{self.namespace}/navigate_through_poses'
        )

        self.logger.info(f"Navigation action client initialized for {self.namespace}")

    # ------------------------------------------------------------------
    # Core Interface Methods
    # ------------------------------------------------------------------

    def get_navigation_status(self) -> NavigationStatus:
        """
        Get the current navigation action status.

        Returns:
            NavigationStatus enum value
        """
        return self.navigation_status

    def get_current_status(self) -> WPFStatus | None:
        """
        Get the current waypoint following status.

        Returns:
            WPFStatus object with current navigation state, or None if no task active
        """
        if not self.sub_task or not self.waypoints_list:
            return None

        current_node_id = self._get_current_node_id()
        target_node_id = self.waypoints_list[-1].node_id if self.waypoints_list else -1

        return WPFStatus(
            status=int(self.navigation_status.value),
            task=self.sub_task.description,
            current_node_id=current_node_id,
            target_node_id=target_node_id
        )

    def is_navigation_active(self) -> bool:
        """
        Check if navigation is currently active.

        Returns:
            True if navigation goal is in progress, False otherwise
        """
        return self.navigation_status in [
            NavigationStatus.SENDING,
            NavigationStatus.ACCEPTED,
            NavigationStatus.ACTIVE
        ]

    def send_goal(self, task: Task) -> bool:
        """
        Send a navigation goal to the action server.

        Args:
            task: Task containing navigation waypoints

        Returns:
            True if goal was sent successfully, False otherwise
        """
        # Prevent duplicate goals
        if self.is_navigation_active():
            self.logger.warning(
                f"Navigation already in progress (status: {self.navigation_status.name}) "
                "- skipping duplicate"
            )
            return False

        # Validate task
        if not task or not task.sub_tasks:
            self.logger.warning(f"Invalid task for navigation: {task}")
            self.navigation_status = NavigationStatus.ERROR
            return False

        self.task = task
        sub_tasks = list(task.sub_tasks)
        self.sub_task = sub_tasks[0]

        # Validate sub-task data
        if not self.sub_task or not isinstance(self.sub_task.data, list):
            self.logger.error("SubTask data must be a list of WayPoints")
            self.navigation_status = NavigationStatus.ERROR
            return False

        # Parse waypoints
        self.waypoints_list = [
            WayPoint(**wp) if isinstance(wp, dict) else wp
            for wp in self.sub_task.data
        ]

        if self.robot_status:
            dest_pos = self.waypoints_list[-1]
            current_pos = self.robot_status.topo_map_position

            dist = ((dest_pos.x - current_pos.x) ** 2 + (dest_pos.y - current_pos.y) ** 2) ** 0.5

            if dist < 0.2:
                pass

        if not self.waypoints_list:
            self.logger.error("No waypoints found in sub-task")
            self.navigation_status = NavigationStatus.ERROR
            return False

        self.logger.debug(f"Waypoints: {[wp.node_id for wp in self.waypoints_list]}")

        goal_poses = self._generate_goal_poses(self.waypoints_list)

        if not goal_poses:
            self.logger.error("Failed to generate goal poses")
            self.navigation_status = NavigationStatus.ERROR
            return False

        self.logger.info(
            f"Sending navigation goal: {self.waypoints_list[0].node_id} → "
            f"{self.waypoints_list[-1].node_id}"
        )

        # Check server availability
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.logger.error("NavigateThroughPoses action server not available")
            self.navigation_status = NavigationStatus.ERROR
            return False

        # Generate poses
        goal_msg = NavigateThroughPoses.Goal()
        goal_msg.poses = goal_poses

        # This NavigateThroughPoses definition is taken directly from ROS2 Kilted
        # Uncomment this section when upgrading to ROS2 Kilted
        # # Generate navigation goal
        # goal_msg: NavigateThroughPoses_Goal = NavigateThroughPoses.Goal()
        # goal_msg.poses.header.stamp = self.node.get_clock().now().to_msg()
        # goal_msg.poses.header.frame_id = 'map'
        # goal_msg.poses.goals = goal_poses

        # Update status
        self.navigation_status = NavigationStatus.SENDING
        self.goal_handle = None

        # Send goal
        future = self.client.send_goal_async(goal_msg, self._feedback_callback)
        future.add_done_callback(self._goal_response_callback)

        return True

    def cancel_goal(self) -> bool:
        """
        Cancel the current navigation goal.

        Returns:
            True if cancellation was initiated, False otherwise
        """
        if not self.is_navigation_active():
            self.logger.info("No goal in progress to cancel")
            return False

        self.logger.info("Canceling navigation goal")

        if self.goal_handle is not None:
            future = self.goal_handle.cancel_goal_async()
            future.add_done_callback(self._goal_canceled_callback)
            return True
        else:
            self.logger.warning("No goal handle available to cancel")
            self.navigation_status = NavigationStatus.CANCELED
            return False

    def reset(self) -> None:
        """Reset navigation client to idle state."""
        self.navigation_status = NavigationStatus.IDLE
        self.goal_handle = None
        self.last_waypoint_index = 0
        self.task = None
        self.sub_task = None
        self.waypoints_list = []
        self.logger.info("Navigation client reset to IDLE")

    # ------------------------------------------------------------------
    # Action Client Callbacks
    # ------------------------------------------------------------------

    def _robot_status_callback(self, msg: RobotStatus) -> None:
        """Update stored robot status."""
        self.robot_status = msg

    def _goal_canceled_callback(self, future) -> None:
        """Handle goal cancellation response."""
        cancel_response: action_msgs.srv.CancelGoal.Response = future.result()

        if cancel_response.return_code == 0:
            self.logger.info("Goal cancellation complete")
            self.navigation_status = NavigationStatus.CANCELED
        else:
            self.logger.warning(f"Goal cancellation failed: {cancel_response.return_code}")
            self.navigation_status = NavigationStatus.ERROR

    def _goal_response_callback(self, future: Future) -> None:
        """Handle goal acceptance/rejection from action server."""
        self.goal_handle = future.result()

        if self.goal_handle is None:
            self.logger.error("Failed to get valid goal handle")
            self.navigation_status = NavigationStatus.FAILED
            return

        if not self.goal_handle.accepted:
            self.logger.error("Goal rejected by action server")
            self.navigation_status = NavigationStatus.REJECTED
            return

        self.logger.info("Goal accepted by action server")
        self.navigation_status = NavigationStatus.ACCEPTED

        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future: Future) -> None:
        """Process the final result of the navigation goal."""
        result = future.result()

        if not result:
            self.logger.error("Failed to get valid result")
            self.navigation_status = NavigationStatus.FAILED
            return

        status = result.status
        self.logger.info(f"Navigation result received. Status: {status}")

        if status == 4:  # SUCCEEDED
            self.logger.info("Destination reached successfully")
            self.navigation_status = NavigationStatus.SUCCEEDED
            self.last_waypoint_index = 0

        elif status == 5:  # CANCELED
            self.logger.warning("Navigation goal was canceled")
            self.navigation_status = NavigationStatus.CANCELED
            self.last_waypoint_index = 0

        elif status == 6:  # ABORTED
            self.logger.warning(
                f"Navigation goal was aborted at waypoint {self._get_current_node_id()}"
            )
            self.navigation_status = NavigationStatus.ABORTED

    def _feedback_callback(
        self, feedback_msg
    ) -> None:
        """
        Process feedback from the action server during navigation.

        Args:
            feedback_msg: Feedback message containing current pose
        """
        # Update status to ACTIVE when receiving feedback
        if self.navigation_status == NavigationStatus.ACCEPTED:
            self.navigation_status = NavigationStatus.ACTIVE

        feedback: NavigateThroughPoses.Feedback = feedback_msg.feedback
        current_pose: Pose = feedback.current_pose.pose

        # Update waypoint tracking
        new_index = self._get_closest_waypoint_index(current_pose)

        if new_index != -1 and new_index != self.last_waypoint_index:
            self.logger.info(
                f"Waypoint progress: {self.last_waypoint_index} → {new_index} "
                f"(Node: {self.waypoints_list[new_index].node_id})"
            )
            self.last_waypoint_index = new_index

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_node_id(self) -> int:
        """
        Get current node ID from last tracked waypoint.

        Returns:
            Current node ID, or -1 if unknown
        """
        if self.last_waypoint_index < len(self.waypoints_list):
            return self.waypoints_list[self.last_waypoint_index].node_id
        return -1

    def _generate_goal_poses(
        self, waypoint_list: list[WayPoint]
    ) -> list[PoseStamped]:
        """
        Generate list of PoseStamped messages for navigation.

        The first pose uses the robot's current position. Subsequent poses
        are oriented towards the next waypoint. The final pose uses its
        specified theta orientation.

        Args:
            waypoint_list: List of waypoints to navigate through

        Returns:
            List of PoseStamped objects for navigation
        """
        if not self.robot_status:
            self.logger.warning("No robot status available for pose generation")
            return []

        waypoint_count = len(waypoint_list)
        poses: list[PoseStamped] = []

        for index, waypoint in enumerate(waypoint_list):
            current_wp = (
                WayPoint(**waypoint) if isinstance(waypoint, dict) else waypoint
            )

            # Convert theta to quaternion
            w, x, y, z = euler.euler2quat(0.0, 0.0, current_wp.theta, 'sxyz')
            # x, y, z, w = quaternion_from_euler(0.0, 0.0, current_wp.theta, 'sxyz')

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.node.get_clock().now().to_msg()

            if index == 0:
                # First waypoint: use current robot position
                pose.pose.position.x = float(
                    self.robot_status.topo_map_position.x
                )
                pose.pose.position.y = float(
                    self.robot_status.topo_map_position.y
                )
                pose.pose.position.z = 0.0

                # Use waypoint orientation
                pose.pose.orientation.x = float(x)
                pose.pose.orientation.y = float(y)
                pose.pose.orientation.z = float(z)
                pose.pose.orientation.w = float(w)

                self.logger.info(
                    f"Start pose: ({pose.pose.position.x:.3f}, "
                    f"{pose.pose.position.y:.3f}) theta: {current_wp.theta:.3f}"
                )
            else:
                # Subsequent waypoints
                pose.pose.position.x = float(current_wp.x)
                pose.pose.position.y = float(current_wp.y)
                pose.pose.position.z = 0.0

                # Calculate orientation
                next_wp = (
                    waypoint_list[index + 1] if index + 1 < waypoint_count else None
                )

                if next_wp:
                    # Point towards next waypoint
                    dx = next_wp.x - current_wp.x
                    dy = next_wp.y - current_wp.y

                    if abs(dx) > 0.01 or abs(dy) > 0.01:
                        # Calculate angle to next waypoint
                        angle = math.atan2(dy, dx)
                        self.logger.debug(
                            f"Waypoint {current_wp.node_id} → angle: {angle:.3f}"
                        )
                        pose.pose.orientation.z = math.sin(angle / 2)
                        pose.pose.orientation.w = math.cos(angle / 2)
                        pose.pose.orientation.x = 0.0
                        pose.pose.orientation.y = 0.0
                    else:
                        # Waypoints too close, use specified orientation
                        pose.pose.orientation.x = float(x)
                        pose.pose.orientation.y = float(y)
                        pose.pose.orientation.z = float(z)
                        pose.pose.orientation.w = float(w)
                else:
                    # Last waypoint: use specified theta
                    pose.pose.orientation.x = float(x)
                    pose.pose.orientation.y = float(y)
                    pose.pose.orientation.z = float(z)
                    pose.pose.orientation.w = float(w)
                    self.logger.info(
                        f"Final waypoint {current_wp.node_id} at "
                        f"({current_wp.x:.3f}, {current_wp.y:.3f}) "
                        f"theta: {current_wp.theta:.3f}"
                    )

            poses.append(pose)

        return poses

    def _get_closest_waypoint_index(self, current_pose: Pose) -> int:
        """
        Find closest waypoint with hysteresis to prevent oscillation.

        Args:
            current_pose: Current robot pose

        Returns:
            Index of closest waypoint
        """
        if not self.waypoints_list:
            return -1

        min_dist = float('inf')
        closest_index = self.last_waypoint_index
        threshold = 0.05 ** 2  # 5cm² - "reached" threshold
        switch_buffer = 0.3  # Must be 0.3m closer to switch waypoints

        # Calculate distance to current tracked waypoint
        current_wp = self.waypoints_list[self.last_waypoint_index]
        current_dist = (
            (current_wp.x - current_pose.position.x) ** 2 +
            (current_wp.y - current_pose.position.y) ** 2
        )

        # Check all waypoints
        for i, waypoint in enumerate(self.waypoints_list):
            dist_squared = (
                (waypoint.x - current_pose.position.x) ** 2 +
                (waypoint.y - current_pose.position.y) ** 2
            )

            # Within threshold - consider reached
            if dist_squared <= threshold:
                if i != self.last_waypoint_index:
                    self.logger.debug(
                        f"Reached waypoint {waypoint.node_id} "
                        f"(dist: {dist_squared**0.5:.3f}m)"
                    )
                return i

            # Only switch if significantly closer (hysteresis)
            if (dist_squared < min_dist and
                dist_squared < (current_dist - switch_buffer ** 2)):
                min_dist = dist_squared
                closest_index = i

        return closest_index