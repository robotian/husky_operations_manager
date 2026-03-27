"""
NOTE: This is the working action client used to run MTC Harvest Action Server
implemented as a part of mtc_tutorial packages.

Harvest Action Client for robot harvesting operations.

This module provides a client interface for sending harvest goals to the
Harvest action server and tracking harvest/loading status.
"""
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.task import Future
from action_msgs.msg import GoalStatus

from status_interfaces.action import Harvest
from status_interfaces.msg import SubTask
from geometry_msgs.msg import Point

from husky_operations_manager.enum import RobotStatusEnum


class HarvestActionClient:
    """
    Action client for robot harvesting operations.

    Handles sending harvest goals, processing feedback, and tracking
    harvest and loading status. Retry logic should be handled by the calling node.
    """

    def __init__(self, node: Node) -> None:
        """
        Initialize the harvest action client.

        Args:
            node: ROS2 node instance for creating action client
        """
        self.logger = RcutilsLogger(self.__class__.__name__)
        self.node = node

        # Task state
        self.sub_task: SubTask | None = None

        # Goal management
        self.goal_handle: ClientGoalHandle | None = None
        self.current_status: RobotStatusEnum = RobotStatusEnum.IDLE

        # Get namespace for topic names
        self.namespace = self.node.get_namespace().rstrip('/')

        # Create action client
        self.client = ActionClient(
            self.node,
            Harvest,
            f'{self.namespace}/harvest'
        )

        self.logger.info(
            f"Harvest action client initialized for {self.namespace}")

    def get_status(self) -> RobotStatusEnum:
        """
        Get the current harvest action status.

        Returns:
            RobotStatusEnum value representing current state
        """
        return self.current_status

    def is_harvest_active(self) -> bool:
        """
        Check if harvest operation is currently active.

        Returns:
            True if harvest goal is in progress, False otherwise
        """
        return self.current_status in [
            RobotStatusEnum.START_HARVESTING,
            RobotStatusEnum.HARVESTING,
            RobotStatusEnum.START_LOADING,
            RobotStatusEnum.LOADING
        ]

    def send_goal(self, subtask: SubTask) -> bool:
        """
        Send a harvest goal to the action server.

        The server will dynamically fetch target_plant_coordinates from
        the computer vision module.

        Args:
            subtask: SubTask containing harvest information

        Returns:
            True if goal was sent successfully, False otherwise
        """
        # Prevent duplicate goals
        if self.is_harvest_active():
            self.logger.warning(
                f"Harvest already in progress (status: {self.current_status.name}) "
                "- skipping duplicate"
            )
            return False

        # Validate subtask
        if not subtask or not isinstance(subtask, SubTask):
            self.logger.error("Invalid SubTask provided for harvest goal")
            self.current_status = RobotStatusEnum.ERROR
            return False

        self.sub_task = subtask

        # Create goal message
        # Server will fetch target_plant_coordinates dynamically from CV module
        goal_msg = Harvest.Goal()
        goal_msg.target_plant_coordinates = Point()  # Empty, filled by server
        # Test Points
        goal_msg.target_plant_coordinates.x = 0.0
        goal_msg.target_plant_coordinates.y = 0.4
        goal_msg.target_plant_coordinates.z = -0.25

        self.logger.info("Sending harvest goal to server")

        # Check server availability
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.logger.error("Harvest action server not available")
            self.current_status = RobotStatusEnum.ERROR
            return False

        # Update status
        self.current_status = RobotStatusEnum.START_HARVESTING
        self.goal_handle = None

        # Send goal
        future = self.client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )
        future.add_done_callback(self._goal_response_callback)

        return True

    def cancel_goal(self) -> bool:
        """
        Cancel the current harvest goal.

        Returns:
            True if cancellation was initiated, False otherwise
        """
        if not self.is_harvest_active():
            self.logger.info("No harvest goal in progress to cancel")
            return False

        self.logger.info("Canceling harvest goal")

        if self.goal_handle is not None:
            future = self.goal_handle.cancel_goal_async()
            future.add_done_callback(self._goal_canceled_callback)
            return True
        else:
            self.logger.warning("No goal handle available to cancel")
            self.current_status = RobotStatusEnum.IDLE
            return False

    def reset(self) -> None:
        """Reset harvest client to idle state."""
        self.current_status = RobotStatusEnum.IDLE
        self.goal_handle = None
        self.sub_task = None
        self.logger.info("Harvest client reset to IDLE")

    def _goal_canceled_callback(self, future: Future) -> None:
        """Handle goal cancellation response."""
        try:
            cancel_response = future.result()
            if cancel_response and cancel_response.return_code == 0:
                self.logger.info("Goal cancellation complete")
                self.current_status = RobotStatusEnum.IDLE
            else:
                self.logger.warning(
                    f"Goal cancellation failed: {cancel_response.return_code if cancel_response else 'No response'}")
                self.current_status = RobotStatusEnum.ERROR
        except Exception as e:
            self.logger.error(f"Error during cancellation: {e}")
            self.current_status = RobotStatusEnum.ERROR

    def _goal_response_callback(self, future: Future) -> None:
        """Handle goal acceptance/rejection from action server."""
        self.goal_handle = future.result()

        if self.goal_handle is None:
            self.logger.error("Failed to get valid goal handle")
            self.current_status = RobotStatusEnum.ERROR
            return

        if not self.goal_handle.accepted:
            self.logger.error("Goal rejected by action server")
            self.current_status = RobotStatusEnum.ERROR
            return

        self.logger.info("Goal accepted by action server")

        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future: Future) -> None:
        """Process the final result of the harvest goal."""
        result = future.result()

        if not result:
            self.logger.error("Failed to get valid result")
            self.current_status = RobotStatusEnum.ERROR
            return

        status = result.status
        server_result: Harvest.Result = result.result

        self.logger.info(
            f"Harvest result received. Status: {status}, "
            f"Success: {server_result.success}, "
            f"Final Status: {server_result.final_status}"
        )

        # Map action status to RobotStatusEnum
        if status == GoalStatus.STATUS_SUCCEEDED:
            # Use server's final_status for more granular state
            if server_result.final_status == Harvest.Feedback.DONE_LOADING:
                self.logger.info("Harvest and loading completed successfully")
                self.current_status = RobotStatusEnum.DONE_LOADING
            elif server_result.final_status == Harvest.Feedback.DONE_HARVESTING:
                self.logger.info("Harvesting completed")
                self.current_status = RobotStatusEnum.DONE_HARVESTING
            else:
                self.logger.info("Harvest operation succeeded")
                self.current_status = RobotStatusEnum.DONE_LOADING

        elif status == GoalStatus.STATUS_CANCELED:
            self.logger.warning("Harvest goal was canceled")
            self.current_status = RobotStatusEnum.IDLE

        elif status == GoalStatus.STATUS_ABORTED:
            self.logger.error("Harvest goal was aborted")
            self.current_status = RobotStatusEnum.ERROR

        else:
            self.logger.error(f"Unknown result status: {status}")
            self.current_status = RobotStatusEnum.ERROR

    def _feedback_callback(self, feedback_msg: Harvest.Feedback) -> None:
        """
        Process feedback from the action server during harvest operation.

        Args:
            feedback_msg: Feedback message containing current status
        """
        feedback: Harvest.Feedback = feedback_msg.feedback

        # Map feedback status to RobotStatusEnum
        status_map = {
            Harvest.Feedback.START_HARVESTING: RobotStatusEnum.START_HARVESTING,
            Harvest.Feedback.HARVESTING: RobotStatusEnum.HARVESTING,
            Harvest.Feedback.DONE_HARVESTING: RobotStatusEnum.DONE_HARVESTING,
            Harvest.Feedback.START_LOADING: RobotStatusEnum.START_LOADING,
            Harvest.Feedback.LOADING: RobotStatusEnum.LOADING,
            Harvest.Feedback.DONE_LOADING: RobotStatusEnum.DONE_LOADING,
            Harvest.Feedback.ERROR: RobotStatusEnum.ERROR,
            Harvest.Feedback.PAUSED: RobotStatusEnum.PAUSED,
        }

        new_status = status_map.get(feedback.status, self.current_status)

        # Log status changes
        if new_status != self.current_status:
            self.logger.info(
                f"Harvest status: {self.current_status.name} → {new_status.name} | "
                f"Task: {feedback.task} | "
                f"Load: {feedback.load_status}% | "
                f"Location: ({feedback.location.x:.2f}, {feedback.location.y:.2f}, {feedback.location.z:.2f})"
            )
            self.current_status = new_status
