"""
This script is used for docking the robot using the docks or docking database configured in the docking config.yaml
"""
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.task import Future
from action_msgs.msg import GoalStatus

from nav2_msgs.action._dock_robot import DockRobot_FeedbackMessage, DockRobot
from status_interfaces.msg import SubTask, DockGoal
from husky_operations_manager.dataclass import DockingFeedback
from husky_operations_manager.enum import RobotStatusEnum


class DockingActionClient:
    """Action client for handling robot docking."""

    def __init__(self, node: Node):
        self.node = node
        self.logger = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip('/')

        # Status mapping for action result codes
        self.status_map = {
            GoalStatus.STATUS_ACCEPTED: RobotStatusEnum.DOCKING,
            GoalStatus.STATUS_EXECUTING: RobotStatusEnum.DOCKING,
            GoalStatus.STATUS_SUCCEEDED: RobotStatusEnum.DONE_DOCKING,
            GoalStatus.STATUS_ABORTED: RobotStatusEnum.ERROR,
            GoalStatus.STATUS_CANCELING: RobotStatusEnum.IDLE,
            GoalStatus.STATUS_CANCELED: RobotStatusEnum.IDLE,
        }

        # Internal state
        self._sub_task: SubTask | None = None
        self._docking_data: DockGoal | None = None
        self._goal_handle = None
        self._current_status = RobotStatusEnum.IDLE
        self._feedback_data: DockingFeedback | None = None
        self._feedback_state: int = 0
        self._docking_time = 0.0

        # Initialize action client
        self.client = ActionClient(
            self.node,
            DockRobot,
            f'{self.namespace}/dock_robot'
        )

        self.logger.info(
            f"Docking action client initialized for {self.namespace}")

    # ------------------------------------------------------------------
    # Core Interface Methods
    # ------------------------------------------------------------------

    def send_docking_goal(self, subtask: SubTask | None) -> bool:
        """Send a docking goal using the given SubTask."""
        if not subtask or not isinstance(subtask, SubTask):
            self.logger.error("Invalid SubTask provided for docking goal.")
            return False

        self._sub_task = subtask
        self._docking_data = subtask.dock_goal if isinstance(
            subtask.dock_goal, DockGoal) else None

        if not self._docking_data:
            self.logger.error("Docking goal data missing in SubTask.")
            return False

        dock_goal_msg = DockRobot.Goal()
        dock_goal_msg.use_dock_id = self._docking_data.use_dock_id
        dock_goal_msg.dock_id = self._docking_data.dock_id
        dock_goal_msg.navigate_to_staging_pose = self._docking_data.navigate_to_staging_pose

        self.logger.info(
            f"Sending docking goal: dock_id={dock_goal_msg.dock_id}, "
            f"navigate_to_staging={dock_goal_msg.navigate_to_staging_pose}"
        )

        if not self.client.wait_for_server(timeout_sec=5.0):
            self.logger.error("Docking action server not available.")
            self._current_status = RobotStatusEnum.ERROR
            return False

        self._current_status = RobotStatusEnum.START_DOCKING
        future = self.client.send_goal_async(
            dock_goal_msg,
            feedback_callback=self._feedback_callback
        )
        future.add_done_callback(self._goal_response_callback)

        return True

    def cancel_goal(self) -> bool:
        """Cancel the current docking goal."""
        if not self._goal_handle:
            self.logger.warning("No docking goal to cancel.")
            return False

        self.logger.info("Canceling docking goal...")
        future = self._goal_handle.cancel_goal_async()
        future.add_done_callback(self._goal_canceled_callback)
        return True

    def reset(self) -> None:
        """Reset the docking client to idle."""
        self._current_status = RobotStatusEnum.IDLE
        self._goal_handle = None
        self._feedback_data = None
        self._feedback_state = 0
        self._docking_time = 0.0
        self.logger.info("Docking client reset to IDLE")

    def get_status(self) -> RobotStatusEnum:
        """Return current docking process status."""
        return self._current_status

    def get_feedback(self) -> DockingFeedback | None:
        """Return the latest DockingFeedback data."""
        return self._feedback_data

    # ------------------------------------------------------------------
    # Action Client Callbacks
    # ------------------------------------------------------------------

    def _goal_response_callback(self, future: Future):
        """Handle response from action server when sending goal."""
        result = future.result()
        if not result:
            self.logger.error("Failed to receive goal handle for docking.")
            self._current_status = RobotStatusEnum.ERROR
            return

        self._goal_handle = result

        if not self._goal_handle.accepted:
            self.logger.error("Docking goal was rejected by server.")
            self._current_status = RobotStatusEnum.ERROR
            return

        self.logger.info("Docking goal accepted by action server.")
        self._current_status = RobotStatusEnum.DOCKING

        # Attach result listener
        self._goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _goal_canceled_callback(self, future: Future):
        """Handle goal cancel response."""
        cancel_response = future.result()
        if cancel_response and cancel_response.return_code == 0:
            self.logger.info("Docking goal successfully canceled.")
            self._current_status = RobotStatusEnum.IDLE
        else:
            self.logger.warning("Docking goal cancellation failed.")
            self._current_status = RobotStatusEnum.ERROR

    def _feedback_callback(self, feedback_msg: DockRobot_FeedbackMessage):
        """Process feedback from the action server during docking."""
        feedback: DockRobot.Feedback = feedback_msg.feedback
        feedback_state = feedback.state

        # Map feedback state string
        feedback_state_str = {
            GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
            GoalStatus.STATUS_EXECUTING: "EXECUTING",
            GoalStatus.STATUS_CANCELING: "CANCELING",
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
            GoalStatus.STATUS_CANCELED: "CANCELED"
        }.get(feedback_state, "UNKNOWN")

        # Log only when state changes
        if self._feedback_state != feedback_state:
            self._feedback_state = feedback_state
            self.logger.info(
                f"Docking feedback: {feedback_state_str} | "
                f"Time: {feedback.docking_time.sec}s | "
                f"Retries: {feedback.num_retries}"
            )
            self._docking_time = feedback.docking_time.sec
            self._current_status = self.status_map.get(
                feedback_state, self._current_status)

        # Update feedback data
        self._feedback_data = DockingFeedback(
            status=self._current_status.value,
            task=self._sub_task.description if self._sub_task else "Docking Task",
            docking_location=self._docking_data.dock_id if self._docking_data else "unknown",
            feedback_message=feedback_state_str,
            docking_time=self._docking_time,
            num_retries=feedback.num_retries
        )

    def _result_callback(self, future: Future):
        """Process final docking result."""
        response = future.result()
        if not response:
            self.logger.error("No result received from docking action.")
            self._current_status = RobotStatusEnum.ERROR
            return

        status = response.status
        result: DockRobot.Result = response.result

        self.logger.debug(f"Docking Result: {result}, Status: {status}")

        status_name = {
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
        }.get(status, "UNKNOWN")

        self.logger.info(
            f"Docking result: {status_name} | "
            f"Success: {result.success} | "
            f"Error Code: {result.error_code} | "
            f"Retries: {result.num_retries}"
        )

        # Map to RobotStatusEnum
        self._current_status = self.status_map.get(
            status, RobotStatusEnum.ERROR)

        # Build final feedback object
        self._feedback_data = DockingFeedback(
            status=self._current_status.value,
            task=self._sub_task.description if self._sub_task else "Docking Task",
            docking_location=self._docking_data.dock_id if self._docking_data else "unknown",
            feedback_message=status_name,
            docking_time=self._docking_time,
            num_retries=result.num_retries,
        )

        self.logger.info(
            f"Docking complete: {self._feedback_data.feedback_message}, "
            f"status={self._feedback_data.status}"
        )