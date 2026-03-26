import math

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.task import Future

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav2_msgs.action._undock_robot import UndockRobot
from status_interfaces.msg import SubTask, UndockGoal

from husky_operations_manager.dataclass import DockingFeedback, Docks
from husky_operations_manager.enum import RobotStatusEnum

class UndockingActionClient:
    """Action client for handling robot undocking."""

    def __init__(self, node: Node):
        self.node = node
        self.logger = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip('/')

        # Status mapping for action result codes
        self.status_map = {
            GoalStatus.STATUS_ACCEPTED: RobotStatusEnum.UNDOCKING,
            GoalStatus.STATUS_EXECUTING: RobotStatusEnum.UNDOCKING,
            GoalStatus.STATUS_SUCCEEDED: RobotStatusEnum.DONE_UNDOCKING,
            GoalStatus.STATUS_ABORTED: RobotStatusEnum.ERROR,
            GoalStatus.STATUS_CANCELING: RobotStatusEnum.IDLE,
            GoalStatus.STATUS_CANCELED: RobotStatusEnum.IDLE,
        }

        # Internal state
        self._sub_task: SubTask | None = None
        self._undocking_data: UndockGoal | None = None
        self._goal_handle = None
        self._current_status = RobotStatusEnum.IDLE
        self._feedback_data: DockingFeedback | None = None
        self._docking_time = 0.0
        
        # Initialize action client
        self.client = ActionClient(
            self.node,
            UndockRobot,
            f'{self.namespace}/undock_robot'
        )

        self.logger.info(f"Undocking action client initialized for {self.namespace}")

    # ------------------------------------------------------------------
    # Core Interface Methods
    # ------------------------------------------------------------------

    def send_undocking_goal(self, subtask: SubTask | None) -> bool:
        """Send an undocking goal using the given SubTask."""
        if not subtask or not isinstance(subtask, SubTask):
            self.logger.error("Invalid SubTask provided for undocking goal.")
            return False

        self._sub_task = subtask
        self._undocking_data = subtask.undock_goal if isinstance(subtask.undock_goal, UndockGoal) else None

        if not self._undocking_data:
            self.logger.error("Undocking goal data missing in SubTask.")
            return False

        undock_goal_msg = UndockRobot.Goal()
        undock_goal_msg.dock_type = self._undocking_data.dock_type
        undock_goal_msg.max_undocking_time = self._undocking_data.max_undocking_time

        self.logger.info(
            f"Sending undocking goal: dock_type={undock_goal_msg.dock_type}, "
            f"max_time={undock_goal_msg.max_undocking_time}s"
        )

        if not self.client.wait_for_server(timeout_sec=5.0):
            self.logger.error("Undocking action server not available.")
            self._current_status = RobotStatusEnum.ERROR
            return False

        self._current_status = RobotStatusEnum.START_UNDOCKING
        future = self.client.send_goal_async(undock_goal_msg)
        future.add_done_callback(self._goal_response_callback)

        return True
    
    def cancel_goal(self) -> bool:
        """Cancel the current undocking goal."""
        if not self._goal_handle:
            self.logger.warning("No undocking goal to cancel.")
            return False

        self.logger.info("Canceling undocking goal...")
        future = self._goal_handle.cancel_goal_async()
        future.add_done_callback(self._goal_canceled_callback)
        return True

    def reset(self) -> None:
        """Reset the undocking client to idle."""
        self._current_status = RobotStatusEnum.IDLE
        self._goal_handle = None
        self._feedback_data = None
        self._docking_time = 0.0
        self.logger.info("Undocking client reset to IDLE")

    def get_status(self) -> RobotStatusEnum:
        """Return current undocking process status."""
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
            self.logger.error("Failed to receive goal handle for undocking.")
            self._current_status = RobotStatusEnum.ERROR
            return

        self._goal_handle = result

        if not self._goal_handle.accepted:
            self.logger.error("Undocking goal was rejected by server.")
            self._current_status = RobotStatusEnum.ERROR
            return

        self.logger.info("Undocking goal accepted by action server.")
        self._current_status = RobotStatusEnum.UNDOCKING

        # Attach feedback listener
        self._goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _goal_canceled_callback(self, future: Future):
        """Handle goal cancel response."""
        cancel_response = future.result()
        if cancel_response and cancel_response.return_code == 0:
            self.logger.info("Undocking goal successfully canceled.")
            self._current_status = RobotStatusEnum.IDLE
        else:
            self.logger.warning("Undocking goal cancellation failed.")
            self._current_status = RobotStatusEnum.ERROR

    def _result_callback(self, future: Future):
        """Process final undocking result."""
        response = future.result()
        if not response:
            self.logger.error("No result received from undocking action.")
            self._current_status = RobotStatusEnum.ERROR
            return

        status = response.status
        result: UndockRobot.Result = response.result

        self.logger.debug(f"Undocking Response: {response}")
        self.logger.debug(f"Undocking Result: {result}, Status: {status}")

        status_name = {
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
        }.get(status, "UNKNOWN")

        self.logger.info(f"Undocking result received: {status_name}")

        # Map to RobotStatusEnum
        self._current_status = self.status_map.get(status, RobotStatusEnum.ERROR)

        # Build feedback object
        self._feedback_data = DockingFeedback(
            status=self._current_status.value,
            task=self._sub_task.description if self._sub_task else "Undocking Task",
            docking_location=self._undocking_data.dock_type if self._undocking_data else "unknown",
            feedback_message=status_name,
            docking_time=self._docking_time,
            num_retries=-1,
        )

        self.logger.info(
            f"Undocking complete: {self._feedback_data.feedback_message}, "
            f"status={self._feedback_data.status}"
        )
