"""
This script is used for controlling the robot arm during manipulator task operations
using the ManipulatorTask action server.
"""
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.task import Future
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped

from status_interfaces.action import ManipulatorTask
from status_interfaces.action._manipulator_task import (
    ManipulatorTask_FeedbackMessage,
)
from status_interfaces.msg import SubTask
from husky_operations_manager.dataclass import ManipulatorTaskFeedback
from husky_operations_manager.enum import RobotStatusEnum, ManipulatorStatus


class ArmCommand:
    GO_STOW       = "GO STOW"
    GO_READY      = "GO READY"
    GO_DROP       = "GO DROP"
    MOVE_EEF      = "MOVE EEF"
    START_HARVEST = "START HARVEST"


# Commands that require a target_eef_pose to be populated in the goal
_POSE_REQUIRED_COMMANDS = {ArmCommand.MOVE_EEF}


class ManipulatorTaskActionClient:
    """Action client for controlling the robot arm."""

    def __init__(self, node: Node):
        self.node = node
        self.logger = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip('/')

        # Status mapping for ManipulatorTask feedback states → RobotStatusEnum
        self.status_map = {
            ManipulatorStatus.IDLE.value:            RobotStatusEnum.IDLE,
            ManipulatorStatus.PLANNING.value:        RobotStatusEnum.START_HARVESTING,
            ManipulatorStatus.MOVING.value:          RobotStatusEnum.HARVESTING,
            ManipulatorStatus.MOVING_COMPLETE.value: RobotStatusEnum.HARVESTING,
            ManipulatorStatus.FAILED.value:          RobotStatusEnum.ERROR,
        }

        # Status mapping for final action result codes → RobotStatusEnum
        self.result_status_map = {
            GoalStatus.STATUS_SUCCEEDED: RobotStatusEnum.DONE_HARVESTING,
            GoalStatus.STATUS_ABORTED:   RobotStatusEnum.ERROR,
            GoalStatus.STATUS_CANCELING: RobotStatusEnum.IDLE,
            GoalStatus.STATUS_CANCELED:  RobotStatusEnum.IDLE,
        }

        # Internal state
        self._sub_task: SubTask | None = None
        self._arm_command: str = ""
        self._goal_handle = None
        self._current_status = RobotStatusEnum.IDLE
        self._feedback_data: ManipulatorTaskFeedback | None = None
        self._feedback_state: int = -1
        self._execution_time: float = 0.0

        # Initialize action client
        self.client = ActionClient(
            self.node,
            ManipulatorTask,
            f'{self.namespace}/manipulator_action_server'
        )

        self.logger.info(
            f"Manipulator Task action client initialized for {self.namespace}"
        )
    
    # ------------------------------------------------------------------
    # ARM Commands Wrapper
    # ------------------------------------------------------------------

    def send_stow_goal(self, subtask: SubTask | None) -> bool:
        """Send GO_STOW command to the manipulator."""
        if not subtask or not isinstance(subtask, SubTask):
            self.logger.error("Invalid SubTask for STOW goal.")
            return False
        self._sub_task = subtask
        self._arm_command = ArmCommand.GO_STOW
        goal_msg = ManipulatorTask.Goal()
        goal_msg.arm_task = self._arm_command
        return self.send_goal(goal_msg)

    def send_ready_goal(self, subtask: SubTask | None) -> bool:
        """Send GO_READY command to the manipulator."""
        if not subtask or not isinstance(subtask, SubTask):
            self.logger.error("Invalid SubTask for READY goal.")
            return False
        self._sub_task = subtask
        self._arm_command = ArmCommand.GO_READY
        goal_msg = ManipulatorTask.Goal()
        goal_msg.arm_task = self._arm_command
        return self.send_goal(goal_msg)

    def send_harvesting_goal(self, subtask: SubTask | None) -> bool:
        """
        Send a manipulator task goal using the given SubTask.

        The arm command is sourced from subtask.description (an ArmCommand
        string).  For MOVE_EEF goals a target_eef_pose is also required;
        supply it via send_move_eef_goal() instead for full pose control.
        """
        if not subtask or not isinstance(subtask, SubTask):
            self.logger.error("Invalid SubTask provided for manipulator task goal.")
            return False

        self._sub_task = subtask
        self._arm_command = ArmCommand.START_HARVEST

        # Build goal message
        goal_msg = ManipulatorTask.Goal()
        goal_msg.arm_task = self._arm_command

        # Warn if this command normally needs a pose but none was supplied
        if self._arm_command in _POSE_REQUIRED_COMMANDS:
            self.logger.warning(
                f"Command '{self._arm_command}' requires a target_eef_pose. "
                "Use send_move_eef_goal() to supply the pose. "
                "Sending with default (zero) pose."
            )
            goal_msg.target_eef_pose = self._get_arm_pose()

        return self.send_goal(goal_msg)

    def send_move_eef_goal(
        self, subtask: SubTask | None, target_pose: PoseStamped
    ) -> bool:
        """
        Send a MOVE_EEF goal with an explicit target end-effector pose.

        Kept as a separate method so future callers can provide a real pose
        without changing the main manipulator task flow.
        """
        if not subtask or not isinstance(subtask, SubTask):
            self.logger.error("Invalid SubTask provided for MOVE_EEF goal.")
            return False

        self._sub_task = subtask
        self._arm_command = ArmCommand.MOVE_EEF

        goal_msg = ManipulatorTask.Goal()
        goal_msg.arm_task = self._arm_command
        goal_msg.target_eef_pose = target_pose

        self.logger.info("Sending MOVE_EEF goal with explicit target pose.")
        return self.send_goal(goal_msg)
    
    # ------------------------------------------------------------------
    # Core Interface Methods
    # ------------------------------------------------------------------

    def send_goal(self, goal_msg: ManipulatorTask.Goal) -> bool:
        """Wait for the action server and dispatch the goal."""
        self.logger.info(
            f"Sending Manipulator Task goal: arm_task='{goal_msg.arm_task}'"
        )

        if not self.client.wait_for_server(timeout_sec=30.0):
            self.logger.error("Manipulator action server not available.")
            self._current_status = RobotStatusEnum.ERROR
            return False

        self._current_status = RobotStatusEnum.START_HARVESTING
        future = self.client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )
        future.add_done_callback(self._goal_response_callback)
        return True

    def cancel_goal(self) -> bool:
        """Cancel the current Manipulator Task goal."""
        if not self._goal_handle:
            self.logger.warning("No manipulator task goal to cancel.")
            return False

        self.logger.info("Canceling manipulator task goal...")
        future = self._goal_handle.cancel_goal_async()
        future.add_done_callback(self._goal_canceled_callback)
        return True

    def reset(self) -> None:
        """Reset the manipulator task client to idle."""
        self._current_status = RobotStatusEnum.IDLE
        self._goal_handle = None
        self._feedback_data = None
        self._feedback_state = -1
        self._execution_time = 0.0
        self.logger.info("Harvesting client reset to IDLE")

    def get_status(self) -> RobotStatusEnum:
        """Return the current manipulator task process status."""
        return self._current_status

    def get_feedback(self) -> ManipulatorTaskFeedback | None:
        """Return the latest HarvestingFeedback data."""
        return self._feedback_data

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_arm_pose(self) -> PoseStamped:
        """
        Return the current end-effector pose.

        TODO: Replace static placeholder with a live TF or topic lookup
              once the pose source is determined.
        """
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        # Static placeholder — orientation set to identity quaternion
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def _build_feedback_object(
        self, status_name: str, num_retries: int = 0
    ) -> ManipulatorTaskFeedback:
        """Construct a HarvestingFeedback dataclass from current state."""
        return ManipulatorTaskFeedback(
            status=self._current_status.value,
            task=self._sub_task.description if self._sub_task else "Manipulator Task",
            arm_task=self._arm_command,
            arm_pose=self._get_arm_pose(),
            feedback_message=status_name,
            execution_time=self._execution_time,
            num_retries=num_retries,
        )

    # ------------------------------------------------------------------
    # Action Client Callbacks
    # ------------------------------------------------------------------

    def _goal_response_callback(self, future: Future):
        """Handle response from the action server when sending goal."""
        result = future.result()
        if not result:
            self.logger.error("Failed to receive goal handle for manipulator task.")
            self._current_status = RobotStatusEnum.ERROR
            return

        self._goal_handle = result

        if not self._goal_handle.accepted:
            self.logger.error("Manipulator Task goal was rejected by server.")
            self._current_status = RobotStatusEnum.ERROR
            return

        self.logger.info("Manipulator Task goal accepted by action server.")
        self._current_status = RobotStatusEnum.HARVESTING

        # Attach result listener
        self._goal_handle.get_result_async().add_done_callback(
            self._result_callback
        )

    def _goal_canceled_callback(self, future: Future):
        """Handle goal cancel response."""
        cancel_response = future.result()
        if cancel_response and cancel_response.return_code == 0:
            self.logger.info("Manipulator Task goal successfully canceled.")
            self._current_status = RobotStatusEnum.IDLE
        else:
            self.logger.warning("Manipulator Task goal cancellation failed.")
            self._current_status = RobotStatusEnum.ERROR

    def _feedback_callback(self, feedback_msg: ManipulatorTask_FeedbackMessage):
        """Process feedback from the action server during Manipulator Task."""
        feedback: ManipulatorTask.Feedback = feedback_msg.feedback
        feedback_state: int = feedback.status

        # Map feedback int → human-readable string
        feedback_state_str = {
            ManipulatorStatus.IDLE.value:            "IDLE",
            ManipulatorStatus.PLANNING.value:        "PLANNING",
            ManipulatorStatus.MOVING.value:          "MOVING",
            ManipulatorStatus.MOVING_COMPLETE.value: "MOVING_COMPLETE",
            ManipulatorStatus.FAILED.value:          "FAILED",
        }.get(feedback_state, "UNKNOWN")

        # Log and update only when state changes
        if self._feedback_state != feedback_state:
            self._feedback_state = feedback_state
            self.logger.info(
                f"Manipulator Task feedback: {feedback_state_str} | "
                f"arm_task='{self._arm_command}'"
            )
            self._current_status = self.status_map.get(
                feedback_state, self._current_status
            )

        self._feedback_data = self._build_feedback_object(feedback_state_str)

    def _result_callback(self, future: Future):
        """Process the final Manipulator Task result."""
        response = future.result()
        if not response:
            self.logger.error("No result received from Manipulator Task action.")
            self._current_status = RobotStatusEnum.ERROR
            return

        status = response.status
        result: ManipulatorTask.Result = response.result

        self.logger.debug(f"Manipulator Task Result: {result}, Status: {status}")

        status_name = {
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_ABORTED:   "ABORTED",
            GoalStatus.STATUS_CANCELED:  "CANCELED",
        }.get(status, "UNKNOWN")

        self.logger.info(
            f"Manipulator Task result: {status_name} | "
            f"Success: {result.success}"
        )

        # Map to RobotStatusEnum
        self._current_status = self.result_status_map.get(
            status, RobotStatusEnum.ERROR
        )

        # Build final feedback object
        self._feedback_data = self._build_feedback_object(status_name)

        self.logger.info(
            f"Manipulator Task complete: {self._feedback_data.feedback_message}, "
            f"status={self._feedback_data.status}"
        )