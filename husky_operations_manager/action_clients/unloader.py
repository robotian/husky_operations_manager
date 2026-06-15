from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.task import Future
from action_msgs.msg import GoalStatus

from status_interfaces.action import OperateUnloader
from husky_operations_manager.types import UnloaderFeedback
from husky_operations_manager.robot_enums import RobotStatusEnum

_SERVER_WAIT_S = 5.0

_TARGET_LABEL = {
    OperateUnloader.Goal.HOME: 'HOME',
    OperateUnloader.Goal.END: 'END',
}

_RESULT_LABEL = {
    OperateUnloader.Result.UNKNOWN: 'UNKNOWN',
    OperateUnloader.Result.AT_HOME: 'AT_HOME',
    OperateUnloader.Result.AT_END: 'AT_END',
    OperateUnloader.Result.ABORTED: 'ABORTED',
    OperateUnloader.Result.CANCELLED: 'CANCELLED',
    OperateUnloader.Result.ERROR: 'ERROR',
}


class UnloaderActionClient:
    """Action client for the bin unloader carriage (OperateUnloader action)."""

    def __init__(self, node: Node):
        self.node = node
        self.logger = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip('/')

        self._goal_handle = None
        self._target_label: str = 'HOME'
        self._current_status = RobotStatusEnum.IDLE
        self._feedback_data: UnloaderFeedback | None = None

        self.client = ActionClient(
            self.node,
            OperateUnloader,
            f'{self.namespace}/operate_unloader',
        )

        self.logger.info(f'UnloaderActionClient initialized for {self.namespace}')

    # ------------------------------------------------------------------
    # Core Interface
    # ------------------------------------------------------------------

    def send_goal(self, target_position: int) -> bool:
        """
        Send a HOME (OperateUnloader.Goal.HOME) or END (OperateUnloader.Goal.END) goal.

        Non-blocking — result is delivered via get_status() / get_feedback().
        Returns False immediately if the server is unavailable or the goal is rejected.
        """
        if target_position not in _TARGET_LABEL:
            self.logger.error(
                f'Invalid target_position {target_position}. '
                f'Use OperateUnloader.Goal.HOME or OperateUnloader.Goal.END.'
            )
            return False

        self._target_label = _TARGET_LABEL[target_position]

        if not self.client.wait_for_server(timeout_sec=_SERVER_WAIT_S):
            self.logger.error(
                f'Unloader action server not available after {_SERVER_WAIT_S:.0f}s.'
            )
            self._current_status = RobotStatusEnum.ERROR
            return False

        goal = OperateUnloader.Goal()
        goal.target_position = target_position

        self.logger.info(f'Sending unloader goal: {self._target_label}')
        self._current_status = RobotStatusEnum.START_UNLOADING

        future = self.client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback,
        )
        future.add_done_callback(self._goal_response_callback)
        return True

    def cancel_goal(self) -> bool:
        """Cancel the currently active unloader goal."""
        if not self._goal_handle:
            self.logger.warning('No active unloader goal to cancel.')
            return False

        self.logger.info('Cancelling unloader goal…')
        future = self._goal_handle.cancel_goal_async()
        future.add_done_callback(self._goal_cancelled_callback)
        return True

    def reset(self) -> None:
        """Reset client to IDLE state."""
        self._current_status = RobotStatusEnum.IDLE
        self._goal_handle = None
        self._feedback_data = None
        self.logger.info('UnloaderActionClient reset to IDLE')

    def get_status(self) -> RobotStatusEnum:
        """Return the current operation status."""
        return self._current_status

    def get_feedback(self) -> UnloaderFeedback | None:
        """Return the latest feedback, or None if no goal has been sent yet."""
        return self._feedback_data

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _goal_response_callback(self, future: Future) -> None:
        result = future.result()
        if not result:
            self.logger.error('Failed to receive goal handle from unloader server.')
            self._current_status = RobotStatusEnum.ERROR
            return

        self._goal_handle = result

        if not self._goal_handle.accepted:
            self.logger.error(
                f'Unloader goal ({self._target_label}) rejected by server — '
                'server may still be homing or another goal is active.'
            )
            self._current_status = RobotStatusEnum.ERROR
            return

        self.logger.info(f'Unloader goal ({self._target_label}) accepted.')
        self._current_status = RobotStatusEnum.UNLOADING

        self._goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _goal_cancelled_callback(self, future: Future) -> None:
        cancel_response = future.result()
        if cancel_response and cancel_response.return_code == 0:
            self.logger.info('Unloader goal successfully cancelled.')
            self._current_status = RobotStatusEnum.IDLE
        else:
            self.logger.warning('Unloader goal cancellation failed.')
            self._current_status = RobotStatusEnum.ERROR

    def _feedback_callback(self, feedback_msg) -> None:
        fb: OperateUnloader.Feedback = feedback_msg.feedback

        direction_label = (
            'MOVING_TO_HOME'
            if fb.direction == OperateUnloader.Feedback.MOVING_TO_HOME
            else 'MOVING_TO_END'
        )

        self.logger.info(
            f'[{fb.progress_percent:5.1f}%] {direction_label} '
            f'steps={fb.step_count} '
            f'home_limit={int(fb.at_home_limit)} '
            f'end_limit={int(fb.at_end_limit)}'
        )

        self._feedback_data = UnloaderFeedback(
            status=self._current_status.value,
            target=self._target_label,
            progress_percent=fb.progress_percent,
            step_count=fb.step_count,
            at_home_limit=fb.at_home_limit,
            at_end_limit=fb.at_end_limit,
            feedback_message=direction_label,
        )

    def _result_callback(self, future: Future) -> None:
        response = future.result()
        if not response:
            self.logger.error('No result received from unloader action server.')
            self._current_status = RobotStatusEnum.ERROR
            return

        status = response.status
        result: OperateUnloader.Result = response.result

        result_label = _RESULT_LABEL.get(result.status, str(result.status))

        if status == GoalStatus.STATUS_SUCCEEDED and result.status in (
            OperateUnloader.Result.AT_HOME,
            OperateUnloader.Result.AT_END,
        ):
            self.logger.info(
                f'Unloader {self._target_label} complete: '
                f'{result_label} — "{result.message}"'
            )
            self._current_status = RobotStatusEnum.DONE_UNLOADING
        elif status == GoalStatus.STATUS_CANCELED:
            self.logger.info(f'Unloader goal cancelled: "{result.message}"')
            self._current_status = RobotStatusEnum.IDLE
        else:
            self.logger.error(
                f'Unloader goal failed: {result_label} — "{result.message}"'
            )
            self._current_status = RobotStatusEnum.ERROR

        self._feedback_data = UnloaderFeedback(
            status=self._current_status.value,
            target=self._target_label,
            progress_percent=100.0,
            step_count=self._feedback_data.step_count if self._feedback_data else 0,
            at_home_limit=result.status == OperateUnloader.Result.AT_HOME,
            at_end_limit=result.status == OperateUnloader.Result.AT_END,
            feedback_message=result_label,
        )
