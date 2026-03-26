"""
TestHarvestNode — integration test for HarvestingActionClient.

Sequence:
    1. GO STOW       — move arm to stow position from any unknown pose
    2. GO READY      — move arm to ready position
    3. START HARVEST — execute harvest; destroy node on success

Each step is sent as an independent goal. The poll timer checks feedback every
POLL_INTERVAL_SEC seconds. A step is considered complete when:
    - ManipulatorStatus in feedback transitions to IDLE
    - AND the last result was success == True

On any ERROR the node logs the failure and shuts down.
"""

import rclpy
from rclpy.node import Node

from status_interfaces.action import ManipulatorTask
from status_interfaces.msg import SubTask
from husky_operations_manager.enum import RobotStatusEnum, ManipulatorStatus
from husky_operations_manager.action_clients.manipulator import ArmCommand, ManipulatorTaskActionClient


# Seconds between each poll of get_feedback()
POLL_INTERVAL_SEC = 0.2


class _Step:
    """Ordered harvesting sequence steps."""
    GO_STOW      = 0
    GO_READY     = 1
    START_HARVEST = 2
    DONE         = 3


# Human-readable labels for logging
_STEP_LABELS = {
    _Step.GO_STOW:       "GO STOW",
    _Step.GO_READY:      "GO READY",
    _Step.START_HARVEST: "START HARVEST",
    _Step.DONE:          "DONE",
}

# Map each step to the ArmCommand string it sends
_STEP_COMMANDS = {
    _Step.GO_STOW:       ArmCommand.GO_STOW,
    _Step.GO_READY:      ArmCommand.GO_READY,
    _Step.START_HARVEST: ArmCommand.START_HARVEST,
}

# Map each step to a SubTask description used in feedback reporting
_STEP_DESCRIPTIONS = {
    _Step.GO_STOW:       "Moving arm to stow position",
    _Step.GO_READY:      "Moving arm to ready position",
    _Step.START_HARVEST: "Executing harvest sequence",
}


class TestHarvestNode(Node):
    """
    ROS2 test node that drives HarvestingActionClient through a fixed
    three-step sequence and self-destructs on success or error.
    """

    def __init__(self):
        super().__init__('test_harvest_node')

        self._client = ManipulatorTaskActionClient(self)

        # Sequence tracking
        self._step: int = _Step.GO_STOW
        self._goal_sent: bool = False
        self._last_feedback_state: int = -1
        self._step_result_success: bool = False

        # Poll timer — drives the entire state machine
        self._poll_timer = self.create_timer(
            POLL_INTERVAL_SEC,
            self._poll_callback
        )

        self.get_logger().info("TestHarvestNode started — beginning sequence from UNKNOWN pose.")
        self.get_logger().info(f"Step 1/{_Step.DONE}: {_STEP_LABELS[_Step.GO_STOW]}")

    # ------------------------------------------------------------------
    # Poll Timer — State Machine
    # ------------------------------------------------------------------

    def _poll_callback(self):
        """Called every POLL_INTERVAL_SEC to drive the step sequence."""

        # ── Finished ────────────────────────────────────────────────────
        if self._step == _Step.DONE: return

        # ── Send goal for current step ───────────────────────────────────
        if not self._goal_sent:
            self._send_step_goal(self._step)
            return

        # ── Wait for result + idle ───────────────────────────────────────
        feedback = self._client.get_feedback()
        robot_status = self._client.get_status()

        # Goal sent but no feedback yet — keep waiting
        if feedback is None: return

        current_feedback_state = feedback.feedback_message

        # Log only on state change
        if current_feedback_state != self._last_feedback_state:
            self._last_feedback_state = current_feedback_state
            self.get_logger().info(
                f"[{_STEP_LABELS[self._step]}] "
                f"feedback={current_feedback_state} | "
                f"robot_status={robot_status.name}"
            )

        # ── Error guard ──────────────────────────────────────────────────
        if robot_status == RobotStatusEnum.ERROR:
            self.get_logger().error(
                f"[{_STEP_LABELS[self._step]}] ERROR reported — aborting test."
            )
            self._shutdown()
            return

        # ── Step complete condition ──────────────────────────────────────
        # Complete when feedback_message is SUCCEEDED (result callback fired)
        # AND robot_status has returned to IDLE or DONE_HARVESTING
        step_complete = (
            current_feedback_state == "SUCCEEDED"
            and robot_status in (
                RobotStatusEnum.IDLE,
                RobotStatusEnum.DONE_HARVESTING,
            )
        )

        if step_complete:
            self.get_logger().info(f"[{_STEP_LABELS[self._step]}] complete ✓")
            self._advance_step()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_step_goal(self, step: int):
        """Build a SubTask for the given step and dispatch the goal."""
        subtask = SubTask()
        subtask.description = _STEP_DESCRIPTIONS[step]

        # Override the fixed ArmCommand the client sends for non-harvest steps
        arm_cmd = _STEP_COMMANDS[step]

        self.get_logger().info(
            f"[{_STEP_LABELS[step]}] Sending arm_task='{arm_cmd}' | "
            f"description='{subtask.description}'"
        )

        # For GO_STOW and GO_READY we need to send a custom arm command
        # rather than the hardcoded START_HARVEST inside send_harvesting_goal.
        # We use the internal _send_goal directly via a thin wrapper.
        success = self._dispatch_goal(subtask, arm_cmd)

        if not success:
            self.get_logger().error(f"[{_STEP_LABELS[step]}] Failed to send goal — aborting test.")
            self._shutdown()
            return

        self._goal_sent = True
        self._last_feedback_state = -1

    def _dispatch_goal(self, subtask: SubTask, arm_cmd: str) -> bool:
        """
        Dispatch a ManipulatorTask goal with the given arm_cmd string.

        Reuses HarvestingActionClient._send_goal() directly so GO_STOW
        and GO_READY can be sent without overriding the START_HARVEST
        default in send_harvesting_goal().
        """
        # Prepare client internal state so callbacks populate feedback correctly
        self._client._sub_task = subtask
        self._client._arm_command = arm_cmd
        self._client._feedback_state = -1
        self._client._feedback_data = None

        goal_msg = ManipulatorTask.Goal()
        goal_msg.arm_task = arm_cmd

        return self._client.send_goal(goal_msg)

    def _advance_step(self):
        """Move to the next step in the sequence, or finish."""
        self._step += 1
        self._goal_sent = False
        self._last_feedback_state = -1
        self._client.reset()

        if self._step == _Step.DONE:
            self.get_logger().info("All steps completed successfully — shutting down test node.")
            self._shutdown()
        else:
            self.get_logger().info(f"Step {self._step + 1}/{_Step.DONE}: {_STEP_LABELS[self._step]}")

    def _shutdown(self):
        """Cancel poll timer and request node shutdown."""
        self._poll_timer.cancel()
        self.get_logger().info("TestHarvestNode shutting down.")
        raise SystemExit


# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = TestHarvestNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()