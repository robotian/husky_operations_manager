#!/usr/bin/env python3
"""
Test node for the full Docking → Unloading → Undocking sequence.

State machine:
  DOCKING        → DockingActionClient sends dock goal; waits for DONE_DOCKING
  UNLOADING      → UnloaderActionClient sends END goal; waits for DONE_UNLOADING
  UNLOAD_WAIT    → one-shot timer holds for post_unload_delay_sec before returning home
  RETURNING_HOME → UnloaderActionClient sends HOME goal; waits for DONE_UNLOADING
  UNDOCKING      → UndockingActionClient sends undock goal; waits for DONE_UNDOCKING
  DONE           → logs result, shuts down

DockingParamFetcher runs in parallel from init so UndockingActionClient
has config ready by the time it's needed.

Usage:
    ros2 run husky_operations_manager test_dock_unload_undock \
        --ros-args -r __ns:=/husky_0
"""

import rclpy
from rclpy.node import Node

from status_interfaces.msg import SubTask, DockGoal, UndockGoal
from status_interfaces.action import OperateUnloader

from husky_operations_manager.robot_enums import RobotStatusEnum
from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.action_clients.unloader import UnloaderActionClient
from husky_operations_manager.action_clients.undocking import UndockingActionClient


class TestDockUnloadUndockNode(Node):
    """
    Integration test for the full dock → unload → undock sequence.

    Phases:
      DOCKING        send dock goal via DockingActionClient; advance on DONE_DOCKING
      UNLOADING      send END goal via UnloaderActionClient; advance on DONE_UNLOADING
      UNLOAD_WAIT    hold for post_unload_delay_sec via one-shot timer; then advance
      RETURNING_HOME send HOME goal via UnloaderActionClient; advance on DONE_UNLOADING
      UNDOCKING      send undock goal via UndockingActionClient; advance on DONE_UNDOCKING
      DONE           cancel timer, log outcome, shut down
    """

    _PHASE_DOCKING = 'DOCKING'
    _PHASE_UNLOADING = 'UNLOADING'
    _PHASE_UNLOAD_WAIT = 'UNLOAD_WAIT'
    _PHASE_RETURNING_HOME = 'RETURNING_HOME'
    _PHASE_UNDOCKING = 'UNDOCKING'
    _PHASE_DONE = 'DONE'

    def __init__(self):
        super().__init__('test_dock_unload_undock')

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'TestDockUnloadUndockNode starting | namespace={self.namespace}')

        self._phase = self._PHASE_DOCKING
        self._test_complete = False
        self._success = False
        self._wait_timer = None

        self.declare_parameter('post_unload_delay_sec', 5.0)
        self._post_unload_delay = float(self.get_parameter('post_unload_delay_sec').value)

        self.declare_parameter('undocking.dock_type', 'simple_charging_dock')
        self.declare_parameter('undocking.staging_x_offset', -0.7)
        self.declare_parameter('undocking.v_linear_min', 0.15)

        self._dock_type = str(self.get_parameter('undocking.dock_type').value)
        self._staging_x_offset = float(self.get_parameter('undocking.staging_x_offset').value)
        self._v_linear_min = float(self.get_parameter('undocking.v_linear_min').value)

        # Action clients
        self.docking_client = DockingActionClient(self)
        self.unloader_client = UnloaderActionClient(self)
        self.undocking_client: UndockingActionClient | None = None

        # 1 Hz main loop
        self._main_timer = self.create_timer(1.0, self._timer_callback)

        # Kick off docking immediately
        self._send_docking_goal()

    # =========================================================================
    # MAIN CONTROL LOOP
    # =========================================================================

    def _timer_callback(self):
        if self._test_complete:
            return

        if self._phase == self._PHASE_DOCKING:
            self._poll_docking()

        elif self._phase == self._PHASE_UNLOADING:
            self._poll_unloading()

        elif self._phase == self._PHASE_RETURNING_HOME:
            self._poll_returning_home()

        elif self._phase == self._PHASE_UNDOCKING:
            self._poll_undocking()

        elif self._phase == self._PHASE_DONE:
            self._finish()

    # =========================================================================
    # PHASE: DOCKING
    # =========================================================================

    def _send_docking_goal(self):
        dock_goal = DockGoal()
        dock_goal.use_dock_id = True
        dock_goal.dock_id = 'unloading_station'
        dock_goal.navigate_to_staging_pose = True

        sub_task = SubTask()
        sub_task.type = SubTask.DOCKING
        sub_task.description = 'Dock for unload sequence'
        sub_task.dock_goal = dock_goal

        self.get_logger().info('Sending docking goal...')
        if not self.docking_client.send_docking_goal(sub_task):
            self.get_logger().error('Failed to send docking goal — aborting sequence')
            self._phase = self._PHASE_DONE

    def _poll_docking(self):
        status = self.docking_client.get_status()
        self.get_logger().info(f'[DOCKING] status={status.name}')

        feedback = self.docking_client.get_feedback()
        if feedback:
            self.get_logger().debug(
                f'  dock={feedback.docking_location} | '
                f'msg={feedback.feedback_message} | '
                f'time={feedback.docking_time:.1f}s | '
                f'retries={feedback.num_retries}'
            )

        if status == RobotStatusEnum.DONE_DOCKING:
            self.get_logger().info('Docking SUCCEEDED — starting unloading')
            self.docking_client.reset()
            self._send_unloading_goal()

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().error('Docking FAILED — aborting sequence')
            self._phase = self._PHASE_DONE

    # =========================================================================
    # PHASE: UNLOADING
    # =========================================================================

    def _send_unloading_goal(self):
        self.get_logger().info('Sending unloader goal: END')
        if not self.unloader_client.send_goal(OperateUnloader.Goal.END):
            self.get_logger().error('Failed to send unloader goal — aborting sequence')
            self._phase = self._PHASE_DONE
            return
        self._phase = self._PHASE_UNLOADING

    def _poll_unloading(self):
        status = self.unloader_client.get_status()
        self.get_logger().info(f'[UNLOADING] status={status.name}')

        feedback = self.unloader_client.get_feedback()
        if feedback:
            self.get_logger().debug(
                f'  target={feedback.target} | '
                f'progress={feedback.progress_percent:.1f}% | '
                f'steps={feedback.step_count} | '
                f'home_limit={int(feedback.at_home_limit)} | '
                f'end_limit={int(feedback.at_end_limit)} | '
                f'msg={feedback.feedback_message}'
            )

        if status == RobotStatusEnum.DONE_UNLOADING:
            self.get_logger().info(
                f'Unloading SUCCEEDED — waiting {self._post_unload_delay:.1f}s before returning HOME'
            )
            self._phase = self._PHASE_UNLOAD_WAIT
            self._wait_timer = self.create_timer(self._post_unload_delay, self._on_unload_wait_done)

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().error('Unloading FAILED — aborting sequence')
            self._phase = self._PHASE_DONE

    def _on_unload_wait_done(self):
        """One-shot callback: fires once after post_unload_delay_sec, then cancels itself."""
        self._wait_timer.cancel()
        self._wait_timer = None
        self.get_logger().info('Post-unload wait complete — sending unloader HOME')
        self._send_return_home_goal()

    def _send_return_home_goal(self):
        self.get_logger().info('Sending unloader goal: HOME')
        if not self.unloader_client.send_goal(OperateUnloader.Goal.HOME):
            self.get_logger().error('Failed to send unloader HOME goal — aborting sequence')
            self._phase = self._PHASE_DONE
            return
        self._phase = self._PHASE_RETURNING_HOME

    def _poll_returning_home(self):
        status = self.unloader_client.get_status()
        self.get_logger().info(f'[RETURNING_HOME] status={status.name}')

        feedback = self.unloader_client.get_feedback()
        if feedback:
            self.get_logger().debug(
                f'  target={feedback.target} | '
                f'progress={feedback.progress_percent:.1f}% | '
                f'steps={feedback.step_count} | '
                f'home_limit={int(feedback.at_home_limit)} | '
                f'end_limit={int(feedback.at_end_limit)} | '
                f'msg={feedback.feedback_message}'
            )

        if status == RobotStatusEnum.DONE_UNLOADING:
            self.get_logger().info('Unloader returned HOME — starting undocking')
            self._init_undocking_client()
            self._send_undocking_goal()

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().error('Unloader HOME return FAILED — aborting sequence')
            self._phase = self._PHASE_DONE

    # =========================================================================
    # PHASE: UNDOCKING
    # =========================================================================

    def _init_undocking_client(self):
        """Build UndockingActionClient from declared parameters."""
        self.undocking_client = UndockingActionClient(self)

    def _send_undocking_goal(self):
        if self.undocking_client is None:
            self.get_logger().error('UndockingActionClient not initialised')
            self._phase = self._PHASE_DONE
            return

        max_undocking_time = (abs(self._staging_x_offset) / max(self._v_linear_min, 0.01)) * 2.0
        dock_type = self._dock_type

        undock_goal = UndockGoal()
        undock_goal.dock_type = dock_type
        undock_goal.max_undocking_time = max_undocking_time

        sub_task = SubTask()
        sub_task.type = SubTask.UNDOCKING
        sub_task.description = 'Undock after unload sequence'
        sub_task.undock_goal = undock_goal

        self.get_logger().info(
            f"Sending undocking goal | dock_type='{dock_type}' | max_undocking_time={max_undocking_time:.1f}s"
        )

        if not self.undocking_client.send_undocking_goal(sub_task):
            self.get_logger().error('Failed to send undocking goal — aborting sequence')
            self._phase = self._PHASE_DONE
            return

        self._phase = self._PHASE_UNDOCKING

    def _poll_undocking(self):
        status = self.undocking_client.get_status()
        self.get_logger().info(f'[UNDOCKING] status={status.name}')

        feedback = self.undocking_client.get_feedback()
        if feedback:
            self.get_logger().debug(
                f"  task='{feedback.task}' | location='{feedback.docking_location}' | msg='{feedback.feedback_message}'"
            )

        if status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info('Undocking SUCCEEDED — sequence complete')
            self.undocking_client.reset()
            self._success = True
            self._phase = self._PHASE_DONE

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().error('Undocking FAILED')
            self.undocking_client.reset()
            self._phase = self._PHASE_DONE

        elif status in (RobotStatusEnum.IDLE, RobotStatusEnum.START_UNDOCKING):
            self.get_logger().debug(f'Waiting for undocking to begin | status={status.name}')

    # =========================================================================
    # PHASE: DONE
    # =========================================================================

    def _finish(self):
        self._test_complete = True
        self._main_timer.cancel()
        outcome = 'SUCCESS' if self._success else 'FAILED'
        self.get_logger().info(
            '══════════════════════════════════════════\n'
            f'  Dock → Unload → Undock sequence: {outcome}\n'
            '══════════════════════════════════════════'
        )
        raise SystemExit

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def cancel_active_goals(self):
        if self._phase == self._PHASE_DOCKING:
            self.docking_client.cancel_goal()
        if self._phase == self._PHASE_UNDOCKING and self.undocking_client:
            self.undocking_client.cancel_goal()


def main(args=None):
    rclpy.init(args=args)
    node = TestDockUnloadUndockNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info('Interrupted — shutting down')
        node.cancel_active_goals()
    finally:
        node.get_logger().info('Test complete — shutting down')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
