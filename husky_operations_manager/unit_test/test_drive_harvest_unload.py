"""
test_drive_harvest_unload.py

Runs a Husky through one full row: harvest 3 bushes, dock, unload, undock.

The node moves through these phases in order:
    STARTUP_HOMING   - send the unloader carriage HOME and wait for it to arrive
    HARVESTING_ROW   - for each of 3 bushes: confirm the arm is READY, drive
                        to the bush, run the harvest, then repeat
    DOCKING          - navigate to and dock at unloading_station
    UNLOADING        - drive the unloader carriage to END
    UNLOAD_WAIT      - wait a few seconds before returning the carriage
    RETURNING_HOME   - drive the unloader carriage back to HOME
    UNDOCKING        - leave the dock
    DONE             - log the result and stop

Before the robot drives anywhere, the arm must be confirmed READY. The
node checks this every time before scan() or resume() and sends a
GO_READY command first if it isn't, so the arm is never mid-harvest
while the robot is moving.

On startup, the unloader is sent HOME even if the row hasn't started
yet. The unloader action server finishes immediately when the carriage
is already at the home limit switch, so this check is safe to run
unconditionally.

Run:
  ros2 run husky_operations_manager test_drive_harvest_unload \\
    --ros-args -r __ns:=/a200_0284 -r /tf:=tf -r /tf_static:=tf_static
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.action_clients.drive import DriveClient
from husky_operations_manager.action_clients.manipulator import ArmCommand, ManipulatorTaskActionClient
from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.action_clients.unloader import UnloaderActionClient
from husky_operations_manager.robot_enums import RobotStatusEnum
from husky_operations_manager.types import DriveConfig
from status_interfaces.action import OperateUnloader
from status_interfaces.msg import DockGoal, DriveFeedback, SubTask, UndockGoal

_STATUS_NAMES = {
    DriveFeedback.IDLE: 'IDLE',
    DriveFeedback.SCANNING: 'SCANNING',
    DriveFeedback.CONTROLLING: 'CONTROLLING',
    DriveFeedback.STOPPED: 'STOPPED',
    DriveFeedback.DEPARTING: 'DEPARTING',
    DriveFeedback.CANCELED: 'CANCELED',
    DriveFeedback.ERROR: 'ERROR',
    DriveFeedback.ABORTED: 'ABORTED',
}


# =============================================================================
# Hardcoded parameters — edit here for field tuning
# =============================================================================

# --- TF frames resolved at startup, merged into DriveConfig once available ---
BASE_FRAME = 'base_link'
CAMERA_FRAME = 'arm_camera_color_frame'
ODOM_FRAME = 'base_mocap'  # TF frame for drive.py's TF-based target-pose lookup

STATIC_DRIVE_PARAMS = {
    # --- Subscriptions ---
    'detection_topic': 'manipulators/arm_detection/image_annotated/detection_pose',
    'odom_topic': 'ground_truth/odom',
    # --- cmd_vel ---
    'base_frame': BASE_FRAME,
    'cmd_vel_rate': 10.0,  # Hz — republish rate between detections
    # --- Stop condition ---
    'ex_tolerance': 0.02,  # m — bush level with arm tolerance
    # --- Speed limits ---
    'v_linear_min': 0.05,  # m/s — minimum speed near stop point
    'v_linear_max': 0.125,  # m/s — speed at first detection
    'v_angular_max': 0.15,  # rad/s — angular correction clamp
    # --- Departure ---
    'departure_clearance': 0.2,  # m — distance past bush before next scan
    # --- No-detection timeout ---
    'no_detection_distance': 0.60,  # m — row end assumed after this distance
    # --- PD target-pose controller (drive.py) ---
    'ang_tol': 0.05,  # rad — final-heading tolerance (~3deg)
    'k_v_p': 0.2,
    'k_v_d': 0.07,
    'k_omega_p': 0.4,
    'k_omega_d': 0.1,
    'k_beta_p': 1.0,
    'k_beta_d': 0.4,
    'a_max': 0.05,  # m/s^2
    'alpha_max': 0.3,  # rad/s^2
    'backward_distance_threshold': 1.0,  # m
    'same_bush_threshold': 0.25,  # m — CONTROLLING re-lock accepted only within this of the currently locked target
    'controlling_timeout': 30.0,  # s — no goal reached within this long -> reset lock, retry same bush
    'max_controlling_retries': 3,  # retry attempts on same bush before giving up (-> ERROR)
    'controlling_retry_delay': 5.0,  # s — stopped wait between ABORTED and re-entering CONTROLLING
    # --- Row geometry (drive.py) ---
    'bushrow_theta': 0.0,  # rad — row orientation in odom frame
}

TOTAL_BUSHES = 3
MAX_HARVEST_RETRIES = 3
MAX_DOCKING_RETRIES = 3

# Unloading dock
DOCK_ID = 'unloading_station'
DOCK_TYPE = 'simple_charging_dock'
STAGING_X_OFFSET = -0.7  # m
V_LINEAR_MIN_UNDOCK = 0.15  # m/s
POST_UNLOAD_DELAY_SEC = 5.0


# =============================================================================
# Node
# =============================================================================

# Ordered phases:
# STARTUP_HOMING → HARVESTING_ROW → DOCKING → UNLOADING →
# UNLOAD_WAIT → RETURNING_HOME → UNDOCKING → DONE
_PHASE_STARTUP_HOMING = 'STARTUP_HOMING'
_PHASE_HARVESTING_ROW = 'HARVESTING_ROW'
_PHASE_DOCKING = 'DOCKING'
_PHASE_UNLOADING = 'UNLOADING'
_PHASE_UNLOAD_WAIT = 'UNLOAD_WAIT'
_PHASE_RETURNING_HOME = 'RETURNING_HOME'
_PHASE_UNDOCKING = 'UNDOCKING'
_PHASE_DONE = 'DONE'


class TestDriveHarvestUnloadNode(Node):
    """
    Test node for one full harvest row on a real robot: drive to each bush,
    harvest it, then dock, unload, and undock.
    """

    def __init__(self):
        """Set up TF listening and action clients, then start the unloader HOME check."""
        super().__init__('test_drive_harvest_unload')

        self._namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'Node namespace: {self._namespace}')

        self._phase = _PHASE_STARTUP_HOMING
        self._bush_count = 0
        self._harvest_pending = False
        self._harvest_retry_count = 0
        self._harvest_poll_timer = None
        self._docking_retry_count = 0
        self._unload_wait_timer = None
        self._start_timer = None

        self._last_confirmed_arm_command: str = ArmCommand.UNKNOWN
        self._arm_ready_wait_timer = None
        self._arm_ready_callback = None

        self._drive_client: DriveClient | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_wait_timer = self.create_timer(0.2, self._wait_for_tf)

        self._docking_client = DockingActionClient(self)
        self._unloader_client = UnloaderActionClient(self)
        self._undocking_client = UndockingActionClient(self)
        self._manipulator_client = ManipulatorTaskActionClient(self)

        self._monitor_timer = self.create_timer(1.0, self._monitor_callback)

        self.get_logger().info('Startup safety check — sending unloader HOME')
        if not self._unloader_client.send_goal(OperateUnloader.Goal.HOME):
            self.get_logger().error('Startup HOME goal failed — aborting')
            self._phase = _PHASE_DONE

    # =========================================================================
    # Startup
    # =========================================================================

    def _wait_for_tf(self) -> None:
        """Poll until camera TF is available, then build DriveClient."""
        now = Time()
        if not self._tf_buffer.can_transform(BASE_FRAME, CAMERA_FRAME, now):
            self.get_logger().info('Waiting for TF...', throttle_duration_sec=1.0)
            return

        self._tf_wait_timer.cancel()
        self._tf_wait_timer = None

        self.get_logger().info('Camera TF resolved — building DriveClient')

        drive_config = DriveConfig(
            **STATIC_DRIVE_PARAMS,
            camera_frame=CAMERA_FRAME,
            odom_frame=ODOM_FRAME,
        )
        self._drive_client = DriveClient(self, drive_config)

    def _wait_for_odom(self) -> None:
        """Poll at 0.2 s until DriveClient confirms odom received, then scan."""
        if not self._drive_client.is_ready():
            self.get_logger().info('Waiting for odom...', throttle_duration_sec=1.0)
            return

        self._start_timer.cancel()
        self._start_timer = None
        self.get_logger().info('Odom received — gating on arm READY before scan()')
        self._ensure_arm_ready(self._drive_client.scan)

    # =========================================================================
    # Arm gating — must be confirmed READY before any scan()/resume()
    # =========================================================================

    def _ensure_arm_ready(self, on_ready) -> None:
        """Run on_ready now if the arm is already READY, otherwise send GO_READY first."""
        if self._last_confirmed_arm_command == ArmCommand.GO_READY:
            on_ready()
            return

        subtask = SubTask()
        subtask.type = SubTask.HARVESTING
        subtask.description = 'Arm ready before DriveClient move'

        self.get_logger().info('Arm not confirmed READY — sending GO_READY')
        if not self._manipulator_client.send_ready_goal(subtask):
            self.get_logger().error('Failed to send GO_READY goal — aborting')
            self._phase = _PHASE_DONE
            rclpy.shutdown()
            return

        self._arm_ready_callback = on_ready
        self._arm_ready_wait_timer = self.create_timer(0.2, self._wait_for_arm_ready)

    def _wait_for_arm_ready(self) -> None:
        """Poll at 0.2s until GO_READY completes, then fire the stashed callback."""
        status = self._manipulator_client.get_status()
        feedback = self._manipulator_client.get_feedback()
        self.get_logger().info(f'[ARM_READY] feedback={feedback.feedback_message if feedback else None}')

        if feedback and feedback.arm_task == ArmCommand.GO_READY and feedback.feedback_message == 'SUCCEEDED':
            self._arm_ready_wait_timer.cancel()
            self._arm_ready_wait_timer = None
            self._manipulator_client.reset()
            self._last_confirmed_arm_command = ArmCommand.GO_READY
            self.get_logger().info('Arm READY confirmed')

            callback = self._arm_ready_callback
            self._arm_ready_callback = None
            callback()
            return

        if status == RobotStatusEnum.ERROR:
            self._arm_ready_wait_timer.cancel()
            self._arm_ready_wait_timer = None
            self.get_logger().error('Arm READY goal failed — aborting')
            self._phase = _PHASE_DONE
            rclpy.shutdown()

    # =========================================================================
    # Main 1 Hz monitor
    # =========================================================================

    def _monitor_callback(self) -> None:
        """Run the poll method for whichever phase the node is currently in."""
        if self._phase == _PHASE_STARTUP_HOMING:
            self._poll_startup_homing()

        elif self._phase == _PHASE_HARVESTING_ROW:
            self._poll_harvesting_row()

        elif self._phase == _PHASE_DOCKING:
            self._poll_docking()

        elif self._phase == _PHASE_UNLOADING:
            self._poll_unloading()

        elif self._phase == _PHASE_RETURNING_HOME:
            self._poll_returning_home()

        elif self._phase == _PHASE_UNDOCKING:
            self._poll_undocking()

        elif self._phase == _PHASE_DONE:
            pass

    # =========================================================================
    # Phase: STARTUP_HOMING
    # =========================================================================

    def _poll_startup_homing(self) -> None:
        """Wait for the unloader HOME goal to finish, then start the harvesting phase."""
        status = self._unloader_client.get_status()
        self.get_logger().info(f'[STARTUP_HOMING] unloader status: {status.name}')

        if status == RobotStatusEnum.DONE_UNLOADING:
            if self._drive_client is None:
                self.get_logger().info(
                    'Unloader HOME confirmed — waiting for TF before starting odom wait',
                    throttle_duration_sec=1.0,
                )
                return

            self._unloader_client.reset()
            self.get_logger().info('Unloader confirmed HOME — starting odom wait')
            self._phase = _PHASE_HARVESTING_ROW
            self._start_timer = self.create_timer(0.2, self._wait_for_odom)

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().error('Startup HOME check failed — aborting')
            self._phase = _PHASE_DONE
            rclpy.shutdown()

    # =========================================================================
    # Phase: HARVESTING_ROW
    # =========================================================================

    def _poll_harvesting_row(self) -> None:
        """Send a START_HARVEST goal each time DriveClient stops at a bush, up to TOTAL_BUSHES times."""
        status = self._drive_client.get_status().status
        self.get_logger().info(
            f'[HARVESTING_ROW] drive={_STATUS_NAMES.get(status, status)} | '
            f'bush={self._bush_count}/{TOTAL_BUSHES} | '
            f'harvest_pending={self._harvest_pending}'
        )

        if status != DriveFeedback.STOPPED or self._harvest_pending:
            return

        self._bush_count += 1
        self._harvest_pending = True
        self._harvest_retry_count = 0

        self.get_logger().info(f'Bush {self._bush_count}/{TOTAL_BUSHES} reached — sending START_HARVEST')
        if not self._send_harvest_goal():
            self.get_logger().error('Failed to send START_HARVEST goal — aborting')
            self._phase = _PHASE_DONE
            rclpy.shutdown()
            return

        self._harvest_poll_timer = self.create_timer(0.2, self._poll_harvest)

    # =========================================================================
    # Harvest
    # =========================================================================

    def _send_harvest_goal(self) -> bool:
        """Build and send a START_HARVEST goal for the current bush."""
        subtask = SubTask()
        subtask.type = SubTask.HARVESTING
        subtask.description = 'Executing harvest sequence'
        return self._manipulator_client.send_harvesting_goal(subtask)

    def _poll_harvest(self) -> None:
        """Poll at 0.2s until START_HARVEST completes, then advance the row."""
        status = self._manipulator_client.get_status()
        feedback = self._manipulator_client.get_feedback()
        self.get_logger().info(f'[HARVEST] feedback={feedback.feedback_message if feedback else None}')

        if status == RobotStatusEnum.ERROR:
            self._harvest_poll_timer.cancel()
            self._harvest_poll_timer = None
            self._manipulator_client.reset()

            if self._harvest_retry_count < MAX_HARVEST_RETRIES:
                self._harvest_retry_count += 1
                self.get_logger().warning(
                    f'Harvest goal failed — retrying ({self._harvest_retry_count}/{MAX_HARVEST_RETRIES})'
                )
                if not self._send_harvest_goal():
                    self.get_logger().error('Retry dispatch failed — aborting')
                    self._phase = _PHASE_DONE
                    rclpy.shutdown()
                    return
                self._harvest_poll_timer = self.create_timer(0.2, self._poll_harvest)
                return

            self.get_logger().error(f'Harvest goal failed after {MAX_HARVEST_RETRIES} retries — shutting down')
            self._phase = _PHASE_DONE
            rclpy.shutdown()
            return

        if not (
            feedback and feedback.arm_task == ArmCommand.START_HARVEST and feedback.feedback_message == 'SUCCEEDED'
        ):
            return

        self._harvest_poll_timer.cancel()
        self._harvest_poll_timer = None
        self._manipulator_client.reset()
        self._last_confirmed_arm_command = ArmCommand.START_HARVEST
        self._on_harvest_complete()

    def _on_harvest_complete(self) -> None:
        """Fires once per bush after START_HARVEST is confirmed complete."""
        if self._bush_count < TOTAL_BUSHES:
            self.get_logger().info(
                f'Harvest {self._bush_count} done — gating on arm READY before resume() for bush {self._bush_count + 1}'
            )
            self._ensure_arm_ready(self._resume_after_harvest)
        else:
            self._harvest_pending = False
            self.get_logger().info(f'All {TOTAL_BUSHES} bushes harvested — starting dock sequence')
            self._start_docking()

    def _resume_after_harvest(self) -> None:
        """Clears harvest_pending and resumes DriveClient in the same tick, closing the _poll_harvesting_row race."""
        self._harvest_pending = False
        self._drive_client.resume()

    # =========================================================================
    # Phase: DOCKING
    # =========================================================================

    def _send_docking_goal(self) -> bool:
        """Build and send the docking goal for unloading_station."""
        dock_goal = DockGoal()
        dock_goal.use_dock_id = True
        dock_goal.dock_id = DOCK_ID
        dock_goal.navigate_to_staging_pose = True

        subtask = SubTask()
        subtask.type = SubTask.DOCKING
        subtask.description = 'Dock for post-harvest unload'
        subtask.dock_goal = dock_goal

        self.get_logger().info(f"Sending docking goal | dock_id='{DOCK_ID}'")
        return self._docking_client.send_docking_goal(subtask)

    def _start_docking(self) -> None:
        """Reset the docking retry count, send the first docking goal, and switch to the DOCKING phase."""
        self._docking_retry_count = 0
        if not self._send_docking_goal():
            self.get_logger().error('Failed to send docking goal — aborting')
            self._phase = _PHASE_DONE
            rclpy.shutdown()
            return

        self._phase = _PHASE_DOCKING

    def _poll_docking(self) -> None:
        """Wait for docking to finish, then start the unloading goal."""
        status = self._docking_client.get_status()
        self.get_logger().info(f'[DOCKING] status={status.name}')

        feedback = self._docking_client.get_feedback()
        if feedback:
            self.get_logger().debug(
                f'  dock={feedback.docking_location} | '
                f'msg={feedback.feedback_message} | '
                f'time={feedback.docking_time:.1f}s | '
                f'retries={feedback.num_retries}'
            )

        if status == RobotStatusEnum.DONE_DOCKING:
            self.get_logger().info('Docking complete — starting unloading')
            self._docking_client.reset()
            self._send_unloading_goal()

        elif status == RobotStatusEnum.ERROR:
            self._docking_client.reset()

            if self._docking_retry_count < MAX_DOCKING_RETRIES:
                self._docking_retry_count += 1
                self.get_logger().warning(
                    f'Docking failed — retrying ({self._docking_retry_count}/{MAX_DOCKING_RETRIES})'
                )
                if not self._send_docking_goal():
                    self.get_logger().error('Retry dispatch failed — aborting')
                    self._phase = _PHASE_DONE
                    rclpy.shutdown()
                return

            self.get_logger().error(f'Docking failed after {MAX_DOCKING_RETRIES} retries — shutting down')
            self._phase = _PHASE_DONE
            rclpy.shutdown()

    # =========================================================================
    # Phase: UNLOADING
    # =========================================================================

    def _send_unloading_goal(self) -> None:
        """Send the unloader END goal and switch to the UNLOADING phase."""
        self.get_logger().info('Sending unloader goal: END')
        if not self._unloader_client.send_goal(OperateUnloader.Goal.END):
            self.get_logger().error('Failed to send unloader END goal — aborting')
            self._phase = _PHASE_DONE
            rclpy.shutdown()
            return

        self._phase = _PHASE_UNLOADING

    def _poll_unloading(self) -> None:
        """Wait for the carriage to reach END, then start the post-unload wait."""
        status = self._unloader_client.get_status()
        self.get_logger().info(f'[UNLOADING] status={status.name}')

        feedback = self._unloader_client.get_feedback()
        if feedback:
            self.get_logger().debug(
                f'  progress={feedback.progress_percent:.1f}% | '
                f'steps={feedback.step_count} | '
                f'end_limit={int(feedback.at_end_limit)}'
            )

        if status == RobotStatusEnum.DONE_UNLOADING:
            self.get_logger().info(f'Unloader at END — waiting {POST_UNLOAD_DELAY_SEC:.0f}s before HOME')
            self._phase = _PHASE_UNLOAD_WAIT
            self._unload_wait_timer = self.create_timer(POST_UNLOAD_DELAY_SEC, self._on_unload_wait_done)

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().error('Unloading failed — aborting sequence')
            self._phase = _PHASE_DONE
            rclpy.shutdown()

    # =========================================================================
    # Unload wait
    # =========================================================================

    def _on_unload_wait_done(self) -> None:
        """Fires once after POST_UNLOAD_DELAY_SEC; commands carriage HOME."""
        if self._unload_wait_timer is not None:
            self._unload_wait_timer.cancel()
            self._unload_wait_timer = None

        self.get_logger().info('Post-unload wait complete — sending unloader HOME')
        if not self._unloader_client.send_goal(OperateUnloader.Goal.HOME):
            self.get_logger().error('Failed to send unloader HOME goal — aborting')
            self._phase = _PHASE_DONE
            rclpy.shutdown()
            return

        self._phase = _PHASE_RETURNING_HOME

    # =========================================================================
    # Phase: RETURNING_HOME
    # =========================================================================

    def _poll_returning_home(self) -> None:
        """Wait for the carriage to reach HOME, then start undocking."""
        status = self._unloader_client.get_status()
        self.get_logger().info(f'[RETURNING_HOME] status={status.name}')

        feedback = self._unloader_client.get_feedback()
        if feedback:
            self.get_logger().debug(
                f'  progress={feedback.progress_percent:.1f}% | '
                f'steps={feedback.step_count} | '
                f'home_limit={int(feedback.at_home_limit)}'
            )

        if status == RobotStatusEnum.DONE_UNLOADING:
            self.get_logger().info('Unloader at HOME — starting undocking')
            self._unloader_client.reset()
            self._send_undocking_goal()

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().error('Unloader HOME return failed — aborting')
            self._phase = _PHASE_DONE
            rclpy.shutdown()

    # =========================================================================
    # Phase: UNDOCKING
    # =========================================================================

    def _send_undocking_goal(self) -> None:
        """Send the undocking goal and switch to the UNDOCKING phase."""
        max_undocking_time = (abs(STAGING_X_OFFSET) / max(V_LINEAR_MIN_UNDOCK, 0.01)) * 2.0

        undock_goal = UndockGoal()
        undock_goal.dock_type = DOCK_TYPE
        undock_goal.max_undocking_time = max_undocking_time

        subtask = SubTask()
        subtask.type = SubTask.UNDOCKING
        subtask.description = 'Undock after unload sequence'
        subtask.undock_goal = undock_goal

        self.get_logger().info(
            f"Sending undocking goal | dock_type='{DOCK_TYPE}' | max_undocking_time={max_undocking_time:.1f}s"
        )
        if not self._undocking_client.send_undocking_goal(subtask):
            self.get_logger().error('Failed to send undocking goal — aborting')
            self._phase = _PHASE_DONE
            rclpy.shutdown()
            return

        self._phase = _PHASE_UNDOCKING

    def _poll_undocking(self) -> None:
        """Wait for undocking to finish and end the sequence, logging success or failure."""
        status = self._undocking_client.get_status()
        self.get_logger().info(f'[UNDOCKING] status={status.name}')

        feedback = self._undocking_client.get_feedback()
        if feedback:
            self.get_logger().debug(f"  task='{feedback.task}' | msg='{feedback.feedback_message}'")

        if status == RobotStatusEnum.DONE_UNDOCKING:
            self.get_logger().info(
                '\n══════════════════════════════════════════════\n'
                '  Harvest row complete — dock → unload → undock: SUCCESS\n'
                '══════════════════════════════════════════════'
            )
            self._undocking_client.reset()
            self._phase = _PHASE_DONE

        elif status == RobotStatusEnum.ERROR:
            self.get_logger().error('Undocking failed')
            self._undocking_client.reset()
            self._phase = _PHASE_DONE
            rclpy.shutdown()


# =============================================================================
# Entry point
# =============================================================================


def main(args=None):
    """Start the node and spin until shutdown."""
    rclpy.init(args=args)
    node = TestDriveHarvestUnloadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
