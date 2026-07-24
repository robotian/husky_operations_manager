"""
test_drive_harvest_unload.py

Test node combining DriveClient bush harvesting with a full dock → unload → undock
sequence after all bushes in the row are done.

Sequence:
  STARTUP_HOMING   — send unloader HOME at boot; wait for AT_HOME confirmation
  HARVESTING_ROW   — DriveClient scans row; each STOPPED fires a simulated harvest
                     timer; after TOTAL_BUSHES harvests the row is complete
  DOCKING          — DockingActionClient navigates to and docks at unloading_station
  UNLOADING        — UnloaderActionClient drives carriage to END
  UNLOAD_WAIT      — one-shot timer holds for POST_UNLOAD_DELAY_SEC
  RETURNING_HOME   — UnloaderActionClient drives carriage back to HOME
  UNDOCKING        — UndockingActionClient departs from dock
  DONE             — logs outcome, node exits

No startup undocking sequence.

Run:
  ros2 run husky_operations_manager test_drive_harvest_unload \\
    --ros-args -r __ns:=/j100_0921 -r /tf:=tf -r /tf_static:=tf_static
"""

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from husky_operations_manager.action_clients.docking import DockingActionClient
from husky_operations_manager.action_clients.drive import DriveClient
from husky_operations_manager.action_clients.undocking import UndockingActionClient
from husky_operations_manager.action_clients.unloader import UnloaderActionClient
from husky_operations_manager.robot_enums import RobotStatusEnum
from husky_operations_manager.types import DriveConfig
from status_interfaces.action import OperateUnloader
from status_interfaces.msg import DockGoal, DriveFeedback, SubTask, UndockGoal

# Human-readable names for DriveFeedback's status constants — used for logging
# the plain int status value (get_status().status), since it's not an enum.
_STATUS_NAMES = {
    DriveFeedback.IDLE: 'IDLE',
    DriveFeedback.SCANNING: 'SCANNING',
    DriveFeedback.CONTROLLING: 'CONTROLLING',
    DriveFeedback.STOPPED: 'STOPPED',
    DriveFeedback.DEPARTING: 'DEPARTING',
    DriveFeedback.CANCELED: 'CANCELED',
    DriveFeedback.ERROR: 'ERROR',
}


# =============================================================================
# Hardcoded parameters — edit here for field tuning
# =============================================================================

# --- TF frames resolved at startup, merged into DriveConfig once available ---
BASE_FRAME = 'base_link'
CAMERA_FRAME = 'arm_camera_color_frame'
ARM_FRAME = 'arm_0_base_link'

# Everything except cam_tx/cam_ty/arm_tx_offset — those are resolved from TF
# in TestDriveHarvestUnloadNode._wait_for_tf and merged in before DriveClient
# is built.
STATIC_DRIVE_PARAMS = {
    # --- Subscriptions ---
    'detection_topic': 'manipulators/arm_detection/image_annotated/detection_pose',
    'odom_topic': 'ground_truth/odom',
    # --- cmd_vel ---
    'base_frame': BASE_FRAME,
    'cmd_vel_rate': 5.0,  # Hz — republish rate between detections
    # --- Stop condition ---
    'ex_tolerance': 0.02,  # m — bush level with arm tolerance
    # --- Speed limits ---
    # Husky A200 speed floors (empirical, this unit):
    #   Deadband (zero-motion threshold): 0.05 m/s — commands below this
    #   produce no physical motion. Locked value, do not go below.
    #   Accurate-tracking floor: 0.1 m/s — control loops track reliably
    #   at/above this; between deadband and floor, robot may move but
    'v_linear_min': 0.05,  # m/s — minimum speed near stop point
    'v_linear_max': 0.125,  # m/s — speed at first detection
    'v_angular_max': 0.15,  # rad/s — angular correction clamp
    # Turn radius floor: 0.02/0.15 = 0.13m
    # --- Departure ---
    'departure_clearance': 0.2,  # m — distance past bush before next scan
    # --- No-detection timeout ---
    # Converted to time: 1.0m / 0.1m/s = 10s
    'no_detection_distance': 1.0,  # m — row end assumed after this distance
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
    # --- Row geometry (drive.py) ---
    'bushrow_theta': 0.0,  # rad — row orientation in odom frame
}

TOTAL_BUSHES = 3
HARVEST_SIMULATION_DURATION_SEC = 10.0

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
    Drives 3 lavender bushes then docks, unloads, and undocks.

    Startup safety: unconditionally homes the unloader carriage before
    starting DriveClient — the action server completes immediately if the
    carriage is already at the home limit switch.
    """

    def __init__(self):
        super().__init__('test_drive_harvest_unload')

        self._namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'Node namespace: {self._namespace}')

        # --- State ---
        self._phase = _PHASE_STARTUP_HOMING
        self._bush_count = 0
        self._harvest_pending = False
        self._harvest_timer = None
        self._unload_wait_timer = None
        self._start_timer = None  # odom poll — created after HOME confirmed

        # DriveClient isn't constructed yet — cam_tx/cam_ty/arm_tx_offset come
        # from TF, resolved via a timer poll (see _wait_for_tf) since a
        # blocking lookup here in __init__ runs before rclpy.spin() ever
        # starts and would never see any /tf or /tf_static callbacks fire.
        # _poll_startup_homing gates the HARVESTING_ROW transition on this
        # being ready, in addition to the unloader HOME confirmation.
        self._drive_client: DriveClient | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_wait_timer = self.create_timer(0.2, self._wait_for_tf)

        # --- Action clients ---
        self._docking_client = DockingActionClient(self)
        self._unloader_client = UnloaderActionClient(self)
        self._undocking_client = UndockingActionClient(self)

        # --- 1 Hz monitor ---
        self._monitor_timer = self.create_timer(1.0, self._monitor_callback)

        # --- Startup safety: home the unloader before anything else ---
        self.get_logger().info('Startup safety check — sending unloader HOME')
        if not self._unloader_client.send_goal(OperateUnloader.Goal.HOME):
            self.get_logger().error('Startup HOME goal failed — aborting')
            self._phase = _PHASE_DONE

    # =========================================================================
    # Startup
    # =========================================================================

    def _wait_for_tf(self) -> None:
        """
        Poll at 0.2s until both the camera and arm static transforms are
        available, then build the full DriveConfig and construct DriveClient.
        Runs as a timer callback so it only executes once rclpy.spin() is
        actually running and servicing the TransformListener's subscriptions.
        """
        now = rclpy.time.Time()
        if not (
            self._tf_buffer.can_transform(BASE_FRAME, CAMERA_FRAME, now)
            and self._tf_buffer.can_transform(BASE_FRAME, ARM_FRAME, now)
        ):
            self.get_logger().info('Waiting for TF...', throttle_duration_sec=1.0)
            return

        self._tf_wait_timer.cancel()
        self._tf_wait_timer = None

        cam_tf = self._tf_buffer.lookup_transform(BASE_FRAME, CAMERA_FRAME, now)
        arm_tf = self._tf_buffer.lookup_transform(BASE_FRAME, ARM_FRAME, now)
        cam_tx, cam_ty = cam_tf.transform.translation.x, cam_tf.transform.translation.y
        arm_tx_offset = arm_tf.transform.translation.x

        self.get_logger().info(f'TF resolved | cam=({cam_tx:.3f}, {cam_ty:.3f}) | arm_tx_offset={arm_tx_offset:.3f}')

        drive_config = DriveConfig(
            **STATIC_DRIVE_PARAMS,
            cam_tx=cam_tx,
            cam_ty=cam_ty,
            arm_tx_offset=arm_tx_offset,
        )
        self._drive_client = DriveClient(self, drive_config)

    def _wait_for_odom(self) -> None:
        """Poll at 0.2 s until DriveClient confirms odom received, then scan."""
        if not self._drive_client.is_ready():
            self.get_logger().info('Waiting for odom...', throttle_duration_sec=1.0)
            return

        self._start_timer.cancel()
        self._start_timer = None
        self.get_logger().info('Odom received — calling DriveClient scan()')
        self._drive_client.scan()

    # =========================================================================
    # Main 1 Hz monitor
    # =========================================================================

    def _monitor_callback(self) -> None:
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
            pass  # terminal — monitor keeps running but does nothing

    # =========================================================================
    # Phase: STARTUP_HOMING
    # =========================================================================

    def _poll_startup_homing(self) -> None:
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

    # =========================================================================
    # Phase: HARVESTING_ROW
    # =========================================================================

    def _poll_harvesting_row(self) -> None:
        """
        On STOPPED (and no harvest already pending) — starts the one-shot
        harvest simulation timer, up to TOTAL_BUSHES times. Count-based cap
        only works here because this is a fixed-row test setup with a known
        bush count (see test_drive_client_lab.py for the same pattern).
        """
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
        self.get_logger().info(
            f'Bush {self._bush_count}/{TOTAL_BUSHES} reached — '
            f'starting simulated harvest | duration={HARVEST_SIMULATION_DURATION_SEC:.0f}s'
        )
        self._harvest_timer = self.create_timer(
            HARVEST_SIMULATION_DURATION_SEC,
            self._on_harvest_complete,
        )

    # =========================================================================
    # Harvest simulation
    # =========================================================================

    def _on_harvest_complete(self) -> None:
        """Fires once per bush after HARVEST_SIMULATION_DURATION_SEC."""
        if self._harvest_timer is not None:
            self._harvest_timer.cancel()
            self._harvest_timer = None

        self._harvest_pending = False

        if self._bush_count < TOTAL_BUSHES:
            self.get_logger().info(f'Harvest {self._bush_count} done — resuming for bush {self._bush_count + 1}')
            self._drive_client.resume()
        else:
            self.get_logger().info(f'All {TOTAL_BUSHES} bushes harvested — starting dock sequence')
            self._send_docking_goal()

    # =========================================================================
    # Phase: DOCKING
    # =========================================================================

    def _send_docking_goal(self) -> None:
        dock_goal = DockGoal()
        dock_goal.use_dock_id = True
        dock_goal.dock_id = DOCK_ID
        dock_goal.navigate_to_staging_pose = True

        subtask = SubTask()
        subtask.type = SubTask.DOCKING
        subtask.description = 'Dock for post-harvest unload'
        subtask.dock_goal = dock_goal

        self.get_logger().info(f"Sending docking goal | dock_id='{DOCK_ID}'")
        if not self._docking_client.send_docking_goal(subtask):
            self.get_logger().error('Failed to send docking goal — aborting')
            self._phase = _PHASE_DONE
            return

        self._phase = _PHASE_DOCKING

    def _poll_docking(self) -> None:
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
            self.get_logger().error('Docking failed — aborting sequence')
            self._phase = _PHASE_DONE

    # =========================================================================
    # Phase: UNLOADING
    # =========================================================================

    def _send_unloading_goal(self) -> None:
        self.get_logger().info('Sending unloader goal: END')
        if not self._unloader_client.send_goal(OperateUnloader.Goal.END):
            self.get_logger().error('Failed to send unloader END goal — aborting')
            self._phase = _PHASE_DONE
            return

        self._phase = _PHASE_UNLOADING

    def _poll_unloading(self) -> None:
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
            return

        self._phase = _PHASE_RETURNING_HOME

    # =========================================================================
    # Phase: RETURNING_HOME
    # =========================================================================

    def _poll_returning_home(self) -> None:
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

    # =========================================================================
    # Phase: UNDOCKING
    # =========================================================================

    def _send_undocking_goal(self) -> None:
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
            return

        self._phase = _PHASE_UNDOCKING

    def _poll_undocking(self) -> None:
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


# =============================================================================
# Entry point
# =============================================================================


def main(args=None):
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
