"""
test_drive_client_lab.py

LAB-ONLY test node for DriveClient (drive.py) — assumes a known, fixed
number of bushes in the row (TOTAL_BUSHES). Auto-resumes STOPPED up to
that many times, then holds for good.

Do NOT use this in the field: real rows have a variable, unknown bush
count, so the count-based cap can't distinguish "goal reached" from
"no-detection timeout / row end" — it just bounds the blast radius for a
controlled lab setup where the bush count is known in advance. For general
use, see test_drive_client.py, which gates resume() on an actual
last-detection-valid flag instead of a hardcoded count.

Simulates the parent harvesting node's interaction with DriveClient:
  1. Calls scan() on startup — robot moves forward
  2. DriveClient heading correction drives msg.center.y → 0 (camera centered)
  3. DriveClient stops when |msg.center.x| <= ex_tolerance (arm level with bush)
  4. On STOPPED — simulates 10s harvest activity then calls resume()
  5. DriveClient departs past bush, resets controller, resumes SCANNING
  6. After TOTAL_BUSHES auto-resumes, holds for good

Parameters are hardcoded for field tuning. No YAML loading.

Run:
  ros2 run husky_operations_manager test_drive_client_lab \
    --ros-args -r __ns:=/a200_0284 -r /tf:=tf -r /tf_static:=tf_static \
    --log-level DriveClient:=debug
"""

import math

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from status_interfaces.msg import DriveFeedback
from husky_operations_manager.types import DriveConfig
from husky_operations_manager.action_clients.drive import DriveClient

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
ARM_FRAME = 'arm_base_link'

# Everything except cam_tx/cam_ty/arm_tx_offset — those are resolved from TF
# in TestDriveV2Node._wait_for_tf and merged in before DriveClient is built.
STATIC_DRIVE_PARAMS = dict(
    # --- Subscriptions ---
    detection_topic='manipulators/arm_detection/image_annotated/detection_pose',
    odom_topic='ground_truth/odom',
    # --- cmd_vel ---
    base_frame=BASE_FRAME,
    cmd_vel_rate=5.0,  # Hz — republish rate between detections
    # --- Stop condition ---
    ex_tolerance=0.02,  # m — bush level with arm tolerance
    # --- Speed limits ---
    # Husky A200 speed floors (empirical, this unit):
    #   Deadband (zero-motion threshold): 0.05 m/s — commands below this
    #   produce no physical motion. Locked value, do not go below.
    #   Accurate-tracking floor: 0.1 m/s — control loops track reliably
    #   at/above this; between deadband and floor, robot may move but
    v_linear_min=0.05,  # m/s — minimum speed near stop point
    v_linear_max=0.125,  # m/s — speed at first detection
    v_angular_max=0.15,  # rad/s — angular correction clamp
    # Turn radius floor: 0.02/0.15 = 0.13m
    # --- Departure ---
    departure_clearance=0.2,  # m — distance past bush before next scan
    # --- No-detection timeout ---
    # Converted to time: 1.0m / 0.1m/s = 10s
    no_detection_distance=1.0,  # m — row end assumed after this distance
    # --- PD target-pose controller (drive.py) ---
    ang_tol=0.05,  # rad — final-heading tolerance (~3deg)
    k_v_p=0.2,
    k_v_d=0.07,
    k_omega_p=0.4,
    k_omega_d=0.1,
    k_beta_p=1.0,
    k_beta_d=0.4,
    a_max=0.05,  # m/s^2
    alpha_max=0.3,  # rad/s^2
    backward_distance_threshold=1.0,  # m
    # --- Row geometry (drive.py) ---
    bushrow_theta=0.0,  # rad — row orientation in odom frame
)

TOTAL_BUSHES = 3  # LAB ONLY — known bush count for this test row
HARVEST_SIMULATION_DURATION_SEC = 30.0  # seconds — simulated harvest activity

# =============================================================================
# Test Node
# =============================================================================


class TestDriveNodeLab(Node):
    """
    LAB-ONLY test harness for DriveClient — known, fixed bush count.

    Monitors DriveClient status at 1Hz. On STOPPED, simulates harvest
    activity for HARVEST_SIMULATION_DURATION_SEC then calls resume() —
    capped at TOTAL_BUSHES auto-resumes, then holds for good.
    """

    def __init__(self):
        super().__init__('test_drive_client_lab')

        self._namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'Node namespace: {self._namespace}')
        self.get_logger().info(
            f'DriveConfig | '
            f'v_linear=[{STATIC_DRIVE_PARAMS["v_linear_min"]}, {STATIC_DRIVE_PARAMS["v_linear_max"]}]m/s | '
            f'v_angular_max={STATIC_DRIVE_PARAMS["v_angular_max"]}rad/s | '
            f'ex_tolerance={STATIC_DRIVE_PARAMS["ex_tolerance"]}m | '
            f'odom_topic={self._namespace}/{STATIC_DRIVE_PARAMS["odom_topic"]} | '
            f'departure_clearance={STATIC_DRIVE_PARAMS["departure_clearance"]}m | '
            f'no_detection_distance={STATIC_DRIVE_PARAMS["no_detection_distance"]}m'
        )

        # DriveClient isn't constructed yet — cam_tx/cam_ty/arm_tx_offset come
        # from TF, resolved below via a timer poll (like _wait_for_odom), since
        # a blocking lookup here in __init__ runs before rclpy.spin() ever
        # starts and would never see any /tf or /tf_static callbacks fire.
        self._drive_client: DriveClient | None = None
        self._monitor_timer = None
        self._start_timer = None
        self._bush_count = 0
        self._harvest_timer = None
        self._harvest_pending = False  # prevents multiple resume() calls per STOPPED
        self._shutdown_timer = None  # one-shot, set once all bushes are done

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._tf_wait_timer = self.create_timer(0.2, self._wait_for_tf)
        self.get_logger().info(f'Waiting for TF: {BASE_FRAME} -> {CAMERA_FRAME}, {ARM_FRAME}...')

    # =========================================================================
    # Startup
    # =========================================================================

    def _wait_for_tf(self) -> None:
        """
        Poll at 0.2s until both the camera and arm static transforms are
        available, then build the full DriveConfig and construct DriveClient.

        Runs as a timer callback (not blocking in __init__) so it only
        executes once rclpy.spin() is actually running and servicing the
        TransformListener's /tf and /tf_static subscriptions.
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

        # 1Hz status monitor
        self._monitor_timer = self.create_timer(1.0, self._monitor_callback)

        # Poll at 0.2s until odom is received, then call scan()
        # Fixed delay is unreliable — odom may not arrive within any given window
        self._start_timer = self.create_timer(0.2, self._wait_for_odom)
        self.get_logger().info('Waiting for odom before starting scan...')

    def _wait_for_odom(self) -> None:
        """
        Poll at 0.2s until DriveClient confirms odom received, then call scan().

        Avoids fixed delay race condition — scan() is only called once odom
        has actually been received and ey_sign can be correctly computed.
        """
        if not self._drive_client.is_ready():
            self.get_logger().info('Waiting for odom...', throttle_duration_sec=1.0)
            return

        self._start_timer.cancel()
        self._start_timer = None
        self.get_logger().info('Odom received — calling DriveClient scan()')
        self._drive_client.scan()

    # =========================================================================
    # Monitor callback
    # =========================================================================

    def _monitor_callback(self) -> None:
        """
        1Hz status monitor.

        On STOPPED (and no harvest already pending) — starts the one-shot
        harvest simulation timer, up to TOTAL_BUSHES times. After that,
        holds for good — DriveClient can't distinguish "goal reached" from
        "no-detection timeout / row end" (both funnel into the same
        DriveFeedback.STOPPED); TOTAL_BUSHES only works here because this is
        a lab setup with a known, fixed bush count.
        """
        status = self._drive_client.get_status().status
        self.get_logger().info(f'DriveClient status: {_STATUS_NAMES.get(status, status)}')

        if status != DriveFeedback.STOPPED or self._harvest_pending:
            return

        if self._bush_count >= TOTAL_BUSHES:
            self.get_logger().warning(
                f'STOPPED — {TOTAL_BUSHES}/{TOTAL_BUSHES} bushes done, holding for good (no more auto-resume)',
                throttle_duration_sec=5.0,
            )
            if self._shutdown_timer is None:
                self._shutdown_timer = self.create_timer(2.0, self._auto_shutdown)
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

    def _auto_shutdown(self) -> None:
        """One-shot, fires 2s after all bushes are done — shuts the node down."""
        self._shutdown_timer.cancel()
        self._shutdown_timer = None
        feedback = self._drive_client.get_status()
        self.get_logger().info(
            f'Auto-shutdown — all bushes done, shutting down node | '
            f'final pose=({feedback.current_x:.3f}, {feedback.current_y:.3f}, '
            f'{math.degrees(feedback.current_yaw):.1f}deg)'
        )
        rclpy.shutdown()

    # =========================================================================
    # Harvest simulation
    # =========================================================================

    def _on_harvest_complete(self) -> None:
        """
        Fires once after HARVEST_SIMULATION_DURATION_SEC.

        Cancels the one-shot timer, clears harvest pending flag,
        calls resume() on DriveClient.
        """
        if self._harvest_timer is not None:
            self._harvest_timer.cancel()
            self._harvest_timer = None

        self._harvest_pending = False
        # Clear before resume() — otherwise a stale True from the bush we
        # just left could wrongly validate a STOPPED for the next one,
        # before any fresh detection has actually arrived for it.
        self._last_detection_valid = False
        self.get_logger().info('Simulated harvest complete — calling resume()')
        self._drive_client.resume()


# =============================================================================
# Entry point
# =============================================================================


def main(args=None):
    rclpy.init(args=args)
    node = TestDriveNodeLab()
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
