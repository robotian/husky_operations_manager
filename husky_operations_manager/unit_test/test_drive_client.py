"""
test_drive_client.py

Standalone test node for DriveClient (drive.py).

Simulates the parent harvesting node's interaction with DriveClient:
  1. Calls scan() on startup — robot moves forward
  2. DriveClient heading correction drives msg.center.y → 0 (camera centered)
  3. DriveClient stops when |msg.center.x| <= ex_tolerance (arm level with bush)
  4. On STOPPED — simulates 10s harvest activity then calls resume()
  5. DriveClient departs past bush, resets controller, resumes SCANNING

Parameters are hardcoded for field tuning. No YAML loading.

Run:
  ros2 run husky_operations_manager test_drive_client \
    --ros-args -r __ns:=/a300_00036 -r /tf:=tf -r /tf_static:=tf_static \
    --log-level DriveClient:=debug
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from husky_operations_manager.action_clients.drive import DriveClient
from husky_operations_manager.types import DriveConfig
from status_interfaces.msg import DriveFeedback

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
CAMERA_FRAME = 'camera_1_color_optical_frame'
ODOM_FRAME = 'map'  # TF frame for drive.py's TF-based target-pose lookup

STATIC_DRIVE_PARAMS = {
    # --- Subscriptions ---
    'detection_topic': 'sensors/camera_1/detection/image_annotated/detection_pose',
    'odom_topic': 'platform/odom/filtered',  # 'ground_truth/odom',
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
    'no_detection_distance': 1.50,  # m — row end assumed after this distance
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
    # 'bushrow_theta': math.pi,  # rad — row orientation in odom frame
}

HARVEST_SIMULATION_DURATION_SEC = 30.0  # seconds — simulated harvest activity

# =============================================================================
# Test Node
# =============================================================================

# TODO: Get bushrow_theta from robots odom at the start of the node.


class TestDriveNode(Node):
    """
    Test node that runs DriveClient through one full cycle: scan for a bush, stop beside it, simulate harvesting, then resume.
    Resumes only if DriveClient's harvested_count went up; otherwise assumes row end and shuts down.
    """

    def __init__(self):
        """Build the TF listener and start the wait-for-TF timer."""
        super().__init__('test_drive_client')

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

        self._drive_client: DriveClient | None = None
        self._monitor_timer = None
        self._start_timer = None
        self._harvest_timer = None
        self._harvest_pending = False
        self._shutdown_timer = None

        self._last_harvested_count = 0

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._tf_wait_timer = self.create_timer(0.2, self._wait_for_tf)
        self.get_logger().info(f'Waiting for TF: {BASE_FRAME} -> {CAMERA_FRAME}...')

    # =========================================================================
    # Startup
    # =========================================================================

    def _wait_for_tf(self) -> None:
        """Poll until camera TF is available, then build DriveConfig and start DriveClient."""
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

        self._monitor_timer = self.create_timer(1.0, self._monitor_callback)

        self._start_timer = self.create_timer(0.2, self._wait_for_odom)
        self.get_logger().info('Waiting for odom before starting scan...')

    def _wait_for_odom(self) -> None:
        """Poll until DriveClient has received odom, then call scan()."""
        if not self._drive_client.is_ready():
            self.get_logger().info('Waiting for odom...', throttle_duration_sec=1.0)
            return

        self._start_timer.cancel()
        self._start_timer = None

        # Row heading taken from the first odom yaw — assumes the robot is parked
        # pointing down the row. Latched once, never re-sampled on resume(): the
        # robot's yaw drifts as it corrects toward each bush, so re-sampling would
        # rotate the row assumption a little further every cycle.
        odom_yaw = self._drive_client.get_status().current_yaw
        self._drive_client.set_bushrow_theta(odom_yaw)
        self.get_logger().info(
            f'bushrow_theta latched from first odom yaw: {odom_yaw:.4f}rad '
            f'({math.degrees(odom_yaw):.1f}deg)'
        )

        self.get_logger().info('Odom received — calling DriveClient scan()')
        self._drive_client.scan()

    # =========================================================================
    # Monitor callback
    # =========================================================================

    def _monitor_callback(self) -> None:
        """Check DriveClient status every second; shut down on ERROR, or gate harvest-resume on a new harvested_count."""
        feedback = self._drive_client.get_status()
        status = feedback.status
        self.get_logger().info(f'DriveClient status: {_STATUS_NAMES.get(status, status)}')

        if status == DriveFeedback.ERROR:
            self.get_logger().info(
                f'ERROR shutdown, shutting down node | '
                f'final pose=({feedback.current_x:.3f}, {feedback.current_y:.3f}, '
                f'{math.degrees(feedback.current_yaw):.1f}deg)'
            )
            rclpy.shutdown()

        if status != DriveFeedback.STOPPED or self._harvest_pending:
            return

        if feedback.harvested_count <= self._last_harvested_count:
            self.get_logger().warning(
                'STOPPED with no new bush harvested — holding for good '
                '(reason unknown: likely row end / sensor gap, not resuming)',
                throttle_duration_sec=5.0,
            )
            if self._shutdown_timer is None:
                self._shutdown_timer = self.create_timer(2.0, self._auto_shutdown)
            return

        self._last_harvested_count = feedback.harvested_count
        self._harvest_pending = True
        self.get_logger().info(
            f'Bush reached — starting simulated harvest | duration={HARVEST_SIMULATION_DURATION_SEC:.0f}s'
        )
        self._harvest_timer = self.create_timer(
            HARVEST_SIMULATION_DURATION_SEC,
            self._on_harvest_complete,
        )

    def _auto_shutdown(self) -> None:
        """Shut the node down 2 seconds after holding for good."""
        self._shutdown_timer.cancel()
        self._shutdown_timer = None
        feedback = self._drive_client.get_status()
        self.get_logger().info(
            f'Auto-shutdown — row end, shutting down node | '
            f'final pose=({feedback.current_x:.3f}, {feedback.current_y:.3f}, '
            f'{math.degrees(feedback.current_yaw):.1f}deg)'
        )
        rclpy.shutdown()

    # =========================================================================
    # Harvest simulation
    # =========================================================================

    def _on_harvest_complete(self) -> None:
        """Clear the harvest-pending flag and call resume() on DriveClient."""
        if self._harvest_timer is not None:
            self._harvest_timer.cancel()
            self._harvest_timer = None

        self._harvest_pending = False
        self.get_logger().info('Simulated harvest complete — calling resume()')
        self._drive_client.resume()


# =============================================================================
# Entry point
# =============================================================================


def main(args=None):
    """Run the node until interrupted."""
    rclpy.init(args=args)
    node = TestDriveNode()
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
