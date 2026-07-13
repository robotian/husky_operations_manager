"""
test_drive_v2.py

Standalone test node for DriveClient v2 (drive_v2.py).

Simulates the parent harvesting node's interaction with DriveClient:
  1. Calls scan() on startup — robot moves forward
  2. DriveClient heading correction drives msg.center.y → 0 (camera centered)
  3. DriveClient stops when |msg.center.x| <= ex_tolerance (arm level with bush)
  4. On STOPPED — simulates 10s harvest activity then calls resume()
  5. DriveClient departs past bush, resets controller, resumes SCANNING

Parameters are hardcoded for field tuning. No YAML loading.

Run:
  ros2 run husky_operations_manager test_drive_v2 \
    --ros-args -r __ns:=/j100_0921 -r /tf:=tf -r /tf_static:=tf_static \
    --log-level DriveClient:=debug
"""

import rclpy
from rclpy.node import Node

from husky_operations_manager.robot_enums import DriveStatus
from husky_operations_manager.types import DriveConfig
from husky_operations_manager.action_clients.drive import DriveClient


# =============================================================================
# Hardcoded parameters — edit here for field tuning
# =============================================================================

DRIVE_CONFIG = DriveConfig(
    # --- Subscriptions ---
    detection_topic='manipulators/arm_0_detection/image_annotated/detection_pose',
    odom_topic='ground_truth/odom',
    # --- cmd_vel ---
    base_frame='base_link',
    cmd_vel_rate=10.0,  # Hz — republish rate between detections
    # --- Stop condition ---
    ex_tolerance=0.01,  # m — bush level with arm tolerance
    stop_lookahead=0.05,  # m — stop 4cm before ex=0 to compensate detection drop coasting
    # Tune to: v_linear_min * expected_drop_gap_seconds
    # At v_linear_min=0.01m/s, 0.04m covers ~400ms drop gap
    # --- Coast gate ---
    ex_coast_gate=0.15,  # m — zero velocity on invalid detections within 10cm
    # --- Angular gate ---
    # Zero angular correction when |ex| <= ex_angular_gate.
    # Robot drives straight for the final approach segment.
    # Prevents yaw at the stop point.
    ex_angular_gate=0.05,  # m — 5x ex_tolerance
    # --- Controller gain ---
    # Single gain on camera Y centering error (heading error signal).
    # At ey=0.09m: angular_z = 0.5 * 0.09 = 0.045 rad/s
    # Turn radius at v_linear_min: 0.02 / 0.045 = 0.44m — gentle curve, no orbiting
    k_rho=0.5,
    # --- Speed limits ---
    v_linear_min=0.007,  # m/s — minimum speed near stop point
    v_linear_max=0.015,  # m/s — speed at first detection
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
    k_v_d=0.2,
    k_omega_p=0.4,
    k_omega_d=0.1,
    k_beta_p=1.0,
    k_beta_d=1.0,
    a_max=0.05,  # m/s^2
    alpha_max=0.3,  # rad/s^2
    backward_distance_threshold=1.0,  # m
    # --- Camera mount / row geometry (drive.py) ---
    cam_tx=-0.239,  # m — camera behind base_link
    cam_ty=-0.500,  # m — camera right of base_link
    bushrow_theta=0.0,  # rad — row orientation in odom frame
    arm_tx_offset=0.214,  # m — unused, kept for parity with reference sim
)

HARVEST_SIMULATION_DURATION_SEC = 10.0  # seconds — simulated harvest activity


# =============================================================================
# Test Node
# =============================================================================


class TestDriveV2Node(Node):
    """
    Test harness for DriveClient v2.

    Monitors DriveClient status at 1Hz. On STOPPED, simulates harvest
    activity for HARVEST_SIMULATION_DURATION_SEC then calls resume().
    Uses a one-shot ROS2 timer for harvest simulation — executor keeps
    spinning during the wait, detection callbacks continue to fire normally.
    """

    def __init__(self):
        super().__init__('test_drive_v2')

        self._namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'Node namespace: {self._namespace}')
        self.get_logger().info(
            f'DriveConfig | '
            f'v_linear=[{DRIVE_CONFIG.v_linear_min}, {DRIVE_CONFIG.v_linear_max}]m/s | '
            f'v_angular_max={DRIVE_CONFIG.v_angular_max}rad/s | '
            f'k_rho={DRIVE_CONFIG.k_rho} | '
            f'ex_tolerance={DRIVE_CONFIG.ex_tolerance}m | '
            f'stop_lookahead={DRIVE_CONFIG.stop_lookahead}m | '
            f'ex_coast_gate={DRIVE_CONFIG.ex_coast_gate}m | '
            f'ex_angular_gate={DRIVE_CONFIG.ex_angular_gate}m | '
            f'odom_topic={self._namespace}/{DRIVE_CONFIG.odom_topic} | '
            f'departure_clearance={DRIVE_CONFIG.departure_clearance}m | '
            f'no_detection_distance={DRIVE_CONFIG.no_detection_distance}m'
        )

        self._drive_client = DriveClient(self, DRIVE_CONFIG)

        # 1Hz status monitor
        self._monitor_timer = self.create_timer(1.0, self._monitor_callback)

        # One-shot harvest simulation timer — created on demand
        self._harvest_timer = None
        self._harvest_pending = False  # prevents multiple resume() calls per STOPPED

        # Poll at 0.2s until odom is received, then call scan()
        # Fixed delay is unreliable — odom may not arrive within any given window
        self._start_timer = self.create_timer(0.2, self._wait_for_odom)
        self.get_logger().info('Waiting for odom before starting scan...')

    # =========================================================================
    # Startup
    # =========================================================================

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
        harvest simulation timer.
        """
        status = self._drive_client.get_status()
        self.get_logger().info(f'DriveClient status: {status.name}')

        if status == DriveStatus.STOPPED and not self._harvest_pending:
            self._harvest_pending = True
            self.get_logger().info(
                f'STOPPED — starting simulated harvest | duration={HARVEST_SIMULATION_DURATION_SEC:.0f}s'
            )
            self._harvest_timer = self.create_timer(
                HARVEST_SIMULATION_DURATION_SEC,
                self._on_harvest_complete,
            )

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
        self.get_logger().info('Simulated harvest complete — calling resume()')
        self._drive_client.resume()


# =============================================================================
# Entry point
# =============================================================================


def main(args=None):
    rclpy.init(args=args)
    node = TestDriveV2Node()
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
