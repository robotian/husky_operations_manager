"""
test_drive_v2.py

Standalone test node for DriveClient v2 (drive_v2.py).

Simulates the parent harvesting node's interaction with DriveClient:
  1. Calls scan() on startup — robot moves forward
  2. DriveClient bearing+lateral controller guides robot to bush level point
  3. On STOPPED — simulates 10s harvest activity then calls resume()
  4. DriveClient departs past bush, resets, resumes SCANNING for next bush

Parameters are hardcoded for field tuning. No YAML loading.

Run:
  ros2 run husky_operations_manager test_drive_v2 --ros-args -r __ns:=/j100_0921
"""

import rclpy
from rclpy.node import Node


from husky_operations_manager.action_clients.drive_v2 import DriveClient, DriveConfig, DriveStatus


# =============================================================================
# Hardcoded parameters — edit here for field tuning
# =============================================================================

DRIVE_CONFIG = DriveConfig(
    # Detection
    detection_topic   = 'sensors/camera_1/detection/image_annotated/detection_pose',

    # cmd_vel
    base_frame        = 'base_link',
    cmd_vel_rate      = 10.0,           # Hz — republish rate between detections

    # Stop condition
    ex_tolerance      = 0.01,           # m — bush level with arm tolerance

    # Lateral convention
    ey_sign           = -1.0,           # bush is to the RIGHT of the robot

    # Controller gains
    k_phi             = 1.5,            # bearing error gain
    k_lateral         = 2.0,            # lateral offset gain

    # Speed limits
    v_linear_min      = 0.02,           # m/s — minimum speed near stop point
    v_linear_max      = 0.1,            # m/s — speed at first detection
    v_angular_max     = 0.25,           # rad/s — angular correction clamp

    # Departure
    departure_clearance    = 0.2,       # m — distance past bush before next scan

    # No-detection timeout
    no_detection_distance  = 1.0,       # m — row end assumed after this distance
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
    """

    def __init__(self):
        super().__init__('test_drive_v2')

        self._namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'Node namespace: {self._namespace}')
        self.get_logger().info(
            f'DriveConfig | '
            f'v_linear=[{DRIVE_CONFIG.v_linear_min}, {DRIVE_CONFIG.v_linear_max}]m/s | '
            f'v_angular_max={DRIVE_CONFIG.v_angular_max}rad/s | '
            f'k_phi={DRIVE_CONFIG.k_phi} k_lateral={DRIVE_CONFIG.k_lateral} | '
            f'ex_tolerance={DRIVE_CONFIG.ex_tolerance}m | '
            f'ey_sign={DRIVE_CONFIG.ey_sign:+.1f} | '
            f'departure_clearance={DRIVE_CONFIG.departure_clearance}m | '
            f'no_detection_distance={DRIVE_CONFIG.no_detection_distance}m'
        )

        self._drive_client = DriveClient(self, DRIVE_CONFIG)

        # 1Hz status monitor — checks DriveStatus and drives harvest simulation
        self._monitor_timer = self.create_timer(1.0, self._monitor_callback)

        # Harvest simulation timer — created on demand, fires once
        self._harvest_timer = None

        # Guard: prevent multiple resume() calls per STOPPED event
        self._harvest_pending = False

        # Start scanning immediately
        self._drive_client.scan()
        self.get_logger().info('DriveClient started — SCANNING')

    # =========================================================================
    # Monitor callback
    # =========================================================================

    def _monitor_callback(self) -> None:
        """
        1Hz status monitor.

        On STOPPED (and no harvest already pending) — starts the harvest
        simulation timer. On all other states — logs current status.
        """
        status = self._drive_client.get_status()
        self.get_logger().info(f'DriveClient status: {status.name}')

        if status == DriveStatus.STOPPED and not self._harvest_pending:
            self._harvest_pending = True
            self.get_logger().info(
                f'STOPPED — starting simulated harvest | '
                f'duration={HARVEST_SIMULATION_DURATION_SEC:.0f}s'
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

        Cancels the one-shot timer, calls resume() on DriveClient, and
        clears the harvest pending flag.
        """
        # Cancel and destroy the one-shot timer
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()