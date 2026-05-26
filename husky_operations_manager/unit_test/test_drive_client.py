"""
test_drive_client.py

Standalone test node for DriveClient sequential bush harvesting simulation.

Responsibilities:
  - Declare and load drive parameters from YAML
  - Instantiate DriveClient (owns detection subscription internally)
  - Subscribe to e-stop
  - Poll DriveClient status:
      SCANNING / CORRECTING / DEPARTING → robot moving, no action
      STOPPED  → start activity simulation timer, call resume() when done
      IDLE     → traversal complete (external stop or no-detection timeout)
      CANCELED → aborted

DriveClient owns all detection and motion logic.
This node owns the activity simulation timer and safety monitoring only.

Initialisation order (safety-critical):
  1. Declare parameters
  2. Read all parameters
  3. Construct DriveClient
  4. Subscribe to e-stop
  5. Create control loop timer
  6. Call start() — robot moves only after all monitors are active
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Bool

from husky_operations_manager.action_clients.drive import DriveClient, DriveConfig, DriveStatus


class TestDriveNode(Node):
    """
    Simulation test harness for DriveClient sequential bush harvesting.

    On STOPPED: starts activity_duration timer, calls drive_client.resume() on expiry.
    On timeout or e-stop: calls drive_client.stop().
    """

    def __init__(self):
        super().__init__('test_drive_node')

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'Node namespace: {self.namespace}')

        # Step 1 + 2: declare and read all parameters
        self._declare_parameters()
        self._no_detection_timeout: float = float(self.get_parameter('no_detection_timeout').value)
        self._control_loop_period: float = float(self.get_parameter('control_loop_period').value)
        self._activity_duration: float = float(self.get_parameter('simulation.activity_duration').value)
        drive_config = self._build_drive_config()

        # Step 3: construct DriveClient
        self._drive_client = DriveClient(self, drive_config)

        # Step 4: e-stop subscription
        self._estop_sub = self.create_subscription(
            Bool,
            f'{self.namespace}/platform/emergency_stop',
            self._estop_callback,
            qos_profile_sensor_data,
        )

        # Activity simulation timer — created on demand when STOPPED detected
        self._activity_timer = None

        # Step 5: control loop timer
        self._drive_started: bool = False
        self._start_time: float = self.get_clock().now().nanoseconds / 1e9
        self._control_timer = self.create_timer(self._control_loop_period, self._control_loop)

        # Step 6: start traversal
        self._drive_started = True
        self._drive_client.start()

        self.get_logger().info(
            f'TestDriveNode ready | '
            f'no_detection_timeout={self._no_detection_timeout}s | '
            f'activity_duration={self._activity_duration}s | '
            f'control_loop_period={self._control_loop_period}s'
        )

    # =========================================================================
    # PARAMETERS
    # =========================================================================

    def _declare_parameters(self) -> None:
        self.declare_parameter('control_loop_period', 0.1)
        self.declare_parameter('no_detection_timeout', 30.0)
        self.declare_parameter('simulation.activity_duration', 5.0)

        self.declare_parameter('drive.base_frame', 'base_link')
        self.declare_parameter('drive.v_linear', 0.1)
        self.declare_parameter('drive.v_angular', 0.2)
        self.declare_parameter('drive.tolerance', 0.05)
        self.declare_parameter('drive.center_y_correction_sign', -1.0)
        self.declare_parameter('drive.noise_margin', 0.02)
        self.declare_parameter('drive.departure_clearance', 0.15)
        self.declare_parameter('drive.detection_topic', 'manipulators/arm_0_detection/image_annotated/detection_pose')

    def _build_drive_config(self) -> DriveConfig:
        config = DriveConfig(
            base_frame=str(self.get_parameter('drive.base_frame').value),
            v_linear=float(self.get_parameter('drive.v_linear').value),
            v_angular=float(self.get_parameter('drive.v_angular').value),
            tolerance=float(self.get_parameter('drive.tolerance').value),
            center_y_correction_sign=float(self.get_parameter('drive.center_y_correction_sign').value),
            noise_margin=float(self.get_parameter('drive.noise_margin').value),
            departure_clearance=float(self.get_parameter('drive.departure_clearance').value),
            detection_topic=str(self.get_parameter('drive.detection_topic').value),
        )
        self.get_logger().info(
            f'DriveConfig | '
            f'v_linear={config.v_linear} v_angular={config.v_angular} | '
            f'tolerance={config.tolerance}m | '
            f'center_y_correction_sign={config.center_y_correction_sign} | '
            f'noise_margin={config.noise_margin}m | '
            f'departure_clearance={config.departure_clearance}m | '
            f'detection_topic={config.detection_topic}'
        )
        return config

    # =========================================================================
    # E-STOP
    # =========================================================================

    def _estop_callback(self, msg: Bool) -> None:
        if not msg.data:
            return
        self.get_logger().error('E-stop received — stopping DriveClient')
        self._cancel_activity_timer()
        self._drive_client.stop()
        self._control_timer.cancel()

    # =========================================================================
    # CONTROL LOOP
    # =========================================================================

    def _control_loop(self) -> None:
        """
        Poll DriveClient status at control_loop_period.

        SCANNING / CORRECTING / DEPARTING — robot moving, no action needed.
        STOPPED   — start activity simulation timer if not already running.
        IDLE      — traversal complete, cancel timer.
        CANCELED  — aborted, cancel timer.

        No-detection timeout covers both cases:
          - detection arrived then stopped (row end)
          - detection never arrived (misconfigured topic)
        """
        status = self._drive_client.get_status()

        if status == DriveStatus.STOPPED:
            if self._activity_timer is None:
                self.get_logger().info(
                    f'Bush aligned — starting activity timer ({self._activity_duration}s simulation)'
                )
                self._activity_timer = self.create_timer(self._activity_duration, self._activity_complete)
            return

        if status == DriveStatus.IDLE and self._drive_started:
            self.get_logger().info('DriveClient IDLE — traversal complete')
            self._control_timer.cancel()
            return

        if status == DriveStatus.CANCELED:
            self.get_logger().warning('DriveClient CANCELED')
            self._cancel_activity_timer()
            self._control_timer.cancel()
            return

        # --- No-detection timeout ---
        now = self.get_clock().now().nanoseconds / 1e9
        last_detection = self._drive_client.last_detection_time()
        elapsed = now - (last_detection if last_detection is not None else self._start_time)

        if elapsed >= self._no_detection_timeout:
            self.get_logger().warning(
                f'No detection for {elapsed:.1f}s '
                f'(timeout={self._no_detection_timeout}s) | '
                f'last_detection={"never" if last_detection is None else f"{elapsed:.1f}s ago"} '
                f'— stopping'
            )
            self._cancel_activity_timer()
            self._drive_client.stop()
            self._control_timer.cancel()

    def _activity_complete(self) -> None:
        """
        Simulation activity timer callback.

        Called when activity_duration has elapsed after bush alignment.
        Cancels the activity timer and calls resume() to begin departure.
        """
        self.get_logger().info('Activity complete (simulated) — calling resume()')
        self._cancel_activity_timer()
        self._drive_client.resume()

    def _cancel_activity_timer(self) -> None:
        """Cancel activity timer if running."""
        if self._activity_timer is not None:
            self._activity_timer.cancel()
            self._activity_timer = None


def main():
    rclpy.init()
    node = TestDriveNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
