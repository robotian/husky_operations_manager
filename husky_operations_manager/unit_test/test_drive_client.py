import rclpy
from rclpy.node import Node

from husky_operations_manager.action_clients.drive_client import DriveClient
from husky_operations_manager.dataclass import DriveConfig
from husky_operations_manager.enum import DriveStatus
from status_interfaces.msg import ImageDetectionPose


class TestDriveNode(Node):
    """
    Standalone unit test node for DriveClient.

    Mirrors how LavenderHarvestNode uses DriveClient:
      - Owns the subscription to ImageDetectionPose
      - Calls drive_client.forward() once to start moving (guarded by is_active())
      - Calls drive_client.stop() event-driven in _detection_callback on detection_valid=True
      - DriveClient handles TF alignment validation and corrective motion internally
      - Control loop monitors status transitions and enforces no-detection timeout

    Changes from original test node:
      1. forward() guarded by is_active() — prevents alignment timer thrash
      2. stop() moved back to _detection_callback — event-driven, not polled
      3. detection_msg removed — detection handled at callback, not in loop
      4. _previous_status tracks transitions — logs only on state change
      5. No-detection timeout guards against unbounded forward drive
      6. DriveConfig read from ROS2 parameters — no hardcoded magic numbers
    """

    def __init__(self):
        super().__init__('test_drive_node')

        self._namespace = self.get_namespace().rstrip('/')

        # --- Declare and read parameters ---
        self._declare_parameters()
        drive_config = self._build_drive_config()

        # --- DriveClient ---
        self._drive_client = DriveClient(self, drive_config)

        # --- State tracking ---
        self._previous_status: DriveStatus = DriveStatus.IDLE
        self._drive_start_time: float | None = None
        self._no_detection_timeout: float = float(
            self.get_parameter('no_detection_timeout').value)

        # --- Subscription to ImageDetectionPose (event-driven) ---
        detection_topic = (
            f'{self._namespace}/manipulators/arm_0_detection'
            f'/image_annotated/detection_pose'
        )
        self._detection_sub = self.create_subscription(
            ImageDetectionPose,
            detection_topic,
            self._detection_callback,
            10
        )
        self.get_logger().info(
            f'Subscribed to detection topic: {detection_topic}')

        # --- Control loop ---
        timer_period = float(self.get_parameter('control_loop_period').value)
        self.create_timer(timer_period, self._control_loop)

        self.get_logger().info('TestDriveNode ready — starting forward drive.')

    # =========================================================================
    # PARAMETERS
    # =========================================================================

    def _declare_parameters(self) -> None:
        """Declare all ROS2 parameters with safe defaults."""
        self.declare_parameter('control_loop_period',   0.2)
        self.declare_parameter('no_detection_timeout',  30.0)

        # DriveClient — match drive_client.yaml field names
        self.declare_parameter('drive.v_linear',           0.1)
        self.declare_parameter('drive.v_angular',          0.2)
        self.declare_parameter('drive.tf_polling_rate',    10.0)
        self.declare_parameter('drive.tolerance',          0.05)
        self.declare_parameter('drive.timeout',            30.0)
        self.declare_parameter('drive.base_frame',         'base_link')
        self.declare_parameter('drive.tf_base_frame',      'arm_0_base_link')
        self.declare_parameter('drive.tf_detection_frame', 'arm_0_detections')

    def _build_drive_config(self) -> DriveConfig:
        """Build DriveConfig from declared parameters — no hardcoded values."""
        config = DriveConfig(
            base_frame         = str(self.get_parameter('drive.base_frame').value),
            v_linear           = float(self.get_parameter('drive.v_linear').value),
            v_angular          = float(self.get_parameter('drive.v_angular').value),
            tf_polling_rate    = float(self.get_parameter('drive.tf_polling_rate').value),
            tolerance          = float(self.get_parameter('drive.tolerance').value),
            timeout            = float(self.get_parameter('drive.timeout').value),
            tf_base_frame      = str(self.get_parameter('drive.tf_base_frame').value),
            tf_detection_frame = str(self.get_parameter('drive.tf_detection_frame').value),
        )
        self.get_logger().info(
            f'DriveConfig loaded | '
            f'v_linear={config.v_linear} v_angular={config.v_angular} | '
            f'tolerance={config.tolerance}m timeout={config.timeout}s | '
            f'base="{config.tf_base_frame}" detection="{config.tf_detection_frame}"'
        )
        return config

    # =========================================================================
    # DETECTION CALLBACK — event-driven stop
    # =========================================================================

    def _detection_callback(self, msg: ImageDetectionPose) -> None:
        """
        Event-driven detection handler.

        Calls drive_client.stop() exactly once per detection event.
        The IDLE guard prevents duplicate stop() calls if detections
        arrive faster than the DriveClient can process them.
        """
        if not msg.detection_valid:
            return

        status = self._drive_client.get_status()

        self.get_logger().info(
            f'Valid detection received | '
            f'center=({msg.center.x:.3f}, {msg.center.y:.3f}, {msg.center.z:.3f}) | '
            f'drive_status={status.name}'
        )

        if status == DriveStatus.IDLE:
            self.get_logger().debug(
                'Robot already stopped — ignoring detection.')
            return

        self.get_logger().info('Calling drive_client.stop() for alignment.')
        self._drive_client.stop()

    # =========================================================================
    # CONTROL LOOP — start motion and monitor status
    # =========================================================================

    def _control_loop(self) -> None:
        """
        Control loop — fires at control_loop_period (default 5Hz).

        Responsibilities:
          1. Start forward drive once if not already active (guarded by is_active())
          2. Enforce no-detection timeout — cancel drive if no detection arrives
          3. Log status transitions only — suppress repeated identical log lines
          4. Handle terminal states (ERROR, CANCELED) with clear diagnostics
        """
        current_status = self._drive_client.get_status()

        # FIX 5: Log only on state transition — not every tick
        if current_status != self._previous_status:
            self.get_logger().info(
                f'DriveClient status: {self._previous_status.name} → {current_status.name}'
            )
            self._previous_status = current_status

        # FIX 4: Terminal states — log and stop; do not restart
        if current_status in (DriveStatus.ERROR, DriveStatus.CANCELED):
            self.get_logger().warning(
                f'DriveClient reached terminal state: {current_status.name} — '
                f'stopping control loop activity. Call reset() to restart.'
            )
            return

        if current_status == DriveStatus.IDLE and self._drive_start_time is None:
            # First tick — start moving
            self.get_logger().info('Starting forward drive.')
            self._drive_start_time = self.get_clock().now().nanoseconds / 1e9
            self._drive_client.forward()
            return

        # FIX 1: Guard forward() — only call if not already active
        # Prevents alignment timer being cancelled and recreated every 200ms
        if not self._drive_client.is_active():
            return

        # FIX 6: No-detection timeout — cancel if detection never arrives
        if self._drive_start_time is not None:
            elapsed = (
                self.get_clock().now().nanoseconds / 1e9
                - self._drive_start_time
            )
            if elapsed >= self._no_detection_timeout:
                self.get_logger().error(
                    f'No detection received after {elapsed:.1f}s '
                    f'(timeout={self._no_detection_timeout:.1f}s) — '
                    f'canceling drive. Check image detection node is publishing on: '
                    f'{self._detection_sub.topic_name}'
                )
                self._drive_client.cancel()
                self._drive_start_time = None


def main(args=None):
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