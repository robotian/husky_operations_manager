"""Unit tests for the Drive Client functionality in Husky Operations Manager."""

from husky_operations_manager.action_clients.drive import DriveClient
from husky_operations_manager.robot_enums import DriveStatus
from husky_operations_manager.types import DriveConfig

import rclpy
from rclpy.node import Node

from status_interfaces.msg import ImageDetectionPose


class TestDriveNode(Node):
    """A ROS 2 node to test drive client operations, subscribing to detection poses and controlling movement."""

    def __init__(self):
        """Initialize the TestDriveNode, set up subscriptions, and start driving."""
        super().__init__('test_drive_node')

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'Node Namespace: {self.namespace}')
        self.detection_msg: ImageDetectionPose = None

        # --- Declare and read parameters ---
        self._declare_parameters()
        drive_config = self._build_drive_config()

        # --- DriveClient ---
        self._drive_client = DriveClient(self, drive_config)

        # --- Subscription to ImageDetectionPose ---
        self.sub = self.create_subscription(
            ImageDetectionPose,
            f'{self.namespace}/{self.image_detection_topic}',
            self._detection_callback,
            10,
        )

        self.get_logger().info(f'Subscribed to detection topic: {self.sub.topic_name}')

    # =========================================================================
    # PARAMETERS
    # =========================================================================

    def _declare_parameters(self) -> None:
        """Declare all ROS2 parameters with safe defaults."""
        self.declare_parameter('image_detection_topic', 'sensors/camera_1/detection/image_annotated/detection_pose')
        self.declare_parameter('control_loop_period', 0.2)
        self.declare_parameter('no_detection_timeout', 30.0)

        # DriveClient — match drive_client.yaml field names
        self.declare_parameter('drive.v_linear', 0.1)
        self.declare_parameter('drive.v_angular', 0.2)
        self.declare_parameter('drive.tf_polling_rate', 10.0)
        self.declare_parameter('drive.tolerance', 0.05)
        self.declare_parameter('drive.timeout', 30.0)
        self.declare_parameter('drive.base_frame', 'base_link')
        self.declare_parameter('drive.tf_base_frame', 'arm_0_base_link')
        self.declare_parameter('drive.tf_detection_frame', 'camera_1_detections')

    def _build_drive_config(self) -> DriveConfig:
        """Build DriveConfig from declared parameters — no hardcoded values."""
        self.image_detection_topic = str(self.get_parameter('image_detection_topic').value)
        self.get_logger().info(f'Detection Topic: {self.image_detection_topic}')

        config = DriveConfig(
            base_frame=str(self.get_parameter('drive.base_frame').value),
            v_linear=float(self.get_parameter('drive.v_linear').value),
            v_angular=float(self.get_parameter('drive.v_angular').value),
            tf_polling_rate=float(self.get_parameter('drive.tf_polling_rate').value),
            tolerance=float(self.get_parameter('drive.tolerance').value),
            timeout=float(self.get_parameter('drive.timeout').value),
            tf_base_frame=str(self.get_parameter('drive.tf_base_frame').value),
            tf_detection_frame=str(self.get_parameter('drive.tf_detection_frame').value),
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
        Process incoming ImageDetectionPose messages.

        If a valid detection is received, log the information and stop the drive client
        if the robot is not already idle.
        """
        if not msg.detection_valid:
            # --- Start moving ---
            self._drive_client.forward()
            self.get_logger().info('DriveClient starts — moving forward.')
            return

        status = self._drive_client.get_status()

        self.get_logger().info(
            f'Valid detection received | '
            f'center=({msg.center.x:.3f}, {msg.center.y:.3f}, {msg.center.z:.3f}) | '
            f'drive_status={status}'
        )

        if status == DriveStatus.IDLE:
            self.get_logger().info('Robot already stopped. Ignoring detection.')
            return

        self._drive_client.stop()


def main():
    """Initialize ROS 2, create the TestDriveNode, spin the node until interrupted, and shut down."""
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
