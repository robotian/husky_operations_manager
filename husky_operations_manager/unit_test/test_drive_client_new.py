import rclpy
from rclpy.node import Node

from husky_operations_manager.action_clients.drive_client import DriveClient
from husky_operations_manager.dataclass import DriveConfig
from husky_operations_manager.enum import DriveStatus
from status_interfaces.msg import ImageDetectionPose


class TestDriveNode(Node):
    """
    Standalone unit test node for DriveClient.

    Mirrors how HuskyOpsManager will use DriveClient:
      - Owns the subscription to ImageDetectionPose
      - Calls drive_client.forward() to start moving
      - Calls drive_client.stop() on detection_valid=True
      - DriveClient handles TF alignment validation and corrective motion internally
    """

    def __init__(self):
        super().__init__('test_drive_node')

        namespace = self.get_namespace().rstrip('/')
        self.detection_msg: ImageDetectionPose = ImageDetectionPose()

        # --- DriveClient config (hardcoded for testing) ---
        drive_config = DriveConfig(
            base_frame="base_link",
            v_linear=0.2,
            v_angular=0.5,
            tf_polling_rate=10.0,
            tolerance=0.05,
            timeout=30.0,
            tf_base_frame="arm_0_base_link",
            tf_detection_frame="arm_0_detections",
        )

        self._drive_client = DriveClient(self, drive_config)

        # --- Subscription to ImageDetectionPose ---
        self.sub = self.create_subscription(
            ImageDetectionPose,
            f"{namespace}/manipulators/arm_0_detection/image_annotated/detection_pose",
            self._detection_callback,
            10
        )
        self.get_logger().info(
            f"Subscribed to detection topic: {self.sub.topic_name}")

        # --- Start moving ---
        self.create_timer(0.2, self._control_loop)

    def _detection_callback(self, msg: ImageDetectionPose) -> None:
        self.detection_msg = msg

        # if not msg.detection_valid:
        #     return

        # status = self._drive_client.get_status()

        # self.get_logger().info(
        #     f"Valid detection received | "
        #     f"center=({msg.center.x:.3f}, {msg.center.y:.3f}, {msg.center.z:.3f}) | "
        #     f"drive_status={status}")

        # if status == DriveStatus.IDLE:
        #     self.get_logger().info(
        #         "Robot already stopped. Ignoring detection.")
        #     return

        # self._drive_client.stop()

    def _control_loop(self):
        status = self._drive_client.get_status()
        self.get_logger().info(f"Current Status: {status}")

        if status in [DriveStatus.ERROR, DriveStatus.CANCELED]:
            return
        
        if self.detection_msg is None:
            self.get_logger().info(f"Please check if Image Detection is publishing on topic {self.sub.topic_name}")
        
        if self.detection_msg.detection_valid:
            self.get_logger().info(
                f"Valid detection received | "
                f"center=({self.detection_msg.center.x:.3f}, {self.detection_msg.center.y:.3f}, {self.detection_msg.center.z:.3f}) | "
                f"drive_status={status}"
            )
            self._drive_client.stop()
        else:
            self._drive_client.forward()
            self.get_logger().info("DriveClient started — moving forward.")


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