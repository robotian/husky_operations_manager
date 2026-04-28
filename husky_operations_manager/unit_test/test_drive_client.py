import rclpy
from rclpy.node import Node

from husky_operations_manager.action_clients.drive_client import DriveClient
from husky_operations_manager.dataclass import DriveConfig
from husky_operations_manager.enum import DriveStatus

from status_interfaces.msg import ImageDetectionPose


class TestDriveNode(Node):
    def __init__(self):
        super().__init__("test_drive_node")

        self.namespace = self.get_namespace().rstrip("/")
        self.detection_msg: ImageDetectionPose = ImageDetectionPose()

        drive_config = DriveConfig(
            base_frame="base_link",
            v_linear=0.2,
            v_angular=0.5,
            tf_polling_rate=10.0,
            timeout=30.0,
            tolerance=0.05,
            tf_base_frame="arm_0_base_link",
            tf_detection_frame="arm_0_detections",
        )

        self.sub = self.create_subscription(
            ImageDetectionPose,
            f"{self.namespace}/manipulators/arm_0_detection/image_annotated/detection_pose",
            self.detection_callback,
            10
        )

        self.drive_client = DriveClient(self, drive_config)
        self.create_timer(0.2, self._control_loop)

    def detection_callback(self, msg):
        self.detection_msg = msg

    def _control_loop(self):
        status = self.drive_client.get_status()
        self.get_logger().info(f"Current Status: {status}")

        if status in [DriveStatus.ERROR, DriveStatus.CANCELED]:
            return
        
        if self.detection_msg is None:
            self.get_logger().info(f"Please check if Image Detection is publishing on topic {self.sub.topic_name}")
        
        if self.detection_msg.detection_valid:
            self.drive_client.stop()
        else:
            self.drive_client.forward()


def main():
    rclpy.init()
    test_node = TestDriveNode()
    rclpy.spin(test_node)  # Keep alive if needed, or remove if not
    rclpy.shutdown()


if __name__ == "__main__":
    main()
