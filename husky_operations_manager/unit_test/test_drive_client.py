import rclpy
from rclpy.node import Node

from husky_operations_manager.action_clients.drive_client_new import DriveClient
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
        self._drive_client = DriveClient(self, drive_config)
 
        # --- Subscription to ImageDetectionPose ---
        self.sub = self.create_subscription(
            ImageDetectionPose,
            f"{self.namespace}/manipulators/arm_0_detection/image_annotated/detection_pose",
            self.detection_callback,
            10
        )

        self.get_logger().info(
            f"Subscribed to detection topic: {self.sub.topic_name}")
 
        # --- Start moving ---
        self._drive_client.forward()
        self.get_logger().info("DriveClient started — moving forward.")
 
    def _detection_callback(self, msg: ImageDetectionPose) -> None:
        """
        Callback for ImageDetectionPose messages.
 
        Mirrors HuskyOpsManager behaviour:
          - Ignores messages where detection_valid=False
          - Calls drive_client.stop() on detection_valid=True
          - DriveClient internally validates pose via TF polling
            before publishing zero velocity
        """
        if not msg.detection_valid:
            return
 
        status = self._drive_client.get_status()
 
        self.get_logger().info(
            f"Valid detection received | "
            f"center=({msg.center.x:.3f}, {msg.center.y:.3f}, {msg.center.z:.3f}) | "
            f"drive_status={status}")
 
        if status == DriveStatus.IDLE:
            self.get_logger().info(
                "Robot already stopped. Ignoring detection.")
            return
 
        self._drive_client.stop()


def main():
    rclpy.init()
    test_node = TestDriveNode()
    rclpy.spin(test_node)  # Keep alive if needed, or remove if not
    rclpy.shutdown()


if __name__ == "__main__":
    main()
