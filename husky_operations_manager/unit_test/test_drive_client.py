import rclpy
from rclpy.node import Node

from husky_operations_manager.action_clients.drive_client import DriveClient
from husky_operations_manager.dataclass import DriveConfig
from husky_operations_manager.enum import DriveStatus


class TestDriveNode(Node):
    def __init__(self):
        super().__init__('test_drive_node')
        
        drive_config = DriveConfig(
            base_frame="base_link",
            fixed_frame="map",
            v_linear=0.2,
            v_angular=0.5,
            tf_polling_rate=10.0,
            alignment_tolerance=0.05,
            tf_target_frame="map",
            tf_base_frame="arm_0_base_link",
            tf_detection_frame="arm_0_detections",
        )

        self.drive_client = DriveClient(self, drive_config)
        self.create_timer(0.2, self._control_loop)

    def _control_loop(self):
        status = self.drive_client.get_status()
        self.get_logger().info(f"Current Status: {status}")

        if status in [DriveStatus.ERROR, DriveStatus.CANCELED]:
            return
        
        self.drive_client.forward()


def main():
    rclpy.init()
    test_node = TestDriveNode()
    rclpy.spin(test_node)  # Keep alive if needed, or remove if not
    rclpy.shutdown()

if __name__ == '__main__':
    main()