"""
test_drive_client.py
 
Standalone test node for the updated DriveClient.
 
Exercises the full DriveClient state machine:
  IDLE → scan() → SCANNING → (detection) → STOPPED
       → resume() → DEPARTING → (clearance) → SCANNING → ...
 
A 1Hz timer monitors DriveStatus and calls resume() when STOPPED,
simulating a harvest activity completing instantly.
 
The node does NOT subscribe to ImageDetectionPose directly —
DriveClient owns that subscription internally.
"""

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from husky_operations_manager.action_clients.drive import DriveClient, DriveConfig, DriveStatus


class TestDriveNode(Node):
    """
    Test harness for DriveClient.

    Starts scanning on boot. When DriveClient reaches STOPPED,
    simulates instant harvest completion and calls resume().
    Logs all status transitions at 1Hz.
    """

    def __init__(self):
        super().__init__('test_drive_node')

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'Node namespace: {self.namespace}')

        self._declare_parameters()
        drive_config = self._build_drive_config()

        # DriveClient owns detection subscription and TF lookup internally
        self._drive_client = DriveClient(self, drive_config)

        # Monitor status and drive the state machine forward
        self._timer = self.create_timer(1.0, self._timer_callback)    

        # Start scanning immediately
        self._drive_client.scan()
        self.get_logger().info('DriveClient started — SCANNING')

    # =========================================================================
    # PARAMETERS
    # =========================================================================

    def _declare_parameters(self) -> None:
        self.declare_parameter('drive.base_frame',             'base_link')
        self.declare_parameter('drive.v_linear',               0.2)
        self.declare_parameter('drive.v_angular',              0.5)
        self.declare_parameter('drive.ex_tolerance',           0.01)
        self.declare_parameter('drive.ey_tolerance',           0.01)
        self.declare_parameter('drive.ey_sign',                -1.0)
        self.declare_parameter('drive.kp',                     1.0)
        self.declare_parameter('drive.departure_clearance',    0.3)
        self.declare_parameter('drive.no_detection_distance',  1.0)
        self.declare_parameter('drive.detection_topic',        'sensors/camera_1/detection/image_annotated/detection_pose')
        self.declare_parameter('drive.cmd_vel_rate',           10.0)
 
    def _build_drive_config(self) -> DriveConfig:
        config = DriveConfig(
            base_frame             = str(self.get_parameter('drive.base_frame').value),
            v_linear               = float(self.get_parameter('drive.v_linear').value),
            v_angular              = float(self.get_parameter('drive.v_angular').value),
            ex_tolerance           = float(self.get_parameter('drive.ex_tolerance').value),
            ey_tolerance           = float(self.get_parameter('drive.ey_tolerance').value),
            ey_sign                = float(self.get_parameter('drive.ey_sign').value),
            kp                     = float(self.get_parameter('drive.kp').value),
            departure_clearance    = float(self.get_parameter('drive.departure_clearance').value),
            no_detection_distance  = float(self.get_parameter('drive.no_detection_distance').value),
            detection_topic        = str(self.get_parameter('drive.detection_topic').value),
            cmd_vel_rate           = float(self.get_parameter('drive.cmd_vel_rate').value),
        )
        self.get_logger().info(
            f'DriveConfig loaded | '
            f'v_linear={config.v_linear} v_angular={config.v_angular} | '
            f'ex_tolerance={config.ex_tolerance}m ey_tolerance={config.ey_tolerance}m | '
            f'ey_sign={config.ey_sign:+.1f} | '
            f'departure_clearance={config.departure_clearance}m | '
            f'no_detection_distance={config.no_detection_distance}m'
        )
        return config


    # =========================================================================
    # TIMER — status monitor and state machine driver
    # =========================================================================

    def _timer_callback(self) -> None:
        """
        1Hz monitor loop.

        Logs current DriveStatus on every tick.
        When STOPPED: calls resume() to simulate instant harvest completion.
        When CANCELED/ERROR: logs and stops the timer.
        """
        status = self._drive_client.get_status()
        self.get_logger().info(f'DriveClient status: {status.name}')

        if status == DriveStatus.STOPPED:
            self.get_logger().info('STOPPED — simulating harvest activity complete, calling resume() after 10sec')
            self.get_clock().sleep_for(Duration(seconds=10.0))           
            self.get_logger().info('Resumed — simulated harvest activity.')
            self._drive_client.resume()

        elif status == DriveStatus.CANCELED:
            self.get_logger().warning('DriveClient CANCELED — stopping monitor')
            self._timer.cancel()

        elif status == DriveStatus.ERROR:
            self.get_logger().error('DriveClient ERROR — call reset() to recover')
            self._timer.cancel()


def main(args=None):
    rclpy.init(args=args)
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
