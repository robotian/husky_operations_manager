import tf2_ros

from geometry_msgs.msg import TwistStamped
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node

from husky_operations_manager.enum import DriveStatus
from husky_operations_manager.dataclass import DriveConfig

# ---------------------------------------------------------------------------
# DriveClient
# ---------------------------------------------------------------------------

class DriveClient:
    def __init__(self, node: Node, config:DriveConfig) -> None:
        self.node      = node   
        self.logger    = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip('/')

        # --- Congig ---
        self.base_frame:  str   = config.base_frame
        self.fixed_frame: str   = config.fixed_frame
        self.v_linear:  float = config.v_linear
        self.v_angular: float = config.v_angular

        # --- Status ---
        self._status:     DriveStatus  = DriveStatus.IDLE

        # --- TF ---
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

        # --- Publishers ---
        self._cmd_vel_pub = self.node.create_publisher(
            TwistStamped, f'{self.namespace}/cmd_vel', 10)
        

        self.logger.info(
            f"DriveClient initialized with Linear Velocity: {self.v_linear} and Angular Velocity: {self.v_angular}")


    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Cancel an active drive and publish zero velocity."""
        if self._status not in (DriveStatus.FORWARD, DriveStatus.REVERSE):
            return
        self.stop()
        self._status = DriveStatus.CANCELED
        self.logger.info(f'Drive canceled, status: {self._status}')


    def get_status(self) -> DriveStatus:
        """Return the current DriveStatus."""
        return self._status

    def is_active(self) -> bool:
        """Return True if a drive is currently in progress."""
        return self._status in (DriveStatus.FORWARD, DriveStatus.REVERSE)

    def reset(self) -> None:
        """Reset to IDLE, clearing all active drive state."""
        self._status       = DriveStatus.IDLE
        self.logger.info("DriveClient reset to IDLE")

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def forward(self):
        self.logger.info(f'Moving forward at speed {self.v_linear}')
        self._status = DriveStatus.FORWARD
        self.__publish_cmd_vel(linear_x=self.v_linear, angular_z=0.0)

    def backward(self):
        self._status = DriveStatus.REVERSE
        self.logger.info(f'Moving backward at speed {self.v_linear}')
        self.__publish_cmd_vel(linear_x=-self.v_linear, angular_z=0.0)

    def turn_right(self):
        self._status = DriveStatus.FORWARD
        self.logger.info(f'Turning right at speed {self.v_linear}')
        self.__publish_cmd_vel(linear_x=0.0, angular_z=-self.v_angular)

    def turn_left(self):
        self.logger.info(f'Turning left at speed {self.v_linear}')
        self.__publish_cmd_vel(linear_x=0.0, angular_z=self.v_angular)
    
    def stop(self):
        self.logger.info('Stopping the robot')
        self.__publish_cmd_vel(linear_x=0.0, angular_z=0.0)

    def __publish_cmd_vel(self, linear_x: float,angular_z: float) -> None:
        """Wrap a Twist in a stamped message and publish to cmd_vel."""
        msg = TwistStamped()
        msg.header.stamp    = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        self._cmd_vel_pub.publish(msg)
