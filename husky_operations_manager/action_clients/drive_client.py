import time

import tf2_ros
from geometry_msgs.msg import TwistStamped
from rclpy.duration import Duration
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node, Timer
from rclpy.time import Time
from tf2_ros import TransformException

from husky_operations_manager.dataclass import DriveConfig
from husky_operations_manager.enum import DriveStatus

# ---------------------------------------------------------------------------
# DriveClient
# ---------------------------------------------------------------------------


class DriveClient:
    def __init__(self, node: Node, config: DriveConfig) -> None:
        self.node = node
        self.logger = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip("/")

        # --- Config ---
        self.base_frame: str = config.base_frame
        self.v_linear: float = config.v_linear
        self.v_angular: float = config.v_angular
        self.tf_polling_rate: float = config.tf_polling_rate
        self.timeout: float = config.timeout
        self.alignment_tolerance: float = config.tolerance
        self.tf_base_frame: str = config.tf_base_frame
        self.tf_detection_frame: str = config.tf_detection_frame

        # --- Status ---
        self._status: DriveStatus = DriveStatus.IDLE
        self._correction_in_progress = False
        self._correction_start_time = None
        self._correction_timer: Timer | None = None

        # --- TF ---
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)
        self._alignment_timer: Timer | None = None

        # --- Publishers ---
        self._cmd_vel_pub = self.node.create_publisher(TwistStamped, f"{self.namespace}/cmd_vel", 10)

        self.logger.info(
            f"DriveClient initialized | "
            f"linear={self.v_linear} angular={self.v_angular} | "
            f"poll_rate={self.tf_polling_rate}Hz "
            f"tolerance={self.alignment_tolerance}m | "
            f"base='{self.tf_base_frame}' "
            f"detection='{self.tf_detection_frame}' "
            f"target='{self.base_frame}'"
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def start_moving(self) -> None:
        self.forward()

    def cancel(self) -> None:
        """Cancel an active drive and publish zero velocity."""
        if self._status not in (DriveStatus.FORWARD, DriveStatus.REVERSE):
            return
        self.stop()
        self._status = DriveStatus.CANCELED
        self.logger.info(f"Drive canceled, status: {self._status}")

    def get_status(self) -> DriveStatus:
        """Return the current DriveStatus."""
        return self._status

    def is_active(self) -> bool:
        """Return True if a drive is currently in progress."""
        return self._status in (DriveStatus.FORWARD, DriveStatus.REVERSE)

    def reset(self) -> None:
        """Reset to IDLE, clearing all active drive state."""
        self._status = DriveStatus.IDLE
        self.logger.info("DriveClient reset to IDLE")

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def forward(self) -> None:
        self.logger.info(f"Moving forward at speed {self.v_linear}")
        self._status = DriveStatus.FORWARD
        self.__publish_cmd_vel(linear_x=self.v_linear, angular_z=0.0)

    def backward(self) -> None:
        self.logger.info(f"Moving backward at speed {self.v_linear}")
        self._status = DriveStatus.REVERSE
        self.__publish_cmd_vel(linear_x=-self.v_linear, angular_z=0.0)

    def turn_right(self) -> None:
        self.logger.info(f"Turning right at angular speed {self.v_angular}")
        self._status = DriveStatus.FORWARD
        self.__publish_cmd_vel(linear_x=0.0, angular_z=-self.v_angular)

    def turn_left(self) -> None:
        self.logger.info(f"Turning left at angular speed {self.v_angular}")
        self._status = DriveStatus.FORWARD
        self.__publish_cmd_vel(linear_x=0.0, angular_z=self.v_angular)

    def stop(self) -> None:
        """Stop the robot, set status to IDLE, and cancel alignment polling."""
        self.logger.info("Stopping the robot")
        self._status = DriveStatus.IDLE
        self.__publish_cmd_vel(linear_x=0.0, angular_z=0.0)
        
        # is_aligned, _ = self._is_aligned()
        # # Check alignment status
        # if is_aligned:
        #     self.logger.info('Aligned: stopping correction and cancel polling')
        #     self.cancel_correction()
        # else:
        #     self.logger.info('Not aligned: starting correction')
        #     self.start_correction()

    # ------------------------------------------------------------------
    # TF alignment Logic In case we need to add pose correction
    # ------------------------------------------------------------------

    def start_correction(self):
        """Begin correction movement and start polling TF."""
        if self._correction_in_progress:
            return
        self._correction_in_progress = True
        self._correction_start_time = time.time()

        is_aligned, x_diff = self._is_aligned()
        
        if is_aligned:
            self.node.get_logger().info('Already aligned, no correction needed')
            self.cancel_correction()
            self.stop() 
            return
        else:
            if x_diff > self.alignment_tolerance:
                self.backward()  # need to move backward
            elif x_diff < -self.alignment_tolerance:
                self.forward()  # need to move forward
            else:
                # within tolerance
                self.node.get_logger().info('Already aligned, no correction needed')
                self.cancel_correction()
                self.stop()  

        # Start polling TF
        self._correction_timer = self.node.create_timer(
            1.0 / self.tf_polling_rate,
            self.correction_callback
        )
    
    def correction_callback(self):
        """Check alignment and timeout during correction."""
        current_time = time.time()
        elapsed = current_time - self._correction_start_time

        # Update alignment status
        is_aligned, _ = self._is_aligned()

        if is_aligned:
            self.node.get_logger().info('Alignment achieved, stopping correction')
            self.stop()
            self.cancel_correction()
        elif elapsed > self.timeout:
            self.node.get_logger().error('Alignment correction timed out')
            self.stop()
            self.cancel_correction()

    def _is_aligned(self):
        """Determine if aligned based on current TF data."""
        try:
            base_tf = self._tf_buffer.lookup_transform(
                self.base_frame, self.tf_base_frame, Time(), Duration(seconds=0.2)
            )
            detection_tf = self._tf_buffer.lookup_transform(
                self.base_frame, self.tf_detection_frame,Time(), Duration(seconds=0.2)
            )
            x_diff = base_tf.transform.translation.x - detection_tf.transform.translation.x
            aligned = abs(x_diff) <= self.alignment_tolerance
            return aligned, x_diff
        except TransformException:
            self.node.get_logger().warn('TF lookup failed during alignment check')
            return False

    def cancel_correction(self):
        """Cancel correction process and stop polling."""
        if self._correction_timer:
            self._correction_timer.cancel()
            self._correction_timer = None
        self._correction_in_progress = False
        self.node.get_logger().info('Correction process canceled')

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def __publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        """Wrap a Twist in a stamped message and publish to cmd_vel."""
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        self._cmd_vel_pub.publish(msg)
