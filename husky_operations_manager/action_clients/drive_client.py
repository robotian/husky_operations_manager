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
    def __init__(self, node: Node, config:DriveConfig) -> None:
        self.node      = node   
        self.logger    = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip('/')

        # --- Congig ---
        self.base_frame:  str   = config.base_frame
        self.fixed_frame: str   = config.fixed_frame
        self.v_linear:  float = config.v_linear
        self.v_angular: float = config.v_angular
        self._tf_polling_rate:    float = config.tf_polling_rate
        self._alignment_tolerance: float = config.alignment_tolerance
        self._tf_target_frame:    str   = config.tf_target_frame
        self._tf_base_frame:      str   = config.tf_base_frame
        self._tf_detection_frame: str   = config.tf_detection_frame

        # --- Status ---
        self._status:     DriveStatus  = DriveStatus.IDLE

        # --- TF ---
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)
        self._alignment_timer: Timer | None = None

        # --- Publishers ---
        self._cmd_vel_pub = self.node.create_publisher(
            TwistStamped, f'{self.namespace}/cmd_vel', 10)
        

        self.logger.info(
                    f"DriveClient initialized | "
                    f"linear={self.v_linear} angular={self.v_angular} | "
                    f"poll_rate={self._tf_polling_rate}Hz "
                    f"tolerance={self._alignment_tolerance}m | "
                    f"base='{self._tf_base_frame}' "
                    f"detection='{self._tf_detection_frame}' "
                    f"target='{self._tf_target_frame}'"
                )


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
        self._stop_alignment_timer()
        self._status       = DriveStatus.IDLE
        self.logger.info("DriveClient reset to IDLE")

    # ------------------------------------------------------------------
    # Detection alignment entry point (called by HuskyOpsManager)
    # ------------------------------------------------------------------
 
    def on_detection_received(self, msg) -> None:
        """
        Called by HuskyOpsManager when an ImageDetectionPose message is received.
        Starts TF alignment polling when detection_valid is True and the robot
        is actively moving.
        """
        if not msg.detection_valid:
            return
 
        if not self.is_active():
            self.logger.debug(
                "Detection received but robot is not moving. "
                "Alignment polling will not start.")
            return
 
        if self._alignment_timer is not None and not self._alignment_timer.is_canceled():
            return  # Already polling — do nothing
 
        self.logger.info(
            f"Valid detection received. Starting TF alignment polling at "
            f"{self._tf_polling_rate} Hz.")
        self._start_alignment_timer()

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def forward(self) -> None:
        self.logger.info(f'Moving forward at speed {self.v_linear}')
        self._status = DriveStatus.FORWARD
        self._start_alignment_timer()
        self.__publish_cmd_vel(linear_x=self.v_linear, angular_z=0.0)
 
    def backward(self) -> None:
        self.logger.info(f'Moving backward at speed {self.v_linear}')
        self._status = DriveStatus.REVERSE
        self._start_alignment_timer()
        self.__publish_cmd_vel(linear_x=-self.v_linear, angular_z=0.0)
 
    def turn_right(self) -> None:
        self.logger.info(f'Turning right at angular speed {self.v_angular}')
        self._status = DriveStatus.FORWARD
        self.__publish_cmd_vel(linear_x=0.0, angular_z=-self.v_angular)
 
    def turn_left(self) -> None:
        self.logger.info(f'Turning left at angular speed {self.v_angular}')
        self._status = DriveStatus.FORWARD
        self.__publish_cmd_vel(linear_x=0.0, angular_z=self.v_angular)
 
    def stop(self) -> None:
        """Stop the robot, set status to IDLE, and cancel alignment polling."""
        self.logger.info('Stopping the robot')
        self._stop_alignment_timer()
        self._status = DriveStatus.IDLE
        self.__publish_cmd_vel(linear_x=0.0, angular_z=0.0)

     # ------------------------------------------------------------------
    # TF alignment polling
    # ------------------------------------------------------------------
 
    def _start_alignment_timer(self) -> None:
        """Cancel any existing timer and start a fresh alignment polling timer."""
        self._stop_alignment_timer()
        self._alignment_timer = self.node.create_timer(
            1.0 / self._tf_polling_rate,
            self._alignment_poll_callback
        )
        self.logger.debug("Alignment polling timer started.")
 
    def _stop_alignment_timer(self) -> None:
        """Cancel the alignment polling timer if it is running."""
        if self._alignment_timer is not None:
            self._alignment_timer.cancel()
            self._alignment_timer = None
            self.logger.debug("Alignment polling timer stopped.")
 
    def _alignment_poll_callback(self) -> None:
        """
        Timer callback — looks up arm_0_base_link and arm_0_detections in the
        target frame (map) and stops the robot when their X axis positions are
        within alignment_tolerance.
        """
 
        # --- Lookup base frame ---
        try:
            base_tf = self._tf_buffer.lookup_transform(
                self._tf_target_frame,
                self._tf_base_frame,
                Time(),
                timeout=Duration(seconds=0.2)
            )
        except TransformException as e:
            self.logger.error(
                f"TF lookup failed for '{self._tf_base_frame}' "
                f"in '{self._tf_target_frame}': {e}")
            return
 
        # --- Lookup detection frame ---
        try:
            detection_tf = self._tf_buffer.lookup_transform(
                self._tf_target_frame,
                self._tf_detection_frame,
                Time(),
                timeout=Duration(seconds=0.2)
            )
        except TransformException as e:
            self.logger.warn(
                f"TF lookup failed for '{self._tf_detection_frame}' "
                f"in '{self._tf_target_frame}'. "
                f"Robot cannot align — check if the image detection node is running. "
                f"Error: {e}")
            return
 
        # --- Compare X axis positions ---
        base_x      = base_tf.transform.translation.x
        detection_x = detection_tf.transform.translation.x
        x_diff      = abs(base_x - detection_x)
 
        self.logger.debug(
            f"Alignment check | "
            f"base_x={base_x:.4f} "
            f"detection_x={detection_x:.4f} "
            f"diff={x_diff:.4f} "
            f"tolerance={self._alignment_tolerance:.4f}"
        )
 
        if x_diff <= self._alignment_tolerance:
            self.logger.info(
                f"Alignment reached | "
                f"x_diff={x_diff:.4f}m <= tolerance={self._alignment_tolerance:.4f}m. "
                f"Stopping robot.")
            self.stop()
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def __publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        """Wrap a Twist in a stamped message and publish to cmd_vel."""
        msg = TwistStamped()
        msg.header.stamp    = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x  = linear_x
        msg.twist.angular.z = angular_z
        self._cmd_vel_pub.publish(msg)
