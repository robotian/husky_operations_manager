import rclpy
import rclpy.time
import rclpy.duration
import tf2_ros
from rclpy.timer import Timer
from geometry_msgs.msg import TwistStamped
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
from tf2_ros import TransformException

from husky_operations_manager.enum import DriveStatus
from husky_operations_manager.dataclass import DriveConfig

# ---------------------------------------------------------------------------
# DriveClient
# ---------------------------------------------------------------------------

class DriveClient:
    def __init__(self, node: Node, config: DriveConfig) -> None:
        self.node      = node
        self.logger    = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip('/')

        # --- Config ---
        self.base_frame:  str   = config.base_frame
        self.v_linear:    float = config.v_linear
        self.v_angular:   float = config.v_angular

        # --- Alignment config ---
        self._tf_polling_rate:     float = config.tf_polling_rate
        self._alignment_tolerance: float = config.tolerance
        self._alignment_timeout:   float = config.timeout
        self._tf_base_frame:       str   = config.tf_base_frame
        self._tf_detection_frame:  str   = config.tf_detection_frame

        # --- Status ---
        self._status:     DriveStatus = DriveStatus.IDLE
        self._is_aligned: bool        = False

        # --- Correction state ---
        self._correcting:        bool         = False
        self._correction_start:  float | None = None

        # --- TF ---
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

        # --- Alignment polling timer ---
        self._alignment_timer: Timer | None = None

        # --- Publishers ---
        self._cmd_vel_pub = self.node.create_publisher(
            TwistStamped, f'{self.namespace}/cmd_vel', 10)

        self.logger.info(
            f"DriveClient initialized | "
            f"linear={self.v_linear} angular={self.v_angular} | "
            f"poll_rate={self._tf_polling_rate}Hz "
            f"tolerance={self._alignment_tolerance}m "
            f"timeout={self._alignment_timeout}s | "
            f"base='{self._tf_base_frame}' "
            f"detection='{self._tf_detection_frame}' "
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Cancel an active drive and publish zero velocity."""
        if self._status not in (DriveStatus.FORWARD, DriveStatus.REVERSE):
            return
        self._stop_alignment_timer()
        self._reset_correction_state()
        self._status = DriveStatus.CANCELED
        self.__publish_cmd_vel(linear_x=0.0, angular_z=0.0)
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
        self._reset_correction_state()
        self._status = DriveStatus.IDLE
        self.logger.info("DriveClient reset to IDLE")

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def forward(self) -> None:
        """Move forward and start TF alignment polling."""
        self.logger.info(f'Moving forward at speed {self.v_linear}')
        self._status = DriveStatus.FORWARD
        self._start_alignment_timer()
        self.__publish_cmd_vel(linear_x=self.v_linear, angular_z=0.0)

    def backward(self) -> None:
        """Move backward and start TF alignment polling."""
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
        """
        Event-driven stop — called externally on detection_valid=True.

        Checks _is_aligned (updated continuously by TF polling):
          - True  → robot is at correct pose → publish zero velocity + cancel timer.
          - False → start corrective motion (forward or backward) until aligned.
                    If correction exceeds alignment_timeout → log error + force stop.
        """
        if self._is_aligned:
            self.logger.info(
                "stop() called — pose validated. Stopping robot.")
            self._stop_alignment_timer()
            self._reset_correction_state()
            self._status = DriveStatus.IDLE
            self.__publish_cmd_vel(linear_x=0.0, angular_z=0.0)
            return

        # Not yet aligned — start corrective motion if not already correcting
        if not self._correcting:
            self.logger.info(
                "stop() called — pose not yet validated. "
                "Starting corrective motion.")
            self._correcting       = True
            self._correction_start = self.node.get_clock().now().nanoseconds / 1e9
            self._apply_correction()

    # ------------------------------------------------------------------
    # TF alignment polling
    # ------------------------------------------------------------------

    def _start_alignment_timer(self) -> None:
        """Cancel any existing timer and start a fresh alignment polling timer."""
        self._stop_alignment_timer()
        self._is_aligned = False
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
        Timer callback — runs continuously while robot is moving.

        Responsibilities:
          1. Lookup arm_0_base_link and arm_0_detections in target frame.
          2. Update _is_aligned flag based on X axis tolerance.
          3. If correcting:
             - Check alignment timeout.
             - If aligned    → publish zero velocity and stop.
             - If not aligned → apply corrective motion.
        """
        t       = rclpy.time.Time()
        timeout = rclpy.duration.Duration(seconds=0.1)

        # --- Lookup base frame ---
        try:
            base_tf = self._tf_buffer.lookup_transform(
                self.base_frame,
                self._tf_base_frame,
                t,
                timeout=timeout
            )
        except TransformException as e:
            self.logger.error(
                f"TF lookup failed for '{self._tf_base_frame}' "
                f"in '{self.base_frame}': {e}")
            return

        # --- Lookup detection frame ---
        try:
            detection_tf = self._tf_buffer.lookup_transform(
                self.base_frame,
                self._tf_detection_frame,
                t,
                timeout=timeout
            )
        except TransformException as e:
            self.logger.warn(
                f"TF lookup failed for '{self._tf_detection_frame}' "
                f"in '{self.base_frame}'. "
                f"Robot cannot align — check if image detection node is running. "
                f"Error: {e}")
            return

        # --- Compute signed X axis difference ---
        # positive → base ahead of detection → need to move backward
        # negative → base behind detection   → need to move forward
        base_x      = base_tf.transform.translation.x
        detection_x = detection_tf.transform.translation.x
        x_diff      = base_x - detection_x

        self._is_aligned = abs(x_diff) <= self._alignment_tolerance

        self.logger.debug(
            f"Alignment poll | "
            f"base_x={base_x:.4f} "
            f"detection_x={detection_x:.4f} "
            f"x_diff={x_diff:.4f} "
            f"aligned={self._is_aligned} "
            f"correcting={self._correcting}"
        )

        # Only run correction logic if stop() was called
        if not self._correcting:
            return

        # --- Check timeout ---
        elapsed = self.node.get_clock().now().nanoseconds / 1e9 - self._correction_start
        if elapsed >= self._alignment_timeout:
            self.logger.error(
                f"Alignment correction timed out after {elapsed:.1f}s "
                f"(timeout={self._alignment_timeout}s). "
                f"Forcing stop. Last x_diff={x_diff:.4f}m.")
            self._stop_alignment_timer()
            self._reset_correction_state()
            self._status = DriveStatus.IDLE
            self.__publish_cmd_vel(linear_x=0.0, angular_z=0.0)
            return

        # --- Aligned — stop ---
        if self._is_aligned:
            self.logger.info(
                f"Correction complete | "
                f"x_diff={abs(x_diff):.4f}m <= "
                f"tolerance={self._alignment_tolerance:.4f}m. "
                f"Stopping robot.")
            self._stop_alignment_timer()
            self._reset_correction_state()
            self._status = DriveStatus.IDLE
            self.__publish_cmd_vel(linear_x=0.0, angular_z=0.0)
            return

        # --- Still not aligned — keep correcting ---
        self._apply_correction(x_diff)

    def _apply_correction(self, x_diff: float = 0.0) -> None:
        """
        Apply corrective motion based on signed X axis difference.

        x_diff = base_x - detection_x:
          positive → base is ahead of detection → move backward
          negative → base is behind detection   → move forward
        """
        if x_diff > 0:
            self.logger.info(
                f"Correcting | base {x_diff:.4f}m ahead → moving backward.")
            self._status = DriveStatus.REVERSE
            self.__publish_cmd_vel(linear_x=-self.v_linear, angular_z=0.0)
        else:
            self.logger.info(
                f"Correcting | base {abs(x_diff):.4f}m behind → moving forward.")
            self._status = DriveStatus.FORWARD
            self.__publish_cmd_vel(linear_x=self.v_linear, angular_z=0.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_correction_state(self) -> None:
        """Reset correction and alignment flags."""
        self._correcting       = False
        self._correction_start = None
        self._is_aligned       = False

    def __publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        """Wrap a Twist in a stamped message and publish to cmd_vel."""
        msg = TwistStamped()
        msg.header.stamp    = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x  = linear_x
        msg.twist.angular.z = angular_z
        self._cmd_vel_pub.publish(msg)