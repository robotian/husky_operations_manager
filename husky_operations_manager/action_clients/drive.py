"""
drive.py

DriveStatus, DriveConfig, and DriveClient for camera-guided lavender harvesting.

State machine:
  IDLE
    └─► scan()      → SCANNING   (forward drive, watching for detections)
          └─► detection + ex/ey within tolerance
                └─► STOPPED     (zero velocity, caller runs harvest activity)
                      └─► resume()  → DEPARTING  (forward drive, ignoring detections
                                                   until departure_clearance is met)
                            └─► clearance met → SCANNING (next bush)
  Any active state:
    └─► cancel()   → CANCELED
    └─► reset()    → IDLE

TF lookup (single parent-child call):
  camera_1_detections expressed in camera_1_color_optical_frame
    ex = translation.x  (forward/backward error)
    ey = translation.y * ey_sign  (lateral error)

Confirmed convention (ey_sign=-1.0):
  Bush is always to the RIGHT of the robot for safe harvesting.
  ey > 0 → robot too far from bush   → move closer (right)
  ey < 0 → robot too close to bush   → move away   (left)
  |ey| <= ey_tolerance → at correct lateral harvesting distance

  TODO: Implement lateral velocity correction publisher that adjusts
        angular_z while SCANNING/CORRECTING to maintain safe harvesting
        distance (ey within tolerance) as the robot approaches the bush.
        Only implement after current approach is tested and robot reliably
        stops when |ex| <= ex_tolerance.
"""

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node

import tf2_ros
from tf2_ros import TransformException

from dataclasses import dataclass
from enum import IntEnum

from geometry_msgs.msg import TwistStamped
from status_interfaces.msg import ImageDetectionPose


# =============================================================================
# DriveStatus
# =============================================================================


class DriveStatus(IntEnum):
    """Combined state enum for DriveClient."""

    IDLE = 0
    SCANNING = 1
    CORRECTING = 2
    STOPPED = 3
    DEPARTING = 4
    CANCELED = 5
    ERROR = 6


# =============================================================================
# DriveConfig
# =============================================================================


@dataclass
class DriveConfig:
    """Configuration for DriveClient."""

    # Detection
    detection_topic: str

    # Velocity
    base_frame: str  # cmd_vel header frame_id
    v_linear: float  # forward/reverse speed (m/s)
    v_angular: float  # turning speed (rad/s)

    # TF frames
    tf_base_frame: str  # source frame for alignment lookup (camera_1_color_optical_frame)
    tf_detection_frame: str  # target frame for alignment lookup (camera_1_detections)

    # Alignment
    ex_tolerance: float  # forward/backward stop tolerance (m)
    ey_tolerance: float  # lateral stop tolerance (m)
    ey_sign: float  # +1.0 or -1.0 ey is negative when bush is to the right

    # Departure
    # distance (m) to travel past a stopped bush before re-enabling detection in DEPARTING state
    departure_clearance: float

    # Legacy — not used internally, retained for callers that set them
    tf_polling_rate: float  # kept for backward compatibility; not used by DriveClient
    timeout: float  # kept for backward compatibility; not used by DriveClient

    # cmd_vel repeat rate (Hz) — republish velocity while active
    cmd_vel_rate: float


# =============================================================================
# DriveClient
# =============================================================================


class DriveClient:
    """
    Component class for camera-guided forward drive.

    Instantiated and owned by HuskyOperationsManager / LavenderHarvestNode.
    Uses the parent node reference for all ROS2 primitives.

    Lifecycle:
      1. scan()    — start SCANNING forward
      2. (auto)    — detection callback fires → STOPPED when ex/ey within tolerance
      3. resume()  — caller calls after harvest activity → DEPARTING
      4. (auto)    — departure clearance met → back to SCANNING
    """

    def __init__(self, node: Node, config: DriveConfig) -> None:
        self.node = node
        self.logger = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip('/')

        # --- Velocity config ---
        self.base_frame: str = config.base_frame
        self.v_linear: float = config.v_linear
        self.v_angular: float = config.v_angular

        # --- Alignment config ---
        self._source_frame: str = config.tf_base_frame
        self._detection_frame: str = config.tf_detection_frame
        self._ex_tolerance: float = config.ex_tolerance
        self._ey_tolerance: float = config.ey_tolerance
        self._ey_sign: float = config.ey_sign

        # --- Departure config ---
        self._departure_clearance: float = config.departure_clearance
        self._departure_start_x: float | None = None

        # --- Status ---
        self._status: DriveStatus = DriveStatus.IDLE

        # --- cmd_vel repeat state ---
        self._current_linear_x: float = 0.0
        self._current_angular_z: float = 0.0

        # --- TF ---
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

        # --- Detection subscription ---
        self._detection_sub = self.node.create_subscription(
            ImageDetectionPose,
            f'{self.namespace}/{config.detection_topic}',
            self._detection_callback,
            10,
        )

        # --- cmd_vel publisher ---
        self._cmd_vel_pub = self.node.create_publisher(TwistStamped, f'{self.namespace}/cmd_vel', 10)

        # --- cmd_vel repeat timer ---
        self._cmd_vel_timer = self.node.create_timer(1.0 / config.cmd_vel_rate, self._cmd_vel_repeat_callback)

        self.logger.info(
            f'DriveClient initialized | '
            f'linear={self.v_linear} angular={self.v_angular} | '
            f'ex_tolerance={self._ex_tolerance}m '
            f'ey_tolerance={self._ey_tolerance}m '
            f'ey_sign={self._ey_sign:+.1f} | '
            f'departure_clearance={self._departure_clearance}m | '
            f"source='{self._source_frame}' "
            f"detection='{self._detection_frame}'"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self) -> None:
        """Start forward scanning drive. Detection callback handles STOPPED transition."""
        self.logger.info(f'SCANNING — moving forward at {self.v_linear}m/s')
        self._status = DriveStatus.SCANNING
        self._current_linear_x = self.v_linear
        self._current_angular_z = 0.0
        self._publish_cmd_vel(linear_x=self.v_linear, angular_z=0.0)

    def resume(self) -> None:
        """
        Called by the parent node after harvest activity completes.
        Transitions STOPPED → DEPARTING and resumes forward drive.
        """
        if self._status != DriveStatus.STOPPED:
            self.logger.warning(f'resume() called in unexpected state: {self._status.name} — ignoring')
            return

        self._departure_start_x = self._get_robot_x()
        self.logger.info(
            f'DEPARTING — moving past bush | clearance={self._departure_clearance}m | start_x={self._departure_start_x}'
        )
        self._status = DriveStatus.DEPARTING
        self._publish_cmd_vel(linear_x=self.v_linear, angular_z=0.0)

    def cancel(self) -> None:
        """Cancel an active drive and publish zero velocity."""
        if self._status in (DriveStatus.IDLE, DriveStatus.CANCELED):
            return
        self._status = DriveStatus.CANCELED
        self._departure_start_x = None        
        self._current_linear_x = 0.0
        self._current_angular_z = 0.0
        self._publish_cmd_vel(linear_x=0.0, angular_z=0.0)
        self.logger.info('Drive CANCELED')

    def reset(self) -> None:
        """Reset to IDLE."""
        self._status = DriveStatus.IDLE
        self._departure_start_x = None
        self._current_linear_x = 0.0
        self._current_angular_z = 0.0
        self.logger.info('DriveClient reset to IDLE')

    def get_status(self) -> DriveStatus:
        """Return the current DriveStatus."""
        return self._status

    def is_active(self) -> bool:
        """Return True if the robot is currently moving."""
        return self._status in (
            DriveStatus.SCANNING,
            DriveStatus.CORRECTING,
            DriveStatus.DEPARTING,
        )

    # ------------------------------------------------------------------
    # Detection callback
    # ------------------------------------------------------------------

    def _detection_callback(self, msg: ImageDetectionPose) -> None:
        """
        Fires on every ImageDetectionPose message.

        SCANNING / CORRECTING:
          Valid detection → compute (ex, ey).
          Within tolerance → STOPPED, zero velocity.
          Outside tolerance → CORRECTING, keep driving.

        DEPARTING:
          Ignore detections until departure_clearance is met.
          Once met → back to SCANNING.

        All other states: no-op.
        """
        self.logger.debug(f"Message received: {msg}")
        self.logger.debug(f"Valid Detection: {msg.detection_valid}")
        if not msg.detection_valid:
            return

        self.logger.debug(f"Detected Pose: {msg.center}")

        if self._status == DriveStatus.DEPARTING:
            self._check_departure_clearance()
            return

        if self._status not in (DriveStatus.SCANNING, DriveStatus.CORRECTING):
            return
        
        result = self._get_alignment_error()
        if result is None:
            return
        ex, ey = result
        self._evaluate_alignment(ex, ey)

        # Treating msg data as error data
        # ex = msg.center.x
        # ey = msg.center.y * self._ey_sign

        # self._evaluate_alignment(ex, ey)

    # ------------------------------------------------------------------
    # Departure clearance
    # ------------------------------------------------------------------

    def _check_departure_clearance(self) -> None:
        """
        During DEPARTING, check if the robot has travelled past the
        departure_clearance distance. If so, resume SCANNING.
        """
        if self._departure_start_x is None:
            return

        current_x = self._get_robot_x()
        if current_x is None:
            return

        # Distance travelled since departure started (X axis in base_link)
        # Both frames share forward X, so simple absolute difference suffices
        travelled = abs(current_x - self._departure_start_x)

        self.logger.debug(f'Departure check | travelled={travelled:.3f}m clearance={self._departure_clearance}m')

        if travelled >= self._departure_clearance:
            self.logger.info(f'Departure clearance met ({travelled:.3f}m) — resuming SCANNING')
            self._departure_start_x = None
            self.scan()

    def _get_robot_x(self) -> float | None:
        """
        Look up the robot's current X position in base_link frame via TF.
        Used for departure clearance measurement.
        Returns float or None on TF failure.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                self._source_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            return tf.transform.translation.x
        except TransformException:
            return None

    # ------------------------------------------------------------------
    # Alignment error
    # ------------------------------------------------------------------

    def _get_alignment_error(self) -> tuple[float, float] | None:
        """
        Compute ex and ey from the difference between camera_1_color_optical_frame
        and camera_1_detections both expressed in map frame.

        ex = source_t.x - detection_t.x
        ey = (source_t.y - detection_t.y) * ey_sign

        Returns (ex, ey) in metres, or None on TF failure.
        """
        result = self._get_frame_poses_wrt_map()
        if result is None:
            return None

        s, d = result
        ex = s.x - d.x
        ey = (s.y - d.y) * self._ey_sign
        return ex, ey

    def _get_frame_poses_wrt_map(self) -> tuple | None:
        """
        Look up camera_1_color_optical_frame and camera_1_detections both wrt map.
        Returns (source_t, detection_t) as translation objects, or None on TF failure.
        """
        try:
            source_tf = self._tf_buffer.lookup_transform(
                self._source_frame,
                'map',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            detection_tf = self._tf_buffer.lookup_transform(
                self._detection_frame,
                'map',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            s = source_tf.transform.translation
            d = detection_tf.transform.translation
            self.logger.debug(
                f"Pose wrt map | '{self._source_frame}' | t=({s.x:+.4f}, {s.y:+.4f}, {s.z:+.4f})"
            )
            self.logger.debug(
                f"Pose wrt map | '{self._detection_frame}' | t=({d.x:+.4f}, {d.y:+.4f}, {d.z:+.4f})"
            )
            return s, d
        except TransformException as e:
            self.logger.debug(
                f"Pose lookup failed wrt 'map': {e}",
                throttle_duration_sec=5.0,
            )
            return None

    def _evaluate_alignment(self, ex: float, ey: float) -> None:
        """
        Evaluate alignment error and transition state accordingly.

        ex: forward/backward alignment — stops the robot when within tolerance
        ey: lateral distance to bush — bush is always to the right
            ey > 0 → too far from bush   → needs to move closer (right)
            ey < 0 → too close to bush   → needs to move away   (left)

        State transitions:
          Both within tolerance → STOPPED, zero velocity
          Otherwise             → CORRECTING, keep driving

        TODO: Use ey to drive angular_z correction once ex-based stopping
              is verified in testing. See module-level TODO for details.
        """
        self.logger.debug(f'Pose | ex={ex:+.4f}m  ey={ey:+.4f}m')
        # --- X axis (forward/backward) ---
        if abs(ex) <= self._ex_tolerance:
            self.logger.info(f'X ALIGNED | ex={ex:+.4f}m within tolerance={self._ex_tolerance:.4f}m')
            self._status = DriveStatus.STOPPED
            self._publish_cmd_vel(linear_x=0.0, angular_z=0.0)
        else:
            self.logger.debug(f'Approaching | ex={ex:+.4f}m exceeds tolerance={self._ex_tolerance:.4f}m')
            self._status = DriveStatus.CORRECTING

        # --- Y axis (lateral harvesting distance) ---
        # TODO: drive angular_z correction based on ey once ex stopping is verified
        if abs(ey) <= self._ey_tolerance:
            self.logger.info(
                f'Y ALIGNED | ey={ey:+.4f}m within tolerance={self._ey_tolerance:.4f}m — at correct harvesting distance'
            )
        elif ey > 0:
            self.logger.info(f'TOO FAR   | ey={ey:+.4f}m — robot too far from bush, move closer (right)')
        else:
            self.logger.info(f'TOO CLOSE | ey={ey:+.4f}m — robot too close to bush, move away (left)')

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cmd_vel_repeat_callback(self) -> None:
        """Republish current velocity at cmd_vel_rate while active."""
        if self.is_active():
            self._publish_cmd_vel(self._current_linear_x, self._current_angular_z)

    def _publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        """Wrap a Twist in a stamped message and publish to cmd_vel."""
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        self._cmd_vel_pub.publish(msg)
