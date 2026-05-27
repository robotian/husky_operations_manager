"""
Drive client for Husky operations.

Manages forward row traversal and sequential bush alignment using
TF-derived spatial error as the correction signal.

Error signal (computed on every valid detection):
  arm_pose    = transform(arm_0_base_link,    base_link)
  camera_pose = transform(camera_1_detections, base_link)

  error_x = arm_pose.x - camera_pose.x   (travel axis, base_link +X = forward)
  error_y = arm_pose.y - camera_pose.y   (lateral axis, base_link +Y = left)

  error_x > 0 → arm ahead of bush  → drive backward (-linear.x)
  error_x < 0 → arm behind bush    → drive forward  (+linear.x)
  error_x ≈ 0 → arm over bush      → check error_y

  error_y > 0 → arm left of bush   → turn right (-angular.z)
  error_y < 0 → arm right of bush  → turn left  (+angular.z)

Stop condition:
  abs(error_x) <= tolerance AND abs(error_y) <= tolerance

State machine (DriveStatus):
  IDLE → SCANNING → CORRECTING → STOPPED → DEPARTING → SCANNING → ...
  Any state → CANCELED (via cancel())
  Any state → IDLE     (via stop() or reset())

Caller interface:
  start()               — begin SCANNING
  resume()              — called after activity completes, begin DEPARTING
  stop()                — external hard stop (e-stop, timeout)
  cancel()              — abort
  get_status()          — poll DriveStatus
  is_active()           — True when robot is moving
  last_detection_time() — for no-detection timeout in calling node
"""

from dataclasses import dataclass
from enum import IntEnum

from geometry_msgs.msg import TwistStamped
from status_interfaces.msg import ImageDetectionPose

import rclpy.duration
import rclpy.time
import tf2_ros
from tf2_ros import TransformException

from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
from rclpy.timer import Timer


class DriveStatus(IntEnum):
    """
    Combined state enum for DriveClient.

    IDLE       — not started, fully reset, or stopped externally
    SCANNING   — moving forward, looking for next bush
    CORRECTING — bush detected, homing in toward center_x=0
    STOPPED    — stop condition met, waiting for resume()
                 caller starts activity (simulation timer or arm action)
                 then calls drive_client.resume()
    DEPARTING  — activity complete, moving past current bush,
                 ignoring detections until departure_clearance is met
    CANCELED   — aborted by cancel()
    ERROR      — unexpected state, caller should call reset()
    """

    IDLE = 0
    SCANNING = 1
    CORRECTING = 2
    STOPPED = 3
    DEPARTING = 4
    CANCELED = 5
    ERROR = 6


@dataclass
class DriveConfig:
    """
    Configuration for DriveClient.

    Error signal accounts for camera mounting offset:
      arm_in_base    = transform(base_link ← arm_0_base_link)
      camera_in_base = transform(base_link ← camera_1_color_optical_frame)
      bush_in_base   = transform(base_link ← camera_1_detections)

      offset_x = arm_in_base.x - camera_in_base.x
      offset_y = arm_in_base.y - camera_in_base.y

      error_x = (arm_in_base.x - bush_in_base.x) - offset_x
      error_y = (arm_in_base.y - bush_in_base.y) - offset_y

    Stop condition: abs(error_x) <= tolerance AND abs(error_y) <= tolerance
    """

    # ── Frames ─────────────────────────────────────────────────────────────────
    base_frame: str  # common reference frame, e.g. 'base_link'
    tf_arm_frame: str  # arm frame, e.g. 'arm_0_base_link'
    tf_detection_frame: str  # bush centroid frame, e.g. 'camera_1_detections'
    tf_camera_optical_frame: str  # camera optical frame for offset, e.g. 'camera_1_color_optical_frame'

    # ── Speed ──────────────────────────────────────────────────────────────────
    v_linear: float  # m/s — constant forward/backward speed
    # no angular correction — lateral positioning via Nav2

    # ── Stop condition ─────────────────────────────────────────────────────────
    tolerance: float  # m — stop when abs(error_x) <= tolerance
    #         AND abs(error_y) <= tolerance

    # ── C3 mitigation — minimum tracker ───────────────────────────────────────
    noise_margin: float  # m — how much abs(error_x) can increase between
    # ticks before rejecting as a bush switch

    # ── Departure clearance ────────────────────────────────────────────────────
    departure_clearance: float  # m — distance past completed bush before scanning resumes

    # ── Detection topic ────────────────────────────────────────────────────────
    detection_topic: str  # relative to namespace


class DriveClient:
    """
    Reactive sensor-driven drive client for sequential lavender row harvesting.

    Owns the full detection-to-motion lifecycle:
      - ImageDetectionPose subscription
      - Sequential bush targeting (SCANNING → CORRECTING → STOPPED → DEPARTING)
      - C3 mitigation via minimum tracker (noise_margin)
      - Departure clearance to prevent re-detecting completed bushes
      - 10Hz watchdog timer for Husky cmd_vel keepalive

    Precondition for start(): arm must be in scan pose (caller's responsibility).
    """

    def __init__(self, node: Node, config: DriveConfig) -> None:
        self.node = node
        self.logger = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip('/')

        # --- Config ---
        self._base_frame = config.base_frame
        self._tf_arm_frame = config.tf_arm_frame
        self._tf_detection_frame = config.tf_detection_frame
        self._tf_camera_optical_frame = config.tf_camera_optical_frame
        self._v_linear = config.v_linear
        self._tolerance = config.tolerance
        self._noise_margin = config.noise_margin
        self._departure_clearance = config.departure_clearance

        # --- Camera mounting offset (base_link frame) ---
        # Computed at start() and resume() — arm must be in scan pose.
        # offset = arm_0_base_link - camera_1_color_optical_frame in base_link frame.
        # Applied to error signal so robot stops with bush under camera, not arm origin.
        self._offset_x: float | None = None
        self._offset_y: float | None = None

        # --- Status ---
        self._status: DriveStatus = DriveStatus.IDLE

        # --- C3 minimum tracker ---
        # Tracks the lowest abs(error_x) seen during current CORRECTING phase.
        # Rejects detections where abs(error_x) rises above this by more than
        # noise_margin — indicates detector switched to a different bush.
        self._min_error_x_seen: float | None = None

        # --- Departure tracking ---
        # Records error_x at the moment resume() is called.
        # Departure is complete when error_x < _departure_start_x - departure_clearance.
        self._departure_start_x: float | None = None

        # --- Detection timing ---
        self._last_detection_time: float | None = None

        # --- Last commanded velocity (watchdog republish) ---
        self._last_linear_x: float = 0.0
        self._last_angular_z: float = 0.0

        # --- Timers ---
        self._watchdog_timer: Timer | None = None

        # --- TF — pose monitoring and error computation ---
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

        # --- Publisher ---
        self._cmd_vel_pub = self.node.create_publisher(TwistStamped, f'{self.namespace}/cmd_vel', 10)

        # --- Detection subscription ---
        self._detection_sub = self.node.create_subscription(
            ImageDetectionPose,
            f'{self.namespace}/{config.detection_topic}',
            self._on_detection,
            10,
        )

        self.logger.info(
            f'DriveClient initialized | '
            f'v_linear={self._v_linear} | '
            f'tolerance={self._tolerance}m | '
            f'noise_margin={self._noise_margin}m | '
            f'departure_clearance={self._departure_clearance}m | '
            f'base_frame={self._base_frame} | '
            f'tf_arm_frame={self._tf_arm_frame} | '
            f'tf_detection_frame={self._tf_detection_frame} | '
            f'tf_camera_optical_frame={self._tf_camera_optical_frame} | '
            f'detection_topic={self.namespace}/{config.detection_topic}'
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Begin forward row traversal — transition to SCANNING.

        Precondition: arm must be in scan pose (caller's responsibility).
        Clears all tracking state from any previous traversal.

        Returns True on success, False if camera offset lookup failed
        (TF not yet available — caller should retry).
        """
        self.logger.info(f'DriveClient start | v_linear={self._v_linear}')
        if not self._lookup_camera_offset():
            return False
        self._clear_tracking_state()
        self._last_detection_time = self.node.get_clock().now().nanoseconds / 1e9
        self._transition(DriveStatus.SCANNING)
        self._start_watchdog_timer()
        self.__publish_cmd_vel(linear_x=self._v_linear, angular_z=0.0)
        return True

    def resume(self) -> None:
        """
        Resume after activity completion — transition to DEPARTING.

        Called by TestDriveNode (simulation timer) or MotionManager (arm action)
        after the bush activity is complete. DriveClient moves forward and ignores
        detections until departure_clearance is met, then transitions to SCANNING
        to look for the next bush.

        No-op if not in STOPPED state.
        """
        if self._status != DriveStatus.STOPPED:
            self.logger.warning(f'resume() called in unexpected state {self._status.name} — ignoring')
            return

        self.logger.info('DriveClient resume — beginning departure')
        if not self._lookup_camera_offset():
            return
        self._min_error_x_seen = None
        self._departure_start_x = None
        self._transition(DriveStatus.DEPARTING)
        self._start_watchdog_timer()
        self.__publish_cmd_vel(linear_x=self._v_linear, angular_z=0.0)

    def stop(self) -> None:
        """
        External hard stop — e-stop, no-detection timeout, task cancel.

        Transitions to IDLE regardless of current state.
        """
        self.logger.info('DriveClient stop() — external hard stop')
        self._stop_watchdog_timer()
        self._clear_tracking_state()
        self._transition(DriveStatus.IDLE)
        self.__publish_cmd_vel(linear_x=0.0, angular_z=0.0)

    def cancel(self) -> None:
        """Abort active drive — transition to CANCELED."""
        if self._status in (DriveStatus.IDLE, DriveStatus.CANCELED):
            return
        self.logger.info('DriveClient cancel()')
        self._stop_watchdog_timer()
        self._clear_tracking_state()
        self._transition(DriveStatus.CANCELED)
        self.__publish_cmd_vel(linear_x=0.0, angular_z=0.0)

    def reset(self) -> None:
        """Reset to IDLE, clearing all active drive state."""
        self._stop_watchdog_timer()
        self._clear_tracking_state()
        self._transition(DriveStatus.IDLE)
        self.logger.info('DriveClient reset to IDLE')

    def get_status(self) -> DriveStatus:
        """Return the current DriveStatus."""
        return self._status

    def is_active(self) -> bool:
        """Return True when robot is moving (SCANNING, CORRECTING, DEPARTING)."""
        return self._status in (
            DriveStatus.SCANNING,
            DriveStatus.CORRECTING,
            DriveStatus.DEPARTING,
        )

    def last_detection_time(self) -> float | None:
        """
        Timestamp (seconds) of the last valid detection.

        Used by calling node for no-detection timeout logic.
        Returns None if no detection has been received since start().
        """
        return self._last_detection_time

    # ------------------------------------------------------------------
    # Detection callback (private)
    # ------------------------------------------------------------------

    def _on_detection(self, msg: ImageDetectionPose) -> None:
        """
        Internal ImageDetectionPose callback.

        On every message: log pose for diagnostics.
        On detection_valid=False: return early.
        On detection_valid=True: compute TF error signal then route by status.

          error_x = arm_0_base_link.x - camera_1_detections.x  (in base_link frame)
          error_y = arm_0_base_link.y - camera_1_detections.y  (in base_link frame)

          SCANNING   — accept first valid detection, transition to CORRECTING
          CORRECTING — minimum tracker C3 mitigation, check stop condition
          STOPPED    — ignore all detections (activity in progress)
          DEPARTING  — ignore until departure_clearance met, then SCANNING
          IDLE / CANCELED / ERROR — ignore
        """
        # Log pose on every message — valid or not — for diagnostics
        self._pose_log_callback()

        if not msg.detection_valid:
            return

        self._last_detection_time = self.node.get_clock().now().nanoseconds / 1e9

        # --- Compute TF error signal ---
        error_x, error_y = self._compute_tf_error()
        if error_x is None:
            # TF lookup failed — skip this detection tick
            return

        self.logger.info(f'Detection | error_x={error_x:.4f}m error_y={error_y:.4f}m | status={self._status.name}')

        if self._status == DriveStatus.SCANNING:
            self._handle_scanning(error_x, error_y)

        elif self._status == DriveStatus.CORRECTING:
            self._handle_correcting(error_x, error_y)

        elif self._status == DriveStatus.STOPPED:
            self.logger.debug('Detection ignored — STOPPED, activity in progress')

        elif self._status == DriveStatus.DEPARTING:
            self._handle_departing(error_x, error_y)

    # ------------------------------------------------------------------
    # Detection phase handlers (private)
    # ------------------------------------------------------------------

    def _handle_scanning(self, error_x: float, error_y: float) -> None:
        """
        SCANNING phase — first valid detection locks onto current bush.
        Initialises minimum tracker and transitions to CORRECTING.
        """
        self.logger.info(f'Bush detected | error_x={error_x:.4f}m error_y={error_y:.4f}m | SCANNING → CORRECTING')
        self._min_error_x_seen = abs(error_x)
        # self._transition(DriveStatus.CORRECTING)
        self._transition(DriveStatus.STOPPED)
        self._correct(error_x, error_y)

    def _handle_correcting(self, error_x: float, error_y: float) -> None:
        """
        CORRECTING phase — minimum tracker C3 mitigation.

        Accepts detections where abs(error_x) is decreasing toward zero.
        Rejects detections where abs(error_x) increases by more than noise_margin
        — this indicates the detector has switched to a different bush (C3).

        On stop condition met → STOPPED.
        """
        # abs_ex = abs(error_x)

        # # --- C3 minimum tracker ---
        # if self._min_error_x_seen is not None:
        #     if abs_ex > self._min_error_x_seen + self._noise_margin:
        #         self.logger.warning(
        #             f'C3 rejected | error_x={error_x:.4f}m '
        #             f'abs={abs_ex:.4f}m > min={self._min_error_x_seen:.4f}m '
        #             f'+ margin={self._noise_margin:.4f}m — bush switch detected'
        #         )
        #         return

        # self._min_error_x_seen = min(self._min_error_x_seen if self._min_error_x_seen is not None else abs_ex, abs_ex)

        # --- Stop condition ---
        if abs(error_x) <= self._tolerance:  # and abs(error_y) <= self._tolerance:
            self.logger.info(
                f'Stop condition met | error_x={error_x:.4f}m error_y={error_y:.4f}m | tolerance={self._tolerance}m'
            )
            self._stop_and_align(error_x)
            return

        self._correct(error_x, error_y)

    def _handle_departing(self, error_x: float, error_y: float) -> None:
        """
        DEPARTING phase — ignore detections until departure_clearance is met.

        Records error_x on first detection after resume() to establish
        the departure reference point. Departure is complete when error_x
        has moved departure_clearance metres more negative than the reference —
        confirming the robot has cleared the completed bush.

        On clearance met → SCANNING (processes this detection immediately).
        """
        # Record departure reference on first detection after resume()
        if self._departure_start_x is None:
            self._departure_start_x = error_x
            self.logger.debug(f'Departure reference set | error_x={error_x:.4f}m')
            return

        clearance_reached = error_x < self._departure_start_x - self._departure_clearance

        if clearance_reached:
            self.logger.info(
                f'Departure clearance met | '
                f'error_x={error_x:.4f}m | '
                f'start={self._departure_start_x:.4f}m '
                f'clearance={self._departure_clearance:.4f}m | '
                f'DEPARTING → SCANNING'
            )
            self._min_error_x_seen = None
            self._departure_start_x = None
            self._transition(DriveStatus.SCANNING)
            self._handle_scanning(error_x, error_y)
        else:
            self.logger.debug(
                f'Departing | error_x={error_x:.4f}m | need {self._departure_start_x - self._departure_clearance:.4f}m'
            )

    # ------------------------------------------------------------------
    # Motion helpers (private)
    # ------------------------------------------------------------------

    def _correct(self, error_x: float, error_y: float) -> None:
        """
        Travel axis correction only — no angular correction.

        error_x drives linear.x:
          error_x > 0 → arm ahead of bush  → drive backward (-linear.x)
          error_x < 0 → arm behind bush    → drive forward  (+linear.x)

        error_y is logged for diagnostics but drives nothing.
        Lateral positioning is Nav2's responsibility.
        """
        linear_x = -(1.0 if error_x > 0 else -1.0) * self._v_linear

        self.logger.debug(
            f'_correct | '
            f'error_x={error_x:.4f}m linear_x={linear_x:.4f} | '
            f'error_y={error_y:.4f}m (lateral — no correction)'
        )

        self.__publish_cmd_vel(linear_x=linear_x, angular_z=0.0)

    def _stop_and_align(self, error_x: float) -> None:
        """
        Called when stop condition is met.

        Publishes zero velocity, stops watchdog, transitions to STOPPED.
        """
        self._stop_watchdog_timer()
        self._transition(DriveStatus.STOPPED)
        self.__publish_cmd_vel(linear_x=0.0, angular_z=0.0)
        self.logger.info(f'Bush aligned | error_x={error_x:.4f}m | waiting for resume()')

    # ------------------------------------------------------------------
    # Watchdog timer
    # ------------------------------------------------------------------

    def _start_watchdog_timer(self) -> None:
        """Start 10Hz timer republishing last commanded velocity."""
        self._stop_watchdog_timer()
        self._watchdog_timer = self.node.create_timer(0.1, self._watchdog_callback)
        self.logger.debug('Watchdog timer started')

    def _stop_watchdog_timer(self) -> None:
        """Cancel watchdog timer if running."""
        if self._watchdog_timer is not None:
            self._watchdog_timer.cancel()
            self._watchdog_timer = None
            self.logger.debug('Watchdog timer stopped')

    def _watchdog_callback(self) -> None:
        """Republish last commanded velocity at 10Hz."""
        if not self.is_active():
            return
        self.__publish_cmd_vel(
            linear_x=self._last_linear_x,
            angular_z=self._last_angular_z,
        )

    # ------------------------------------------------------------------
    # Pose monitoring (private)
    # ------------------------------------------------------------------

    def _pose_log_callback(self) -> None:
        """
        TF pose logger — called on every ImageDetectionPose message.

        Logs base_link, arm_0_base_link, and camera_1_detections positions
        in the map frame. Fires on every message (valid or not) so pose is
        always correlated with detection pipeline activity.

        Logs ERROR if any frame is unavailable — map frame missing indicates
        navigation stack is not running or map → odom chain is broken.
        camera_1_detections is only broadcast when detection_valid=True —
        UNAVAILABLE between bushes is expected and normal.
        """
        t = rclpy.time.Time()
        timeout = rclpy.duration.Duration(seconds=0.1)
        frames = {
            'arm_0_base_link': self._tf_arm_frame,
            'camera_1_detections': self._tf_detection_frame,
        }
        poses = {}

        for label, frame_id in frames.items():
            try:
                tf = self._tf_buffer.lookup_transform(frame_id, self._base_frame, t, timeout=timeout)
                tx = tf.transform.translation.x
                ty = tf.transform.translation.y
                tz = tf.transform.translation.z
                poses[label] = f'({tx:.4f}, {ty:.4f}, {tz:.4f})'
            except TransformException as e:
                self.logger.error(
                    f'TF lookup failed | '
                    f'frame={frame_id} in {self._base_frame} | '
                    f'error={e}',
                    throttle_duration_sec=2.0
                )
                poses[label] = 'UNAVAILABLE'

        self.logger.info(
            f'Pose wrt {self._base_frame} | status={self._status.name} | '
            f'arm_0_base_link={poses["arm_0_base_link"]} | '
            f'camera_1_detections={poses["camera_1_detections"]}'
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _compute_tf_error(
        self,
    ) -> tuple[float, float] | tuple[None, None]:
        """
        Compute spatial error between arm and bush accounting for camera offset.

        Looks up at latest available time (rclpy.time.Time()) rather than
        message stamp — using message stamp causes extrapolation errors because
        camera_1_detections TF is broadcast at detection time but the TF buffer
        lags slightly behind the message header stamp.

        Applies pre-computed camera mounting offset (computed at start/resume):
          error_x = (arm.x - bush.x) - offset_x
          error_y = (arm.y - bush.y) - offset_y

        When error_x=0 and error_y=0, the bush is correctly positioned
        under the camera at the expected offset from arm_0_base_link.

        Returns (None, None) on TF lookup failure — caller skips detection tick.
        """
        if self._offset_x is None or self._offset_y is None:
            self.logger.error('Camera offset not computed — call start() or resume() first')
            return None, None

        t = rclpy.time.Time()  # latest available — avoids extrapolation errors
        timeout = rclpy.duration.Duration(seconds=0.1)

        try:
            arm_tf = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._tf_arm_frame,
                t,
                timeout=timeout,
            )
        except TransformException as e:
            self.logger.error(f'TF lookup failed | frame={self._tf_arm_frame} in {self._base_frame} | error={e}')
            return None, None

        try:
            bush_tf = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._tf_detection_frame,
                t,
                timeout=timeout,
            )
        except TransformException as e:
            self.logger.error(f'TF lookup failed | frame={self._tf_detection_frame} in {self._base_frame} | error={e}')
            return None, None

        raw_error_x = arm_tf.transform.translation.x - bush_tf.transform.translation.x
        raw_error_y = arm_tf.transform.translation.y - bush_tf.transform.translation.y

        error_x = raw_error_x - self._offset_x
        error_y = raw_error_y - self._offset_y

        self.logger.debug(
            f'TF error | '
            f'arm=({arm_tf.transform.translation.x:.4f}, '
            f'{arm_tf.transform.translation.y:.4f}) | '
            f'bush=({bush_tf.transform.translation.x:.4f}, '
            f'{bush_tf.transform.translation.y:.4f}) | '
            f'offset=({self._offset_x:.4f}, {self._offset_y:.4f}) | '
            f'error=({error_x:.4f}, {error_y:.4f})'
        )

        return error_x, error_y

    def _lookup_camera_offset(self) -> bool:
        """
        Compute and cache camera mounting offset in base_link frame.

        Called at start() and resume() — arm must be in scan pose.

        offset = arm_0_base_link - camera_1_color_optical_frame in base_link frame.

        This offset accounts for the physical distance between the arm origin
        and the camera optical centre. Applied in _compute_tf_error() so the
        robot stops with the bush correctly positioned under the camera.

        Returns True on success, False on failure (caller should not proceed).
        """
        timeout = rclpy.duration.Duration(seconds=0.5)
        t = rclpy.time.Time()

        try:
            arm_tf = self._tf_buffer.lookup_transform(                
                self._tf_arm_frame,
                self._base_frame,
                t,
                timeout=timeout,
            )
        except TransformException as e:
            self.logger.error(
                f'Camera offset lookup failed | '
                f'frame={self._tf_arm_frame} in {self._base_frame} | '
                f'error={e} | '
                f'ensure arm is in scan pose before calling start()/resume()'
            )
            return False

        try:
            camera_tf = self._tf_buffer.lookup_transform(
                self._tf_camera_optical_frame,
                self._base_frame,
                t,
                timeout=timeout,
            )
        except TransformException as e:
            self.logger.error(
                f'Camera offset lookup failed | '
                f'frame={self._tf_camera_optical_frame} in {self._base_frame} | '
                f'error={e} | '
                f'ensure arm is in scan pose before calling start()/resume()'
            )
            return False

        self._offset_x = arm_tf.transform.translation.x - camera_tf.transform.translation.x
        self._offset_y = arm_tf.transform.translation.y - camera_tf.transform.translation.y

        self.logger.info(f'Camera offset computed | offset_x={self._offset_x:.4f}m offset_y={self._offset_y:.4f}m')
        return True

    def _clear_tracking_state(self) -> None:
        """Reset all per-traversal tracking variables."""
        self._min_error_x_seen = None
        self._departure_start_x = None

    def _transition(self, new_status: DriveStatus) -> None:
        """Log and apply a status transition."""
        if self._status != new_status:
            self.logger.info(f'Status: {self._status.name} → {new_status.name}')
            self._status = new_status

    def __publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        """Stamp and publish TwistStamped. Caches values for watchdog."""
        self._last_linear_x = linear_x
        self._last_angular_z = angular_z
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        self._cmd_vel_pub.publish(msg)
