"""
Drive client for Husky operations.

Manages forward row traversal and sequential bush alignment using
ImageDetectionPose as the sole error signal.

Axis mapping (confirmed from ground truth image — top-down view):
  base_link:           +X = robot forward, +Y = left, +Z = up
  camera_1_detections: +X = opposite base_link +X, +Y = left, +Z = down

  Bushes are arranged along the robot travel axis.
  As robot moves forward, bush moves in camera_1_detections +X direction.
  center_x is the TRAVEL axis. center_y is the LATERAL axis.

  center_x < 0 → bush ahead  → drive forward  (+linear.x)
  center_x > 0 → bush passed → drive backward (-linear.x)
  center_x ≈ 0 → bush centred → STOP

  center_y → angular.z correction (sign field-validated)

Physical assumption:
  arm_0_base_link X offset from camera along travel axis is ZERO.
  center_x = 0 means arm is directly over bush.
  tolerance directly determines arm positioning accuracy.

Stop condition:
  abs(center_x) <= tolerance AND abs(center_y) <= tolerance

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

    Physical assumption:
      arm_0_base_link X offset from camera along travel axis is ZERO.
      When center_x = 0, both camera and arm are directly over the bush.
      tolerance therefore directly determines arm positioning accuracy.
      If arm pose changes, this assumption must be revalidated.
    """

    # ── Frame ──────────────────────────────────────────────────────────────────
    base_frame: str

    # ── Speed ──────────────────────────────────────────────────────────────────
    v_linear: float  # m/s   — constant forward/backward speed
    v_angular: float  # rad/s — constant lateral correction speed

    # ── Stop condition ─────────────────────────────────────────────────────────
    tolerance: float  # m — stop when abs(center_x) <= tolerance
    #         AND abs(center_y) <= tolerance

    # ── Lateral correction direction ───────────────────────────────────────────
    center_y_correction_sign: float  # lateral axis → angular.z
    # flip to 1.0 if robot turns wrong way

    # ── C3 mitigation — minimum tracker ───────────────────────────────────────
    noise_margin: float  # m — wobble budget: how much abs(center_x) can
    # increase between ticks before rejecting as bush switch

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
        self._v_linear = config.v_linear
        self._v_angular = config.v_angular
        self._tolerance = config.tolerance
        self._center_y_correction_sign = config.center_y_correction_sign
        self._noise_margin = config.noise_margin
        self._departure_clearance = config.departure_clearance

        # --- Status ---
        self._status: DriveStatus = DriveStatus.IDLE

        # --- C3 minimum tracker ---
        # Tracks the lowest abs(center_x) seen during current CORRECTING phase.
        # Rejects detections where abs(center_x) rises above this by more than
        # noise_margin — indicates detector switched to a different bush.
        self._min_center_x_seen: float | None = None

        # --- Departure tracking ---
        # Records center_x at the moment resume() is called.
        # Departure is complete when center_x < _departure_start_x - departure_clearance.
        self._departure_start_x: float | None = None

        # --- Detection timing ---
        self._last_detection_time: float | None = None

        # --- Last commanded velocity (watchdog republish) ---
        self._last_linear_x: float = 0.0
        self._last_angular_z: float = 0.0

        # --- Timers ---
        self._watchdog_timer: Timer | None = None

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
            f'v_linear={self._v_linear} v_angular={self._v_angular} | '
            f'tolerance={self._tolerance}m | '
            f'center_y_correction_sign={self._center_y_correction_sign} | '
            f'noise_margin={self._noise_margin}m | '
            f'departure_clearance={self._departure_clearance}m | '
            f'detection_topic={self.namespace}/{config.detection_topic}'
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Begin forward row traversal — transition to SCANNING.

        Precondition: arm must be in scan pose (caller's responsibility).
        Clears all tracking state from any previous traversal.
        """
        self.logger.info(f'DriveClient start | v_linear={self._v_linear}')
        self._clear_tracking_state()
        self._last_detection_time = self.node.get_clock().now().nanoseconds / 1e9
        self._transition(DriveStatus.SCANNING)
        self._start_watchdog_timer()
        self.__publish_cmd_vel(linear_x=self._v_linear, angular_z=0.0)

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
        self._min_center_x_seen = None
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

        Routes to handler based on current DriveStatus:

          SCANNING   — accept first valid detection, transition to CORRECTING
          CORRECTING — minimum tracker C3 mitigation, check stop condition
          STOPPED    — ignore all detections (activity in progress)
          DEPARTING  — ignore until departure_clearance met, then SCANNING
          IDLE / CANCELED / ERROR — ignore
        """
        if not msg.detection_valid:
            return

        center_x = msg.center.x
        center_y = msg.center.y
        self._last_detection_time = self.node.get_clock().now().nanoseconds / 1e9

        self.logger.debug(f'Detection | center_x={center_x:.4f}m center_y={center_y:.4f}m | status={self._status.name}')

        if self._status == DriveStatus.SCANNING:
            self._handle_scanning(center_x, center_y)

        elif self._status == DriveStatus.CORRECTING:
            self._handle_correcting(center_x, center_y)

        elif self._status == DriveStatus.STOPPED:
            # Activity in progress — ignore all detections
            self.logger.debug('Detection ignored — STOPPED, activity in progress')

        elif self._status == DriveStatus.DEPARTING:
            self._handle_departing(center_x, center_y)

    # ------------------------------------------------------------------
    # Detection phase handlers (private)
    # ------------------------------------------------------------------

    def _handle_scanning(self, center_x: float, center_y: float) -> None:
        """
        SCANNING phase — first valid detection locks onto current bush.

        Initialises minimum tracker and transitions to CORRECTING.
        """
        self.logger.info(f'Bush detected | center_x={center_x:.4f}m center_y={center_y:.4f}m | SCANNING → CORRECTING')
        self._min_center_x_seen = abs(center_x)
        self._transition(DriveStatus.CORRECTING)
        self._correct(center_x, center_y)

    def _handle_correcting(self, center_x: float, center_y: float) -> None:
        """
        CORRECTING phase — minimum tracker C3 mitigation.

        Accepts detections where abs(center_x) is decreasing toward zero.
        Rejects detections where abs(center_x) increases by more than noise_margin
        — this indicates the detector has switched to a different bush (C3).

        On stop condition met → STOPPED.
        """
        abs_cx = abs(center_x)

        # --- C3 minimum tracker ---
        if self._min_center_x_seen is not None:
            if abs_cx > self._min_center_x_seen + self._noise_margin:
                self.logger.warning(
                    f'C3 rejected | center_x={center_x:.4f}m '
                    f'abs={abs_cx:.4f}m > min={self._min_center_x_seen:.4f}m '
                    f'+ margin={self._noise_margin:.4f}m — bush switch detected'
                )
                return

        self._min_center_x_seen = min(
            self._min_center_x_seen if self._min_center_x_seen is not None else abs_cx, abs_cx
        )

        # --- Stop condition ---
        if abs(center_x) <= self._tolerance and abs(center_y) <= self._tolerance:
            self.logger.info(
                f'Stop condition met | center_x={center_x:.4f}m center_y={center_y:.4f}m | tolerance={self._tolerance}m'
            )
            self._stop_and_align(center_x)
            return

        self._correct(center_x, center_y)

    def _handle_departing(self, center_x: float, center_y: float) -> None:
        """
        DEPARTING phase — ignore detections until departure_clearance is met.

        Records center_x on first detection after resume() to establish
        the departure reference point. Departure is complete when center_x
        has moved departure_clearance metres more negative than the reference —
        confirming the robot has cleared the completed bush.

        On clearance met → SCANNING (processes this detection immediately).
        """
        # Record departure reference on first detection after resume()
        if self._departure_start_x is None:
            self._departure_start_x = center_x
            self.logger.debug(f'Departure reference set | center_x={center_x:.4f}m')
            return

        clearance_reached = center_x < self._departure_start_x - self._departure_clearance

        if clearance_reached:
            self.logger.info(
                f'Departure clearance met | '
                f'center_x={center_x:.4f}m | '
                f'start={self._departure_start_x:.4f}m '
                f'clearance={self._departure_clearance:.4f}m | '
                f'DEPARTING → SCANNING'
            )
            self._min_center_x_seen = None
            self._departure_start_x = None
            self._transition(DriveStatus.SCANNING)
            self._handle_scanning(center_x, center_y)
        else:
            self.logger.debug(
                f'Departing | center_x={center_x:.4f}m | '
                f'need {self._departure_start_x - self._departure_clearance:.4f}m'
            )

    # ------------------------------------------------------------------
    # Motion helpers (private)
    # ------------------------------------------------------------------

    def _correct(self, center_x: float, center_y: float) -> None:
        """
        Two-axis correction toward bush centre.

        center_x → linear.x  (travel axis, negated: center_x < 0 = ahead = forward)
        center_y → angular.z (lateral axis, sign field-validated)
        Angular correction only applied when center_y is outside tolerance.
        """
        linear_x = -(1.0 if center_x > 0 else -1.0) * self._v_linear

        angular_z = (
            self._center_y_correction_sign * (1.0 if center_y > 0 else -1.0) * self._v_angular
            if abs(center_y) > self._tolerance
            else 0.0
        )

        self.logger.debug(
            f'_correct | '
            f'center_x={center_x:.4f}m linear_x={linear_x:.4f} | '
            f'center_y={center_y:.4f}m angular_z={angular_z:.4f}'
        )

        self.__publish_cmd_vel(linear_x=linear_x, angular_z=angular_z)

    def _stop_and_align(self, center_x: float) -> None:
        """
        Called when stop condition is met.

        Publishes zero velocity, stops watchdog, transitions to STOPPED.
        Watchdog is stopped — robot is stationary. Caller is responsible
        for feeding the Husky watchdog during the activity if required.
        """
        self._stop_watchdog_timer()
        self._transition(DriveStatus.STOPPED)
        self.__publish_cmd_vel(linear_x=0.0, angular_z=0.0)
        self.logger.info(f'Bush aligned | center_x={center_x:.4f}m | waiting for resume()')

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
    # Utility
    # ------------------------------------------------------------------

    def _clear_tracking_state(self) -> None:
        """Reset all per-traversal tracking variables."""
        self._min_center_x_seen = None
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
