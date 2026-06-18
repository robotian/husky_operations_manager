"""
drive_v2.py

Camera-guided drive controller for lavender row harvesting.

Owned by a parent harvesting node which provides the ROS2 node handle
and drives the lifecycle via the public API (scan / resume / cancel / reset).

-------------------------------------------------------------------------------
Coordinate Frames
-------------------------------------------------------------------------------

  camera_1_detections (= camera_1_color_optical_frame orientation):
    Confirmed via tf2_echo and rviz.
    +X → backward  (negative when bush is ahead, zero when level, positive when passed)
    +Y → left
    +Z → down

  base_link:
    +X → forward
    +Y → left
    +Z → up

  Camera is mounted upside-down and backward-facing (~179° roll + ~179° yaw).
  camera_1_detections is a child of camera_1_color_optical_frame with same orientation.

-------------------------------------------------------------------------------
Error Signals
-------------------------------------------------------------------------------

  ex = msg.center.x
      Forward/backward centering error in camera_1_detections frame.
      Negative  → bush is ahead of the arm (approaching).
      Zero      → bush is level with the arm (stop point reached).
      Positive  → robot has passed the bush (overshoot).
      Sign is consistent across all robot headings — no correction needed.

  ey = msg.center.y * ey_sign
      Camera Y centering error — heading error signal.
      msg.center.y = 0 means bush is centered in camera Y axis.
      Since camera is backward-facing, camera Y centered on bush means
      robot heading is parallel to the row — correct harvesting position.
      msg.center.y sign depends on robot map-frame heading (see ey_sign below).

-------------------------------------------------------------------------------
Dynamic ey_sign
-------------------------------------------------------------------------------

  Bush is always physically to the robot's right (arm mounting constraint).
  Camera Y = base_link Y (both left). Bush to the right = -Y base_link.
  msg.center.y sign depends on robot map-frame heading:

    right_y_map = -cos(yaw)   [robot's right vector, Y component in map frame]

    If right_y_map >= 0: bush appears at positive center.y → ey_sign = -1.0
    If right_y_map <  0: bush appears at negative center.y → ey_sign = +1.0

  East/West boundary (right_y_map ≈ 0): resolved via right_x_map = sin(yaw).

  Verified against all 4 cardinals:
    North (+0°):   right_y_map=-1.0 → ey_sign=+1.0
    East  (+90°):  right_y_map=0.0  → right_x_map=+1.0 → ey_sign=-1.0
    South (+180°): right_y_map=+1.0 → ey_sign=-1.0
    West  (-90°):  right_y_map=0.0  → right_x_map=-1.0 → ey_sign=+1.0

  Computed once at scan() from live odom yaw. Stable for the entire approach
  since the robot's map-frame heading does not change significantly during
  a single bush approach.

-------------------------------------------------------------------------------
Controller
-------------------------------------------------------------------------------

  Activated on first valid detection of each new bush (ex < 0 only).
  Resets on scan() call and on DEPARTING → SCANNING transition.

  Forward speed scaling (linear_x):
    Starts at v_linear_max when bush is first detected (lookahead_distance).
    Reduces linearly to v_linear_min as ex approaches zero.
    lookahead_distance = abs(ex) at first detection — set once per bush.

    linear_x = v_linear_min + (v_linear_max - v_linear_min) * (abs(ex) / lookahead_distance)
    linear_x = clamp(linear_x, v_linear_min, v_linear_max)

  Heading correction (angular_z):
    angular_z = clamp(k_rho * ey, -v_angular_max, +v_angular_max)

    k_rho: single gain on camera Y centering error (heading error signal).
    Gated off when |ex| <= ex_angular_gate — robot drives straight for
    final approach segment, preventing yaw at the stop point.
    Gated off when ex >= 0 — bush is level or passed, no correction needed.

  Stop condition:
    |ex| <= ex_tolerance → STOPPED, zero velocity published.

  Overshoot detection:
    ex sign change negative → positive → hard stop (STOPPED), logged as WARNING.

-------------------------------------------------------------------------------
State Machine
-------------------------------------------------------------------------------

  IDLE
    └─► scan() → SCANNING  (forward drive, controller inactive)
          ├─► first valid detection (ex < 0) → controller activates
          ├─► |ex| <= ex_tolerance + stop_lookahead → STOPPED
          ├─► ex sign flip (overshoot) → STOPPED  (WARNING logged)
          ├─► no detection for no_detection_timeout → STOPPED  (row end)
          └─► cancel() → CANCELED

  STOPPED
    └─► resume() → DEPARTING  (detections ignored, drive past bush)
          └─► clearance elapsed → SCANNING  (controller reset for next bush)

  Any active state: cancel() → CANCELED | reset() → IDLE

-------------------------------------------------------------------------------
Notes
-------------------------------------------------------------------------------

  - cmd_vel is republished at cmd_vel_rate between detections using last
    computed (linear_x, angular_z). Prevents robot stopping due to cmd_vel
    timeout between ~5Hz detection messages.
  - _cmd_vel_timer is the sole ROS2 timer. No-detection timeout and departure
    clearance are implemented as absolute deadlines checked inside
    _cmd_vel_repeat_callback. Resolution = 1 / cmd_vel_rate (100ms at 10Hz).
    Worst-case departure overshoot at v_linear_max=0.1m/s: 1cm — acceptable.
  - Odom subscription uses qos_profile_sensor_data (BEST_EFFORT).
  - DriveConfig has no default values — all fields must be explicitly supplied.
  - TF lookup is NOT used — ex/ey derived directly from msg.center with
    ey_sign correction from odom yaw.
"""

import math

from husky_operations_manager.robot_enums import DriveStatus
from husky_operations_manager.types import DriveConfig
from nav_msgs.msg import Odometry
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion

from geometry_msgs.msg import TwistStamped
from status_interfaces.msg import ImageDetectionPose

class DriveClient:
    """
    Component class for camera-guided forward drive with heading correction.

    Instantiated and owned by the parent harvesting node.
    Uses the parent node reference for all ROS2 primitives.

    Lifecycle:
      1. scan()    — start SCANNING forward, controller inactive
      2. (auto)    — first valid detection (ex < 0) activates controller
      3. (auto)    — |ex| <= ex_tolerance → STOPPED
      4. resume()  — called by parent after harvest → DEPARTING
      5. (auto)    — departure clearance met → SCANNING, controller reset
    """

    def __init__(self, node: Node, config: DriveConfig) -> None:
        self._node   = node
        self._logger = RcutilsLogger(self.__class__.__name__)
        self._ns     = self._node.get_namespace().rstrip('/')

        # --- Config ---
        self._base_frame            = config.base_frame
        self._ex_tolerance          = config.ex_tolerance
        self._stop_lookahead        = config.stop_lookahead
        self._ex_coast_gate         = config.ex_coast_gate
        self._ex_angular_gate       = config.ex_angular_gate
        self._k_rho                 = config.k_rho
        self._v_linear_min          = config.v_linear_min
        self._v_linear_max          = config.v_linear_max
        self._v_angular_max         = config.v_angular_max
        self._departure_clearance   = config.departure_clearance

        # Convert no_detection_distance to timeout using v_linear_max
        self._no_detection_timeout: float = (
            config.no_detection_distance / max(config.v_linear_max, 0.01)
        )

        # --- Odom state ---
        # ey_sign is computed once at scan() from live odom yaw.
        # Determines correct angular_z direction for camera Y centering.
        self._current_yaw:   float = 0.0
        self._odom_received: bool  = False
        self._ey_sign:       float = 0.0   # set at scan() time

        # --- Controller state ---
        self._controller_active:  bool         = False
        self._lookahead_distance: float | None = None  # set on first detection
        self._last_ex:            float | None = None  # overshoot detection

        # --- Deadline state ---
        # Absolute wall-clock timestamps (nanoseconds / 1e9) checked inside
        # _cmd_vel_repeat_callback. None means the respective feature is inactive.
        self._no_detection_deadline: float | None = None
        self._departure_deadline:    float | None = None

        # --- Drive state ---
        self._status:           DriveStatus = DriveStatus.IDLE
        self._current_linear_x: float       = 0.0
        self._current_angular_z: float      = 0.0

        # --- Odom subscription ---
        self._odom_sub = self._node.create_subscription(
            Odometry,
            f'{self._ns}/{config.odom_topic}',
            self._odom_callback,
            qos_profile_sensor_data,
        )

        # --- Detection subscription ---
        self._detection_sub = self._node.create_subscription(
            ImageDetectionPose,
            f'{self._ns}/{config.detection_topic}',
            self._detection_callback,
            qos_profile_sensor_data,
        )

        # --- cmd_vel publisher ---
        self._cmd_vel_pub = self._node.create_publisher(
            TwistStamped,
            f'{self._ns}/cmd_vel',
            10,
        )

        # --- cmd_vel repeat timer ---
        self._cmd_vel_timer = self._node.create_timer(
            1.0 / config.cmd_vel_rate,
            self._cmd_vel_repeat_callback,
        )

        self._logger.info(
            f'DriveClient v2 initialized | '
            f'v_linear=[{self._v_linear_min}, {self._v_linear_max}]m/s | '
            f'v_angular_max={self._v_angular_max}rad/s | '
            f'k_rho={self._k_rho} | '
            f'ex_tolerance={self._ex_tolerance}m | '
            f'stop_lookahead={self._stop_lookahead}m | '
            f'ex_coast_gate={self._ex_coast_gate}m | '
            f'ex_angular_gate={self._ex_angular_gate}m | '
            f'departure_clearance={self._departure_clearance}m | '
            f'no_detection_timeout={self._no_detection_timeout:.1f}s'
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def scan(self) -> None:
        """
        Start forward scanning drive.

        Computes ey_sign from current odom yaw. Guards against starting
        if odom has not yet been received. Controller is inactive until
        the first valid detection with ex < 0.
        """
        if not self._odom_received:
            self._logger.error(
                'scan() called but odom not yet received — '
                'cannot compute ey_sign, aborting scan'
            )
            return

        self._ey_sign = self._compute_ey_sign()
        self._logger.info(
            f'SCANNING | '
            f'yaw={math.degrees(self._current_yaw):+.2f}deg | '
            f'ey_sign={self._ey_sign:+.1f} | '
            f'v_max={self._v_linear_max}m/s | '
            f'controller inactive until first detection'
        )

        self._status = DriveStatus.SCANNING
        self._reset_controller()
        self._current_linear_x      = self._v_linear_max
        self._current_angular_z     = 0.0
        self._no_detection_deadline = None
        self._departure_deadline    = None
        self._publish_cmd_vel(self._v_linear_max, 0.0)

    def resume(self) -> None:
        """
        Called by the parent node after harvest activity completes.

        Transitions to DEPARTING. Detections are ignored until the robot
        has moved departure_clearance metres past the bush. Controller is
        reset at the DEPARTING → SCANNING transition via scan().
        """
        if self._status != DriveStatus.STOPPED:
            self._logger.warning(
                f'resume() called in unexpected state: {self._status.name} — ignoring'
            )
            return

        departure_duration = self._departure_clearance / max(self._v_linear_max, 0.01)
        self._departure_deadline = (
            self._node.get_clock().now().nanoseconds / 1e9 + departure_duration
        )

        self._logger.info(
            f'DEPARTING | '
            f'clearance={self._departure_clearance}m | '
            f'estimated_duration={departure_duration:.1f}s'
        )

        self._status            = DriveStatus.DEPARTING
        self._current_linear_x  = self._v_linear_max
        self._current_angular_z = 0.0
        self._publish_cmd_vel(self._v_linear_max, 0.0)

    def cancel(self) -> None:
        """Cancel active drive and publish zero velocity."""
        if self._status in (DriveStatus.IDLE, DriveStatus.CANCELED):
            return
        self._status                = DriveStatus.CANCELED
        self._current_linear_x      = 0.0
        self._current_angular_z     = 0.0
        self._no_detection_deadline = None
        self._departure_deadline    = None
        self._reset_controller()
        self._publish_cmd_vel(0.0, 0.0)
        self._logger.info('Drive CANCELED')

    def reset(self) -> None:
        """Reset to IDLE — clears all state."""
        self._status                = DriveStatus.IDLE
        self._current_linear_x      = 0.0
        self._current_angular_z     = 0.0
        self._no_detection_deadline = None
        self._departure_deadline    = None
        self._reset_controller()
        self._logger.info('DriveClient reset to IDLE')

    def get_status(self) -> DriveStatus:
        """Return the current DriveStatus."""
        return self._status

    def is_active(self) -> bool:
        """Return True if the robot is currently moving."""
        return self._status in (DriveStatus.SCANNING, DriveStatus.DEPARTING)

    def is_ready(self) -> bool:
        """Return True if odom has been received and scan() can be safely called."""
        return self._odom_received

    # =========================================================================
    # Odom callback
    # =========================================================================

    def _odom_callback(self, msg: Odometry) -> None:
        """
        Extract yaw from odom quaternion and store as _current_yaw.

        Called continuously. _current_yaw is only consumed at scan() time
        to compute ey_sign — not used during the approach itself.

        Throttled debug log at 5s — once per bush approach, not per message.
        """
        q = msg.pose.pose.orientation
        _, _, self._current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._odom_received = True

        self._logger.debug(
            f'Odom | '
            f'position=({msg.pose.pose.position.x:.3f}, {msg.pose.pose.position.y:.3f}) | '
            f'yaw={math.degrees(self._current_yaw):+.2f}deg',
            throttle_duration_sec=5.0,
        )

    # =========================================================================
    # ey_sign computation
    # =========================================================================

    def _compute_ey_sign(self) -> float:
        """
        Compute ey_sign from current robot map-frame yaw.

        Bush is always physically to the robot's right (arm mounting constraint).
        Camera Y = base_link Y (both left). Bush to right = -Y base_link = -Y camera.
        msg.center.y sign depends on robot map-frame heading:

          right_y_map = -cos(yaw)   [Y component of robot's right vector in map]

          right_y_map >= 0: bush appears at positive center.y → ey_sign = -1.0
          right_y_map <  0: bush appears at negative center.y → ey_sign = +1.0

        East/West boundary (right_y_map ≈ 0):
          Resolved via right_x_map = sin(yaw) [X component of robot's right vector]
          right_x_map > 0 → ey_sign = -1.0
          right_x_map < 0 → ey_sign = +1.0

        Verified against all 4 cardinals:
          North (0°):   right_y_map=-1.0          → ey_sign=+1.0
          East  (+90°): right_y_map=0 boundary,   right_x_map=+1.0 → ey_sign=-1.0
          South (+180°): right_y_map=+1.0         → ey_sign=-1.0
          West  (-90°): right_y_map=0 boundary,   right_x_map=-1.0 → ey_sign=+1.0
        """
        right_y_map = -math.cos(self._current_yaw)

        if abs(right_y_map) > 0.01:
            # Clear case — use Y component directly
            sign = -1.0 if right_y_map >= 0.0 else +1.0
        else:
            # East/West boundary — resolve via X component
            right_x_map = math.sin(self._current_yaw)
            sign = -1.0 if right_x_map > 0.0 else +1.0

        self._logger.debug(
            f'_compute_ey_sign | '
            f'yaw={math.degrees(self._current_yaw):+.2f}deg | '
            f'right_y_map={right_y_map:+.4f} | '
            f'ey_sign={sign:+.1f}'
        )
        return sign

    # =========================================================================
    # Detection callback
    # =========================================================================

    def _detection_callback(self, msg: ImageDetectionPose) -> None:
        """
        Fires on every ImageDetectionPose message.

        SCANNING:
          Invalid detection within ex_coast_gate → zero linear_x to hold
            position. Prevents coasting past stop point during detection drops.
          Invalid detection outside ex_coast_gate → no-op (deadline keeps running).
          First valid detection with ex < 0 → activate controller,
            set lookahead_distance = abs(ex).
          First valid detection with ex >= 0 → trailing edge of previous
            bush after DEPARTING — skip, not a new bush.
          Subsequent valid detections → compute velocity, check stop/overshoot.

        DEPARTING:
          All detections → ignored, clearance handled by _cmd_vel_repeat_callback.

        All other states: no-op.
        """
        self._logger.debug(
            f'Detection | valid={msg.detection_valid} | '
            f'center=({msg.center.x:.3f}, {msg.center.y:.3f}, {msg.center.z:.3f}) | '
            f'status={self._status.name}'
        )

        if self._status == DriveStatus.DEPARTING:
            return

        if self._status != DriveStatus.SCANNING:
            return

        if not msg.detection_valid:
            # Within coast gate — hold position to prevent coasting past stop point
            if (
                self._controller_active
                and self._last_ex is not None
                and abs(self._last_ex) <= self._ex_coast_gate
            ):
                self._current_linear_x = 0.0
                self._logger.debug(
                    f'Invalid detection within coast gate | '
                    f'last_ex={self._last_ex:+.3f}m — holding position'
                )
            return

        # Valid detection in SCANNING state — reset no-detection deadline
        # (only if controller already active; deadline starts on first detection)

        ex = msg.center.x
        ey = msg.center.y * self._ey_sign

        # --- Controller activation ---
        if not self._controller_active:
            if ex >= 0.0:
                self._logger.debug(
                    f'Skipping activation — trailing edge detected | ex={ex:+.3f}m'
                )
                return

            # First valid detection of new bush (ex < 0) — activate controller
            self._lookahead_distance    = abs(ex)
            self._controller_active     = True
            self._last_ex               = ex
            self._no_detection_deadline = (
                self._node.get_clock().now().nanoseconds / 1e9 + self._no_detection_timeout
            )
            self._logger.info(
                f'Controller activated | '
                f'lookahead_distance={self._lookahead_distance:.3f}m | '
                f'ex={ex:+.3f}m ey={ey:+.3f}m | '
                f'ey_sign={self._ey_sign:+.1f}'
            )

        # Controller active — reset no-detection deadline on every valid detection
        self._no_detection_deadline = (
            self._node.get_clock().now().nanoseconds / 1e9 + self._no_detection_timeout
        )

        # --- Overshoot detection ---
        if self._last_ex is not None and self._last_ex < 0.0 and ex > 0.0:
            self._logger.warning(
                f'Overshoot | '
                f'ex crossed zero: last={self._last_ex:+.3f}m current={ex:+.3f}m — '
                f'hard stop'
            )
            self._hard_stop()
            return

        self._last_ex = ex

        # --- Stop condition ---
        stop_threshold = self._ex_tolerance + self._stop_lookahead
        if abs(ex) <= stop_threshold:
            self._logger.info(
                f'Bush level with arm | '
                f'ex={ex:+.4f}m within threshold={stop_threshold}m '
                f'(tolerance={self._ex_tolerance}m + lookahead={self._stop_lookahead}m) | '
                f'ey={ey:+.4f}m — STOPPED'
            )
            self._hard_stop()
            return

        # --- Compute and apply velocity ---
        linear_x, angular_z = self._compute_velocity(ex, ey)
        self._current_linear_x  = linear_x
        self._current_angular_z = angular_z
        self._publish_cmd_vel(linear_x, angular_z)

        self._logger.debug(
            f'Controller | '
            f'ex={ex:+.4f}m ey={ey:+.4f}m | '
            f'linear_x={linear_x:.4f}m/s angular_z={angular_z:+.4f}rad/s'
        )

    # =========================================================================
    # Controller
    # =========================================================================

    def _compute_velocity(self, ex: float, ey: float) -> tuple[float, float]:
        """
        Compute (linear_x, angular_z) from forward and heading errors.

        Forward speed scaling:
          linear_x scales linearly from v_linear_max at first detection
          (lookahead_distance) down to v_linear_min as ex → 0.
          Clamped to [v_linear_min, v_linear_max].

        Heading correction:
          angular_z = clamp(k_rho * ey, -v_angular_max, +v_angular_max)
          ey = msg.center.y * ey_sign = camera Y centering error.
          ey = 0 → bush centered in camera Y → robot heading parallel to row.
          Zeroed when |ex| <= ex_angular_gate (final approach, drive straight)
          or when ex >= 0 (bush level or passed, no correction).
        """
        # --- Forward speed scaling ---
        lookahead = self._lookahead_distance if self._lookahead_distance else abs(ex)
        if lookahead > 0.0:
            fraction = abs(ex) / lookahead
        else:
            fraction = 0.0

        linear_x = self._v_linear_min + (self._v_linear_max - self._v_linear_min) * fraction
        linear_x = max(self._v_linear_min, min(self._v_linear_max, linear_x))

        # --- Heading correction ---
        if abs(ex) <= self._ex_angular_gate or ex >= 0.0:
            # Final approach segment or bush passed — drive straight
            angular_z = 0.0
        else:
            raw_angular = self._k_rho * ey
            angular_z   = max(-self._v_angular_max, min(self._v_angular_max, raw_angular))

        self._logger.debug(
            f'_compute_velocity | '
            f'ex={ex:+.4f}m ey={ey:+.4f}m | '
            f'linear_x={linear_x:.4f}m/s angular_z={angular_z:+.4f}rad/s | '
            f'lookahead={lookahead:.3f}m fraction={fraction:.3f}'
        )

        return linear_x, angular_z

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _hard_stop(self) -> None:
        """Publish zero velocity and transition to STOPPED."""
        self._status                = DriveStatus.STOPPED
        self._current_linear_x      = 0.0
        self._current_angular_z     = 0.0
        self._no_detection_deadline = None
        self._publish_cmd_vel(0.0, 0.0)

    def _reset_controller(self) -> None:
        """
        Reset all per-bush controller state.

        Called on scan() (new row start or post-departure) and cancel()/reset().
        Each new bush gets a fresh lookahead_distance and ey_sign is already
        set by scan() before this is called.
        """
        self._controller_active  = False
        self._lookahead_distance = None
        self._last_ex            = None

    def _cmd_vel_repeat_callback(self) -> None:
        """
        Fires at cmd_vel_rate. Serves three purposes:

        1. SCANNING — republish last linear_x to prevent cmd_vel timeout between
           ~5Hz detections. Check no_detection_deadline; on expiry → hard stop
           (row end signal).

        2. DEPARTING — republish v_linear_max. Check departure_deadline; on expiry
           → call scan() to reset controller and resume SCANNING.

        3. Both — angular_z is always zeroed. Publishing stale angular_z between
           detections causes yaw drift when the detection pipeline drops out.

        Deadline resolution = 1 / cmd_vel_rate (100ms at 10Hz).
        Worst-case departure overshoot at v_linear_max=0.1m/s: ~1cm.
        """
        if self._status == DriveStatus.SCANNING:
            if (
                self._no_detection_deadline is not None
                and self._node.get_clock().now().nanoseconds / 1e9 >= self._no_detection_deadline
            ):
                self._no_detection_deadline = None
                self._logger.info(
                    f'No detection for {self._no_detection_timeout:.1f}s — '
                    f'row end assumed, stopping'
                )
                self._hard_stop()
                return
            self._publish_cmd_vel(self._current_linear_x, 0.0)

        elif self._status == DriveStatus.DEPARTING:
            if (
                self._departure_deadline is not None
                and self._node.get_clock().now().nanoseconds / 1e9 >= self._departure_deadline
            ):
                self._departure_deadline = None
                self._logger.info(
                    'Departure clearance met — resuming SCANNING | '
                    'controller reset for next bush'
                )
                self.scan()
                return
            self._publish_cmd_vel(self._current_linear_x, 0.0)

    def _publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        """Wrap velocity in TwistStamped and publish to cmd_vel."""
        msg                  = TwistStamped()
        msg.header.stamp     = self._node.get_clock().now().to_msg()
        msg.header.frame_id  = self._base_frame
        msg.twist.linear.x   = linear_x
        msg.twist.angular.z  = angular_z
        self._cmd_vel_pub.publish(msg)
