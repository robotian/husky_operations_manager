"""
drive_v2.py

Camera-guided drive controller for crop-row harvesting.
Replaces drive.py's simple proportional logic with a bearing+lateral
controller that couples heading correction and lateral error into a
single angular_z command, and scales linear_x proportionally with
forward distance to the bush.

-------------------------------------------------------------------------------
Coordinate Convention
-------------------------------------------------------------------------------

  ex = msg.center.x              Forward/backward offset to bush centroid (m).
                                   Negative  → bush is ahead of the arm.
                                   Zero      → bush is level with the arm (stop).
                                   Positive  → robot has passed the bush.

  ey = msg.center.y * ey_sign    Lateral offset after sign correction (m).
                                   Positive  → too far from bush (steer closer).
                                   Negative  → too close to bush (steer away).

  ey_sign = -1.0 by default      Bush is to the RIGHT of the robot. Positive
                                   camera y → robot is too far → steer right.

-------------------------------------------------------------------------------
Controller
-------------------------------------------------------------------------------

  Activated on first valid detection of each new bush.
  Resets on scan() call and on DEPARTING → SCANNING transition.

  phi = atan2(ey, abs(ex))
      Bearing angle to the bush centroid from the robot forward axis.
      Serves dual purpose:
        1. Bearing correction  — drives the robot toward the level point.
        2. Heading correction  — keeps the robot parallel to the row.
           A correctly headed robot sees ey constant and ex decreasing;
           heading drift shows up as phi deviation from the expected trajectory.

  angular_z = clamp(k_phi * phi + k_lateral * ey, -v_angular_max, +v_angular_max)
      k_phi    — gain on bearing angle (heading + bearing correction).
      k_lateral — gain on raw lateral offset (fine lateral position correction).

  linear_x = clamp(
      v_linear_min + (v_linear_max - v_linear_min) * (abs(ex) / lookahead_distance),
      v_linear_min,
      v_linear_max
  )
      lookahead_distance = abs(ex) at first detection of this bush.
      Robot starts at v_linear_max, reduces linearly to v_linear_min as ex → 0.

  Gating:
    - Controller inactive until first valid detection per bush.
    - angular_z zeroed when ex >= 0 (bush level or passed — no lateral steering
      toward a bush that is behind the robot).
    - Hard stop on ex sign change negative → positive (overshoot detected).

-------------------------------------------------------------------------------
State Machine
-------------------------------------------------------------------------------

  IDLE
    └─► scan() → SCANNING  (forward drive + controller inactive)
          ├─► first detection → controller activates
          ├─► ex within ex_tolerance → STOPPED
          ├─► ex sign flip (overshoot) → STOPPED  (logged as WARNING)
          ├─► no detection for no_detection_timeout → STOPPED  (row end)
          └─► cancel() → CANCELED

  STOPPED
    └─► resume() → DEPARTING  (detections ignored, drive past bush)
          └─► clearance elapsed → SCANNING  (controller reset for next bush)

  Any active state: cancel() → CANCELED | reset() → IDLE

-------------------------------------------------------------------------------
Notes
-------------------------------------------------------------------------------

  - cmd_vel is republished at cmd_vel_rate between detections using the last
    computed (linear_x, angular_z). This prevents the robot from stopping
    due to cmd_vel timeout between ~5Hz detection messages.
  - DriveConfig is populated by the parent node from YAML parameters.
  - No TF lookup — raw msg.center.x/y used directly.
"""

import math
from dataclasses import dataclass
from enum import IntEnum

from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import TwistStamped
from status_interfaces.msg import ImageDetectionPose


# =============================================================================
# DriveStatus
# =============================================================================


class DriveStatus(IntEnum):
    """Operational states for DriveClient."""

    IDLE      = 0
    SCANNING  = 1   # Moving forward, controller may or may not be active
    STOPPED   = 2   # Bush level with arm — waiting for resume()
    DEPARTING = 3   # Moving past bush before re-enabling detection
    CANCELED  = 4
    ERROR     = 5


# =============================================================================
# DriveConfig
# =============================================================================


@dataclass
class DriveConfig:
    """
    Configuration for DriveClient.

    All velocity and gain values are loaded from YAML by the parent node.
    No defaults are set here — every field must be explicitly supplied to
    prevent silent misconfiguration.
    """

    # --- Detection subscription ---
    detection_topic: str           # Topic name relative to namespace

    # --- cmd_vel ---
    base_frame: str                # frame_id for TwistStamped header
    cmd_vel_rate: float            # Republish rate between detections (Hz)

    # --- Stop condition ---
    ex_tolerance: float            # |ex| <= ex_tolerance → STOPPED (m)

    # --- Lateral convention ---
    ey_sign: float                 # +1.0 or -1.0; corrects camera y axis

    # --- Controller gains ---
    k_phi: float                   # Bearing error gain
    k_lateral: float               # Lateral offset gain

    # --- Speed limits ---
    v_linear_min: float            # Minimum forward speed during approach (m/s)
    v_linear_max: float            # Maximum forward speed / speed at first detection (m/s)
    v_angular_max: float           # Clamp on angular_z output (rad/s)

    # --- Departure ---
    departure_clearance: float     # Distance past bush before re-enabling detection (m)

    # --- No-detection timeout ---
    no_detection_distance: float   # Distance without detection before row-end assumed (m)


# =============================================================================
# DriveClient
# =============================================================================


class DriveClient:
    """
    Component class for camera-guided forward drive with bearing+lateral control.

    Instantiated and owned by the parent harvesting node.
    Uses the parent node reference for all ROS2 primitives.

    Lifecycle:
      1. scan()    — start SCANNING forward, controller inactive
      2. (auto)    — first valid detection activates controller
      3. (auto)    — ex within ex_tolerance → STOPPED
      4. resume()  — called by parent after harvest → DEPARTING
      5. (auto)    — departure clearance met → SCANNING, controller reset
    """

    def __init__(self, node: Node, config: DriveConfig) -> None:
        self._node   = node
        self._logger = RcutilsLogger(self.__class__.__name__)
        self._ns     = self._node.get_namespace().rstrip('/')

        # --- Config ---
        self._base_frame          = config.base_frame
        self._ex_tolerance        = config.ex_tolerance
        self._ey_sign             = config.ey_sign
        self._k_phi               = config.k_phi
        self._k_lateral           = config.k_lateral
        self._v_linear_min        = config.v_linear_min
        self._v_linear_max        = config.v_linear_max
        self._v_angular_max       = config.v_angular_max
        self._departure_clearance = config.departure_clearance

        # Convert no_detection_distance to timeout (seconds)
        self._no_detection_timeout: float = (
            config.no_detection_distance / max(config.v_linear_max, 0.01)
        )

        # --- State ---
        self._status: DriveStatus = DriveStatus.IDLE

        # --- Controller state ---
        self._controller_active:   bool        = False
        self._lookahead_distance:  float | None = None   # set on first detection
        self._last_ex:             float | None = None   # overshoot detection

        # --- Departure state ---
        self._departure_start_time: float | None = None

        # --- Last computed velocity (republished between detections) ---
        self._current_linear_x:  float = 0.0
        self._current_angular_z: float = 0.0

        # --- ROS2 interfaces ---
        self._cmd_vel_pub = self._node.create_publisher(
            TwistStamped,
            f'{self._ns}/cmd_vel',
            10,
        )

        self._detection_sub = self._node.create_subscription(
            ImageDetectionPose,
            f'{self._ns}/{config.detection_topic}',
            self._detection_callback,
            qos_profile_sensor_data,
        )

        self._cmd_vel_timer = self._node.create_timer(
            1.0 / config.cmd_vel_rate,
            self._cmd_vel_repeat_callback,
        )

        self._no_detection_timer = None

        self._logger.info(
            f'DriveClient v2 initialized | '
            f'v_linear=[{self._v_linear_min}, {self._v_linear_max}]m/s | '
            f'v_angular_max={self._v_angular_max}rad/s | '
            f'k_phi={self._k_phi} k_lateral={self._k_lateral} | '
            f'ex_tolerance={self._ex_tolerance}m | '
            f'ey_sign={self._ey_sign:+.1f} | '
            f'departure_clearance={self._departure_clearance}m | '
            f'no_detection_timeout={self._no_detection_timeout:.1f}s'
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def scan(self) -> None:
        """
        Start forward scanning drive.

        Controller is inactive until the first valid detection arrives.
        Resets all controller state from the previous bush.
        """
        self._logger.info(
            f'SCANNING — moving forward at v_max={self._v_linear_max}m/s | '
            f'controller inactive until first detection'
        )
        self._status = DriveStatus.SCANNING
        self._reset_controller()
        self._current_linear_x  = self._v_linear_max
        self._current_angular_z = 0.0
        self._publish_cmd_vel(self._v_linear_max, 0.0)
        self._reset_no_detection_timer()

    def resume(self) -> None:
        """
        Called by the parent node after harvest activity completes.

        Transitions to DEPARTING — detections are ignored until the robot
        has moved departure_clearance metres past the bush. Controller is
        reset at the DEPARTING → SCANNING transition, not here.
        """
        if self._status != DriveStatus.STOPPED:
            self._logger.warning(
                f'resume() called in unexpected state: {self._status.name} — ignoring'
            )
            return

        self._departure_start_time = self._node.get_clock().now().nanoseconds / 1e9
        departure_duration = self._departure_clearance / max(self._v_linear_max, 0.01)

        self._logger.info(
            f'DEPARTING — moving past bush | '
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
        self._status            = DriveStatus.CANCELED
        self._current_linear_x  = 0.0
        self._current_angular_z = 0.0
        self._cancel_no_detection_timer()
        self._reset_controller()
        self._publish_cmd_vel(0.0, 0.0)
        self._logger.info('Drive CANCELED')

    def reset(self) -> None:
        """Reset to IDLE — clears all state."""
        self._status = DriveStatus.IDLE
        self._reset_controller()
        self._current_linear_x  = 0.0
        self._current_angular_z = 0.0
        self._cancel_no_detection_timer()
        self._departure_start_time = None
        self._logger.info('DriveClient reset to IDLE')

    def get_status(self) -> DriveStatus:
        """Return the current DriveStatus."""
        return self._status

    def is_active(self) -> bool:
        """Return True if the robot is currently moving."""
        return self._status in (DriveStatus.SCANNING, DriveStatus.DEPARTING)

    # =========================================================================
    # Detection callback
    # =========================================================================

    def _detection_callback(self, msg: ImageDetectionPose) -> None:
        """
        Fires on every ImageDetectionPose message.

        SCANNING:
          Invalid detection → no-op (no_detection_timer keeps running).
          First valid detection → activate controller, set lookahead_distance.
          Subsequent valid detections → compute velocity, check stop/overshoot.

        DEPARTING:
          All detections ignored — check departure clearance only.

        All other states: no-op.
        """
        self._logger.debug(
            f'Detection | valid={msg.detection_valid} | '
            f'center=({msg.center.x:.3f}, {msg.center.y:.3f}, {msg.center.z:.3f}) | '
            f'status={self._status.name}'
        )

        if self._status == DriveStatus.DEPARTING:
            self._check_departure_clearance()
            return

        if self._status != DriveStatus.SCANNING:
            return

        if not msg.detection_valid:
            return

        # Valid detection in SCANNING state
        self._reset_no_detection_timer()

        ex = msg.center.x
        ey = msg.center.y * self._ey_sign

        # First detection of this bush — activate controller
        if not self._controller_active:
            self._lookahead_distance = abs(ex)
            self._controller_active  = True
            self._last_ex            = ex
            self._logger.info(
                f'Controller activated | '
                f'lookahead_distance={self._lookahead_distance:.3f}m | '
                f'ex={ex:+.3f}m ey={ey:+.3f}m'
            )

        # Overshoot detection: ex crossed from negative to positive
        if (self._last_ex is not None and
                self._last_ex < 0 and ex > 0):
            self._logger.warning(
                f'Overshoot detected | '
                f'ex crossed zero: last={self._last_ex:+.3f}m current={ex:+.3f}m — '
                f'hard stop'
            )
            self._hard_stop()
            return

        self._last_ex = ex

        # Stop condition
        if abs(ex) <= self._ex_tolerance:
            self._logger.info(
                f'Bush level with arm | '
                f'ex={ex:+.4f}m within tolerance={self._ex_tolerance}m — STOPPED'
            )
            self._hard_stop()
            return

        # Compute and apply controller output
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
        Bearing + lateral controller.

        phi = atan2(ey, abs(ex))
            Bearing angle to bush centroid from robot forward axis.
            Corrects both heading (keeps robot parallel to row) and
            bearing (guides robot to the level point).

        angular_z = clamp(k_phi * phi + k_lateral * ey, -v_angular_max, +v_angular_max)
            Combined heading and lateral correction.
            Zeroed when ex >= 0 — no steering toward a bush behind the robot.

        linear_x = v_linear_min + (v_linear_max - v_linear_min) * (|ex| / lookahead_distance)
            Linear speed scaling: v_linear_max at first detection, reduces
            linearly to v_linear_min as ex approaches zero.
            Clamped to [v_linear_min, v_linear_max].
        """
        # Bearing angle to bush
        phi = math.atan2(ey, abs(ex))

        # Angular correction — gated on ex sign
        if ex >= 0.0:
            # Bush is level or passed — no lateral correction
            angular_z = 0.0
        else:
            raw_angular = self._k_phi * phi + self._k_lateral * ey
            angular_z   = max(-self._v_angular_max,
                              min(self._v_angular_max, raw_angular))

        # Linear speed — proportional to remaining forward distance
        lookahead = self._lookahead_distance if self._lookahead_distance else abs(ex)
        if lookahead > 0.0:
            speed_fraction = abs(ex) / lookahead
        else:
            speed_fraction = 0.0

        linear_x = self._v_linear_min + (
            (self._v_linear_max - self._v_linear_min) * speed_fraction
        )
        linear_x = max(self._v_linear_min, min(self._v_linear_max, linear_x))

        self._logger.debug(
            f'_compute_velocity | '
            f'phi={math.degrees(phi):+.2f}deg | '
            f'ex={ex:+.4f}m ey={ey:+.4f}m | '
            f'linear_x={linear_x:.4f}m/s angular_z={angular_z:+.4f}rad/s'
        )

        return linear_x, angular_z

    # =========================================================================
    # Departure clearance
    # =========================================================================

    def _check_departure_clearance(self) -> None:
        """
        Called from _detection_callback during DEPARTING state.

        Computes elapsed time since resume() was called. When elapsed
        time exceeds departure_clearance / v_linear_max, transitions back
        to SCANNING and resets the controller for the next bush.
        """
        if self._departure_start_time is None:
            return

        elapsed            = self._node.get_clock().now().nanoseconds / 1e9 - self._departure_start_time
        departure_duration = self._departure_clearance / max(self._v_linear_max, 0.01)

        self._logger.debug(
            f'Departure check | elapsed={elapsed:.2f}s / {departure_duration:.2f}s'
        )

        if elapsed >= departure_duration:
            self._logger.info(
                f'Departure clearance met ({elapsed:.2f}s) — resuming SCANNING | '
                f'controller reset for next bush'
            )
            self._departure_start_time = None
            self.scan()

    # =========================================================================
    # No-detection timeout
    # =========================================================================

    def _reset_no_detection_timer(self) -> None:
        """Restart the no-detection watchdog timer."""
        if self._no_detection_timer is None:
            self._no_detection_timer = self._node.create_timer(
                self._no_detection_timeout,
                self._on_no_detection_timeout,
            )
        else:
            self._no_detection_timer.reset()

    def _cancel_no_detection_timer(self) -> None:
        """Cancel the no-detection timer if running."""
        if self._no_detection_timer is not None:
            self._no_detection_timer.cancel()

    def _on_no_detection_timeout(self) -> None:
        """
        Fires when no valid detection received within no_detection_timeout.

        Stops the robot and transitions to STOPPED. The parent node interprets
        STOPPED after a timeout as row-end and calls cancel() to move on.
        """
        self._cancel_no_detection_timer()
        if self._status != DriveStatus.SCANNING:
            return
        self._logger.info(
            f'No detection for {self._no_detection_timeout:.1f}s — '
            f'row end assumed, stopping'
        )
        self._hard_stop()

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _hard_stop(self) -> None:
        """Publish zero velocity and transition to STOPPED."""
        self._status            = DriveStatus.STOPPED
        self._current_linear_x  = 0.0
        self._current_angular_z = 0.0
        self._cancel_no_detection_timer()
        self._publish_cmd_vel(0.0, 0.0)

    def _reset_controller(self) -> None:
        """
        Reset all per-bush controller state.

        Called on scan() (new row start) and at DEPARTING → SCANNING
        transition (next bush on the same row).
        """
        self._controller_active  = False
        self._lookahead_distance = None
        self._last_ex            = None

    def _cmd_vel_repeat_callback(self) -> None:
        """
        Republish last computed velocity at cmd_vel_rate.

        Prevents the robot from stopping due to cmd_vel timeout between
        detection messages (~5Hz). Only publishes while actively moving.
        """
        if self.is_active():
            self._publish_cmd_vel(self._current_linear_x, self._current_angular_z)

    def _publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        """Wrap velocity in a TwistStamped and publish to cmd_vel."""
        msg                     = TwistStamped()
        msg.header.stamp        = self._node.get_clock().now().to_msg()
        msg.header.frame_id     = self._base_frame
        msg.twist.linear.x      = linear_x
        msg.twist.angular.z     = angular_z
        self._cmd_vel_pub.publish(msg)