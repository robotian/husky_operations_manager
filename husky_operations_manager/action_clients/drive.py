"""
drive.py

Camera-guided drive controller for crop-row harvesting.

Owned by a parent harvesting node which provides the ROS2 node handle
and drives the lifecycle via the public API (scan / resume / cancel / reset).

-------------------------------------------------------------------------------
State Machine
-------------------------------------------------------------------------------

  IDLE
    └─► scan() → SCANNING  (forward drive + yaw correction)
          ├─► detection within ex_tolerance → STOPPED  (parent harvests)
          │     └─► resume() → DEPARTING  (detections ignored until clearance)
          │           └─► clearance elapsed → SCANNING  (next plant)
          └─► no detection for no_detection_timeout → STOPPED  (row end)

  Any active state: cancel() → CANCELED | reset() → IDLE
  CORRECTING: transient sub-state of SCANNING (detection valid, ex out of tolerance)

-------------------------------------------------------------------------------
Alignment Convention  (implemented in _evaluate_alignment)
-------------------------------------------------------------------------------

  ex = center.x              forward/backward error (m)
  ey = center.y * ey_sign    lateral error after sign correction (m)

  Platform default (ey_sign = -1.0, bush always to the RIGHT):
    ey > 0  →  too far   →  steer right (+angular_z)
    ey < 0  →  too close →  steer left  (-angular_z)

  Stop condition: |ex| <= ex_tolerance  (yaw correction runs independently)
  Yaw correction: angular_z = clamp(ey * kp, -v_angular, +v_angular)

  NOTE: Yaw correction is under active tuning — see _evaluate_alignment.

-------------------------------------------------------------------------------
Notes
-------------------------------------------------------------------------------

  - No-detection timer is created once and reused via reset() (no allocation churn).
  - Detection subscriber uses qos_profile_sensor_data (BEST_EFFORT).
  - DriveConfig is populated by the parent node from YAML parameters.
"""

from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

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
    no_detection_distance: float  # distance (m) without detection before stopping

    # Velocity
    base_frame: str  # cmd_vel header frame_id
    cmd_vel_rate: float  # cmd_vel repeat rate (Hz) — republish velocity while active
    v_linear: float  # forward/reverse speed (m/s)
    v_angular: float  # turning speed (rad/s)
    kp: float  # proportional gain for yaw correction

    # Alignment
    ex_tolerance: float  # forward/backward stop tolerance (m)
    ey_tolerance: float  # lateral stop tolerance (m)
    ey_sign: float  # +1.0 or -1.0, ey is negative when bush is to the right

    # Departure
    departure_clearance: float  # distance (m) past bush before re-enabling detection


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
        self._kp: float = config.kp

        # --- Alignment config ---
        self._ex_tolerance: float = config.ex_tolerance
        self._ey_tolerance: float = config.ey_tolerance
        self._ey_sign: float = config.ey_sign

        # --- Departure config ---
        self._departure_clearance: float = config.departure_clearance
        self._departure_start_time: float | None = None

        # --- No detection timeout ---
        # Convert distance to time: timeout = distance / v_linear
        self._no_detection_timeout: float = config.no_detection_distance / max(config.v_linear, 0.01)
        self._no_detection_timer = None

        # --- Status ---
        self._status: DriveStatus = DriveStatus.IDLE

        # --- cmd_vel repeat state ---
        self._current_linear_x: float = 0.0
        self._current_angular_z: float = 0.0

        # --- Detection subscription ---
        self._detection_sub = self.node.create_subscription(
            ImageDetectionPose,
            f'{self.namespace}/{config.detection_topic}',
            self._detection_callback,
            qos_profile_sensor_data,
        )

        # --- cmd_vel ---
        self._cmd_vel_pub = self.node.create_publisher(TwistStamped, f'{self.namespace}/cmd_vel', 10)
        self._cmd_vel_timer = self.node.create_timer(1.0 / config.cmd_vel_rate, self._cmd_vel_repeat_callback)

        self.logger.info(
            f'DriveClient initialized | '
            f'linear={self.v_linear} angular={self.v_angular} | '
            f'ex_tolerance={self._ex_tolerance}m '
            f'ey_tolerance={self._ey_tolerance}m '
            f'ey_sign={self._ey_sign:+.1f} | '
            f'departure_clearance={self._departure_clearance}m | '
            f'no_detection_distance={config.no_detection_distance}m '
            f'(timeout={self._no_detection_timeout:.1f}s)'
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
        self._reset_no_detection_timer()

    def resume(self) -> None:
        """Called by the parent node after harvest activity completes."""
        if self._status != DriveStatus.STOPPED:
            self.logger.warning(f'resume() called in unexpected state: {self._status.name} — ignoring')
            return

        self._departure_start_time = self.node.get_clock().now().nanoseconds / 1e9
        departure_duration = self._departure_clearance / max(self.v_linear, 0.01)
        self.logger.info(
            f'DEPARTING — moving past bush | clearance={self._departure_clearance}m | '
            f'estimated_duration={departure_duration:.1f}s'
        )

        self._status = DriveStatus.DEPARTING
        self._current_linear_x = self.v_linear
        self._current_angular_z = 0.0
        self._publish_cmd_vel(linear_x=self.v_linear, angular_z=0.0)

    def cancel(self) -> None:
        """Cancel an active drive and publish zero velocity."""
        if self._status in (DriveStatus.IDLE, DriveStatus.CANCELED):
            return
        self._status = DriveStatus.CANCELED
        self._departure_start_time = None
        self._current_linear_x = 0.0
        self._current_angular_z = 0.0
        self._cancel_no_detection_timer()
        self._publish_cmd_vel(linear_x=0.0, angular_z=0.0)
        self.logger.info('Drive CANCELED')

    def reset(self) -> None:
        """Reset to IDLE."""
        self._status = DriveStatus.IDLE
        self._departure_start_time = None
        self._current_linear_x = 0.0
        self._current_angular_z = 0.0
        self._cancel_no_detection_timer()
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
          Valid detection → compute (ex, ey) from msg.center.
          Within tolerance → STOPPED, zero velocity.
          Outside tolerance → CORRECTING, keep driving.

        DEPARTING:
          Ignore detections until departure_clearance time is met.
          Once met → back to SCANNING.

        All other states: no-op.
        """
        self.logger.debug(f'Message received. Detection Valid: {msg.detection_valid} at'
         f'Position: x = {msg.center.x}, y = {msg.center.y},z = {msg.center.y}')

        if not msg.detection_valid:
            return

        self.logger.debug(f'Detected Pose: {msg.center}')

        if self._status == DriveStatus.DEPARTING:
            self._check_departure_clearance()
            return

        if self._status not in (DriveStatus.SCANNING, DriveStatus.CORRECTING):
            return

        self._reset_no_detection_timer()
        ex = msg.center.x
        ey = msg.center.y * self._ey_sign
        self._evaluate_alignment(ex, ey)

    # ------------------------------------------------------------------
    # Departure clearance
    # ------------------------------------------------------------------

    def _check_departure_clearance(self) -> None:
        """
        During DEPARTING, check if enough time has elapsed to clear the bush.
        Clearance time = departure_clearance / v_linear.
        """
        if self._departure_start_time is None:
            return

        elapsed = self.node.get_clock().now().nanoseconds / 1e9 - self._departure_start_time
        departure_duration = self._departure_clearance / max(self.v_linear, 0.01)

        self.logger.debug(f'Departure check | elapsed={elapsed:.2f}s duration={departure_duration:.2f}s')

        if elapsed >= departure_duration:
            self.logger.info(f'Departure clearance met ({elapsed:.2f}s) — resuming SCANNING')
            self._departure_start_time = None
            self.scan()

    # ------------------------------------------------------------------
    # No detection timeout
    # ------------------------------------------------------------------

    def _reset_no_detection_timer(self) -> None:
        """Restart the no-detection watchdog timer."""
        if self._no_detection_timer is None:
            self._no_detection_timer = self.node.create_timer(
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
        Stops the robot and transitions to STOPPED.
        """
        self._cancel_no_detection_timer()
        if self._status not in (DriveStatus.SCANNING, DriveStatus.CORRECTING):
            return
        self.logger.info(f'No detection for {self._no_detection_timeout:.1f}s — row end assumed, stopping')
        self._status = DriveStatus.STOPPED
        self._current_linear_x = 0.0
        self._current_angular_z = 0.0
        self._publish_cmd_vel(linear_x=0.0, angular_z=0.0)

    # ------------------------------------------------------------------
    # Alignment error
    # ------------------------------------------------------------------

    def _evaluate_alignment(self, ex: float, ey: float) -> None:
        """
        Evaluate alignment error, apply yaw correction, and transition state.

        ex: forward/backward — stops the robot when within ex_tolerance
        ey: lateral distance — bush is always to the right
            ey > 0 → too far   → move closer (positive angular_z)
            ey < 0 → too close → move away   (negative angular_z)

        Yaw correction: angular_z = clamp(ey * kp, -v_angular, +v_angular)
        Applied when |ey| > ey_tolerance, zero when laterally aligned.

        Robot stops only when BOTH ex and ey are within tolerance simultaneously.
        """
        self.logger.debug(f'Pose | ex={ex:+.4f}m  ey={ey:+.4f}m')

        # --- Y axis — yaw correction (always runs) ---
        if abs(ey) <= self._ey_tolerance:
            angular_z = 0.0
            self.logger.info(f'Y ALIGNED | ey={ey:+.4f}m within tolerance={self._ey_tolerance:.4f}m')
        else:
            angular_z = max(-self.v_angular, min(self.v_angular, ey * self._kp))
            direction = 'TOO FAR (move closer)' if ey > 0 else 'TOO CLOSE (move away)'
            self.logger.info(f'{direction} | ey={ey:+.4f}m angular_z={angular_z:+.4f}rad/s')
 
        # --- X axis — stop decision ---
        if abs(ex) <= self._ex_tolerance:
            self.logger.info(f'X ALIGNED | ex={ex:+.4f}m within tolerance={self._ex_tolerance:.4f}m — STOPPED')
            self._status            = DriveStatus.STOPPED
            self._current_linear_x  = 0.0
            self._current_angular_z = 0.0
            self._publish_cmd_vel(linear_x=0.0, angular_z=0.0)
        else:
            self._status            = DriveStatus.CORRECTING
            self._current_linear_x  = self.v_linear
            self._current_angular_z = angular_z
            self.logger.debug(f'Correcting | ex={ex:+.4f}m ey={ey:+.4f}m angular_z={angular_z:+.4f}rad/s')
            self._publish_cmd_vel(linear_x=self._current_linear_x, angular_z=angular_z)


        # # ex: forward/backward alignment — stops the robot when within tolerance
        # # TODO: Use ey to drive angular_z correction once ex-based stopping
        # # --- X axis (forward/backward) ---
        # if abs(ex) <= self._ex_tolerance:
        #     self.logger.info(f'X ALIGNED | ex={ex:+.4f}m within tolerance={self._ex_tolerance:.4f}m')
        #     self._status = DriveStatus.STOPPED
        #     self._current_linear_x = 0.0
        #     self._current_angular_z = 0.0
        #     self._publish_cmd_vel(linear_x=0.0, angular_z=0.0)
        # else:
        #     self.logger.debug(f'Approaching | ex={ex:+.4f}m exceeds tolerance={self._ex_tolerance:.4f}m')
        #     self._status = DriveStatus.CORRECTING

        # # --- Y axis (lateral harvesting distance) ---
        # # TODO: drive angular_z correction based on ey once ex stopping is verified
        # if abs(ey) <= self._ey_tolerance:
        #     self.logger.info(
        #         f'Y ALIGNED | ey={ey:+.4f}m within tolerance={self._ey_tolerance:.4f}m — at correct harvesting distance'
        #     )
        # elif ey > 0:
        #     self.logger.info(f'TOO FAR   | ey={ey:+.4f}m — robot too far from bush, move closer (right)')
        # else:
        #     self.logger.info(f'TOO CLOSE | ey={ey:+.4f}m — robot too close to bush, move away (left)')

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cmd_vel_repeat_callback(self) -> None:
        """Republish current velocity at cmd_vel_rate while active."""
        if self.is_active():
            self._publish_cmd_vel(self._current_linear_x, self._current_angular_z)

    def _publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        """Wrap a Twist in a stamped message and publish to cmd_vel."""
        self._current_linear_x = linear_x
        self._current_angular_z = angular_z

        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        self._cmd_vel_pub.publish(msg)
