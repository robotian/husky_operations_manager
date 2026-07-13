"""
drive.py

Camera-guided PD drive controller for lavender row harvesting.

Ported from the reference simulation (`DriveClientSimulatorV1.ipynb`,
`ROS2DriveClientNode`) — a target-pose + PD position/heading controller,
as opposed to the direct ex/ey proportional controller in drive_v2.py.

Owned by a parent harvesting node which provides the ROS2 node handle
and drives the lifecycle via the public API (scan / resume).

-------------------------------------------------------------------------------
Coordinate Frames
-------------------------------------------------------------------------------

  camera_1_detections (= camera_1_color_optical_frame orientation):
    +X → backward  (bush ahead → negative center.x)
    +Y → left
    +Z → down

  base_link:
    +X → forward
    +Y → left
    +Z → up

  Camera is mounted upside-down and backward-facing (~179° roll + ~179° yaw).

-------------------------------------------------------------------------------
Target Pose Calculation
-------------------------------------------------------------------------------

  On every valid detection, the bush position (msg.center.x, msg.center.y —
  camera-frame local offsets) is rotated into the odom frame using the
  camera mount offset (cam_tx, cam_ty) and the robot's current odom pose,
  producing an absolute (target_x, target_y, target_yaw) — mirrors the sim's
  `camera_frame_to_reference_frame()` / `set_target_pose()`.

  This is a manual 2-D rigid-body transform, not a TF lookup — matches the
  sim and drive_v2.py, both of which avoid TF for this calculation.

  NOTE: the sim declares an ARM_TX_OFFSET (0.214 m — arm_0_base_link.tx
  relative to base_link) documented as "the arm-stop goal is placed past
  the bush along the row", but the sim code never actually applies it —
  the target pose is the bush position itself. Reproduced faithfully here
  (arm_tx_offset is carried in DriveConfig but unused), not fixed.

-------------------------------------------------------------------------------
Controller
-------------------------------------------------------------------------------

  Runs on a fixed-rate control timer (1 / config.cmd_vel_rate), consuming
  the latest cached odom + detection each tick — mirrors the sim's
  `update_loop_10hz()` cadence, rather than being driven off the detection
  callback directly (as in drive_v2.py).

  Phase 1 (distance_error > ex_tolerance): PD drive-to-point.
    heading_error = atan2(dy, dx) - theta
    Reverse motion allowed if distance_error < backward_distance_threshold
    and |heading_error| > pi/2.
    v = K_v_p * distance_error + K_v_d * d(distance_error)/dt
    omega = K_omega_p * heading_error + K_omega_d * d(heading_error)/dt

  Phase 2 (position reached, final heading not yet within ang_tol): PD
  final-orientation correction.
    omega = K_beta_p * final_heading_error + K_beta_d * d(...)/dt

  Both phases clamped to [v_linear_min/max?, v_linear_max] / [-omega_max,
  omega_max] then accel-limited to ±a_max*dt / ±alpha_max*dt around the
  previous command.

-------------------------------------------------------------------------------
State Machine
-------------------------------------------------------------------------------

  IDLE
    └─► scan() → SCANNING  (forward at v_linear_min, no target yet)
          ├─► valid detection → set target pose → CONTROLLING
          └─► no detection for no_detection_timeout → STOPPED  (row end)

  CONTROLLING
    ├─► valid detection → update target pose (re-locks every tick)
    ├─► PD goal reached (position + final heading) → STOPPED
    └─► (no explicit timeout — mirrors sim, which has none in this state)

  STOPPED
    └─► resume() → DEPARTING  (drive at v_linear_min, no detection handling)
          └─► departure_clearance elapsed → SCANNING  (via scan())

-------------------------------------------------------------------------------
Notes
-------------------------------------------------------------------------------

  - Detection subscription uses the real ImageDetectionPose message
    (status_interfaces) — msg.detection_valid, msg.center.x, msg.center.y.
    msg.center.x/y are the raw camera-frame bush offsets consumed directly
    by the target-pose transform, NOT the ex/ey pre-computed error signal
    used in drive_v2.py.
  - ey_sign (present in the sim and drive_v2.py) is intentionally omitted:
    it is computed in the sim but never consumed by the PD math — the
    camera→odom transform already accounts for heading via the robot's
    actual odom yaw. Dropping it changes nothing behaviorally.
  - Odom and detection subscriptions use qos_profile_sensor_data
    (BEST_EFFORT), matching drive_v2.py's convention for sensor topics.
  - cmd_vel is published as TwistStamped every control tick — this doubles
    as the "republish between detections" behavior drive_v2.py implements
    with a separate timer, since here the control timer itself is the
    sole driver of both the state machine and the publish rate.
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
    Component class for camera-guided PD target-pose drive control.

    Instantiated and owned by the parent harvesting node.
    Uses the parent node reference for all ROS2 primitives.

    Lifecycle:
      1. scan()    — start SCANNING forward, no target locked
      2. (auto)    — first valid detection locks a target pose → CONTROLLING
      3. (auto)    — PD goal reached (position + final heading) → STOPPED
      4. resume()  — called by parent after harvest → DEPARTING
      5. (auto)    — departure clearance met → SCANNING (via scan())
    """

    def __init__(self, node: Node, config: DriveConfig) -> None:
        self._node   = node
        self._logger = RcutilsLogger(self.__class__.__name__)
        self._ns     = self._node.get_namespace().rstrip('/')

        # --- Config ---
        self._base_frame          = config.base_frame
        self._stop_threshold      = config.ex_tolerance
        self._ang_tol             = config.ang_tol
        self._v_linear_min        = config.v_linear_min
        self._v_linear_max        = config.v_linear_max
        self._omega_max           = config.v_angular_max
        self._k_v_p               = config.k_v_p
        self._k_v_d               = config.k_v_d
        self._k_omega_p           = config.k_omega_p
        self._k_omega_d           = config.k_omega_d
        self._k_beta_p            = config.k_beta_p
        self._k_beta_d            = config.k_beta_d
        self._a_max                = config.a_max
        self._alpha_max            = config.alpha_max
        self._backward_dist_thresh = config.backward_distance_threshold
        self._cam_pose: tuple[float, float, float] = (
            config.cam_tx,
            config.cam_ty,
            math.pi,  # camera faces backward
        )
        self._bushrow_theta       = config.bushrow_theta

        self._dt = 1.0 / config.cmd_vel_rate

        # Convert distance-based config fields to time, matching the
        # derivation already established for DriveConfig by drive_v2.py.
        self._no_detection_timeout: float = (
            config.no_detection_distance / max(config.v_linear_max, 0.01)
        )
        self._departure_duration: float = (
            config.departure_clearance / max(config.v_linear_max, 0.01)
        )

        # --- Odom state ---
        self._odom_received: bool = False
        self._current_pose: list[float] = [0.0, 0.0, 0.0]     # x, y, yaw
        self._current_velocity: list[float] = [0.0, 0.0]      # v, omega

        # --- Latest cached detection ---
        # (detected, bush_x, bush_y) in camera frame — consumed once per
        # control tick, same as the sim's camera_data input.
        self._latest_detection: tuple[bool, float, float] = (False, 0.0, 0.0)

        # --- Drive state ---
        self._status: DriveStatus = DriveStatus.IDLE
        self._state_timer: float = 0.0

        # --- Target pose (odom frame) ---
        self._target_pose: tuple[float, float, float] | None = None

        # --- PD state memory ---
        self._v_prev: float = 0.0
        self._omega_prev: float = 0.0
        self._prev_distance_error: float | None = None
        self._prev_heading_error: float | None = None
        self._prev_final_heading_error: float | None = None

        self._cmd_linear_x: float = 0.0
        self._cmd_angular_z: float = 0.0

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

        # --- Control timer — sole driver of the state machine + cmd_vel ---
        self._control_timer = self._node.create_timer(
            self._dt,
            self._control_callback,
        )

        self._logger.info(
            f'DriveClient (PD) initialized | '
            f'v_linear=[{self._v_linear_min}, {self._v_linear_max}]m/s | '
            f'omega_max={self._omega_max}rad/s | '
            f'stop_threshold={self._stop_threshold}m | '
            f'ang_tol={self._ang_tol}rad | '
            f'control_rate={config.cmd_vel_rate}Hz | '
            f'no_detection_timeout={self._no_detection_timeout:.1f}s | '
            f'departure_duration={self._departure_duration:.1f}s'
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def scan(self) -> None:
        """
        Start forward scanning drive. No target is locked until the first
        valid detection arrives on the control timer.
        """
        if not self._odom_received:
            self._logger.error(
                'scan() called but odom not yet received — aborting scan'
            )
            return

        self._status = DriveStatus.SCANNING
        self._state_timer = 0.0
        self._target_pose = None
        self._reset_pd_state()

        self._logger.info('SCANNING — controller inactive until first detection')

    def resume(self) -> None:
        """
        Called by the parent node after harvest activity completes.

        Transitions STOPPED → DEPARTING. Departure duration is derived
        from config.departure_clearance (m) / v_linear_max.
        """
        if self._status != DriveStatus.STOPPED:
            self._logger.warning(
                f'resume() called in unexpected state: {self._status.name} — ignoring'
            )
            return

        self._status = DriveStatus.DEPARTING
        self._state_timer = 0.0
        self._logger.info(
            f'DEPARTING — clearing bush area | duration={self._departure_duration:.1f}s'
        )

    def get_status(self) -> DriveStatus:
        """Return the current DriveStatus."""
        return self._status

    def is_active(self) -> bool:
        """Return True if the robot is currently moving."""
        return self._status in (
            DriveStatus.SCANNING,
            DriveStatus.CONTROLLING,
            DriveStatus.DEPARTING,
        )

    def is_ready(self) -> bool:
        """Return True if odom has been received and scan() can be safely called."""
        return self._odom_received

    # =========================================================================
    # Odom callback
    # =========================================================================

    def _odom_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._current_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self._current_velocity = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        self._odom_received = True

    # =========================================================================
    # Detection callback — caches only, state machine runs on control timer
    # =========================================================================

    def _detection_callback(self, msg: ImageDetectionPose) -> None:
        self._latest_detection = (
            bool(msg.detection_valid),
            msg.center.x,
            msg.center.y,
        )

    # =========================================================================
    # Control timer — sole driver of the state machine + cmd_vel
    # =========================================================================

    def _control_callback(self) -> None:
        """Fixed-rate tick — mirrors the sim's update_loop_10hz()."""
        if not self._odom_received:
            return

        self._state_timer += self._dt
        curr_yaw = self._current_pose[2]
        img_detected, bush_x, bush_y = self._latest_detection

        if self._status == DriveStatus.IDLE:
            self._cmd_linear_x = 0.0
            self._cmd_angular_z = 0.0

        elif self._status == DriveStatus.SCANNING:
            self._cmd_linear_x = self._v_linear_min
            self._cmd_angular_z = 0.0

            if self._state_timer >= self._no_detection_timeout:
                self._logger.warning(
                    f'No-detection timeout ({self._no_detection_timeout:.1f}s) '
                    '— row end or gap. Stopping.'
                )
                self._hard_stop()

            elif img_detected:
                self._set_target_pose(bush_x, bush_y)
                self._status = DriveStatus.CONTROLLING
                self._state_timer = 0.0
                tx, ty, tyaw = self._target_pose
                self._logger.info(
                    f'Target locked — bush cam=({bush_x:.3f}, {bush_y:.3f}) '
                    f'target odom=({tx:.3f}, {ty:.3f}, {math.degrees(tyaw):.1f}deg)'
                )

        elif self._status == DriveStatus.CONTROLLING:
            if img_detected:
                self._set_target_pose(bush_x, bush_y)

            cmd_v, cmd_w, goal_reached = self._compute_commands_pd()
            self._cmd_linear_x = cmd_v
            self._cmd_angular_z = cmd_w

            if goal_reached:
                self._logger.info('Goal reached — STOPPED (awaiting resume())')
                self._hard_stop()

        elif self._status == DriveStatus.STOPPED:
            self._cmd_linear_x = 0.0
            self._cmd_angular_z = 0.0

        elif self._status == DriveStatus.DEPARTING:
            self._cmd_linear_x = self._v_linear_min
            self._cmd_angular_z = 0.0

            if self._state_timer >= self._departure_duration:
                self._logger.info('Departure complete — re-scanning')
                self.scan()

        self._publish_cmd_vel(self._cmd_linear_x, self._cmd_angular_z)

    # =========================================================================
    # Target pose (camera frame → odom frame)
    # =========================================================================

    def _camera_frame_to_odom(self, local_x: float, local_y: float) -> tuple[float, float]:
        """Manual 2-D rigid-body transform — matches the sim, no TF lookup."""
        cx, cy, cyaw = self._cam_pose
        base_x = cx + local_x * math.cos(cyaw) + local_y * math.sin(cyaw)
        base_y = cy + local_x * math.sin(cyaw) - local_y * math.cos(cyaw)

        rx, ry, ryaw = self._current_pose
        odom_x = rx + base_x * math.cos(ryaw) - base_y * math.sin(ryaw)
        odom_y = ry + base_x * math.sin(ryaw) + base_y * math.cos(ryaw)
        return odom_x, odom_y

    def _set_target_pose(self, bush_x: float, bush_y: float) -> None:
        """
        Convert camera-frame bush offsets into an odom-frame target pose.

        NOTE: arm_tx_offset is intentionally NOT applied — matches the
        sim, which declares but never uses it.
        """
        cam_x, cam_y = self._camera_frame_to_odom(bush_x, bush_y)
        cam_yaw = self._bushrow_theta + self._cam_pose[2]

        target_x = cam_x + self._cam_pose[0] * math.cos(cam_yaw) - self._cam_pose[1] * math.sin(cam_yaw)
        target_y = cam_y + self._cam_pose[0] * math.sin(cam_yaw) + self._cam_pose[1] * math.cos(cam_yaw)
        target_yaw = cam_yaw - self._cam_pose[2]

        self._target_pose = (target_x, target_y, target_yaw)

    # =========================================================================
    # PD controller
    # =========================================================================

    def _compute_commands_pd(self) -> tuple[float, float, bool]:
        """PD position + heading controller. Returns (v, omega, goal_reached)."""
        x, y, theta = self._current_pose
        x_d, y_d, theta_d = self._target_pose if self._target_pose is not None else (0.0, 0.0, 0.0)

        self._v_prev = self._current_velocity[0]
        self._omega_prev = self._current_velocity[1]

        dx = x_d - x
        dy = y_d - y
        distance_error = math.hypot(dx, dy)

        v_raw = 0.0
        omega_raw = 0.0
        reached_goal = False

        if distance_error > self._stop_threshold:
            target_heading = math.atan2(dy, dx)
            heading_error = self._normalize_angle(target_heading - theta)

            allow_backward = distance_error < self._backward_dist_thresh
            if allow_backward and abs(heading_error) > math.pi / 2.0:
                heading_error = self._normalize_angle(heading_error - math.pi)
                direction = -1.0
            else:
                direction = 1.0

            if self._prev_distance_error is None:
                self._prev_distance_error = distance_error
                self._prev_heading_error = heading_error

            d_distance = (distance_error - self._prev_distance_error) / self._dt
            d_heading = self._normalize_angle(heading_error - self._prev_heading_error) / self._dt

            pd_v = self._k_v_p * distance_error + self._k_v_d * d_distance
            v_raw = direction * pd_v * math.cos(heading_error)
            if direction == 1.0 and v_raw < 0:
                v_raw = 0.0

            omega_raw = self._k_omega_p * heading_error + self._k_omega_d * d_heading

            self._prev_distance_error = distance_error
            self._prev_heading_error = heading_error
            self._prev_final_heading_error = None

        else:
            final_heading_error = self._normalize_angle(theta_d - theta)

            if abs(final_heading_error) > self._ang_tol:
                if self._prev_final_heading_error is None:
                    self._prev_final_heading_error = final_heading_error

                d_final = self._normalize_angle(
                    final_heading_error - self._prev_final_heading_error
                ) / self._dt

                v_raw = 0.0
                omega_raw = self._k_beta_p * final_heading_error + self._k_beta_d * d_final
                self._prev_final_heading_error = final_heading_error
            else:
                v_raw = 0.0
                omega_raw = 0.0
                reached_goal = True
                self._reset_pd_state()

        v_limited = self._clamp(v_raw, -self._v_linear_max, self._v_linear_max)
        omega_limited = self._clamp(omega_raw, -self._omega_max, self._omega_max)

        max_dv = self._a_max * self._dt
        max_domega = self._alpha_max * self._dt

        v_cmd = self._clamp(v_limited, self._v_prev - max_dv, self._v_prev + max_dv)
        omega_cmd = self._clamp(omega_limited, self._omega_prev - max_domega, self._omega_prev + max_domega)

        return v_cmd, omega_cmd, reached_goal

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _hard_stop(self) -> None:
        self._status = DriveStatus.STOPPED
        self._cmd_linear_x = 0.0
        self._cmd_angular_z = 0.0

    def _reset_pd_state(self) -> None:
        self._prev_distance_error = None
        self._prev_heading_error = None
        self._prev_final_heading_error = None

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(value, hi))

    def _publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        self._cmd_vel_pub.publish(msg)
