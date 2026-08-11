"""
drive.py

Drives the robot toward lavender bushes using camera detections, then
stops the robot next to each bush so the arm can harvest it.

Owned by a parent node, which starts the driving with scan() and tells
it to keep going after each bush with resume().

-------------------------------------------------------------------------------
Coordinate Frames
-------------------------------------------------------------------------------

  A "frame" is just a coordinate system with its own origin and axis
  directions. The camera and the robot body (base_link) don't share the
  same one, so positions have to be converted between them.

  Camera frame (config.camera_frame, e.g. arm_camera_color_frame):
    +X → backward  (bush ahead → negative center.x)
    +Y → left
    +Z → down

  base_link (the robot's own frame):
    +X → forward
    +Y → left
    +Z → up

  The camera is mounted upside-down and facing backward, which is why its
  axes look flipped compared to the robot's.

-------------------------------------------------------------------------------
Target Pose Calculation
-------------------------------------------------------------------------------

  Every time the camera detects a bush (msg.center.x/y, in the camera's
  own frame), the code converts that position into the odom frame (the
  robot's fixed starting-point frame) using TF — ROS2's built-in system
  for looking up how frames relate to each other. See
  _camera_frame_to_odom.

  _set_target_pose then nudges that point slightly further along the
  row (using cam_tx/cam_ty and bushrow_theta) to get the final target
  the robot will drive to: (target_x, target_y, target_yaw).

  cam_tx/cam_ty/cam_yaw come from _cam_pose, a base_frame->camera_frame
  TF lookup refreshed on every scan()/resume() (see _refresh_cam_pose) —
  none are fixed constants, since the camera rides on the arm and
  rotates with it, not just translates.

  Refreshing only at scan()/resume() is sufficient, not a gap: the arm
  is held fixed for the entire SCANNING/CONTROLLING/DEPARTING cycle and
  only repositions while STOPPED (harvest). scan()/resume() bracket
  every arm move, so _cam_pose is never stale when it's used.

-------------------------------------------------------------------------------
Controller
-------------------------------------------------------------------------------

  Runs on a timer, checking the latest odom + detection every tick.

  Phase 1 — drive to the bush: while still far away, drive toward it
  (backing up instead, if it's close but facing the wrong way), turning
  to face it as it goes.

  Phase 2 — final turn: once close enough, stop moving forward and just
  turn in place until facing the bush correctly.

  Speed and turn commands are capped, and can only change gradually
  tick-to-tick, so the robot doesn't jerk around.

-------------------------------------------------------------------------------
State Machine
-------------------------------------------------------------------------------

  The robot moves through these stages one at a time:

  IDLE
    └─► scan() → SCANNING  (drives forward slowly, no bush picked yet)
          ├─► sees a bush → locks it as the target → CONTROLLING
          └─► sees nothing for a while → STOPPED  (assumes it's the row's end)

  CONTROLLING
    ├─► sees the bush again → updates the locked target
    ├─► reaches the bush (position + facing it) → STOPPED
    └─► takes too long without reaching it →
          ├─► under the retry limit → ABORTED
          └─► retry limit hit → ERROR  (gives up on this bush)

  ABORTED  (stopped, target kept, ignores new detections)
    └─► after a short wait → CONTROLLING  (tries the same bush again)

  STOPPED
    └─► resume() → DEPARTING  (drives forward a bit, ignoring detections)
          └─► far enough clear → SCANNING  (via scan())

-------------------------------------------------------------------------------
Notes
-------------------------------------------------------------------------------

  - Detections come in as ImageDetectionPose messages — detection_valid
    tells you if a bush was actually seen, center.x/y is where it is in
    the camera frame.
  - Odom and detection topics use BEST_EFFORT QoS, matching how sensor
    topics are usually configured in ROS2.
  - cmd_vel is sent out every tick, whether or not anything changed — this
    keeps the robot's speed controller fed even if no new bush shows up.
  - Husky A200 speed floors (measured on this specific robot):
      Below 0.05 m/s, the robot doesn't move at all (friction). Above
      0.1 m/s, it tracks speed commands reliably. In between, it might
      move, but jerkily.
"""

import math

from geometry_msgs.msg import PointStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from tf_transformations import euler_from_quaternion

from husky_operations_manager.types import DriveConfig
from status_interfaces.msg import DriveFeedback, ImageDetectionPose

_STATUS_NAMES = {
    DriveFeedback.IDLE: 'IDLE',
    DriveFeedback.SCANNING: 'SCANNING',
    DriveFeedback.CONTROLLING: 'CONTROLLING',
    DriveFeedback.STOPPED: 'STOPPED',
    DriveFeedback.DEPARTING: 'DEPARTING',
    DriveFeedback.CANCELED: 'CANCELED',
    DriveFeedback.ERROR: 'ERROR',
    DriveFeedback.ABORTED: 'ABORTED',
}

D_ALPHA = 0.3


class DriveClient:
    """
    Component class for camera-guided PD target-pose drive control.

    Instantiated and owned by the parent harvesting node, which supplies
    the ROS2 node handle and drives the lifecycle via scan()/resume().

    Lifecycle:
      1. scan()    — start SCANNING forward, no target locked
      2. (auto)    — first valid detection locks a target pose → CONTROLLING
      3. (auto)    — PD goal reached (position + final heading) → STOPPED
      4. resume()  — called by parent after harvest → DEPARTING
      5. (auto)    — departure clearance met → SCANNING (via scan())
    """

    def __init__(self, node: Node, config: DriveConfig) -> None:
        """Build DriveClient from node + config: state, TF buffer, subs/pubs, control timer."""
        self._node = node
        self._logger = RcutilsLogger(self.__class__.__name__)
        self._ns = self._node.get_namespace().rstrip('/')

        # --- Config ---
        self._base_frame = config.base_frame
        self._camera_frame = config.camera_frame
        self._odom_frame = config.odom_frame
        self._stop_threshold = config.ex_tolerance
        self._ang_tol = config.ang_tol
        self._v_linear_min = config.v_linear_min
        self._v_linear_max = config.v_linear_max
        self._omega_max = config.v_angular_max
        self._k_v_p = config.k_v_p
        self._k_v_d = config.k_v_d
        self._k_omega_p = config.k_omega_p
        self._k_omega_d = config.k_omega_d
        self._k_beta_p = config.k_beta_p
        self._k_beta_d = config.k_beta_d
        self._a_max = config.a_max
        self._alpha_max = config.alpha_max
        self._backward_dist_thresh = config.backward_distance_threshold
        self._same_bush_threshold = config.same_bush_threshold
        self._controlling_timeout = config.controlling_timeout
        self._max_controlling_retries = config.max_controlling_retries
        self._controlling_retry_delay = config.controlling_retry_delay

        self._cam_pose: tuple[float, float, float] | None = None

        self._bushrow_theta = config.bushrow_theta

        self._dt = 1.0 / config.cmd_vel_rate

        # Convert distance-based config fields to time
        self._no_detection_timeout: float = config.no_detection_distance / config.v_linear_min
        self._departure_duration: float = config.departure_clearance / config.v_linear_min

        # --- Odom state ---
        self._odom_received: bool = False
        self._current_pose: list[float] = [0.0, 0.0, 0.0]  # x, y, yaw

        # --- Latest cached detection ---
        self._latest_detection: tuple[bool, float, float] = (False, 0.0, 0.0)

        # --- Detection staleness guard ---
        self._latest_detection_stamp: tuple[int, int] | None = None
        self._last_used_detection_stamp: tuple[int, int] | None = None
        self._scan_start_stamp: Time | None = None

        # --- Odom-frame positions of already-harvested bushes ---
        self._harvested_positions: list[tuple[float, float]] = []

        # --- Drive state ---
        self._status: DriveFeedback = DriveFeedback.IDLE
        self._state_timer: float = 0.0

        # --- Tracks the status cmd_vel was last published under ---
        self._last_published_status: DriveFeedback | None = None

        # --- Target pose (odom frame) ---
        self._target_pose: tuple[float, float, float] | None = None
        self._lock_anchor: tuple[float, float] | None = None

        # --- TF (camera->odom lookup) ---
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._node)

        # --- CONTROLLING timeout retry count — reset on fresh lock (scan()/resume()) ---
        self._controlling_retry_count: int = 0

        # --- PD state memory ---
        self._v_prev: float = 0.0
        self._omega_prev: float = 0.0
        self._prev_distance_error: float | None = None
        self._prev_heading_error: float | None = None
        self._prev_final_heading_error: float | None = None
        self._distance_error: float = 0.0
        self._heading_error: float = 0.0

        # --- Derivative EMA filter (alpha=1.0 disables; lag ~ dt/alpha) ---
        self._d_alpha: float = D_ALPHA
        self._d_distance_filt: float = 0.0
        self._d_heading_filt: float = 0.0
        self._d_final_filt: float = 0.0

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

        self._drive_feedback_pub = self._node.create_publisher(
            DriveFeedback,
            f'{self._ns}/drive_feedback',
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
        """Start SCANNING forward from a fresh (unlocked) state; no-op with an error log if odom isn't ready or camera TF is unavailable."""
        if not self._odom_received:
            self._logger.error(
                'scan() called but odom not yet received — aborting scan',
                throttle_duration_sec=2.0,
            )
            return

        if not self._refresh_cam_pose():
            self._logger.error(
                'scan() called but camera TF unavailable — aborting scan',
                throttle_duration_sec=2.0,
            )
            return

        self._status = DriveFeedback.SCANNING
        self._state_timer = 0.0
        self._target_pose = None
        self._lock_anchor = None
        self._latest_detection = (False, 0.0, 0.0)
        self._controlling_retry_count = 0
        self._reset_pd_state()

        # Detection latency is ~2-3s, so detections arriving just after
        # scan() were imaged before/during departure and can point at the
        # already-harvested bush. Only lock on imagery taken after this.
        self._scan_start_stamp = self._node.get_clock().now()

        self._logger.info('SCANNING - controller inactive until first detection')

    def resume(self) -> None:
        """Transition STOPPED to DEPARTING after harvest completes; warns and no-ops from any other state or if camera TF is unavailable."""
        if self._status != DriveFeedback.STOPPED:
            self._logger.warning(
                f'resume() called in unexpected state: {_STATUS_NAMES.get(self._status, self._status)} — ignoring',
                throttle_duration_sec=2.0,
            )
            return

        if not self._refresh_cam_pose():
            self._logger.error(
                'resume() called but camera TF unavailable — aborting resume',
                throttle_duration_sec=2.0,
            )
            return

        self._status = DriveFeedback.DEPARTING
        self._state_timer = 0.0
        self._target_pose = None
        self._lock_anchor = None
        self._latest_detection = (False, 0.0, 0.0)
        self._latest_detection_stamp = None
        self._last_used_detection_stamp = None
        self._controlling_retry_count = 0
        self._reset_pd_state()
        self._logger.info(f'DEPARTING — clearing bush area | duration={self._departure_duration:.1f}s')

    def cancel(self) -> None:
        """Force-stop immediately into CANCELED, publishing a zero cmd_vel regardless of current state."""
        self._status = DriveFeedback.CANCELED
        self._state_timer = 0.0
        self._target_pose = None
        self._latest_detection = (False, 0.0, 0.0)
        self._latest_detection_stamp = None
        self._last_used_detection_stamp = None
        self._reset_pd_state()
        self._cmd_linear_x = 0.0
        self._cmd_angular_z = 0.0
        self._publish_cmd_vel(0.0, 0.0, publish_to_topic=True)
        self._logger.warning('CANCELED — forced stop by parent node')

    def reset(self) -> None:
        """Clear back to IDLE after a natural STOPPED row-end (robot already stopped)."""
        self._status = DriveFeedback.IDLE
        self._state_timer = 0.0
        self._target_pose = None
        self._latest_detection = (False, 0.0, 0.0)
        self._latest_detection_stamp = None
        self._last_used_detection_stamp = None
        self._reset_pd_state()
        self._logger.info('DriveClient reset to IDLE')

    def get_status(self) -> DriveFeedback:
        """Return the current full feedback state as a DriveFeedback message."""
        return self._build_feedback_msg()

    def is_active(self) -> bool:
        """Return True if status is SCANNING, CONTROLLING, or DEPARTING."""
        return self._status in (
            DriveFeedback.SCANNING,
            DriveFeedback.CONTROLLING,
            DriveFeedback.DEPARTING,
        )

    def is_ready(self) -> bool:
        """Return True if odom has been received and scan() can be safely called."""
        return self._odom_received

    def is_camera_tf_ready(self) -> bool:
        """Return True if base_frame->camera_frame TF is currently resolvable."""
        return self._tf_buffer.can_transform(self._base_frame, self._camera_frame, Time())

    # =========================================================================
    # Odom callback
    # =========================================================================

    def _odom_callback(self, msg: Odometry) -> None:
        """Update current pose/velocity of the robot."""
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._current_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        current_velocity = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

        if not self._odom_received:
            self._logger.info(
                f'First odom received | pose=({self._current_pose[0]:.3f}, '
                f'{self._current_pose[1]:.3f}, {math.degrees(self._current_pose[2]):.1f}deg)'
            )

        self._odom_received = True

        self._logger.debug(
            f'Odom | pose=({self._current_pose[0]:.3f}, {self._current_pose[1]:.3f}, '
            f'{math.degrees(self._current_pose[2]):.1f}deg) | '
            f'velocity=(v={current_velocity[0]:.3f}, omega={current_velocity[1]:.3f})',
            throttle_duration_sec=2.0,
        )

    # =========================================================================
    # Detection callback — caches only, state machine runs on control timer
    # =========================================================================

    def _detection_callback(self, msg: ImageDetectionPose) -> None:
        """The latest detection and stamp for the control timer."""
        detected = bool(msg.detection_valid)
        bush_x, bush_y = msg.center.x, msg.center.y

        self._logger.debug(
            f'Received Detection - Valid:{msg.detection_valid} at {bush_x, bush_y}',
            throttle_duration_sec=2.0,
        )

        self._latest_detection = (detected, bush_x, bush_y)
        self._latest_detection_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)

    def _detection_predates_scan(self) -> bool:
        """True if the latest detection's image was taken before scan() started — i.e. before/during departure, likely showing the already-harvested bush."""
        if self._latest_detection_stamp is None or self._scan_start_stamp is None:
            return False
        stamp_ns = self._latest_detection_stamp[0] * 1_000_000_000 + self._latest_detection_stamp[1]
        if stamp_ns < self._scan_start_stamp.nanoseconds:
            self._logger.info(
                'Ignoring pre-scan detection — imaged before departure completed, '
                'likely the harvested bush, continuing SCANNING',
                throttle_duration_sec=2.0,
            )
            return True
        return False

    # =========================================================================
    # Control timer — sole driver of the state machine + cmd_vel
    # =========================================================================

    def _control_callback(self) -> None:
        """Drive the state machine, PD controller, and cmd_vel/feedback publishing on a fixed-rate tick."""
        if not self._odom_received:
            return

        self._state_timer += self._dt
        curr_yaw = self._current_pose[2]
        img_detected, bush_x, bush_y = self._latest_detection
        is_new_detection = img_detected and self._latest_detection_stamp != self._last_used_detection_stamp

        if self._status == DriveFeedback.IDLE:
            self._cmd_linear_x = 0.0
            self._cmd_angular_z = 0.0

        elif self._status == DriveFeedback.SCANNING:
            self._cmd_linear_x = self._v_linear_min
            self._cmd_angular_z = 0.0

            if is_new_detection and bush_x < 0 and not self._detection_predates_scan():
                self._set_target_pose(bush_x, bush_y)
                if self._target_pose is None:
                    self._logger.warning(
                        'Target lock skipped — TF unavailable, continuing SCANNING',
                        throttle_duration_sec=2.0,
                    )
                else:
                    self._status = DriveFeedback.CONTROLLING
                    self._state_timer = 0.0
                    self._last_used_detection_stamp = self._latest_detection_stamp
                    tx, ty, tyaw = self._target_pose
                    self._logger.info(
                        f'Target locked — bush cam=({bush_x:.3f}, {bush_y:.3f}) '
                        f'target odom=({tx:.3f}, {ty:.3f}, {math.degrees(tyaw):.1f}deg)'
                    )

            elif self._state_timer >= self._no_detection_timeout:
                self._logger.warning(
                    f'No-detection timeout ({self._no_detection_timeout:.1f}s) - row end or gap. Stopping.'
                )
                self._hard_stop()

            elif img_detected:
                self._logger.info(
                    f'Ignoring detection at bush_x={bush_x:.3f} (not ahead) — '
                    f'likely the departed bush, continuing SCANNING',
                    throttle_duration_sec=2.0,
                )

        elif self._status == DriveFeedback.CONTROLLING:
            # bush_x >= 0 (level/behind) can't improve the lock — skip re-lock
            if is_new_detection and bush_x < 0:
                self._set_target_pose(bush_x, bush_y)
                self._last_used_detection_stamp = self._latest_detection_stamp

                # Only for logging
                tx, ty, tyaw = self._target_pose
                cx, cy, cyaw = self._current_pose
                self._logger.info(
                    f'Target re-locked | target odom=({tx:.3f}, {ty:.3f}, {math.degrees(tyaw):.1f}deg) | '
                    f'current pose=({cx:.3f}, {cy:.3f}, {math.degrees(cyaw):.1f}deg)'
                )

            cmd_v, cmd_w, goal_reached = self._compute_commands_pd()
            self._cmd_linear_x = cmd_v
            self._cmd_angular_z = cmd_w

            if goal_reached:
                self._harvested_positions.append((self._target_pose[0], self._target_pose[1]))
                self._logger.info(f'Goal reached STOPPED (awaiting resume()) at {self._current_pose} ')
                self._hard_stop()

            elif self._state_timer >= self._controlling_timeout:
                if self._controlling_retry_count >= self._max_controlling_retries:
                    self._logger.error(
                        f'ERROR: CONTROLLING timeout ({self._controlling_timeout:.1f}s) retries exhausted '
                        f'({self._controlling_retry_count}/{self._max_controlling_retries}).'
                    )
                    self._error_stop()
                else:
                    self._logger.warning(
                        f' ABORTED: CONTROLLING timeout ({self._controlling_timeout:.1f}s), retrying in '
                        f'{self._controlling_retry_delay:.1f}s ({self._controlling_retry_count + 1}/{self._max_controlling_retries})'
                    )
                    self._status = DriveFeedback.ABORTED
                    self._state_timer = 0.0
                    self._cmd_linear_x = 0.0
                    self._cmd_angular_z = 0.0
                    self._reset_pd_state()

        elif self._status == DriveFeedback.ABORTED:
            self._cmd_linear_x = 0.0
            self._cmd_angular_z = 0.0

            if self._state_timer >= self._controlling_retry_delay:
                self._controlling_retry_count += 1
                self._logger.info(
                    f'Retry {self._controlling_retry_count}/{self._max_controlling_retries} '
                    'resuming CONTROLLING on same target'
                )
                self._status = DriveFeedback.CONTROLLING
                self._state_timer = 0.0
                self._reset_pd_state()

        elif self._status == DriveFeedback.STOPPED or self._status == DriveFeedback.CANCELED:
            self._cmd_linear_x = 0.0
            self._cmd_angular_z = 0.0

        elif self._status == DriveFeedback.DEPARTING:
            self._cmd_linear_x = self._v_linear_min
            self._cmd_angular_z = 0.0

            if self._state_timer >= self._departure_duration:
                self._logger.info('Departure complete re-scanning')
                self.scan()

        idle_states = (
            DriveFeedback.IDLE,
            DriveFeedback.STOPPED,
            DriveFeedback.CANCELED,
            DriveFeedback.ERROR,
            DriveFeedback.ABORTED,
        )
        publish_to_topic = self._status not in idle_states or self._status != self._last_published_status
        self._publish_cmd_vel(self._cmd_linear_x, self._cmd_angular_z, publish_to_topic)
        if publish_to_topic:
            self._last_published_status = self._status

        self._publish_feedback()

    # =========================================================================
    # Target pose (camera frame → odom frame)
    # =========================================================================
    def _refresh_cam_pose(self) -> bool:
        """TF-lookup base_frame->camera_frame translation and refresh _cam_pose; returns False on TF failure."""
        try:
            transform = self._tf_buffer.lookup_transform(self._base_frame, self._camera_frame, Time())
        except TransformException as e:
            self._logger.warning(
                f'TF lookup failed ({self._base_frame} <- {self._camera_frame}): {e}',
                throttle_duration_sec=2.0,
            )
            return False

        cam_tx = transform.transform.translation.x
        cam_ty = transform.transform.translation.y
        q = transform.transform.rotation
        _, _, cam_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._cam_pose = (cam_tx, cam_ty, cam_yaw)
        self._logger.info(f'Camera pose refreshed | cam_pose={self._cam_pose}')
        return True

    def _camera_frame_to_odom(self, local_x: float, local_y: float) -> tuple[float, float] | None:
        """TF-lookup camera-to-odom point transform for (local_x, local_y); returns None on TF failure."""
        point = PointStamped()
        point.header.frame_id = self._camera_frame
        point.header.stamp = Time().to_msg()
        point.point.x = local_x
        point.point.y = local_y
        point.point.z = 0.0

        try:
            transform = self._tf_buffer.lookup_transform(self._odom_frame, self._camera_frame, Time())
        except TransformException as e:
            self._logger.warning(
                f'TF lookup failed ({self._odom_frame} <- {self._camera_frame}): {e}',
                throttle_duration_sec=2.0,
            )
            return None

        odom_point = do_transform_point(point, transform)
        return odom_point.point.x, odom_point.point.y

    def _set_target_pose(self, bush_x: float, bush_y: float) -> None:
        """Lock or re-lock target pose from a camera-frame bush offset, rejecting re-locks past same_bush_threshold."""
        result = self._camera_frame_to_odom(bush_x, bush_y)

        if result is None:
            return

        cam_x, cam_y = result
        cam_yaw = self._bushrow_theta + self._cam_pose[2]

        target_x = cam_x + self._cam_pose[0] * math.cos(cam_yaw) - self._cam_pose[1] * math.sin(cam_yaw)
        target_y = cam_y + self._cam_pose[0] * math.sin(cam_yaw) + self._cam_pose[1] * math.cos(cam_yaw)
        target_yaw = cam_yaw - self._cam_pose[2]

        if self._target_pose is not None:
            shift = math.hypot(target_x - self._lock_anchor[0], target_y - self._lock_anchor[1])
            if shift > self._same_bush_threshold:
                self._logger.info(
                    f'Re-lock rejected — candidate odom=({target_x:.3f}, {target_y:.3f}) is {shift:.3f}m from '
                    f'original lock ({self._lock_anchor[0]:.3f}, {self._lock_anchor[1]:.3f}) > '
                    f'same_bush_threshold={self._same_bush_threshold:.3f}m — likely a different bush, keeping current lock'
                )
                return
        else:
            self._lock_anchor = (target_x, target_y)

        self._target_pose = (target_x, target_y, target_yaw)
        self._logger.info(f'Target Pose as {self._target_pose}')

    # =========================================================================
    # PD controller
    # =========================================================================

    def _compute_commands_pd(self) -> tuple[float, float, bool]:
        """Compute PD position and heading commands; return (v_cmd, omega_cmd, goal_reached)."""
        x, y, theta = self._current_pose
        x_d, y_d, theta_d = self._target_pose if self._target_pose is not None else (0.0, 0.0, 0.0)

        # self._v_prev = self._current_velocity[0]
        # self._omega_prev = self._current_velocity[1]
        self._v_prev = self._cmd_linear_x
        self._omega_prev = self._cmd_angular_z

        dx = x_d - x
        dy = y_d - y
        distance_error = math.hypot(dx, dy)
        self._distance_error = distance_error

        v_raw = 0.0
        omega_raw = 0.0
        reached_goal = False

        if distance_error > self._stop_threshold:
            target_heading = math.atan2(dy, dx)
            heading_error = self._normalize_angle(target_heading - theta)
            self._heading_error = heading_error

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

            # EMA-filter the derivatives: per-tick odom jitter divided by
            # dt=0.05s otherwise dominates the d-term and chatters omega.
            self._d_distance_filt = self._d_alpha * d_distance + (1.0 - self._d_alpha) * self._d_distance_filt
            self._d_heading_filt = self._d_alpha * d_heading + (1.0 - self._d_alpha) * self._d_heading_filt
            d_distance = self._d_distance_filt
            d_heading = self._d_heading_filt

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
            self._heading_error = final_heading_error

            if abs(final_heading_error) > self._ang_tol:
                if self._prev_final_heading_error is None:
                    self._prev_final_heading_error = final_heading_error

                d_final = self._normalize_angle(final_heading_error - self._prev_final_heading_error) / self._dt
                self._d_final_filt = self._d_alpha * d_final + (1.0 - self._d_alpha) * self._d_final_filt

                v_raw = 0.0
                omega_raw = self._k_beta_p * final_heading_error + self._k_beta_d * self._d_final_filt
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
        """Force status to STOPPED and zero cmd_vel."""
        self._status = DriveFeedback.STOPPED
        self._cmd_linear_x = 0.0
        self._cmd_angular_z = 0.0

    def _error_stop(self) -> None:
        """Force status to ERROR and zero cmd_vel; distinct from STOPPED, which callers treat as success."""
        self._status = DriveFeedback.ERROR
        self._cmd_linear_x = 0.0
        self._cmd_angular_z = 0.0

    def _reset_pd_state(self) -> None:
        """Clear PD error-history state so the next tick's derivative terms start fresh."""
        self._prev_distance_error = None
        self._prev_heading_error = None
        self._prev_final_heading_error = None
        self._d_distance_filt = 0.0
        self._d_heading_filt = 0.0
        self._d_final_filt = 0.0

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Wrap an angle to [-pi, pi]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        """Clamp value to the closed interval [lo, hi]."""
        return max(lo, min(value, hi))

    def _publish_cmd_vel(self, linear_x: float, angular_z: float, publish_to_topic: bool = True) -> None:
        """Publish a TwistStamped cmd_vel; publish_to_topic=False suppresses the actual topic publish."""
        msg = TwistStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z

        # Gatekeeping to avoid continuous publishing of zero velocity when robot is STOPPED or still
        if publish_to_topic:
            self._cmd_vel_pub.publish(msg)

        self._logger.debug(f'Published command velocity: {msg.twist}')

    def _build_feedback_msg(self) -> DriveFeedback:
        """Assemble the current DriveFeedback message from target/current pose and PD/status state."""
        tx, ty, tyaw = self._target_pose if self._target_pose is not None else (0.0, 0.0, 0.0)
        cx, cy, cyaw = self._current_pose

        feedback = DriveFeedback()
        feedback.header.stamp = self._node.get_clock().now().to_msg()
        feedback.header.frame_id = self._base_frame
        feedback.status = self._status
        feedback.target_x = tx
        feedback.target_y = ty
        feedback.target_yaw = tyaw
        feedback.current_x = cx
        feedback.current_y = cy
        feedback.current_yaw = cyaw
        feedback.distance_error = self._distance_error
        feedback.heading_error = self._heading_error
        feedback.cmd_linear_x = self._cmd_linear_x
        feedback.cmd_angular_z = self._cmd_angular_z
        feedback.harvested_count = len(self._harvested_positions)
        feedback.state_timer = self._state_timer
        return feedback

    def _publish_feedback(self) -> None:
        """Publish the current DriveFeedback message."""
        self._drive_feedback_pub.publish(self._build_feedback_msg())
