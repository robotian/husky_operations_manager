"""
DriveClient — closed-loop TF drive to staging pose or dock pose.

Merges the previous ReverseDriveClient into a single class that handles
both directions. Direction is resolved from DockingConfig at construction
time (hybrid approach) and passed into the shared drive_to() core so that
_compute_command never reads dock_backwards directly.

Public entry points:
    drive_to_staging() — drives to staging pose (away from dock)
    drive_to_dock()    — drives to dock pose (into dock)

Both are thin wrappers around drive_to(target, forward) which holds all
shared state-setting logic.

TODO: Update the drive client to work with general navigation instead of 
just docking and undocking.
"""
import math
import time

import tf2_ros
from transforms3d import euler

from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.impl.rcutils_logger import RcutilsLogger

from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, Quaternion

from husky_operations_manager.dataclass import DockingConfig
from husky_operations_manager.enum import DriveStatus


# ---------------------------------------------------------------------------
# Module-level helpers (unchanged from ReverseDriveClient)
# ---------------------------------------------------------------------------

def _yaw_from_quaternion(q: Quaternion) -> float:
    """Extract yaw from a ROS Quaternion message."""
    _, _, yaw = euler.quat2euler([q.w, q.x, q.y, q.z], 'sxyz')
    return yaw


def _quaternion_from_yaw(yaw: float) -> Quaternion:
    """Build a ROS Quaternion message from a yaw angle in radians."""
    w, x, y, z = euler.euler2quat(0.0, 0.0, yaw, 'sxyz')
    q = Quaternion()
    q.w, q.x, q.y, q.z = float(w), float(x), float(y), float(z)
    return q


def _shortest_angular_distance(from_yaw: float, to_yaw: float) -> float:
    """Return the signed shortest angular distance between two yaw angles."""
    return math.atan2(math.sin(to_yaw - from_yaw), math.cos(to_yaw - from_yaw))


# ---------------------------------------------------------------------------
# DriveClient
# ---------------------------------------------------------------------------

class DriveClient:
    """
    Closed-loop TF drive client for staging pose and dock pose.

    Drives the robot forward or in reverse using cmd_vel based on TF feedback
    from the odom → base_link transform. A single control timer runs at
    controller_frequency and acts only while a drive is active.

    Direction is resolved from DockingConfig at construction time:
        dock_backwards=True  → staging approach is REVERSE, dock approach is FORWARD
        dock_backwards=False → staging approach is FORWARD, dock approach is REVERSE

    This means drive_to_staging() and drive_to_dock() are always called with the
    correct direction — the caller never needs to read dock_backwards.
    Direction-explicit DriveStatus values allow the caller to distinguish
    which drive completed without a separate get_direction() call.

    Usage:
        client = DriveClient(node, docking_config)
        client.drive_to_staging()   # non-blocking, poll get_status()
        client.drive_to_dock()      # non-blocking, poll get_status()
        client.cancel()
        client.reset()
    """

    def __init__(self, node: Node, config: DockingConfig) -> None:
        """
        Initialise DriveClient.

        Resolves direction flags, pre-computes both target poses, and starts
        the idle control timer.
        """
        self.node      = node
        self.config    = config
        self.logger    = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip('/')

        # --- Status ---
        self._status:     DriveStatus  = DriveStatus.IDLE
        self._start_time: float | None = None

        # --- Active drive state (set by drive_to()) ---
        # Target pose for the current drive — either _staging_pose or _dock_pose
        self._target_pose: PoseStamped | None = None
        # True = drive forward (positive linear.x), False = drive in reverse
        self._drive_forward: bool = False
        # Human-readable label used in log messages for the current drive
        self._drive_label: str = ''

        # --- Resolve direction from dock_backwards at construction time ---
        # dock_backwards=True  → robot backs into dock → staging=REVERSE, dock=FORWARD
        # dock_backwards=False → robot drives into dock → staging=FORWARD, dock=REVERSE
        # These flags are the single source of truth — dock_backwards is not read
        # again anywhere else in this class.
        self._staging_forward: bool = not config.dock_backwards
        self._dock_forward: bool    = config.dock_backwards

        # --- Resolve active dock and plugin ---
        self._dock   = config.dock_configs[config.docks[0]]
        self._plugin = config.plugin_configs[config.dock_plugins[0]]

        # --- Speed and tolerances from config ---
        self._lin_speed = config.controller_v_linear_max
        self._ang_speed = config.controller_v_angular_max
        self._lin_tol   = config.undock_linear_tolerance
        self._ang_tol   = config.undock_angular_tolerance

        # Timeout: distance to travel / minimum speed * safety factor 2.0
        # Uses staging_x_offset as the characteristic travel distance for both
        # directions since staging and dock are separated by that offset
        self._timeout = (abs(self._plugin.staging_x_offset) / self._lin_speed) * 2.0

        # --- TF ---
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

        # --- Publishers ---
        self._cmd_vel_pub = self.node.create_publisher(
            TwistStamped, f'{self.namespace}/cmd_vel', 10)

        # Separate visualisation publishers so both poses can be monitored in RViz
        self._staging_pose_pub = self.node.create_publisher(
            PoseStamped, f'{self.namespace}/drive/staging_pose', 10)
        self._dock_pose_pub = self.node.create_publisher(
            PoseStamped, f'{self.namespace}/drive/dock_pose', 10)

        # --- Pre-compute and latch both target poses ---
        # Both are computed once at construction and reused on every drive.
        # Staging pose: dock pose offset along dock_theta by staging_x_offset
        self._staging_pose = self._compute_staging_pose()
        # Dock pose: dock_x, dock_y, dock_theta directly from DockInstanceConfig
        self._dock_pose    = self._compute_dock_pose()

        # --- Control timer — always running, acts only when REVERSING or FORWARDING ---
        self._control_timer = self.node.create_timer(
            1.0 / config.controller_frequency,
            self._control_loop
        )

        self.logger.info(
            f"DriveClient init | dock='{self._dock.instance_name}' | "
            f"dock_backwards={config.dock_backwards} | "
            f"staging_direction={'FORWARD' if self._staging_forward else 'REVERSE'} | "
            f"dock_direction={'FORWARD' if self._dock_forward else 'REVERSE'} | "
            f"speed={self._lin_speed} m/s | timeout={self._timeout:.1f}s"
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def drive_to_staging(self) -> bool:
        """
        Start a drive to the staging pose.

        Direction is determined by dock_backwards resolved at construction:
            dock_backwards=True  → reverse (robot backs away from dock)
            dock_backwards=False → forward (robot drives away from dock)

        Returns True if drive started, False if already active.
        """
        return self._drive_to(
            target=self._staging_pose,
            forward=self._staging_forward,
            label='staging pose'
        )

    def drive_to_dock(self) -> bool:
        """
        Start a drive to the dock pose.

        Direction is determined by dock_backwards resolved at construction:
            dock_backwards=True  → forward (robot drives into dock)
            dock_backwards=False → reverse (robot backs into dock)

        Returns True if drive started, False if already active.
        """
        return self._drive_to(
            target=self._dock_pose,
            forward=self._dock_forward,
            label='dock pose'
        )

    def cancel(self) -> None:
        """Cancel an active drive and publish zero velocity."""
        if self._status not in (DriveStatus.REVERSING, DriveStatus.FORWARDING):
            return
        # _shutdown sets ERROR_* — override to CANCELED_* since this is not a fault
        self._shutdown(success=False, msg=f"Drive to {self._drive_label} canceled")
        self._status = (DriveStatus.CANCELED_FORWARDING
                        if self._drive_forward else DriveStatus.CANCELED_REVERSING)

    def get_status(self) -> DriveStatus:
        """Return the current DriveStatus."""
        return self._status

    def is_active(self) -> bool:
        """Return True if a drive is currently in progress."""
        return self._status in (DriveStatus.REVERSING, DriveStatus.FORWARDING)

    def reset(self) -> None:
        """Reset to IDLE, clearing all active drive state."""
        self._status       = DriveStatus.IDLE
        self._start_time   = None
        self._target_pose  = None
        self._drive_forward = False
        self._drive_label  = ''
        self.logger.info("DriveClient reset to IDLE")

    # ------------------------------------------------------------------
    # Shared drive core
    # ------------------------------------------------------------------

    def _drive_to(self, target: PoseStamped, forward: bool, label: str) -> bool:
        """
        Core drive entry — shared by drive_to_staging() and drive_to_dock().

        Sets the target pose, direction flag, and drive label, then transitions
        to REVERSING or FORWARDING. The control timer picks up the active
        drive on the next tick.

        Args:
            target:  Pre-computed PoseStamped to drive toward.
            forward: True = positive linear.x (forward), False = negative (reverse).
            label:   Human-readable name used in log messages.

        Returns:
            True if drive started successfully, False if already active.
        """
        if self._status in (DriveStatus.REVERSING, DriveStatus.FORWARDING):
            self.logger.warning(
                f"drive_to({label}) ignored — already driving to {self._drive_label}"
            )
            return False

        # Set active drive state — consumed by _control_loop and _compute_command
        self._target_pose   = target
        self._drive_forward = forward
        self._drive_label   = label
        self._start_time = None
        # Direction-explicit active status — lets caller distinguish which
        # drive is in progress from get_status() alone
        self._status = (DriveStatus.FORWARDING
                        if forward else DriveStatus.REVERSING)

        direction_str = 'forward' if forward else 'reverse'
        self.logger.info(f"Drive started | target={label} | direction={direction_str}")
        self.logger.debug(
            f"Drive target | "
            f"x={target.pose.position.x:.3f} y={target.pose.position.y:.3f} | "
            f"frame={target.header.frame_id}"
        )
        return True

    # ------------------------------------------------------------------
    # Target pose computation
    # ------------------------------------------------------------------

    def _compute_staging_pose(self) -> PoseStamped:
        """
        Pre-compute the staging pose from dock pose and plugin staging offset.

        staging_x = dock_x + cos(dock_theta) * staging_x_offset
        staging_y = dock_y + sin(dock_theta) * staging_x_offset
        staging_theta = dock_theta + staging_yaw_offset
        """
        staging_x     = (self._dock.dock_x
                         + math.cos(self._dock.dock_theta) * self._plugin.staging_x_offset)
        staging_y     = (self._dock.dock_y
                         + math.sin(self._dock.dock_theta) * self._plugin.staging_x_offset)
        staging_theta = self._dock.dock_theta + self._plugin.staging_yaw_offset

        pose = PoseStamped()
        pose.header.frame_id  = self._dock.frame
        pose.header.stamp     = self.node.get_clock().now().to_msg()
        pose.pose.position.x  = staging_x
        pose.pose.position.y  = staging_y
        pose.pose.orientation = _quaternion_from_yaw(staging_theta)

        self._staging_pose_pub.publish(pose)
        self.logger.info(
            f"Staging pose: ({staging_x:.3f}, {staging_y:.3f}, "
            f"{math.degrees(staging_theta):.1f}deg)"
        )
        return pose

    def _compute_dock_pose(self) -> PoseStamped:
        """
        Pre-compute the dock pose directly from DockInstanceConfig.

        No offset is applied — dock_x, dock_y, dock_theta are the physical
        dock position and orientation.
        """
        pose = PoseStamped()
        pose.header.frame_id  = self._dock.frame
        pose.header.stamp     = self.node.get_clock().now().to_msg()
        pose.pose.position.x  = self._dock.dock_x
        pose.pose.position.y  = self._dock.dock_y
        pose.pose.orientation = _quaternion_from_yaw(self._dock.dock_theta)

        self._dock_pose_pub.publish(pose)
        self.logger.info(
            f"Dock pose: ({self._dock.dock_x:.3f}, {self._dock.dock_y:.3f}, "
            f"{math.degrees(self._dock.dock_theta):.1f}deg)"
        )
        return pose

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """
        Shared control loop — runs at controller_frequency.

        Acts only when status is REVERSING or FORWARDING.

        On each tick:
          1. Publish target pose for RViz visualisation
          2. Get current robot pose via TF
          3. Check timeout
          4. Compute and publish cmd_vel, or shut down on goal reached
        """
        if self._status not in (DriveStatus.REVERSING, DriveStatus.FORWARDING)\
                or self._target_pose is None:
            return

        # Refresh timestamp on target pose before publishing for visualisation
        self._target_pose.header.stamp = self.node.get_clock().now().to_msg()
        if self._drive_forward == self._staging_forward:
            self._staging_pose_pub.publish(self._target_pose)
        else:
            self._dock_pose_pub.publish(self._target_pose)

        robot_pose = self._get_robot_pose(self._target_pose.header.frame_id)
        if robot_pose is None:
            return

        # Log start position once TF becomes available
        if self._start_time is None:
            self._start_time = time.monotonic()
            self.logger.info(
                f"TF ready | drive to {self._drive_label} | "
                f"start: ({robot_pose.pose.position.x:.3f}, "
                f"{robot_pose.pose.position.y:.3f})"
            )

        if time.monotonic() - self._start_time >= self._timeout:
            self._shutdown(
                success=False,
                msg=f"Drive to {self._drive_label} timed out after {self._timeout:.1f}s"
            )
            return

        cmd = Twist()
        if self._compute_command(cmd, self._target_pose, robot_pose):
            elapsed = time.monotonic() - self._start_time
            self._shutdown(
                success=True,
                msg=f"{self._drive_label} reached in {elapsed:.2f}s"
            )
            return

        self._publish_cmd_vel(cmd)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _get_robot_pose(self, frame: str) -> PoseStamped | None:
        """
        Look up the current robot pose in the given frame via TF.

        Returns None on lookup failure — the control loop skips that tick.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                frame,
                self.config.base_frame,
                Time(),
                timeout=Duration(seconds=0.2)
            )
            pose = PoseStamped()
            pose.header.frame_id  = frame
            pose.pose.position.x  = tf.transform.translation.x
            pose.pose.position.y  = tf.transform.translation.y
            pose.pose.orientation = tf.transform.rotation
            return pose
        except Exception as e:
            self.logger.warning(f"TF lookup failed: {e}", throttle_duration_sec=2.0)
            return None

    def _compute_command(
        self, cmd: Twist, target: PoseStamped, robot_pose: PoseStamped
    ) -> bool:
        """
        Compute cmd_vel to drive toward target. Returns True when goal is reached.

        Direction is controlled by self._drive_forward (set in _drive_to()):
            forward=True  → heading reference = 0.0,  linear.x = +speed
            forward=False → heading reference = pi,    linear.x = -speed

        This is the only place in the class where _drive_forward affects behaviour.
        dock_backwards is never read here — direction is fully resolved upstream.

        Args:
            cmd:        Twist message to fill in-place.
            target:     Target PoseStamped to drive toward.
            robot_pose: Current robot PoseStamped from TF.

        Returns:
            True if within linear and angular tolerance of target (goal reached).
        """
        dist = math.hypot(
            robot_pose.pose.position.x - target.pose.position.x,
            robot_pose.pose.position.y - target.pose.position.y,
        )
        yaw_err = _shortest_angular_distance(
            _yaw_from_quaternion(robot_pose.pose.orientation),
            _yaw_from_quaternion(target.pose.orientation),
        )

        self.logger.debug(
            f"_compute_command | dist={dist:.3f}m yaw_err={math.degrees(yaw_err):.1f}deg | "
            f"forward={self._drive_forward} | "
            f"lin_tol={self._lin_tol} ang_tol={math.degrees(self._ang_tol):.1f}deg"
        )

        if dist < self._lin_tol and abs(yaw_err) < self._ang_tol:
            return True

        try:
            tf = self._tf_buffer.lookup_transform(
                self.config.base_frame,
                target.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.1)
            )
        except Exception as e:
            self.logger.error(f"TF base_frame lookup failed: {e}")
            return False

        # Transform target position into base_frame
        tf_yaw = _yaw_from_quaternion(tf.transform.rotation)
        t      = tf.transform.translation
        lx = (math.cos(tf_yaw) * target.pose.position.x
              - math.sin(tf_yaw) * target.pose.position.y + t.x)
        ly = (math.sin(tf_yaw) * target.pose.position.x
              + math.cos(tf_yaw) * target.pose.position.y + t.y)

        # Heading reference and linear velocity sign are the only two lines
        # that differ between forward and reverse drive.
        # forward=True:  robot faces target  → heading ref = 0.0, speed = +v
        # forward=False: robot backs to target → heading ref = pi,  speed = -v
        heading_ref   = 0.0 if self._drive_forward else math.pi
        heading_error = _shortest_angular_distance(heading_ref, math.atan2(ly, lx))

        cmd.linear.x  = self._lin_speed if self._drive_forward else -self._lin_speed
        cmd.angular.z = max(
            -self._ang_speed,
            min(self._ang_speed, self._ang_speed * (heading_error / math.pi))
        )
        return False

    def _publish_cmd_vel(self, twist: Twist) -> None:
        """Wrap a Twist in a stamped message and publish to cmd_vel."""
        msg = TwistStamped()
        msg.header.stamp    = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.config.base_frame
        msg.twist           = twist
        self._cmd_vel_pub.publish(msg)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self, success: bool, msg: str) -> None:
        """
        Stop the drive — publish zero velocity and set terminal status.

        Terminal status is direction-explicit so the caller can identify both
        the outcome and which drive was running from get_status() alone:
            success + forward  → DONE_FORWARDING
            success + reverse  → DONE_REVERSING
            failure + forward  → ERROR_FORWARDING
            failure + reverse  → ERROR_REVERSING
        cancel() overrides ERROR_* to CANCELED_* after calling this method.
        The control loop exits on the next tick because status is no longer
        REVERSING or FORWARDING.
        """
        self._publish_cmd_vel(Twist())  # zero velocity — stop the robot immediately
        if success:
            self._status = (DriveStatus.DONE_FORWARDING
                            if self._drive_forward else DriveStatus.DONE_REVERSING)
            self.logger.info(msg)
        else:
            self._status = (DriveStatus.ERROR_FORWARDING
                            if self._drive_forward else DriveStatus.ERROR_REVERSING)
            self.logger.warning(msg)