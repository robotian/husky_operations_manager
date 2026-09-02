import warnings

import math
import time

import tf2_geometry_msgs
import tf2_ros
from tf_transformations import euler_from_quaternion, quaternion_from_euler

from geometry_msgs.msg import PoseStamped, Quaternion, Twist, TwistStamped
from rclpy.duration import Duration
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
from rclpy.time import Time

from husky_operations_manager.dataclass import DockInstanceConfig, ReverseDriveConfig
from husky_operations_manager.enum import ReverseDriveStatus

warnings.filterwarnings("ignore", category=SyntaxWarning,module="angles.*")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    from angles import shortest_angular_distance

def _yaw_from_quaternion(q: Quaternion) -> float:
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw


class ReverseDriveClient:
    """Drives robot in reverse to a staging pose using TF closed-loop feedback.

    Holds no dock identity between runs. The caller selects a dock per run via
    drive_to_staging(dock_name); the staging pose is computed at that point.
    """

    def __init__(self, node: Node, config: ReverseDriveConfig) -> None:
        self.node      = node
        self.config    = config
        self.logger    = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip('/')

        self._status:       ReverseDriveStatus         = ReverseDriveStatus.IDLE
        self._done:         bool                       = False
        self._start_time:   float | None               = None
        self._active_dock:  DockInstanceConfig | None  = None
        self._staging_pose: PoseStamped | None         = None

        # Speed and tolerances directly from config
        self._lin_speed = config.v_linear_min
        self._ang_speed = config.v_angular_max
        self._lin_tol   = config.linear_tolerance
        self._ang_tol   = config.angular_tolerance

        # Timeout: abs(staging_x_offset) / v_linear_min * 1.25
        # max() guards against a zero v_linear_min in config.
        self._timeout = (abs(config.staging_x_offset) / max(config.v_linear_min, 0.01)) * 1.25

        # TF
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

        # Publishers
        self._cmd_vel_pub = self.node.create_publisher(
            TwistStamped, f'{self.namespace}/cmd_vel', 10
        )
        self._staging_pose_pub = self.node.create_publisher(
            PoseStamped, f'{self.namespace}/reverse_nav/staging_pose', 10
        )

        # Control timer — idles until drive_to_staging() is called
        self._control_timer = self.node.create_timer(
            1.0 / config.controller_frequency,
            self._control_loop
        )

        self.logger.info(
            f"ReverseDriveClient Started | "
            f"docks={list(config.dock_configs.keys())} | "
            f"speed={self._lin_speed} m/s | timeout={self._timeout:.1f}s"
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def drive_to_staging(self, dock_name: str) -> bool:
        if self._status == ReverseDriveStatus.REVERSING:
            self.logger.warning("drive_to_staging() ignored — already active")
            return False

        if self.config.dock_backwards:
            self.logger.error(
                "dock_backwards=True — reverse drive unsafe. "
                "Manual intervention required."
            )
            self._status = ReverseDriveStatus.ERROR
            return False

        dock = self.config.dock_configs.get(dock_name)
        if dock is None:
            self.logger.error(
                f"drive_to_staging() — unknown dock '{dock_name}' | "
                f"known={list(self.config.dock_configs.keys())}"
            )
            self._status = ReverseDriveStatus.ERROR
            return False

        self._active_dock  = dock
        self._staging_pose = self._compute_staging_pose(dock)
        self._done         = False
        self._start_time   = None
        self._status       = ReverseDriveStatus.REVERSING
        self.logger.info(f"Reverse drive started | dock='{dock_name}'")
        return True

    def cancel(self) -> None:
        if self._status != ReverseDriveStatus.REVERSING:
            return
        self._shutdown(success=False, msg="Reverse drive canceled")
        self._status = ReverseDriveStatus.CANCELED

    def get_status(self) -> ReverseDriveStatus:
        return self._status

    def is_active(self) -> bool:
        return self._status == ReverseDriveStatus.REVERSING

    def reset(self) -> None:
        self._status       = ReverseDriveStatus.IDLE
        self._done         = False
        self._start_time   = None
        self._active_dock  = None
        self._staging_pose = None
        self.logger.info("ReverseDriveClient reset to IDLE")

    # ------------------------------------------------------------------
    # Staging pose
    # ------------------------------------------------------------------

    def _compute_staging_pose(self, dock: DockInstanceConfig) -> PoseStamped:
        staging_x     = dock.pose.x + math.cos(dock.pose.theta) * self.config.staging_x_offset
        staging_y     = dock.pose.y + math.sin(dock.pose.theta) * self.config.staging_x_offset
        staging_theta = dock.pose.theta + self.config.staging_yaw_offset

        # Converting euler angle to Quaternion
        x, y, z, w = quaternion_from_euler(0.0, 0.0, staging_theta)
        q = Quaternion()
        q.w, q.x, q.y, q.z = float(w), float(x), float(y), float(z)

        pose = PoseStamped()
        pose.header.frame_id  = dock.frame
        pose.header.stamp     = self.node.get_clock().now().to_msg()
        pose.pose.position.x  = staging_x
        pose.pose.position.y  = staging_y
        pose.pose.orientation = q

        self._staging_pose_pub.publish(pose)
        self.logger.info(
            f"Staging pose: ({staging_x:.3f}, {staging_y:.3f}, "
            f"{math.degrees(staging_theta):.1f}deg)"
        )
        return pose

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        if self._status != ReverseDriveStatus.REVERSING or self._staging_pose is None:
            return

        self._staging_pose.header.stamp = self.node.get_clock().now().to_msg()
        self._staging_pose_pub.publish(self._staging_pose)

        robot_pose = self._get_robot_pose(self._staging_pose.header.frame_id)
        if robot_pose is None:
            return

        if self._start_time is None:
            self._start_time = time.monotonic()
            self.logger.info(
                f"TF ready | start: ({robot_pose.pose.position.x:.3f}, "
                f"{robot_pose.pose.position.y:.3f})"
            )

        if time.monotonic() - self._start_time >= self._timeout:
            self._shutdown(success=False, msg=f"Timeout after {self._timeout:.1f}s")
            return

        cmd = Twist()
        if self._compute_command(cmd, self._staging_pose, robot_pose):
            elapsed = time.monotonic() - self._start_time
            self._shutdown(success=True, msg=f"Staging pose reached in {elapsed:.2f}s")
            return

        self._publish_cmd_vel(cmd)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _get_robot_pose(self, frame: str) -> PoseStamped | None:
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
        """Compute reverse cmd_vel toward target. Returns True when goal reached."""
        dist = math.hypot(
            robot_pose.pose.position.x - target.pose.position.x,
            robot_pose.pose.position.y - target.pose.position.y,
        )
        yaw_err = shortest_angular_distance(
            _yaw_from_quaternion(robot_pose.pose.orientation),
            _yaw_from_quaternion(target.pose.orientation),
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

        local_target = tf2_geometry_msgs.do_transform_pose(target.pose, tf)
        lx = local_target.position.x
        ly = local_target.position.y

        heading_error = shortest_angular_distance(math.pi, math.atan2(ly, lx))

        cmd.linear.x  = -self._lin_speed
        cmd.angular.z = max(-self._ang_speed,
                            min(self._ang_speed,
                                self._ang_speed * (heading_error / math.pi)))
        return False

    def _publish_cmd_vel(self, twist: Twist) -> None:
        msg = TwistStamped()
        msg.header.stamp    = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.config.base_frame
        msg.twist           = twist
        self.logger.info(f"Sending msg: {msg}", throttle_duration_sec=1.0)
        self._cmd_vel_pub.publish(msg)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self, success: bool, msg: str) -> None:
        self._done   = True
        self._status = ReverseDriveStatus.DONE if success else ReverseDriveStatus.ERROR
        self._publish_cmd_vel(Twist())
        self.logger.info(msg) if success else self.logger.warning(msg)
