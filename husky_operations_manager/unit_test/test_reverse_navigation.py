"""Reverse navigation node — drives the robot backwards to a staging pose."""
 
import math
import time
 
import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.duration import Duration
 
import tf2_ros
 
from geometry_msgs.msg import Twist, PoseStamped, Pose, Quaternion, TwistStamped

# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _yaw_from_quaternion(q: Quaternion) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))
 
def _quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q
 
def _angular_distance(from_yaw: float, to_yaw: float) -> float:
    return math.atan2(math.sin(to_yaw - from_yaw), math.cos(to_yaw - from_yaw))


# ── Node ─────────────────────────────────────────────────────────────────────

class ReverseDriveClient(Node):

    def __init__(self) -> None:
        super().__init__("reverse_navigation_node")
        self.logger = RcutilsLogger(self.__class__.__name__)
 
        self.declare_parameter("goal_x",             0.5)
        self.declare_parameter("goal_y",             0.4)
        self.declare_parameter("goal_yaw",           0.0)
        self.declare_parameter("staging_x_offset",  -0.5)
        self.declare_parameter("staging_yaw_offset", 0.0)
        self.declare_parameter("fixed_frame",        "map")
        self.declare_parameter("base_frame",         "base_link")
        self.declare_parameter("linear_speed",       0.15)
        self.declare_parameter("angular_speed",      0.3)
        self.declare_parameter("linear_tolerance",   0.05)
        self.declare_parameter("angular_tolerance",  0.05)
        self.declare_parameter("timeout",            30.0)
        self.declare_parameter("cmd_vel_topic",      "cmd_vel")
        self.declare_parameter("control_frequency",  20.0)
 
        self._goal_x             = self.get_parameter("goal_x").value
        self._goal_y             = self.get_parameter("goal_y").value
        self._goal_yaw           = self.get_parameter("goal_yaw").value
        self._staging_x_offset   = self.get_parameter("staging_x_offset").value
        self._staging_yaw_offset = self.get_parameter("staging_yaw_offset").value
        self._fixed_frame        = self.get_parameter("fixed_frame").value
        self._base_frame         = self.get_parameter("base_frame").value
        self._lin_speed          = abs(self.get_parameter("linear_speed").value)
        self._ang_speed          = abs(self.get_parameter("angular_speed").value)
        self._lin_tol            = self.get_parameter("linear_tolerance").value
        self._ang_tol            = self.get_parameter("angular_tolerance").value
        self._timeout            = self.get_parameter("timeout").value
        cmd_vel_topic            = self.get_parameter("cmd_vel_topic").value
        ctrl_freq                = self.get_parameter("control_frequency").value
 
        self._done:       bool         = False
        self._start_time: float | None = None
 
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
 
        # Prefix base_frame with namespace to match Nav2 TF frame names
        ns = self.get_namespace().rstrip("/")
        self._cmd_vel_pub = self.create_publisher(TwistStamped, f"{ns}/{cmd_vel_topic}", 10)
        self._staging_pose_pub = self.create_publisher( PoseStamped, f"{ns}/reverse_nav/staging_pose", 10)
        self._control_timer = self.create_timer(1.0 / ctrl_freq, self._control_loop)
 
        # Compute and latch the staging pose (publisher must exist first)
        dock_pose = Pose()
        dock_pose.position.x  = self._goal_x
        dock_pose.position.y  = self._goal_y
        dock_pose.orientation = _quaternion_from_yaw(self._goal_yaw)
        self._staging_pose = self._get_staging_pose(dock_pose, self._fixed_frame)
 
        self.logger.info(
            f"Ready | goal: ({self._goal_x:.3f}, {self._goal_y:.3f}, "
            f"{math.degrees(self._goal_yaw):.1f}deg) | "
            f"staging: ({self._staging_pose.pose.position.x:.3f}, "
            f"{self._staging_pose.pose.position.y:.3f}) | "
            f"base_frame: {self._base_frame} | timeout: {self._timeout:.0f} s"
        )
    
    # ── Staging Pose ────────────────────────────

    def _get_staging_pose(self, dock_pose: Pose, fixed_frame: str) -> PoseStamped:
        yaw = _yaw_from_quaternion(dock_pose.orientation)
 
        staging_pose = PoseStamped()
        staging_pose.header.frame_id  = fixed_frame
        staging_pose.header.stamp     = self.get_clock().now().to_msg()
        staging_pose.pose.position.x  = dock_pose.position.x + math.cos(yaw) * self._staging_x_offset
        staging_pose.pose.position.y  = dock_pose.position.y + math.sin(yaw) * self._staging_x_offset
        staging_pose.pose.orientation = _quaternion_from_yaw(yaw + self._staging_yaw_offset)
 
        self._staging_pose_pub.publish(staging_pose)
        self.logger.info(
            f"Staging pose: ({staging_pose.pose.position.x:.3f}, "
            f"{staging_pose.pose.position.y:.3f}), "
            f"yaw: {math.degrees(yaw + self._staging_yaw_offset):.1f}deg"
        )
        return staging_pose

    # ── Robot Pose ────────────────────────────

    def _get_robot_pose_in_frame(self, fixed_frame: str) -> PoseStamped | None:
        try:
            robot_pose_tf = self._tf_buffer.lookup_transform(
                fixed_frame,
                self._base_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )

            robot_pose = PoseStamped()
            robot_pose.header.frame_id = self._base_frame
            robot_pose.header.stamp    = Time().to_msg()
            robot_pose.pose.position.x    = robot_pose_tf.transform.translation.x
            robot_pose.pose.position.y    = robot_pose_tf.transform.translation.y
            robot_pose.pose.orientation   = robot_pose_tf.transform.rotation

            return robot_pose
        except Exception as e:
            self.logger.warning(f"get_robot_pose_in_frame failed: {e}",
                                throttle_duration_sec=2.0)
            return None
        
    def _transform_pose_to_frame(
        self, pose: PoseStamped, target_frame: str
    ) -> PoseStamped | None:
        """Transform a PoseStamped into target_frame via lookup_transform (2-D)."""
        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame, 
                pose.header.frame_id,
                Time(), 
                timeout=Duration(seconds=0.1),
            )

            yaw   = _yaw_from_quaternion(tf.transform.rotation)
            t     = tf.transform.translation
            px, py = pose.pose.position.x, pose.pose.position.y
    
            result = PoseStamped()
            result.header.frame_id    = target_frame
            result.header.stamp       = tf.header.stamp
            result.pose.position.x    = math.cos(yaw) * px - math.sin(yaw) * py + t.x
            result.pose.position.y    = math.sin(yaw) * px + math.cos(yaw) * py + t.y
            result.pose.orientation   = _quaternion_from_yaw(
                yaw + _yaw_from_quaternion(pose.pose.orientation)
            )
            return result

        except Exception as e:
            self.logger.error(f"transform_pose_to_frame failed: {e}")
            return None

    # ── Velocity Command ──────────────────────────────────────────

    def _get_command_to_pose(
        self,
        cmd: Twist,
        target_pose: PoseStamped,
        linear_tolerance: float,
        angular_tolerance: float,
    ) -> bool:
        cmd.linear.x = cmd.angular.z = 0.0
 
        robot_pose = self._get_robot_pose_in_frame(target_pose.header.frame_id)
        if robot_pose is None:
            return False
 
        dist = math.hypot(
            robot_pose.pose.position.x - target_pose.pose.position.x,
            robot_pose.pose.position.y - target_pose.pose.position.y,
        )
        yaw_err = _angular_distance(
            _yaw_from_quaternion(robot_pose.pose.orientation),
            _yaw_from_quaternion(target_pose.pose.orientation),
        )
 
        if dist < linear_tolerance and abs(yaw_err) < angular_tolerance:
            return True
 
        local = self._transform_pose_to_frame(target_pose, self._base_frame)
        if local is None:
            return False
 
        heading_error = _angular_distance(
            math.pi, math.atan2(local.pose.position.y, local.pose.position.x)
        )
        cmd.linear.x  = -self._lin_speed
        cmd.angular.z = max(-self._ang_speed,
                            min(self._ang_speed,
                                self._ang_speed * (heading_error / math.pi)))
 
        self.logger.debug(
            f"dist: {dist:.3f} m | yaw_err: {math.degrees(yaw_err):.1f}deg | "
            f"lin: {cmd.linear.x:.3f} | ang: {cmd.angular.z:.3f}"
        )
        return False
    
    def _publish_cmd_vel(self, twist: Twist) -> None:
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.twist           = twist
        self._cmd_vel_pub.publish(msg)
        # self.get_logger().info(f"Publishing to {self._cmd_vel_pub.topic_name}: {twist}")

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        if self._done:
            return
 
        self._staging_pose.header.stamp = self.get_clock().now().to_msg()
        self._staging_pose_pub.publish(self._staging_pose)
 
        robot_pose = self._get_robot_pose_in_frame(self._staging_pose.header.frame_id)
        if robot_pose is None:
            return
 
        if self._start_time is None:
            self._start_time = time.monotonic()
            self.logger.info(
                f"TF ready — start: ({robot_pose.pose.position.x:.3f}, "
                f"{robot_pose.pose.position.y:.3f}) | timeout: {self._timeout:.0f} s"
            )
 
        elapsed = time.monotonic() - self._start_time
 
        if elapsed >= self._timeout:
            self._shutdown(success=False, msg=f"Timeout after {elapsed:.1f} s")
            return
 
        cmd = Twist()
        
        reached = self._get_command_to_pose(
            cmd, self._staging_pose, self._lin_tol, self._ang_tol
        )
 
        if reached:
            self._shutdown(success=True, msg=f"Goal reached in {elapsed:.2f} s")
 
        # Only publish if goal not yet reached — _done flag stops further publishing
        if not self._done:
            self._publish_cmd_vel(cmd)       

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _shutdown(self, success: bool, msg: str) -> None:
        self._done = True
        self._publish_cmd_vel(Twist())
        self.logger.info(msg) if success else self.logger.warning(msg)
        raise SystemExit

# ── Entry point ──────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ReverseDriveClient()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node._publish_cmd_vel(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()