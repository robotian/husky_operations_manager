"""
frame_align.py

Standalone dev/test script for the TF-based target-pose calc. Build
and validate cam_tx/cam_ty/target-pose math here in isolation, without
spinning up the full DriveClient state machine — once validated, copy
_camera_frame_to_odom_tf / _set_target_pose_tf into drive.py.

cam_tx/cam_ty resolved from TF at startup (base_link -> camera_frame). 
bushrow_theta is a hardcoded constant here, not a ROS param — edit BUSHROW_THETA below
for row-angle testing.

Run:
  ros2 run husky_operations_manager frame_align \
    --ros-args -r __ns:=/a200_0284 -r /tf:=tf -r /tf_static:=tf_static
"""

import math

import rclpy
import tf2_ros
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_point
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion

from status_interfaces.msg import ImageDetectionPose

# =============================================================================
# Hardcoded parameters — edit here for testing
# =============================================================================

BUSHROW_THETA = 0.0  # rad — row orientation, hardcoded for now


class DetectionAlignmentNode(Node):
    def __init__(self):
        super().__init__('detection_alignment_node')

        self._declare_parameters()
        self._read_parameters()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._bushrow_theta = BUSHROW_THETA
        self._cam_pose: tuple[float, float, float] | None = None
        self._odom_received: bool = False
        self._current_pose: list[float] = [0.0, 0.0, 0.0]  # x, y, yaw
        self._target_pose_tf: tuple[float, float, float] | None = None
        self._detection_sub = None

        self._tf_wait_timer = self.create_timer(0.2, self._wait_for_tf)
        self.get_logger().info(f'Waiting for TF: {self._base_frame} -> {self._camera_frame}...')

    # =========================================================================
    # PARAMETERS
    # =========================================================================

    def _declare_parameters(self):
        self.declare_parameter(
            'detection_topic',
            'manipulators/arm_detection/image_annotated/detection_pose',
        )
        self.declare_parameter('odom_topic', 'ground_truth/odom')
        self.declare_parameter('camera_frame', 'arm_camera_color_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('odom_frame', 'base_mocap')
        self.declare_parameter('ex_tolerance', 0.01)  # metres
        self.declare_parameter('ey_tolerance', 0.02)  # metres

    def _read_parameters(self):
        self._detection_topic = str(self.get_parameter('detection_topic').value)
        self._odom_topic = str(self.get_parameter('odom_topic').value)
        self._camera_frame = str(self.get_parameter('camera_frame').value)
        self._base_frame = str(self.get_parameter('base_frame').value)
        self._odom_frame = str(self.get_parameter('odom_frame').value)
        self._ex_tolerance = float(self.get_parameter('ex_tolerance').value)
        self._ey_tolerance = float(self.get_parameter('ey_tolerance').value)

    # =========================================================================
    # Startup — resolve cam_pose from TF before subscribing to detections
    # =========================================================================

    def _wait_for_tf(self) -> None:
        """Poll at 0.2s until the camera static transform is available, then resolve cam_pose and start the detection subscription."""
        now = Time()
        if not self._tf_buffer.can_transform(self._base_frame, self._camera_frame, now):
            self.get_logger().info('Waiting for TF...', throttle_duration_sec=1.0)
            return

        self._tf_wait_timer.cancel()
        self._tf_wait_timer = None

        cam_tf = self._tf_buffer.lookup_transform(self._base_frame, self._camera_frame, now)
        cam_tx, cam_ty = cam_tf.transform.translation.x, cam_tf.transform.translation.y
        self._cam_pose = (cam_tx, cam_ty, math.pi)  # camera faces backward
        self.get_logger().info(f'TF resolved | cam_pose={self._cam_pose}')

        # --- Odom subscription ---
        self._odom_sub = self.create_subscription(
            Odometry,
            self._odom_topic,
            self._odom_callback,
            QoSPresetProfiles.SENSOR_DATA.value,
        )

        self._detection_sub = self.create_subscription(
            ImageDetectionPose,
            self._detection_topic,
            self._detection_callback,
            QoSPresetProfiles.SENSOR_DATA.value,
        )

    # =========================================================================
    # Odom callback
    # =========================================================================

    def _odom_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._current_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self._odom_received = True
        
        self.get_logger().info(
            f'ODOM| pose=({self._current_pose[0]:.3f}, '
            f'{self._current_pose[1]:.3f}, {math.degrees(self._current_pose[2]):.1f}deg)',
            throttle_duration_sec=2.0
        )

    # =========================================================================
    # DETECTION CALLBACK
    # =========================================================================

    def _detection_callback(self, msg: ImageDetectionPose):
        if not msg.detection_valid:
            return

        self.get_logger().info(
            f'Detection center | x={msg.center.x:.4f} y={msg.center.y:.4f}'
        )

        if self._odom_received:
            self._set_target_pose(msg.center.x, msg.center.y)
            self._set_target_pose_tf(msg.center.x, msg.center.y)
        else:
            self.get_logger().warn(
                'Skipping, no odom received yet',
                throttle_duration_sec=2.0,
            )

    # =========================================================================
    # Target pose (camera frame -> odom frame, TF-based)
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

    def _camera_frame_to_odom_tf(self, local_x: float, local_y: float) -> tuple[float, float] | None:
        """TF-based camera->odom transform — single lookup + do_transform_point. Returns None on TF failure."""
        point = PointStamped()
        point.header.frame_id = self._camera_frame
        point.header.stamp = Time().to_msg()
        point.point.x = local_x
        point.point.y = local_y
        point.point.z = 0.0

        try:
            transform = self._tf_buffer.lookup_transform(self._odom_frame, self._camera_frame, Time())
        except TransformException as e:
            self.get_logger().warn(
                f"TF lookup failed ('{self._odom_frame}' <- '{self._camera_frame}'): {e}",
                throttle_duration_sec=2.0,
            )
            return None

        odom_point = do_transform_point(point, transform)
        return odom_point.point.x, odom_point.point.y

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
        self.get_logger().info(
            'Manual Target Pose | '
            f'x={target_x:.4f}m  y={target_y:.4f}m  yaw={math.degrees(target_yaw):.2f}deg'
        )

    def _set_target_pose_tf(self, bush_x: float, bush_y: float) -> None:
        """Convert camera-frame bush offsets into an odom-frame target pose — ported from drive.py, no re-lock guard."""
        result = self._camera_frame_to_odom_tf(bush_x, bush_y)
        if result is None:
            return
        cam_x, cam_y = result
        cam_yaw = self._bushrow_theta + self._cam_pose[2]

        target_x = cam_x + self._cam_pose[0] * math.cos(cam_yaw) - self._cam_pose[1] * math.sin(cam_yaw)
        target_y = cam_y + self._cam_pose[0] * math.sin(cam_yaw) + self._cam_pose[1] * math.cos(cam_yaw)
        target_yaw = cam_yaw - self._cam_pose[2]

        self._target_pose_tf = (target_x, target_y, target_yaw)
        self.get_logger().info(
            f'TF Target pose [{self._odom_frame}] | '
            f'x={target_x:.4f}m  y={target_y:.4f}m  yaw={math.degrees(target_yaw):.2f}deg'
        )


def main(args=None):
    rclpy.init(args=args)
    node = DetectionAlignmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
