"""
frame_align.py

Computes desired robot pose to center arm_0 detection in camera FOV.

TF lookups:
  1. arm_0_camera_color_frame -> arm_0_detections  : detection offset in camera frame
  2. base_link -> arm_0_camera_color_frame          : rotate offset into base_link axes
  3. odom -> base_link (optional)                   : current robot pose in world frame

Desired pose:
  odom available  : x = odom_x + ex_bl, y = odom_y + ey_bl, yaw = current heading (odom frame)
  odom broken     : x = ex_bl, y = ey_bl, yaw = 0.0 (relative to base_link)

Switches to odom automatically when odom -> base_link becomes available. No code change needed.
"""

import math

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node

import tf2_ros
from tf2_ros import TransformException

from rclpy.qos import QoSPresetProfiles
from status_interfaces.msg import ImageDetectionPose


class DetectionAlignmentNode(Node):
    def __init__(self):
        super().__init__('detection_alignment_node')

        self._declare_parameters()
        self._read_parameters()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._detection_sub = self.create_subscription(
            ImageDetectionPose,
            self._detection_topic,
            self._detection_callback,
            QoSPresetProfiles.SENSOR_DATA.value,
        )

        self.get_logger().info(
            f'DetectionAlignmentNode ready | '
            f"topic='{self._detection_topic}' | "
            f"camera='{self._camera_frame}' | "
            f"detection='{self._detection_frame}' | "
            f"robot='{self._robot_frame}' | "
            f"odom='{self._odom_frame}'"
        )

    # =========================================================================
    # PARAMETERS
    # =========================================================================

    def _declare_parameters(self):
        self.declare_parameter(
            'detection_topic',
            '/a200_0284/manipulators/arm_0_detection/image_annotated/detection_pose',
        )
        self.declare_parameter('camera_frame', 'arm_0_camera_color_frame')
        self.declare_parameter('detection_frame', 'arm_0_camera_detections')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('ex_tolerance', 0.01)  # metres
        self.declare_parameter('ey_tolerance', 0.02)  # metres

    def _read_parameters(self):
        self._detection_topic = str(self.get_parameter('detection_topic').value)
        self._camera_frame = str(self.get_parameter('camera_frame').value)
        self._detection_frame = str(self.get_parameter('detection_frame').value)
        self._robot_frame = str(self.get_parameter('robot_frame').value)
        self._odom_frame = str(self.get_parameter('odom_frame').value)
        self._ex_tolerance = float(self.get_parameter('ex_tolerance').value)
        self._ey_tolerance = float(self.get_parameter('ey_tolerance').value)

    # =========================================================================
    # DETECTION CALLBACK
    # =========================================================================

    def _detection_callback(self, msg: ImageDetectionPose):
        if not msg.detection_valid:
            return

        # 1. Detection offset in camera frame
        tf_cam = self._lookup_tf(self._camera_frame, self._detection_frame)
        if tf_cam is None:
            return
        ex_cam = tf_cam.transform.translation.x
        ey_cam = tf_cam.transform.translation.y

        # 2. Rotate offset into base_link axes
        tf_base_to_cam = self._lookup_tf(self._robot_frame, self._camera_frame)
        if tf_base_to_cam is None:
            return
        ex_bl, ey_bl = self._rotate_2d(ex_cam, ey_cam, tf_base_to_cam.transform.rotation)

        # 3. Desired pose — use odom if available, fall back to base_link
        tf_odom = self._try_lookup_tf(self._odom_frame, self._robot_frame)

        if tf_odom is not None:
            desired_x = tf_odom.transform.translation.x + ex_bl
            desired_y = tf_odom.transform.translation.y + ey_bl
            desired_yaw = self._yaw_from_quat(tf_odom.transform.rotation)
            frame_label = self._odom_frame
        else:
            self.get_logger().warn(
                f"'{self._odom_frame}' → '{self._robot_frame}' unavailable — "
                f"desired pose in '{self._robot_frame}' frame",
                throttle_duration_sec=5.0,
            )
            desired_x = ex_bl
            desired_y = ey_bl
            desired_yaw = 0.0
            frame_label = self._robot_frame

        self.get_logger().info(
            f'Desired pose [{frame_label}] | '
            f'x={desired_x:.4f}m  '
            f'y={desired_y:.4f}m  '
            f'yaw={math.degrees(desired_yaw):.2f}deg'
        )

    # =========================================================================
    # TF HELPERS
    # =========================================================================

    def _lookup_tf(self, target: str, source: str):
        """Lookup TF, log warn on failure."""
        try:
            return self._tf_buffer.lookup_transform(
                target,
                source,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException as e:
            self.get_logger().warn(
                f"TF lookup failed | '{target}' ← '{source}': {e}",
                throttle_duration_sec=5.0,
            )
            return None

    def _try_lookup_tf(self, target: str, source: str):
        """Lookup TF silently — returns None without warning on failure."""
        try:
            return self._tf_buffer.lookup_transform(
                target,
                source,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except TransformException:
            return None

    # =========================================================================
    # MATH HELPERS
    # =========================================================================

    @staticmethod
    def _rotate_2d(x: float, y: float, q) -> tuple:
        """Rotate 2D vector (x, y) by quaternion. Z ignored (downward camera)."""
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        rx = (1 - 2 * (qy * qy + qz * qz)) * x + (2 * (qx * qy - qw * qz)) * y
        ry = (2 * (qx * qy + qw * qz)) * x + (1 - 2 * (qx * qx + qz * qz)) * y
        return rx, ry

    @staticmethod
    def _yaw_from_quat(q) -> float:
        """Extract yaw (Z rotation) from quaternion."""
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main(args=None):
    rclpy.init(args=args)
    node = DetectionAlignmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
