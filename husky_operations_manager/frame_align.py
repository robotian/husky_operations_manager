"""
detection_alignment_node.py

Computes alignment error (ex, ey) on every valid ImageDetectionPose message.

TF lookup: camera_1_color_optical_frame <- camera_1_detections
  — shortest possible chain (direct parent-child)
  — eliminates static arm chain offset from ey
  — near-zero baseline from calibration (Section 3: Y=-0.017m)

  ex = translation.x  (forward/backward error)
  ey = translation.y * ey_sign  (lateral error)
       ey_sign=+1.0 (default) or -1.0 (flip) — set via YAML after testing

Logic:
  |ex| <= ex_tolerance  → STOP
  ey > +ey_tolerance    → TURN RIGHT (or LEFT if sign flipped)
  ey < -ey_tolerance    → TURN LEFT  (or RIGHT if sign flipped)
"""

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node

import tf2_ros
from tf2_ros import TransformException

from status_interfaces.msg import ImageDetectionPose


class DetectionAlignmentNode(Node):
    """
    Subscribes to ImageDetectionPose and on every valid detection:
      1. Looks up camera_1_detections in camera_1_color_optical_frame
         (direct parent-child — shortest chain, no arm offset)
      2. Computes ex (forward error) and ey (lateral error)
      3. Logs STOP or TURN direction based on error values
    """

    def __init__(self):
        super().__init__('detection_alignment_node')

        self.namespace = self.get_namespace().rstrip('/')

        self._declare_parameters()
        self._read_parameters()

        # --- TF ---
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # --- Subscription ---
        self._detection_sub = self.create_subscription(
            ImageDetectionPose,
            "/j100_0921/sensors/camera_1/detection/image_annotated/detection_pose",
            self._detection_callback,
            10,
        )

        self.get_logger().info(
            f'DetectionAlignmentNode ready | '
            f"source='{self._source_frame}' | "
            f"detection='{self._detection_frame}' | "
            f'ex_tolerance={self._ex_tolerance:.4f}m | '
            f'ey_tolerance={self._ey_tolerance:.4f}m | '
            f'ey_sign={self._ey_sign:+.1f} | '
            f'subscribed to: {self._detection_sub.topic_name}'
        )

    # =========================================================================
    # PARAMETERS
    # =========================================================================

    def _declare_parameters(self):
        # Option B: direct parent-child lookup eliminates arm chain offset
        self.declare_parameter('source_frame', 'camera_1_color_optical_frame')
        self.declare_parameter('detection_frame', 'camera_1_detections')
        self.declare_parameter('ex_tolerance', 0.01)  # metres
        self.declare_parameter('ey_tolerance', 0.02)  # metres
        # +1.0 = default convention, -1.0 = flip if steering is inverted on robot
        self.declare_parameter('ey_sign', -1.0)

    def _read_parameters(self):
        self._source_frame = str(self.get_parameter('source_frame').value)
        self._detection_frame = str(self.get_parameter('detection_frame').value)
        self._ex_tolerance = float(self.get_parameter('ex_tolerance').value)
        self._ey_tolerance = float(self.get_parameter('ey_tolerance').value)
        self._ey_sign = float(self.get_parameter('ey_sign').value)

        self.get_logger().info(
            f'Parameters loaded | '
            f"source='{self._source_frame}' | "
            f"detection='{self._detection_frame}' | "
            f'ex_tolerance={self._ex_tolerance:.4f}m | '
            f'ey_tolerance={self._ey_tolerance:.4f}m | '
            f'ey_sign={self._ey_sign:+.1f}'
        )

    # =========================================================================
    # DETECTION CALLBACK
    # =========================================================================

    def _detection_callback(self, msg: ImageDetectionPose):
        """
        Fires on every ImageDetectionPose message.
        Ignores invalid detections. Computes ex/ey on valid ones.
        """
        if not msg.detection_valid:
            return

        tf = self._lookup_detection()
        if tf is None:
            return

        ex = tf.transform.translation.x
        # Apply sign convention — determined by testing on robot
        ey = tf.transform.translation.y * self._ey_sign

        self.get_logger().info(f'Detection alignment | ex={ex:+.4f}m  ey={ey:+.4f}m (sign={self._ey_sign:+.1f})')

        self._evaluate_ex(ex)
        self._evaluate_ey(ey)

    # =========================================================================
    # ALIGNMENT EVALUATION
    # =========================================================================

    def _evaluate_ex(self, ex: float):
        """
        Log STOP if ex is within tolerance of zero.
        ex ≈ 0 → detection frame is aligned with source frame on X axis.
        """
        if abs(ex) <= self._ex_tolerance:
            self.get_logger().info(
                f'STOP | ex={ex:+.4f}m within tolerance={self._ex_tolerance:.4f}m — aligned on X axis'
            )
        else:
            self.get_logger().debug(f'Moving | ex={ex:+.4f}m exceeds tolerance={self._ex_tolerance:.4f}m')

    def _evaluate_ey(self, ey: float):
        """
        Log turn direction based on signed ey.

        Convention (with ey_sign=+1.0, adjust via YAML if inverted):
          ey > +ey_tolerance → TURN RIGHT
          ey < -ey_tolerance → TURN LEFT
          |ey| <= ey_tolerance → LATERAL ALIGNED
        """
        if abs(ey) <= self._ey_tolerance:
            self.get_logger().info(f'LATERAL ALIGNED | ey={ey:+.4f}m within tolerance={self._ey_tolerance:.4f}m')
        elif ey > 0:
            self.get_logger().info(f'TURN RIGHT | ey={ey:+.4f}m — bush is to the right')
        else:
            self.get_logger().info(f'TURN LEFT  | ey={ey:+.4f}m — bush is to the left')

    # =========================================================================
    # TF LOOKUP
    # =========================================================================

    def _lookup_detection(self):
        """
        Direct parent-child lookup:
          camera_1_detections expressed in camera_1_color_optical_frame.

        Equivalent to: tf2_echo camera_1_color_optical_frame camera_1_detections
          → lookup_transform(
                target_frame='camera_1_detections',
                source_frame='camera_1_color_optical_frame',
            )

        Shortest possible chain — eliminates static arm offset from ey.
        Returns TransformStamped or None on failure.
        """
        try:
            return self._tf_buffer.lookup_transform(
                self._detection_frame,  # target_frame
                self._source_frame,  # source_frame
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException as e:
            self.get_logger().warn(
                f"TF lookup failed | '{self._detection_frame}' ← '{self._source_frame}': {e}",
                throttle_duration_sec=5.0,
            )
            return None


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
