"""
Lavender Detection ROS 2 Node (Jazzy / Ubuntu 24.04)
=====================================================
Single-file, 4-class design:
  1. Config          - all tunable variables (no ROS, no torch imports)
  2. DepthUtils      - static 3-D geometry helpers (no RealSense SDK)
  3. DetectionModel  - model load + inference wrapper
  4. LavenderDetection(Node) - ROS 2 node

Deprecated-method notes
-----------------------
  torch.load(...) without weights_only= is deprecated since PyTorch 2.0 and
  raises a FutureWarning.  We pass weights_only=True explicitly (safe because
  we are loading only a state-dict, not a full pickled object).
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import models, transforms

# ROS 2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ROS 2 messages
from sensor_msgs.msg import CameraInfo, Image

# cv_bridge: converts sensor_msgs/Image <-> numpy array
# apt: ros-jazzy-cv-bridge
from cv_bridge import CvBridge
import cv2

from ament_index_python import get_package_share_directory


# ══════════════════════════════════════════════════════════════════════════════
# 1. Config
#    All tunable variables live here as plain attributes.
#    To migrate to a YAML config file later, replace the defaults with values
#    loaded from rclpy parameters or a yaml.safe_load() call — no other file
#    needs to change.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Model ────────────────────────────────────────────────────────────────
    # Path to the trained checkpoint.
    # Override at runtime via the ROS 2 parameter 'model_path' or edit here.
    model_path: str = os.path.join(
        get_package_share_directory("husky_operations_manager"),
        "data",
        "best_model.pth",
    )

    # Number of classes the model was trained with (background + lavender = 2)
    n_classes: int = 2

    # Minimum confidence score to accept a detection
    score_threshold: float = 0.8

    # PyTorch device — auto-selects CUDA if available, otherwise CPU
    # Python 3.10+ / PyTorch 2.x compatible
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Timer ────────────────────────────────────────────────────────────────
    # Inference runs at this rate (Hz). Increase if your hardware allows.
    timer_hz: float = 0.5 

    # ── Topics (relative to namespace) ───────────────────────────────────────
    color_image_topic: str    = "manipulators/arm_0_color_camera/image_raw"
    depth_image_topic: str    = "manipulators/arm_0_depth_camera/image_raw"
    camera_info_topic: str    = "manipulators/arm_0_color_camera/camera_info"
    annotated_image_topic: str = "manipulators/arm_0_detection/image_annotated"

    # ── Depth helpers ─────────────────────────────────────────────────────────
    # Window size (pixels) for median depth sampling around a pixel
    depth_window: int = 7

    # Minimum valid depth readings required inside a window
    depth_min_valid: int = 20

    # Pixels to inset bbox corners inward before sampling depth
    # (avoids picking up background depth at the very edge of the box)
    corner_inset_px: int = 5

    # Step size (pixels) for the bbox 3-D extent scan
    bbox_scan_step: int = 2

    # Minimum 3-D points required before computing extents
    bbox_min_points: int = 10

    # Percentile bounds used when computing bbox extents (removes outliers)
    extent_percentile_low: float  = 5.0
    extent_percentile_high: float = 95.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. DepthUtils
#    Pure-static geometry helpers.
#    All methods accept a plain IntrinsicsLike namedtuple (see below) so the
#    class works with any camera — RealSense, USB, RTSP, RPi — as long as
#    CameraInfo is available on ROS 2.
#    No pyrealsense2 import required.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Intrinsics:
    """
    Minimal pinhole camera intrinsics.
    Populated from sensor_msgs/CameraInfo (K matrix):
        K = [fx,  0, cx,
              0, fy, cy,
              0,  0,  1]
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @staticmethod
    def from_camera_info(msg: CameraInfo) -> "Intrinsics":
        """
        Build an Intrinsics from a sensor_msgs/CameraInfo message.

        CameraInfo.K is a row-major 3x3 matrix stored as a flat list of 9
        floats:  [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        """
        return Intrinsics(
            fx=float(msg.k[0]),
            fy=float(msg.k[4]),
            cx=float(msg.k[2]),
            cy=float(msg.k[5]),
            width=int(msg.width),
            height=int(msg.height),
        )


class DepthUtils:
    """
    Static 3-D geometry helpers.
    Replaces all pyrealsense2 deprojection calls with standard pinhole math:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """

    @staticmethod
    def deproject_pixel(
        u: float,
        v: float,
        depth_m: float,
        intr: Intrinsics,
    ) -> Tuple[float, float, float]:
        """
        Back-project a 2-D pixel + depth into a 3-D camera-frame point.

        Parameters
        ----------
        u, v     : pixel coordinates (column, row)
        depth_m  : depth in metres
        intr     : camera intrinsics

        Returns
        -------
        (X, Y, Z) in metres in the camera optical frame
          X: rightward, Y: downward, Z: into the scene
        """
        X = (u - intr.cx) * depth_m / intr.fx
        Y = (v - intr.cy) * depth_m / intr.fy
        return float(X), float(Y), float(depth_m)

    @staticmethod
    def robust_depth_at_pixel(
        depth_image: np.ndarray,
        u: int,
        v: int,
        depth_scale: float,
        box: Optional[Tuple[int, int, int, int]] = None,
        win: int = 7,
        min_valid: int = 20,
    ) -> Optional[float]:
        """
        Return the median depth (metres) inside a win×win window centred on
        (u, v).  Zero-valued pixels are treated as invalid.

        Parameters
        ----------
        depth_image : uint16 numpy array (raw depth units)
        u, v        : pixel coordinates (column, row)
        depth_scale : metres per raw depth unit
                      For RealSense: depth_sensor.get_depth_scale()
                      For ROS depth images (16UC1): typically 0.001 (mm→m)
        box         : optional (x1,y1,x2,y2) to restrict sampling inside bbox
        win         : window diameter in pixels
        min_valid   : minimum non-zero pixels required; returns None otherwise

        Returns
        -------
        Depth in metres, or None if insufficient valid readings.
        """
        h, w = depth_image.shape
        half = win // 2

        u0, u1 = max(0, u - half), min(w, u + half + 1)
        v0, v1 = max(0, v - half), min(h, v + half + 1)

        patch = depth_image[v0:v1, u0:u1].copy()

        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            uu = np.arange(u0, u1)
            vv = np.arange(v0, v1)
            U, V = np.meshgrid(uu, vv)
            mask_inside = (U >= x1) & (U < x2) & (V >= y1) & (V < y2)
            patch = patch[mask_inside]

        patch = patch[patch > 0]
        if patch.size < min_valid:
            return None

        return float(np.median(patch)) * depth_scale

    @staticmethod
    def pixel_to_3d_camera(
        u: float,
        v: float,
        depth_image: np.ndarray,
        intr: Intrinsics,
        depth_scale: float,
        box: Optional[Tuple[int, int, int, int]] = None,
        win: int = 7,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Deproject pixel (u, v) to (X, Y, Z) in the camera optical frame.

        Returns
        -------
        (X, Y, Z, depth_m) in metres, or None if depth is invalid.
        """
        depth_m = DepthUtils.robust_depth_at_pixel(
            depth_image, int(u), int(v), depth_scale, box=box, win=win
        )
        if depth_m is None:
            return None

        X, Y, Z = DepthUtils.deproject_pixel(u, v, depth_m, intr)
        return float(X), float(Y), float(Z), float(depth_m)

    @staticmethod
    def bbox_keypoints_3d_camera(
        box: Tuple[int, int, int, int],
        depth_image: np.ndarray,
        intr: Intrinsics,
        depth_scale: float,
        win: int = 7,
        inset_px: int = 5,
    ) -> dict:
        """
        Compute 3-D camera-frame points for the centre + 4 corners of a bbox.

        Corner pixels are inset inward by inset_px to avoid sampling depth
        from the background just outside the object.

        Returns
        -------
        dict with keys: 'center', 'tl', 'tr', 'bl', 'br'
        Each value is (X, Y, Z, depth_m) in metres, or None if invalid.
        """
        x1, y1, x2, y2 = map(int, box)

        x1i = x1 + inset_px
        y1i = y1 + inset_px
        x2i = x2 - inset_px
        y2i = y2 - inset_px

        uc = int(0.5 * (x1 + x2))
        vc = int(0.5 * (y1 + y2))

        pts_px = {
            "center": (uc, vc),
            "tl":     (x1i, y1i),
            "tr":     (x2i, y1i),
            "bl":     (x1i, y2i),
            "br":     (x2i, y2i),
        }

        return {
            k: DepthUtils.pixel_to_3d_camera(
                u, v, depth_image, intr, depth_scale, box=box, win=win
            )
            for k, (u, v) in pts_px.items()
        }

    @staticmethod
    def bbox_3d_extents(
        box: Tuple[float, float, float, float],
        depth_image: np.ndarray,
        intr: Intrinsics,
        depth_scale: float,
        step: int = 2,
        min_points: int = 10,
        pct_low: float = 5.0,
        pct_high: float = 95.0,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Estimate 3-D physical extents (width, height, depth) of the object
        inside a bounding box by deprojecting every step-th pixel.

        Returns
        -------
        (X_size, Y_size, Z_size) in metres, or (None, None, None).
        """
        x1, y1, x2, y2 = map(int, box)
        h_img, w_img = depth_image.shape

        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        if x2 <= x1 or y2 <= y1:
            return None, None, None

        xs, ys, zs = [], [], []

        for v in range(y1, y2, step):
            for u in range(x1, x2, step):
                z_raw = depth_image[v, u]
                if z_raw == 0:
                    continue
                Z = float(z_raw) * depth_scale
                X, Y, _ = DepthUtils.deproject_pixel(u, v, Z, intr)
                xs.append(X)
                ys.append(Y)
                zs.append(Z)

        if len(xs) < min_points:
            return None, None, None

        xs_arr = np.array(xs)
        ys_arr = np.array(ys)
        zs_arr = np.array(zs)

        x_min, x_max = np.percentile(xs_arr, [pct_low, pct_high])
        y_min, y_max = np.percentile(ys_arr, [pct_low, pct_high])
        z_min, z_max = np.percentile(zs_arr, [pct_low, pct_high])

        return (x_max - x_min), (y_max - y_min), (z_max - z_min)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DetectionModel
#    Loads the Faster R-CNN checkpoint and wraps inference.
#    Isolated from ROS 2 — can be unit-tested independently.
# ══════════════════════════════════════════════════════════════════════════════

class DetectionModel:
    """
    Loads a Faster R-CNN ResNet-50 FPN model from a checkpoint file and
    exposes a single run() method for inference on a PIL image.

    Deprecated-method fix
    ---------------------
    torch.load() without weights_only= emits a FutureWarning in PyTorch ≥ 2.0
    and will change default behaviour in a future release.
    We pass weights_only=True explicitly — this is safe here because the
    checkpoint contains only a state-dict (tensors + primitive Python types),
    not arbitrary pickled Python objects.

    Compatible with Python 3.10+ and PyTorch 2.x.
    """

    def __init__(self, cfg: Config) -> None:
        self.device = torch.device(cfg.device)
        self.score_threshold = cfg.score_threshold
        self._to_tensor = transforms.ToTensor()

        # Build model skeleton
        model = models.detection.fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, cfg.n_classes
            )
        )

        # ── Deprecated fix ────────────────────────────────────────────────
        # Old (PyTorch < 2.0, deprecated):
        #   checkpoint = torch.load(path, map_location=device)
        #
        # New (PyTorch ≥ 2.0, works on both versions):
        #   checkpoint = torch.load(path, map_location=device, weights_only=True)
        #
        # weights_only=True restricts unpickling to safe tensor/primitive types,
        # preventing arbitrary code execution from untrusted checkpoint files.
        # ─────────────────────────────────────────────────────────────────
        checkpoint = torch.load(
            cfg.model_path,
            map_location=self.device,
            weights_only=True,        # ← replaces deprecated default
        )
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()

        self._model = model

    def run(
        self, pil_img: PILImage.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a single PIL RGB image.

        Returns
        -------
        boxes  : FloatTensor shape (N, 4)  — [x1, y1, x2, y2] in pixels
        scores : FloatTensor shape (N,)
        Both tensors are on CPU.
        """
        img_tensor = self._to_tensor(pil_img).to(self.device)

        with torch.no_grad():
            outputs = self._model([img_tensor])[0]

        return outputs["boxes"].cpu(), outputs["scores"].cpu()


# ══════════════════════════════════════════════════════════════════════════════
# 4. LavenderDetection  (ROS 2 Node)
# ══════════════════════════════════════════════════════════════════════════════

class LavenderDetection(Node):
    """
    ROS 2 node that:
      • Subscribes to colour image, depth image, and CameraInfo topics.
      • On every timer tick, runs detection + 3-D measurement if all three
        messages have been received at least once.
      • Publishes an annotated sensor_msgs/Image with bounding boxes drawn.
      • Logs 3-D keypoint data via self.get_logger() (no OpenCV window).

    Depth image convention
    ----------------------
    Expects a 16UC1 depth image where each value is depth in millimetres
    (standard for Intel RealSense ROS 2 driver and most depth cameras).
    depth_scale = 0.001  →  raw_value × 0.001 = metres.
    Change DEPTH_SCALE below if your camera uses a different unit.
    """

    # Depth scale: metres per raw uint16 unit.
    # 0.001 = millimetres (standard ROS 2 depth image convention).
    DEPTH_SCALE: float = 0.001

    def __init__(self, cfg: Config) -> None:
        super().__init__("lavender_detection")

        self._cfg = cfg
        self._bridge = CvBridge()

        # ── ROS 2 parameter: namespace ────────────────────────────────────
        self.declare_parameter("namespace", cfg.namespace if hasattr(cfg, "namespace") else "")
        self._namespace: str = (
            self.get_parameter("namespace")
            .get_parameter_value()
            .string_value
        )

        # ── Cached latest messages (None until first message arrives) ─────
        self._color_camera_info: Optional[CameraInfo] = None
        self._color_image: Optional[Image]            = None
        self._depth_image: Optional[Image]            = None

        # ── Lazy-initialised intrinsics (built once from first CameraInfo) ─
        self._intrinsics: Optional[Intrinsics] = None

        # ── Load detection model ──────────────────────────────────────────
        self.get_logger().info(f"Loading model from: {cfg.model_path}")
        self._detector = DetectionModel(cfg)
        self.get_logger().info("Model loaded.")

        # ── Topic names with namespace prefix ────────────────────────────
        ns = self._namespace.rstrip("/")

        def full_topic(relative: str) -> str:
            return f"{ns}/{relative}" if ns else relative

        color_topic  = full_topic(cfg.color_image_topic)
        depth_topic  = full_topic(cfg.depth_image_topic)
        info_topic   = full_topic(cfg.camera_info_topic)
        output_topic = full_topic(cfg.annotated_image_topic)

        # ── QoS: best-effort matches most camera drivers ──────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, info_topic, self._color_camera_info_callback, qos
        )
        self.create_subscription(
            Image, color_topic, self._color_image_callback, qos
        )
        self.create_subscription(
            Image, depth_topic, self._depth_image_callback, qos
        )

        # ── Publisher ─────────────────────────────────────────────────────
        self._annotated_pub = self.create_publisher(Image, output_topic, 10)

        # ── Timer ─────────────────────────────────────────────────────────
        timer_period = 1.0 / cfg.timer_hz
        self.create_timer(timer_period, self._timer_callback)

        self.get_logger().info(
            f"LavenderDetection started | namespace='{self._namespace}' | "
            f"timer={cfg.timer_hz} Hz"
        )
        self.get_logger().info(f"  colour  → {color_topic}")
        self.get_logger().info(f"  depth   → {depth_topic}")
        self.get_logger().info(f"  info    → {info_topic}")
        self.get_logger().info(f"  output  ← {output_topic}")

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    def _color_camera_info_callback(self, msg: CameraInfo) -> None:
        self._color_camera_info = msg
        # Build intrinsics once; CameraInfo is static for a fixed camera
        if self._intrinsics is None:
            self._intrinsics = Intrinsics.from_camera_info(msg)
            self.get_logger().info(
                f"Intrinsics received: fx={self._intrinsics.fx:.2f}, "
                f"fy={self._intrinsics.fy:.2f}, "
                f"cx={self._intrinsics.cx:.2f}, "
                f"cy={self._intrinsics.cy:.2f}"
            )

    def _color_image_callback(self, msg: Image) -> None:
        self._color_image = msg

    def _depth_image_callback(self, msg: Image) -> None:
        self._depth_image = msg

    # ── Timer callback (main processing loop) ─────────────────────────────────

    def _timer_callback(self) -> None:
        """
        Called at cfg.timer_hz.  Guards that all three data sources are ready
        before running inference.
        """
        if self._color_image is None:
            self.get_logger().warn("Waiting for colour image…", throttle_duration_sec=5.0)
            return
        if self._depth_image is None:
            self.get_logger().warn("Waiting for depth image…", throttle_duration_sec=5.0)
            return
        if self._intrinsics is None:
            self.get_logger().warn("Waiting for CameraInfo…", throttle_duration_sec=5.0)
            return

        # ── Convert ROS images to numpy ───────────────────────────────────
        try:
            # colour: BGR8 → numpy uint8 (H×W×3)
            bgr_np = self._bridge.imgmsg_to_cv2(
                self._color_image, desired_encoding="bgr8"
            )
            # depth: 16UC1 → numpy uint16 (H×W)
            depth_np: np.ndarray = self._bridge.imgmsg_to_cv2(
                self._depth_image, desired_encoding="passthrough"
            )

        except Exception as exc:
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return

        # ── BGR → RGB PIL image for the model ────────────────────────────
        rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb_np)

        # ── Run detection ─────────────────────────────────────────────────
        boxes, scores = self._detector.run(pil_img)

        cfg = self._cfg
        intr = self._intrinsics

        # ── Process each detection ────────────────────────────────────────
        for box, score in zip(boxes, scores):
            if float(score) < cfg.score_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            bbox_tuple = (x1, y1, x2, y2)

            # 3-D keypoints (centre + corners)
            pts3d = DepthUtils.bbox_keypoints_3d_camera(
                bbox_tuple,
                depth_np,
                intr,
                self.DEPTH_SCALE,
                win=cfg.depth_window,
                inset_px=cfg.corner_inset_px,
            )

            # 3-D bbox extents
            X_size, Y_size, Z_size = DepthUtils.bbox_3d_extents(
                box.numpy(),
                depth_np,
                intr,
                self.DEPTH_SCALE,
                step=cfg.bbox_scan_step,
                min_points=cfg.bbox_min_points,
                pct_low=cfg.extent_percentile_low,
                pct_high=cfg.extent_percentile_high,
            )

            # ── ROS 2 logger output (replaces console print) ──────────────
            self.get_logger().info("=" * 40)
            self.get_logger().info(f"Detection score : {float(score):.3f}")
            self.get_logger().info(
                f"BBox pixels     : x1={x1}, y1={y1}, x2={x2}, y2={y2}"
            )

            center = pts3d.get("center")
            if center is not None:
                Xc, Yc, Zc, dc = center
                dist = float(np.sqrt(Xc**2 + Yc**2 + Zc**2))
                self.get_logger().info(
                    f"Center (m)      : X={Xc:.4f}, Y={Yc:.4f}, Z={Zc:.4f}"
                )
                self.get_logger().info(
                    f"  Euclidean dist: {dist:.4f} m  |  depth={dc:.4f} m"
                )
            else:
                self.get_logger().warn("Center          : NO VALID DEPTH")

            for name in ("tl", "tr", "bl", "br"):
                p = pts3d.get(name)
                if p is not None:
                    X, Y, Z, d = p
                    self.get_logger().info(
                        f"Corner {name}       : X={X:.4f}, Y={Y:.4f}, "
                        f"Z={Z:.4f}  depth={d:.4f} m"
                    )
                else:
                    self.get_logger().warn(f"Corner {name}       : NO VALID DEPTH")

            if X_size is not None:
                self.get_logger().info(
                    f"BBox extents (m): X={X_size:.4f}, "
                    f"Y={Y_size:.4f}, Z={Z_size:.4f}"
                )
            else:
                self.get_logger().warn("BBox extents    : NO VALID DEPTH")

            self.get_logger().info("=" * 40)

            # ── Draw bounding box on BGR image ────────────────────────────
            cv2.rectangle(bgr_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

            uc = int(0.5 * (x1 + x2))
            vc = int(0.5 * (y1 + y2))
            cv2.circle(bgr_np, (uc, vc), 4, (0, 255, 255), -1)

            cv2.putText(
                bgr_np,
                f"{float(score):.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # ── Publish annotated image ───────────────────────────────────────
        try:
            annotated_msg: Image = self._bridge.cv2_to_imgmsg(
                bgr_np, encoding="bgr8"
            )
            # Preserve the original header timestamp for downstream nodes
            annotated_msg.header = self._color_image.header
            self._annotated_pub.publish(annotated_msg)
        except Exception as exc:
            self.get_logger().error(f"Failed to publish annotated image: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None) -> None:
    rclpy.init(args=args)

    cfg = Config()  # ← edit Config fields at the top of the file

    node = LavenderDetection(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()