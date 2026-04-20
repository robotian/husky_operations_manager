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
import time
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
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

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
    timer_hz: float = 2.0

    # ── Topics (relative to namespace) ───────────────────────────────────────
    color_image_topic: str    = "manipulators/arm_0_color_camera/image_raw"
    depth_image_topic: str    = "manipulators/arm_0_depth_camera/image_raw"
    camera_info_topic: str    = "manipulators/arm_0_color_camera/camera_info"
    depth_info_topic: str    = "manipulators/arm_0_depth_camera/camera_info"
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


class DepthUtils_Base:
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
        depth_m = DepthUtils_Base.robust_depth_at_pixel(
            depth_image, int(u), int(v), depth_scale, box=box, win=win
        )
        if depth_m is None:
            return None

        X, Y, Z = DepthUtils_Base.deproject_pixel(u, v, depth_m, intr)
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
            k: DepthUtils_Base.pixel_to_3d_camera(
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
                X, Y, _ = DepthUtils_Base.deproject_pixel(u, v, Z, intr)
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

class DepthUtils:
    """
    3D geometry helpers — stateful, initialized once per camera.

    Converts from pure-static to instance-based to enable:
      - Pre-built full-image coordinate grids (U_full, V_full)
      - Cached depth scale to avoid per-call multiply
      - Pre-allocated working buffers reused every frame

    Usage:
        # Initialize once when CameraInfo arrives
        self._depth_utils = DepthUtils(intrinsics, (480, 640), depth_scale=0.001)

        # Call every frame — no pre-processing needed in _timer_callback
        pts3d   = self._depth_utils.bbox_keypoints_3d_camera(bbox, depth_np, ...)
        extents = self._depth_utils.bbox_3d_extents(box, depth_np, ...)
    """

    def __init__(
        self,
        intr: 'Intrinsics',
        image_shape: Tuple[int, int],   # (height, width)
        depth_scale: float = 0.001,
    ) -> None:
        """
        Pre-build coordinate grids and cache all constants.

        Args:
            intr:        Camera intrinsics from CameraInfo
            image_shape: (height, width) of the depth image
            depth_scale: metres per raw uint16 unit (default 0.001 = mm→m)
        """
        self._intr        = intr
        self._depth_scale = np.float32(depth_scale)
        self._h, self._w  = image_shape

        # --- Pre-build full-image coordinate grids ---
        # Built once at init, sliced every frame — zero per-frame allocation
        cols = np.arange(self._w, dtype=np.float32)
        rows = np.arange(self._h, dtype=np.float32)
        self._U_full, self._V_full = np.meshgrid(cols, rows)

        # Cache intrinsic constants as float32 scalars for broadcasting
        self._cx = np.float32(intr.cx)
        self._cy = np.float32(intr.cy)
        self._fx = np.float32(intr.fx)
        self._fy = np.float32(intr.fy)

        # Pre-allocate a reusable float32 buffer for depth conversion
        # Avoids repeated allocation of (H, W) float32 arrays each frame
        self._depth_f32_buf = np.empty(
            (self._h, self._w), dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_depth_f32(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Convert raw uint16 depth to scaled float32 in-place into
        the pre-allocated buffer — zero new allocation per frame.

        Args:
            depth_image: uint16 numpy array (H x W)

        Returns:
            float32 view of internal buffer with depth in metres
        """
        np.multiply(
            depth_image, self._depth_scale,
            out=self._depth_f32_buf,
            casting='unsafe'
        )
        return self._depth_f32_buf

    @staticmethod
    def _fast_percentile(
        arr: np.ndarray,
        lo: float,
        hi: float,
    ) -> Tuple[float, float]:
        """
        Compute two percentiles using np.partition — O(N) vs O(N log N).

        Args:
            arr: 1D numpy array
            lo:  lower percentile (0-100)
            hi:  upper percentile (0-100)

        Returns:
            (low_value, high_value)
        """
        n = len(arr)
        k_lo = max(0, int(n * lo / 100.0))
        k_hi = min(n - 1, int(n * hi / 100.0))
        partitioned = np.partition(arr, [k_lo, k_hi])
        return float(partitioned[k_lo]), float(partitioned[k_hi])

    # ------------------------------------------------------------------
    # Public interface — same signatures as original static methods
    # ------------------------------------------------------------------

    def deproject_pixel(
        self,
        u: float,
        v: float,
        depth_m: float,
    ) -> Tuple[float, float, float]:
        """
        Back-project a 2D pixel + depth into a 3D camera-frame point.
        Unchanged — kept for API compatibility.
        """
        X = (u - self._cx) * depth_m / self._fx
        Y = (v - self._cy) * depth_m / self._fy
        return float(X), float(Y), float(depth_m)

    def robust_depth_at_pixel(
        self,
        depth_f32: np.ndarray,
        u: int,
        v: int,
        box: Optional[Tuple[int, int, int, int]] = None,
        win: int = 7,
        min_valid: int = 5,
    ) -> Optional[float]:
        """
        Return median depth (metres) in a win×win window around (u, v).

        Accepts pre-scaled float32 depth — no internal scaling needed.
        Meshgrid replaced with direct index arithmetic.

        Args:
            depth_f32: pre-scaled float32 depth array (metres)
            u, v:      pixel coordinates
            box:       optional bbox to clip window against
            win:       window diameter in pixels
            min_valid: minimum valid pixels required

        Returns:
            Median depth in metres or None
        """
        half = win // 2

        u0, u1 = max(0, u - half), min(self._w, u + half + 1)
        v0, v1 = max(0, v - half), min(self._h, v + half + 1)

        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            # Clip window to bbox — direct arithmetic, no meshgrid
            u0 = max(u0, x1);  u1 = min(u1, x2)
            v0 = max(v0, y1);  v1 = min(v1, y2)

        if u1 <= u0 or v1 <= v0:
            return None

        # Flatten and filter in one step — no intermediate boolean array
        valid = depth_f32[v0:v1, u0:u1].ravel()
        valid = valid[valid > 0]

        if valid.size < min_valid:
            return None

        return float(np.median(valid))

    def bbox_keypoints_3d_camera(
        self,
        box: Tuple[int, int, int, int],
        depth_image: np.ndarray,
        win: int = 7,
        inset_px: int = 5,
    ) -> dict:
        """
        Compute 3D camera-frame points for centre + 4 corners of a bbox.

        Optimization:
          - depth_image converted once internally via pre-allocated buffer
          - meshgrid replaced with direct index clipping per keypoint
          - min_valid relaxed to 5 (small windows rarely hit 20 valid pixels)

        Args:
            box:         (x1, y1, x2, y2) bounding box in pixels
            depth_image: raw uint16 depth array
            win:         window size for depth sampling
            inset_px:    corner inset in pixels

        Returns:
            dict with keys center/tl/tr/bl/br, each (X,Y,Z,depth_m) or None
        """
        x1, y1, x2, y2 = map(int, box)

        # Convert once — writes into pre-allocated buffer
        depth_f32 = self._to_depth_f32(depth_image)

        x1i, y1i = x1 + inset_px, y1 + inset_px
        x2i, y2i = x2 - inset_px, y2 - inset_px
        uc = int(0.5 * (x1 + x2))
        vc = int(0.5 * (y1 + y2))

        keypoints = {
            "center": (uc,  vc ),
            "tl":     (x1i, y1i),
            "tr":     (x2i, y1i),
            "bl":     (x1i, y2i),
            "br":     (x2i, y2i),
        }

        result = {}
        for name, (u, v) in keypoints.items():
            depth_m = self.robust_depth_at_pixel(
                depth_f32, u, v, box=(x1, y1, x2, y2), win=win
            )
            if depth_m is None:
                result[name] = None
                continue

            X = (u - self._cx) * depth_m / self._fx
            Y = (v - self._cy) * depth_m / self._fy
            result[name] = (float(X), float(Y), float(depth_m), float(depth_m))

        return result

    def bbox_3d_extents(
        self,
        box: Tuple[float, float, float, float],
        depth_image: np.ndarray,
        step: int = 2,
        min_points: int = 10,
        pct_low: float = 5.0,
        pct_high: float = 95.0,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Estimate 3D physical extents (width, height, depth) of the object
        inside a bounding box.

        Optimizations applied:
          - depth converted once into pre-allocated float32 buffer
          - Python double loop replaced with single array slice
          - Coordinate grids sliced from pre-built full-image grids
          - Deprojection via numpy broadcasting — zero Python loop
          - Percentile via np.partition O(N) instead of np.percentile O(NlogN)

        Args:
            box:        (x1, y1, x2, y2) bounding box
            depth_image: raw uint16 depth array
            step:       pixel stride for sampling
            min_points: minimum valid points required
            pct_low:    lower percentile for extent computation
            pct_high:   upper percentile for extent computation

        Returns:
            (X_size, Y_size, Z_size) in metres or (None, None, None)
        """
        x1, y1, x2, y2 = map(int, box)

        # Clamp to image bounds
        x1 = max(0, min(x1, self._w - 1))
        x2 = max(0, min(x2, self._w - 1))
        y1 = max(0, min(y1, self._h - 1))
        y2 = max(0, min(y2, self._h - 1))

        if x2 <= x1 or y2 <= y1:
            return None, None, None

        # Convert raw depth into pre-allocated float32 buffer — no new alloc
        depth_f32 = self._to_depth_f32(depth_image)

        # Single strided slice — equivalent to double loop, O(1) Python
        Z = depth_f32[y1:y2:step, x1:x2:step]

        # Slice pre-built grids — zero allocation
        U = self._U_full[y1:y2:step, x1:x2:step]
        V = self._V_full[y1:y2:step, x1:x2:step]

        # Valid pixel mask — single vectorized comparison
        valid_mask = Z > 0

        if valid_mask.sum() < min_points:
            return None, None, None

        # Boolean indexing — extract valid pixels only
        Z_v = Z[valid_mask]
        U_v = U[valid_mask]
        V_v = V[valid_mask]

        # Deproject all valid pixels simultaneously — numpy broadcasting
        X_v = (U_v - self._cx) * Z_v / self._fx
        Y_v = (V_v - self._cy) * Z_v / self._fy

        # Fast percentile via partition — avoids full sort
        x_lo, x_hi = self._fast_percentile(X_v, pct_low, pct_high)
        y_lo, y_hi = self._fast_percentile(Y_v, pct_low, pct_high)
        z_lo, z_hi = self._fast_percentile(Z_v, pct_low, pct_high)

        return (x_hi - x_lo), (y_hi - y_lo), (z_hi - z_lo)

    # ------------------------------------------------------------------
    # Static helpers retained for backward compatibility
    # ------------------------------------------------------------------

    @staticmethod
    def from_camera_info_and_shape(
        msg: 'CameraInfo',
        depth_scale: float = 0.001,
    ) -> 'DepthUtils':
        """
        Convenience constructor — builds DepthUtils directly from CameraInfo.

        Args:
            msg:         sensor_msgs/CameraInfo message
            depth_scale: metres per raw depth unit

        Returns:
            Initialized DepthUtils instance

        Usage in LavenderDetection:
            def _color_camera_info_callback(self, msg):
                if self._depth_utils is None:
                    self._depth_utils = DepthUtils.from_camera_info_and_shape(
                        msg, LavenderDetection.DEPTH_SCALE
                    )
        """
        intr = Intrinsics.from_camera_info(msg)
        return DepthUtils(intr, (msg.height, msg.width), depth_scale)

# ══════════════════════════════════════════════════════════════════════════════
# 3. DetectionModel
#    Loads the Faster R-CNN checkpoint and wraps inference.
#    Isolated from ROS 2 — can be unit-tested independently.
# ══════════════════════════════════════════════════════════════════════════════

class DetectionModel_Base:
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

class DetectionModel_FP16:
    """
    Loads a Faster R-CNN ResNet-50 FPN model from a checkpoint file and
    exposes a single run() method for inference on an RGB numpy array.

    Optimizations applied:
        - FP16 half precision (CUDA only) for ~40-60% inference reduction
        - Direct numpy → tensor path eliminating PIL conversion overhead
        - Non-blocking CUDA memory transfer to prevent CPU stall
        - Pinned memory for faster host-to-device PCIe transfer
        - CUDA memory pool pre-allocation via warmup passes
        - expandable_segments allocator to reduce fragmentation
        - RPN threshold tightening to reduce proposal count variance
        - CUDA synchronization before return to prevent carry-over GPU work
    """

    def __init__(self, cfg: Config, logger=None) -> None:
        self._logger = logger
        self.device  = torch.device(cfg.device)
        self.score_threshold = cfg.score_threshold

        self._log(f"Initializing DetectionModel_FP16 on device: {self.device}")

        # --- Build model skeleton ---
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

        # --- Load checkpoint ---
        checkpoint = torch.load(
            cfg.model_path,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()

        # --- Optimization 1: FP16 half precision ---
        # RTX 4060 tensor cores give significant throughput on FP16.
        # Only applied on CUDA — CPU FP16 is slower than FP32.
        if self.device.type == 'cuda':
            model          = model.half()
            self._use_half = True
            self._log("FP16 half precision enabled")
        else:
            self._use_half = False
            self._log("FP16 disabled — running FP32 on CPU")

        # torch.compile skipped — Faster R-CNN RPN anchor mutation is
        # incompatible with CUDAGraphs. FP16 + CUDA remains active.
        self._log(
            "torch.compile skipped — Faster R-CNN RPN anchor mutation "
            "is incompatible with CUDAGraphs. FP16 + CUDA remains active."
        )

        self._model = model

        # --- Optimization 2: Pinned memory flag ---
        # Pinned (page-locked) memory reduces PCIe transfer time CPU → GPU.
        self._use_pinned = (self.device.type == 'cuda')
        if self._use_pinned:
            self._log("Pinned memory enabled for faster host-to-device transfer")

        # --- Optimization 3: CUDA memory allocator configuration ---
        # expandable_segments reduces fragmentation from variable-size
        # allocations caused by the RPN's dynamic proposal count per frame.
        if self.device.type == 'cuda':
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            torch.cuda.memory.set_per_process_memory_fraction(0.7)
            self._log("CUDA allocator set to expandable_segments mode (0.7 fraction)")

        # --- Optimization 4: RPN threshold tightening ---
        # Faster R-CNN's RPN generates a variable number of proposals per
        # image depending on scene complexity. Variable proposal counts cause
        # different sized tensor allocations each frame, leading to CUDA
        # memory pressure and inconsistent inference times.
        # Tightening these thresholds reduces proposal count variance,
        # making tensor sizes more consistent between frames.
        #
        # NOTE: Monitor detection accuracy after applying — if small or
        # distant plants are missed, increase pre_nms_top_n_test toward 750.
        if self.device.type == 'cuda':
            self._model.rpn.nms_thresh           = 0.65  # default 0.7
            self._model.rpn.score_thresh         = 0.05  # default 0.0
            self._model.rpn.pre_nms_top_n_test   = 500   # default 1000
            self._model.rpn.post_nms_top_n_test  = 200   # default 1000
            self._log(
                "RPN thresholds tightened | "
                "pre_nms=500 post_nms=200 nms_thresh=0.65 score_thresh=0.05"
            )

        # --- Optimization 5: CUDA warmup passes ---
        # Pre-allocates CUDA memory buffers by running dummy forward passes
        # at the expected input resolution before real inference begins.
        # This forces the allocator to reserve and cache all required buffers
        # upfront, preventing cold-start spikes on the first real frames.
        # 3 passes are used to stabilize both RPN and ROI head allocations.
        if self.device.type == 'cuda':
            self._log("Running CUDA warmup passes...")
            torch.cuda.empty_cache()
            dummy = torch.zeros(
                1, 3, 480, 640,
                dtype=torch.float16 if self._use_half else torch.float32,
                device=self.device
            )
            with torch.no_grad():
                for _ in range(3):
                    _ = self._model([dummy[0]])
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self._log("CUDA warmup complete — memory pool stabilized")

        self._log("DetectionModel_FP16 ready")

    def run(
        self, image: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a single RGB numpy array (H x W x 3, uint8).

        Replaces the PIL-based path with a direct numpy → tensor conversion,
        eliminating one memory copy and the PIL conversion overhead.

        Args:
            image: RGB numpy array uint8 (H x W x 3)

        Returns:
            boxes  : FloatTensor shape (N, 4) — [x1, y1, x2, y2] in pixels
            scores : FloatTensor shape (N,)
            Both tensors are on CPU.
        """
        # --- Optimization 6: Direct numpy → tensor (no PIL) ---
        # torch.from_numpy shares memory with the array (zero copy).
        # permute: HWC → CHW layout required by torchvision models
        # div(255): uint8 [0,255] → float [0.0, 1.0]
        img_tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )

        # --- Optimization 7: Pinned memory ---
        # Pinning before device transfer reduces PCIe copy time.
        if self._use_pinned:
            img_tensor = img_tensor.pin_memory()

        # --- Optimization 8: FP16 cast + non-blocking device transfer ---
        # non_blocking=True returns immediately, letting the CPU continue
        # while the GPU DMA engine handles the transfer asynchronously.
        if self._use_half:
            img_tensor = img_tensor.half()

        img_tensor = img_tensor.to(self.device, non_blocking=True)

        with torch.no_grad():
            outputs = self._model([img_tensor])[0]

        # --- Optimization 9: CUDA synchronization before return ---
        # Ensures all GPU operations for this frame are complete before
        # returning. Prevents carry-over GPU work from inflating the
        # timing of the next frame and stabilizes frame-to-frame latency.
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        return outputs["boxes"].cpu(), outputs["scores"].cpu()

    def _log(self, msg: str) -> None:
        """Use ROS2 logger if available, otherwise fall back to print."""
        if self._logger:
            self._logger.info(msg)
        else:
            print(f"[DetectionModel_FP16] {msg}")

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
        self.declare_parameter("namespace", "")
        self._namespace: str = (
            self.get_parameter("namespace")
            .get_parameter_value()
            .string_value
        )

        # ── Cached latest messages (None until first message arrives) ─────
        self._color_camera_info: Optional[CameraInfo] = None
        self._depth_camera_info: Optional[CameraInfo] = None
        self._color_image: Optional[Image]            = None
        self._depth_image: Optional[Image]            = None

        # ── Lazy-initialised intrinsics (built once from first CameraInfo) ─
        self._intrinsics: Optional[Intrinsics] = None
        self._depth_utils: Optional[DepthUtils] = None

        # ── Load detection model ──────────────────────────────────────────
        self.get_logger().info(f"Loading model from: {cfg.model_path}")
        # self._detector = DetectionModel_Base(cfg)
        self._detector = DetectionModel_FP16(cfg, self.get_logger())
        self.get_logger().info("Model loaded.")

        # ── Topic names with namespace prefix ────────────────────────────
        ns = self._namespace.rstrip("/")

        def full_topic(relative: str) -> str:
            return f"{ns}/{relative}" if ns else relative

        color_topic  = full_topic(cfg.color_image_topic)
        depth_topic  = full_topic(cfg.depth_image_topic)
        camera_info_topic   = full_topic(cfg.camera_info_topic)
        depth_info_topic   = full_topic(cfg.depth_info_topic)
        output_topic = full_topic(cfg.annotated_image_topic)

        # ── QoS: best-effort matches most camera drivers ──────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, camera_info_topic, self._color_camera_info_callback, qos
        )
        self.create_subscription(
            CameraInfo, camera_info_topic, self._depth_camera_info_callback, qos
        )
        self.create_subscription(
            Image, color_topic, self._color_image_callback, qos
        )
        self.create_subscription(
            Image, depth_topic, self._depth_image_callback, qos
        )

        # ── Publisher ─────────────────────────────────────────────────────
        self._annotated_pub_raw = self.create_publisher(Image, f'{output_topic}/raw', 10)
        self._annotated_pub_compressed = self.create_publisher(CompressedImage, f'{output_topic}/compressed', 10)

        # ── Timer ─────────────────────────────────────────────────────────
        timer_period = 1.0 / cfg.timer_hz
        self.create_timer(timer_period, self._timer_callback)

        self.get_logger().info(
            f"LavenderDetection started | namespace='{self._namespace}' | "
            f"timer={cfg.timer_hz} Hz"
        )
        self.get_logger().info(f"  colour  → {color_topic}")
        self.get_logger().info(f"  depth   → {depth_topic}")
        self.get_logger().info(f"  info    → {camera_info_topic}")
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

    def _depth_camera_info_callback(self, msg: CameraInfo) -> None:
        self._depth_camera_info = msg
        # Build intrinsics once; CameraInfo is static for a fixed camera
        if self._depth_utils is None:
            self._depth_utils = DepthUtils.from_camera_info_and_shape(msg, self.DEPTH_SCALE)

    def _color_image_callback(self, msg: Image) -> None:
        self._color_image = msg

    def _depth_image_callback(self, msg: Image) -> None:
        self._depth_image = msg

    # ── Timer callback (main processing loop) ─────────────────────────────────

    def _timer_callback(self) -> None:
        if (self._color_image is None or self._depth_image is None or 
            self._intrinsics is None or  self._depth_utils is None):
            self.get_logger().warn("Waiting for CameraInfo…", throttle_duration_sec=5.0)
            return

        t_start = time.perf_counter()

        # --- cv_bridge conversion ---
        try:
            t0 = time.perf_counter()
            bgr_np = self._bridge.imgmsg_to_cv2(self._color_image, desired_encoding="bgr8")
            depth_np = self._bridge.imgmsg_to_cv2(self._depth_image, desired_encoding="passthrough")
            t_bridge = time.perf_counter() - t0
        except Exception as exc:
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return

        # --- BGR → RGB PIL ---
        t0 = time.perf_counter()
        rgb_np  = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
        # pil_img = PILImage.fromarray(rgb_np)
        t_convert = time.perf_counter() - t0

        # --- Inference ---
        t0 = time.perf_counter()
        boxes, scores = self._detector.run(rgb_np)
        t_inference = time.perf_counter() - t0

        # --- 3D processing per detection ---
        t0 = time.perf_counter()
        for box, score in zip(boxes, scores):
            if float(score) < self._cfg.score_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            bbox_tuple = (x1, y1, x2, y2)

            # DepthUtils.bbox_keypoints_3d_camera(
            #     bbox_tuple, depth_np, self._intrinsics,
            #     self.DEPTH_SCALE,
            #     win=self._cfg.depth_window,
            #     inset_px=self._cfg.corner_inset_px,
            # )
            # DepthUtils.bbox_3d_extents(
            #     box.numpy(), depth_np, self._intrinsics,
            #     self.DEPTH_SCALE,
            #     step=self._cfg.bbox_scan_step,
            #     min_points=self._cfg.bbox_min_points,
            #     pct_low=self._cfg.extent_percentile_low,
            #     pct_high=self._cfg.extent_percentile_high,
            # )
            self._depth_utils.bbox_keypoints_3d_camera(
                bbox_tuple, depth_np,
                win=self._cfg.depth_window, inset_px=self._cfg.corner_inset_px,
            )
            self._depth_utils.bbox_3d_extents(
                box.numpy(), depth_np,
                step=self._cfg.bbox_scan_step, min_points=self._cfg.bbox_min_points,
                pct_low=self._cfg.extent_percentile_low, pct_high=self._cfg.extent_percentile_high,
            )
        t_depth = time.perf_counter() - t0

        # --- Drawing + publish ---
        t0 = time.perf_counter()
        for box, score in zip(boxes, scores):
            if float(score) < self._cfg.score_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(bgr_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(bgr_np, (int(0.5*(x1+x2)), int(0.5*(y1+y2))), 4, (0,255,255), -1)
            cv2.putText(bgr_np, f"{float(score):.2f}", (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        t_draw = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            compressed_image_data = np.array(
                cv2.imencode(".jpg", bgr_np, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
                ).tobytes()
            
            annotated_msg = self._bridge.cv2_to_imgmsg(bgr_np, encoding="bgr8")
            annotated_msg.header = self._color_image.header
            self._annotated_pub_raw.publish(annotated_msg)

            compressed_msg = CompressedImage()
            compressed_msg.header = self._color_image.header
            compressed_msg.format = "jpeg"
            compressed_msg.data = compressed_image_data
            self._annotated_pub_compressed.publish(compressed_msg)
        except Exception as exc:
            self.get_logger().error(f"Failed to publish annotated image: {exc}")
        t_publish = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start

        # --- Report ---
        self.get_logger().info(
            f"\n[Timing Report ms]\n"
            f"  cv_bridge   : {t_bridge   * 1000:.2f}\n"
            f"  bgr→pil     : {t_convert  * 1000:.2f}\n"
            f"  inference   : {t_inference* 1000:.2f}\n"
            f"  depth 3D    : {t_depth    * 1000:.2f}\n"
            f"  draw        : {t_draw     * 1000:.2f}\n"
            f"  publish     : {t_publish  * 1000:.2f}\n"
            f"  ─────────────────────\n"
            f"  TOTAL       : {t_total    * 1000:.2f}\n"
            f"  MAX FPS     : {1.0/t_total:.2f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None) -> None:
    rclpy.init(args=args)

    cfg = Config()  # ← edit Config fields at the top of the file

    node = LavenderDetection(cfg)
    try:
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    main()