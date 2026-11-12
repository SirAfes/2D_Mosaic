"""
features.py

Registry of interchangeable feature methods.

Every method turns a grayscale uint8 working image into keypoints (N x 2,
float32, pixel coordinates), descriptors and a descriptor norm. Methods that
need opencv-contrib (in OpenCV 5 AKAZE, KAZE, BRISK, BEBLID, ... moved to
``cv2.xfeatures2d``) or optional deep-learning packages are reported as
unavailable instead of failing.

Built-in methods
----------------
sift        SIFT (Lowe 2004), the detector family Metashape-type tools use.
rootsift    SIFT with RootSIFT descriptors (Arandjelovic & Zisserman 2012):
            Hellinger kernel, usually more matches at the same precision.
orb         ORB = oriented FAST + rotated BRIEF over a scale pyramid.
fast        FAST-9 corners (Rosten & Drummond 2006) with grid-bucketed
            non-maximum suppression; FAST has no descriptor, upright ORB
            (rBRIEF) descriptors are computed on the corners. No scale
            invariance -- good for sequential frames at constant distance.
fast_sift   FAST corners + upright SIFT descriptors (fixed support size).
gftt_sift   Shi-Tomasi corners + upright SIFT descriptors.
akaze       AKAZE (Alcantarilla 2013), nonlinear scale space, binary MLDB.
kaze        KAZE, nonlinear scale space, float descriptors.
brisk       BRISK (Leutenegger 2011), AGAST corners, binary descriptors.
orb_beblid  ORB keypoints + BEBLID descriptors (Suarez 2020).
fast_beblid FAST corners + BEBLID descriptors.

Learned features (SuperPoint/DISK + LightGlue) can be added as further
entries of the registry; they are not bundled because they need PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

_XF = getattr(cv2, "xfeatures2d", None)


def _factory(name):
    """cv2.<name>_create or cv2.xfeatures2d.<name>_create, else None."""
    for mod in (cv2, _XF):
        if mod is not None and hasattr(mod, name + "_create"):
            return getattr(mod, name + "_create")
    return None


@dataclass
class Features:
    pts: np.ndarray          # N x 2 float32
    desc: np.ndarray | None  # N x D
    norm: int                # cv2.NORM_L2 or cv2.NORM_HAMMING
    method: str
    size: tuple = (0, 0)     # (w, h) of the image they were detected on


@dataclass
class Method:
    name: str
    family: str              # "float" | "binary"
    build: Callable | None
    note: str = ""

    @property
    def available(self) -> bool:
        return self.build is not None


def _grid_select(kps, img_shape, n_max, grid=(8, 8)):
    """Keep the strongest keypoints per grid cell (spatially even coverage)."""
    if len(kps) <= n_max:
        return list(kps)
    h, w = img_shape[:2]
    gx, gy = grid
    per_cell = max(1, int(np.ceil(n_max / (gx * gy))))
    cells = {}
    for kp in kps:
        cx = min(gx - 1, int(kp.pt[0] * gx / w))
        cy = min(gy - 1, int(kp.pt[1] * gy / h))
        cells.setdefault((cx, cy), []).append(kp)
    out = []
    for lst in cells.values():
        lst.sort(key=lambda k: -k.response)
        out.extend(lst[:per_cell])
    out.sort(key=lambda k: -k.response)
    return out[:n_max]


def _to_features(kps, desc, norm, name, img):
    if desc is None or len(kps) == 0:
        return Features(np.zeros((0, 2), np.float32), None, norm, name, img.shape[1::-1])
    pts = np.array([k.pt for k in kps], np.float32)
    return Features(pts, desc, norm, name, img.shape[1::-1])


# ----------------------------------------------------------------- builders

def _sift(n, root=False):
    f = _factory("SIFT")
    if f is None:
        return None

    def run(img, mask=None):
        det = f(nfeatures=n, contrastThreshold=0.02)
        kps, desc = det.detectAndCompute(img, mask)
        if desc is not None and root:
            desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-7)
            desc = np.sqrt(desc).astype(np.float32)
        return _to_features(kps, desc, cv2.NORM_L2, "rootsift" if root else "sift", img)
    return run


def _orb(n):
    f = _factory("ORB")
    if f is None:
        return None

    def run(img, mask=None):
        det = f(nfeatures=n * 2, scaleFactor=1.2, nlevels=8, fastThreshold=10)
        kps = det.detect(img, mask)
        kps = _grid_select(kps, img.shape, n)
        kps, desc = det.compute(img, kps)
        return _to_features(kps, desc, cv2.NORM_HAMMING, "orb", img)
    return run


def _fast_kps(img, mask, n, threshold=12):
    det = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)
    kps = det.detect(img, mask)
    # Adapt the threshold to low-contrast (e.g. underwater) images.
    while len(kps) < n and threshold > 4:
        threshold = max(4, threshold // 2)
        det.setThreshold(threshold)
        kps = det.detect(img, mask)
    return _grid_select(kps, img.shape, n, grid=(12, 12))


def _with_desc(detector_name, n, desc_name, size=31.0):
    """Corner detector + descriptor extractor ("detector_descriptor" methods)."""
    if desc_name == "orb":
        dfac, norm = _factory("ORB"), cv2.NORM_HAMMING
    elif desc_name == "sift":
        dfac, norm = _factory("SIFT"), cv2.NORM_L2
    elif desc_name == "beblid":
        dfac, norm = _factory("BEBLID"), cv2.NORM_HAMMING
    else:
        dfac, norm = None, None
    if dfac is None:
        return None
    if detector_name == "gftt" and _factory("GFTTDetector") is None:
        return None
    name = f"{detector_name}_{desc_name}" if not (detector_name == "fast" and desc_name == "orb") else "fast"

    def run(img, mask=None):
        if detector_name == "fast":
            kps = _fast_kps(img, mask, n)
        elif detector_name == "gftt":
            det = cv2.GFTTDetector_create(maxCorners=n * 2, qualityLevel=0.005, minDistance=5,
                                          blockSize=5)
            kps = _grid_select(det.detect(img, mask), img.shape, n)
        else:  # orb keypoints
            det = cv2.ORB_create(nfeatures=n * 2, fastThreshold=10)
            kps = _grid_select(det.detect(img, mask), img.shape, n)
        for k in kps:
            k.size = max(k.size, size) if detector_name != "orb" else k.size
            if detector_name != "orb":
                k.angle = 0.0
        if desc_name == "orb":
            ext = dfac(edgeThreshold=int(size) + 1, patchSize=int(size))
            # rBRIEF needs a border; drop keypoints too close to the edge
            h, w = img.shape[:2]
            b = int(size) + 2
            kps = [k for k in kps if b <= k.pt[0] < w - b and b <= k.pt[1] < h - b]
        elif desc_name == "beblid":
            ext = dfac(1.0)
        else:
            ext = dfac()
        kps, desc = ext.compute(img, kps)
        return _to_features(kps, desc, norm, name, img)
    return run


def _contrib(name, n, norm, **kw):
    f = _factory(name)
    if f is None:
        return None

    def run(img, mask=None):
        det = f(**kw)
        kps = det.detect(img, mask)
        kps = _grid_select(kps, img.shape, n)
        kps, desc = det.compute(img, kps)
        return _to_features(kps, desc, norm, name.lower(), img)
    return run


def build_registry(n_features: int = 8000) -> dict[str, Method]:
    n = n_features
    reg = [
        Method("sift", "float", _sift(n), "Scale/rotation invariant, reference standard"),
        Method("rootsift", "float", _sift(n, root=True), "SIFT with Hellinger-normalised descriptors"),
        Method("orb", "binary", _orb(n), "oriented FAST + rotated BRIEF, very fast"),
        Method("fast", "binary", _with_desc("fast", n, "orb"),
               "FAST-9 corners + upright rBRIEF, no scale invariance"),
        Method("fast_sift", "float", _with_desc("fast", n, "sift"),
               "FAST-9 corners + upright SIFT descriptor"),
        Method("gftt_sift", "float", _with_desc("gftt", n, "sift"),
               "Shi-Tomasi corners + upright SIFT descriptor"),
        Method("akaze", "binary", _contrib("AKAZE", n, cv2.NORM_HAMMING, threshold=0.0005),
               "nonlinear scale space, binary descriptor (needs opencv-contrib in OpenCV 5)"),
        Method("kaze", "float", _contrib("KAZE", n, cv2.NORM_L2, threshold=0.0005),
               "nonlinear scale space, float descriptor (contrib)"),
        Method("brisk", "binary", _contrib("BRISK", n, cv2.NORM_HAMMING, thresh=20),
               "AGAST corners, scale space, binary (contrib)"),
        Method("orb_beblid", "binary", _with_desc("orb", n, "beblid"),
               "ORB keypoints + BEBLID descriptor (contrib)"),
        Method("fast_beblid", "binary", _with_desc("fast", n, "beblid"),
               "FAST corners + BEBLID descriptor (contrib)"),
    ]
    return {m.name: m for m in reg}


DEFAULT_ORDER = ["sift", "rootsift", "akaze", "orb", "brisk", "kaze", "fast_sift",
                 "fast", "gftt_sift", "orb_beblid", "fast_beblid"]


def resolve_methods(spec: str | list, registry: dict[str, Method]) -> list[str]:
    """'auto' / 'all' / comma list -> list of available method names."""
    if isinstance(spec, str):
        spec = spec.strip().lower()
        if spec in ("auto", "all"):
            names = [m for m in DEFAULT_ORDER if m in registry]
        else:
            names = [s.strip() for s in spec.split(",") if s.strip()]
    else:
        names = list(spec)
    out = []
    for nm in names:
        if nm not in registry:
            raise ValueError(f"Unknown feature method '{nm}'. Known: {', '.join(registry)}")
        if registry[nm].available:
            out.append(nm)
    return out
