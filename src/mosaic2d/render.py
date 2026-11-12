"""
render.py

Out-of-core mosaic rendering.

The canvas lives in a disk-backed NumPy memmap, so the mosaic can be much
larger than memory. Frames are decoded one at a time (frame-major order) and
written strip by strip.

Which frame supplies a pixel (the *seam* layout) is decided analytically: the
frame in which the pixel lies closest to the image centre wins (elliptical
distance, normalised by the valid image size). This is the rule Metashape uses
when blending is disabled; it keeps every pixel from its least distorted, best
lit view and needs no global label image.

Blending modes (names follow Metashape):

disabled  every pixel comes unmodified from the winning frame. Radiometry is
          preserved exactly -- preferred for quantitative multispectral work.
mosaic    (default) two-band blending: high frequencies from the winning frame,
          low frequencies harmonised across overlaps. Implemented as a smooth
          per-frame gain field  g_f = L / L_f  where L is the feather-weighted
          low-resolution mosaic and L_f the frame's own low-resolution image.
          Removes brightness steps at seams (vignetting, strobe falloff,
          exposure changes) without ghosting.
average   feather-weighted average of all frames at every pixel (smooth, but
          blurs/ghosts where the scene has relief or moved).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RenderFrame:
    index: int
    path: str
    H: np.ndarray                 # ref-lens px (undistorted) -> output px
    ref_size: tuple               # (w, h) of the reference lens image, native px
    valid: tuple                  # (x0, y0, x1, y1) valid rect in ref px (all lenses)
    lens_A: dict                  # lens -> 3x3, lens px -> ref px
    thumb: np.ndarray | None = None   # h_t x w_t x B (in ref px * thumb_scale)
    thumb_scale: float = 1.0
    Hinv: np.ndarray = field(default=None)
    bbox: tuple = (0, 0, 0, 0)    # x0, y0, x1, y1 in output px

    def __post_init__(self):
        self.Hinv = np.linalg.inv(self.H)


@dataclass
class Distortion:
    k: np.ndarray
    center: np.ndarray
    radius: float

    @property
    def active(self) -> bool:
        return bool(np.any(np.abs(self.k) > 1e-9))

    def distort(self, u, v):
        """Undistorted -> distorted coordinates (inverse of bundle.undistort_points)."""
        if not self.active:
            return u, v
        cx, cy = self.center
        du, dv = u - cx, v - cy
        x, y = du.copy(), dv.copy()
        for _ in range(5):
            r2 = (x * x + y * y) / (self.radius ** 2)
            f = 1 + self.k[0] * r2 + self.k[1] * r2 * r2
            x, y = du / f, dv / f
        return x + cx, y + cy

    def undistort(self, pts):
        if not self.active:
            return pts
        d = (pts - self.center) / self.radius
        r2 = (d ** 2).sum(1, keepdims=True)
        return self.center + (pts - self.center) * (1 + self.k[0] * r2 + self.k[1] * r2 * r2)


def footprint(fr: RenderFrame, dist: Distortion, n=12) -> np.ndarray:
    x0, y0, x1, y1 = fr.valid
    t = np.linspace(0, 1, n, endpoint=False)
    pts = np.concatenate([
        np.stack([x0 + (x1 - x0) * t, np.full(n, y0)], 1),
        np.stack([np.full(n, x1), y0 + (y1 - y0) * t], 1),
        np.stack([x1 - (x1 - x0) * t, np.full(n, y1)], 1),
        np.stack([np.full(n, x0), y1 - (y1 - y0) * t], 1)])
    pts = dist.undistort(pts.astype(np.float64))
    return cv2.perspectiveTransform(pts[None], fr.H)[0]


def _map(fr: RenderFrame, X, Y):
    """Output px -> undistorted ref px."""
    Hi = fr.Hinv
    w = Hi[2, 0] * X + Hi[2, 1] * Y + Hi[2, 2]
    u = (Hi[0, 0] * X + Hi[0, 1] * Y + Hi[0, 2]) / w
    v = (Hi[1, 0] * X + Hi[1, 1] * Y + Hi[1, 2]) / w
    return u, v


def _score(fr: RenderFrame, u, v, dist: Distortion):
    """(distance score, valid mask, distorted u, distorted v)."""
    ud, vd = dist.distort(u, v)
    x0, y0, x1, y1 = fr.valid
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
    s = ((ud - cx) / hw) ** 2 + ((vd - cy) / hh) ** 2
    valid = (ud >= x0) & (ud <= x1) & (vd >= y0) & (vd <= y1)
    if dist.active:
        # the inverse distortion is only meaningful near the image; far outside
        # the fixed-point iteration can land back inside the frame
        valid &= (np.abs(u - cx) < 1.4 * hw) & (np.abs(v - cy) < 1.4 * hh)
    return s, valid, ud, vd


def _cheb_weight(fr, ud, vd):
    x0, y0, x1, y1 = fr.valid
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d = np.maximum(np.abs(ud - cx) / ((x1 - x0) / 2), np.abs(vd - cy) / ((y1 - y0) / 2))
    return np.clip(1 - d, 0, 1) ** 1.5


def _remap_multi(img, mx, my, interp=cv2.INTER_LINEAR):
    """cv2.remap for any channel count (OpenCV handles at most 4 at once)."""
    if img.ndim == 2:
        return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
    C = img.shape[2]
    parts = []
    for c0 in range(0, C, 4):
        sub = np.ascontiguousarray(img[..., c0:c0 + 4])
        r = cv2.remap(sub, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        parts.append(r if r.ndim == 3 else r[..., None])
    return np.concatenate(parts, axis=2)


class Renderer:
    def __init__(self, frames: list[RenderFrame], dist: Distortion, canvas_size: tuple,
                 band_sources: list, n_bands: int, dtype=np.uint16, blend="mosaic",
                 work_dir=None, strip=512, log=print, progress=None):
        """
        band_sources: per output band (lens, channel).
        """
        self.frames = frames
        self.dist = dist
        self.W, self.H = canvas_size
        self.band_sources = band_sources
        self.B = n_bands
        self.dtype = np.dtype(dtype)
        self.blend = blend
        self.strip = strip
        self.log = log
        self.progress = progress
        self.work_dir = Path(work_dir or tempfile.mkdtemp(prefix="mosaic2d_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for fr in frames:
            fp = footprint(fr, dist)
            x0, y0 = np.floor(fp.min(0)).astype(int)
            x1, y1 = np.ceil(fp.max(0)).astype(int) + 1
            fr.bbox = (max(0, x0), max(0, y0), min(self.W, x1), min(self.H, y1))
        self.maxv = np.iinfo(self.dtype).max
        self.gain = {}

    # ---------------------------------------------------------- low-res pass
    def low_frequency_pass(self):
        """Feather-weighted low-resolution mosaic and per-frame gain fields."""
        ls = min(1.0, 1600.0 / max(self.W, self.H))
        Wl, Hl = max(1, int(self.W * ls)), max(1, int(self.H * ls))
        acc = np.zeros((Hl, Wl, self.B), np.float64)
        wsum = np.zeros((Hl, Wl), np.float64)
        own = {}
        sig = 2.0
        for fr in self.frames:
            if fr.thumb is None:
                continue
            bx0, by0, bx1, by1 = [int(v * ls) for v in fr.bbox]
            bx1, by1 = min(Wl, bx1 + 2), min(Hl, by1 + 2)
            if bx1 <= bx0 or by1 <= by0:
                continue
            X, Y = np.meshgrid((np.arange(bx0, bx1) + 0.5) / ls, (np.arange(by0, by1) + 0.5) / ls)
            u, v = _map(fr, X, Y)
            s, valid, ud, vd = _score(fr, u, v, self.dist)
            w = _cheb_weight(fr, ud, vd) * valid
            mx = (ud * fr.thumb_scale).astype(np.float32)
            my = (vd * fr.thumb_scale).astype(np.float32)
            t = _remap_multi(fr.thumb.astype(np.float32), mx, my)
            t = np.where(valid[..., None], t, 0)
            acc[by0:by1, bx0:bx1] += t * w[..., None]
            wsum[by0:by1, bx0:bx1] += w
            own[fr.index] = (bx0, by0, t, valid)
        L = acc / np.maximum(wsum, 1e-9)[..., None]
        Lb = cv2.GaussianBlur(L.astype(np.float32), (0, 0), sig)
        for fr in self.frames:
            if fr.index not in own:
                continue
            bx0, by0, t, valid = own[fr.index]
            h, w = valid.shape
            vm = valid.astype(np.float32)
            # normalised (masked) blur of the frame's own low-res image
            tb = cv2.GaussianBlur(t.astype(np.float32) * vm[..., None], (0, 0), sig)
            nb = cv2.GaussianBlur(vm, (0, 0), sig)
            tb = tb.reshape(h, w, -1) / np.maximum(nb, 1e-6)[..., None]
            Lf = Lb[by0:by0 + h, bx0:bx0 + w].reshape(h, w, -1)
            g = Lf / np.maximum(tb, 1e-3 * max(float(tb.mean()), 1e-6))
            g = np.clip(np.where(valid[..., None], g, 1.0), 0.33, 3.0).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 1.0).reshape(h, w, -1)
            self.gain[fr.index] = (bx0, by0, ls, g)
        self.low_mosaic = L

    def _gain_at(self, fr, X, Y):
        bx0, by0, ls, g = self.gain[fr.index]
        mx = (X * ls - bx0 - 0.5).astype(np.float32)
        my = (Y * ls - by0 - 0.5).astype(np.float32)
        out = _remap_multi(g, mx, my)
        return np.where(out > 0, out, 1.0)

    # ---------------------------------------------------------- full pass
    def _others(self, fr):
        x0, y0, x1, y1 = fr.bbox
        return [g for g in self.frames if g is not fr and g.bbox[0] < x1 and g.bbox[2] > x0
                and g.bbox[1] < y1 and g.bbox[3] > y0]

    def render(self, load_lenses, out_name="mosaic"):
        """load_lenses(frame) -> {lens: H x W x C array}. Returns the memmap."""
        shape = (self.H, self.W, self.B)
        need = shape[0] * shape[1] * shape[2] * self.dtype.itemsize
        self.log(f"  canvas {self.W} x {self.H} px, {self.B} band(s), "
                 f"{need / 1e9:.2f} GB ({self.blend} blending)")
        canvas_path = self.work_dir / f"{out_name}.canvas"
        out = np.memmap(canvas_path, dtype=self.dtype, mode="w+", shape=shape)
        acc = wsum = None
        if self.blend == "average":
            acc = np.memmap(self.work_dir / "acc.f32", np.float32, "w+", shape)
            wsum = np.memmap(self.work_dir / "wsum.f32", np.float32, "w+", shape[:2])
        if self.blend == "mosaic":
            self.low_frequency_pass()

        for n, fr in enumerate(self.frames):
            x0, y0, x1, y1 = fr.bbox
            if x1 <= x0 or y1 <= y0:
                continue
            lenses = load_lenses(fr)
            others = self._others(fr)
            Ainv = {ln: np.linalg.inv(fr.lens_A[ln]) for ln in fr.lens_A}
            for ys in range(y0, y1, self.strip):
                ye = min(y1, ys + self.strip)
                X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float64) + 0.0,
                                   np.arange(ys, ye, dtype=np.float64))
                u, v = _map(fr, X, Y)
                s, valid, ud, vd = _score(fr, u, v, self.dist)
                if not valid.any():
                    continue
                if self.blend == "average":
                    sel = valid
                else:
                    sel = valid.copy()
                    for g in others:
                        ug, vg = _map(g, X, Y)
                        sg, vgm, _, _ = _score(g, ug, vg, self.dist)
                        beats = vgm & ((sg < s) | ((sg == s) & (g.index < fr.index)))
                        sel &= ~beats
                        if not sel.any():
                            break
                if not sel.any():
                    continue
                rows = np.where(sel.any(1))[0]
                cols = np.where(sel.any(0))[0]
                r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
                sel = sel[r0:r1, c0:c1]
                udc, vdc = ud[r0:r1, c0:c1], vd[r0:r1, c0:c1]
                vals = np.zeros(sel.shape + (self.B,), np.float32)
                for ln, A in Ainv.items():
                    bands = [b for b, (bl, _) in enumerate(self.band_sources) if bl == ln]
                    if not bands:
                        continue
                    mx = (A[0, 0] * udc + A[0, 1] * vdc + A[0, 2]).astype(np.float32)
                    my = (A[1, 0] * udc + A[1, 1] * vdc + A[1, 2]).astype(np.float32)
                    img = lenses[ln]
                    chans = [self.band_sources[b][1] for b in bands]
                    sub = img[..., chans] if img.ndim == 3 else img[..., None]
                    r = _remap_multi(np.ascontiguousarray(sub), mx, my)
                    vals[..., bands] = r
                Xc, Yc = X[r0:r1, c0:c1], Y[r0:r1, c0:c1]
                if self.blend == "mosaic" and fr.index in self.gain:
                    vals *= self._gain_at(fr, Xc, Yc)
                oy, ox = ys + r0, x0 + c0
                if self.blend == "average":
                    w = _cheb_weight(fr, udc, vdc) * sel + 1e-6 * sel
                    acc[oy:oy + sel.shape[0], ox:ox + sel.shape[1]] += vals * w[..., None]
                    wsum[oy:oy + sel.shape[0], ox:ox + sel.shape[1]] += w
                else:
                    tgt = out[oy:oy + sel.shape[0], ox:ox + sel.shape[1]]
                    v = np.clip(vals, 1, self.maxv).astype(self.dtype)
                    tgt[sel] = v[sel]
            del lenses
            if self.progress:
                self.progress(n + 1, len(self.frames))
        if self.blend == "average":
            for ys in range(0, self.H, self.strip):
                w = np.asarray(wsum[ys:ys + self.strip])
                a = np.asarray(acc[ys:ys + self.strip]) / np.maximum(w, 1e-9)[..., None]
                o = np.clip(a, 1, self.maxv).astype(self.dtype)
                o[w <= 0] = 0
                out[ys:ys + self.strip] = o
            del acc, wsum
            for f in ("acc.f32", "wsum.f32"):
                try:
                    os.remove(self.work_dir / f)
                except OSError:
                    pass
        out.flush()
        return out

    # ---------------------------------------------------------- previews
    def seam_map(self, max_side=1600):
        """Label image (frame index per pixel, -1 = empty) at preview scale."""
        s = min(1.0, max_side / max(self.W, self.H))
        Wp, Hp = max(1, int(self.W * s)), max(1, int(self.H * s))
        X, Y = np.meshgrid((np.arange(Wp) + 0.5) / s, (np.arange(Hp) + 0.5) / s)
        best = np.full((Hp, Wp), np.inf)
        label = np.full((Hp, Wp), -1, np.int32)
        for fr in self.frames:
            u, v = _map(fr, X, Y)
            sc, valid, _, _ = _score(fr, u, v, self.dist)
            m = valid & (sc < best)
            best[m] = sc[m]
            label[m] = fr.index
        return label, s


def downsample(arr, max_side=3000):
    """Area-downsample a (memmap) H x W x C array in row chunks."""
    H, W = arr.shape[:2]
    f = max(1, int(np.ceil(max(H, W) / max_side)))
    Ho, Wo = H // f, W // f
    out = np.zeros((Ho, Wo, arr.shape[2]), np.float32)
    rows = max(f, (4096 // f) * f)
    for y in range(0, Ho * f, rows):
        blk = np.asarray(arr[y:min(y + rows, Ho * f), :Wo * f]).astype(np.float32)
        h = blk.shape[0] // f
        blk = blk[:h * f].reshape(h, f, Wo, f, -1).mean(axis=(1, 3))
        out[y // f:y // f + h] = blk
    return out


def visualise(img: np.ndarray, channels=(0, 1, 2), gamma=0.5) -> np.ndarray:
    """Float H x W x C -> uint8 BGR for JPEG, per-channel 1-99.5 % stretch."""
    rgb = img[..., list(channels)].astype(np.float32)
    valid = (img > 0).any(axis=2)
    out = np.zeros(rgb.shape, np.float32)
    for c in range(3):
        ch = rgb[..., c]
        vals = ch[valid]
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, [1, 99.5])
        out[..., c] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma
    out[~valid] = 0
    return (out[..., ::-1] * 255).astype(np.uint8)
