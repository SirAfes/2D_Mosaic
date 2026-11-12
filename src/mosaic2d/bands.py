"""
bands.py

Multispectral band handling.

1. Intra-frame lens alignment (multi-lens cameras). Each lens image of a frame
   is registered to the reference lens of the same frame:
   phase correlation on gradient-magnitude images (works across spectral
   bands) gives the translation, an ECC refinement (Evangelidis & Psarakis
   2008) on the same gradient images adds a small affine part (focal-length
   and mounting differences between lenses). Results are checked against
   the series: lens offsets of a rigid lens array change only with distance,
   so a frame whose offset is far from the series median *and* that
   correlates poorly is re-estimated around the median (or set to it).

2. Reference band selection. The band used to register frames to each other
   is chosen by what actually matters: on a sample of neighbouring frame
   pairs, every band is matched with the primary feature method, and the band
   with the most verified inliers wins (texture, signal-to-noise and water
   attenuation are all reflected in that number).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .sources import Band, CameraProfile, FrameData, band_image, normalise_for_features


def gradient_image(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    g = img.astype(np.float32)
    g = g / (cv2.GaussianBlur(g, (0, 0), max(g.shape) / 20.0) + 1e-3 * (g.mean() + 1e-6))
    g = cv2.GaussianBlur(g, (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    step = max(1, m.size // 200000)
    hi = np.percentile(m.ravel()[::step], 99.5)
    return np.clip(m / max(hi, 1e-6), 0, 1).astype(np.float32)


def lens_gray(fd: FrameData, prof: CameraProfile, lens: int) -> np.ndarray:
    """Mean of the used channels of a lens, each scaled to its own mean."""
    chans = [band_image(fd, b).astype(np.float32) for b in prof.lens_bands(lens)]
    chans = [c / max(float(c.mean()), 1e-6) for c in chans]
    return np.mean(chans, axis=0)


def _ncc(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


@dataclass
class LensResult:
    A: np.ndarray            # 3x3: lens px -> reference-lens px
    method: str
    ncc: float
    response: float = 0.0


@dataclass
class FrameLensAlignment:
    ref_lens: int
    lenses: dict = field(default_factory=dict)   # lens -> LensResult

    def A(self, lens: int) -> np.ndarray:
        return np.eye(3) if lens == self.ref_lens else self.lenses[lens].A


def _warp_ncc(ref_g, mov_g, W):
    """NCC of ref with mov warped by W (2x3, ref -> mov coords)."""
    h, w = ref_g.shape
    warped = cv2.warpAffine(mov_g, W, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    valid = cv2.warpAffine(np.ones_like(mov_g), W, (w, h),
                           flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0.5
    m = cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    return _ncc(ref_g, warped, m) if m.sum() > 1000 else 0.0


def align_lens(ref_g: np.ndarray, mov_g: np.ndarray, prior_shift=None, max_dev=None,
               affine=True) -> LensResult:
    """Register gradient image mov_g to ref_g. Returns lens->ref transform."""
    h = min(ref_g.shape[0], mov_g.shape[0])
    w = min(ref_g.shape[1], mov_g.shape[1])
    ref_g, mov_g = ref_g[:h, :w], mov_g[:h, :w]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    if prior_shift is not None:
        # search only around the prior: pre-shift, then measure the residual
        M0 = np.float32([[1, 0, prior_shift[0]], [0, 1, prior_shift[1]]])
        pre = cv2.warpAffine(mov_g, M0, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        (rx, ry), resp = cv2.phaseCorrelate(ref_g, pre, win)
        if max_dev is not None and np.hypot(rx, ry) > max_dev:
            rx = ry = 0.0
            resp = 0.0
        dx, dy = prior_shift[0] + rx, prior_shift[1] + ry
    else:
        (dx, dy), resp = cv2.phaseCorrelate(ref_g, mov_g, win)
    W = np.float32([[1, 0, dx], [0, 1, dy]])
    best_W, best_m = W, "phasecorr"
    best = _warp_ncc(ref_g, mov_g, W)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    modes = [(cv2.MOTION_AFFINE, "ecc_affine")] if affine else []
    modes.append((cv2.MOTION_TRANSLATION, "ecc_translation"))
    for mode, name in modes:
        try:
            _, Wn = cv2.findTransformECC(ref_g, mov_g, W.copy(), mode, crit, None, 5)
        except cv2.error:
            continue
        L = Wn[:, :2]
        sv = np.linalg.svd(L, compute_uv=False)
        rot = np.degrees(np.arctan2(L[1, 0], L[0, 0]))
        if sv.max() > 1.04 or sv.min() < 0.96 or abs(rot) > 3:
            continue
        if np.hypot(Wn[0, 2] - dx, Wn[1, 2] - dy) > 25:
            continue
        s = _warp_ncc(ref_g, mov_g, Wn)
        if s > best + 0.005:
            best_W, best_m, best = Wn, name, s
            break
    A = np.linalg.inv(np.vstack([best_W, [0, 0, 1]]).astype(np.float64))
    return LensResult(A, best_m, best, float(resp))


def align_frame_lenses(fd: FrameData, prof: CameraProfile, ref_lens: int,
                       scale: float = 1.0, priors: dict | None = None, max_dev=None,
                       affine=True) -> FrameLensAlignment:
    """All lenses of one frame -> reference lens. scale < 1 works on reduced images."""
    def prep(lens):
        g = lens_gray(fd, prof, lens)
        if scale != 1.0:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gradient_image(g)

    ref_g = prep(ref_lens)
    out = FrameLensAlignment(ref_lens)
    S = np.diag([scale, scale, 1.0])
    Si = np.diag([1 / scale, 1 / scale, 1.0])
    for lens in range(1, prof.n_lenses + 1):
        if lens == ref_lens:
            continue
        prior = None
        if priors and lens in priors:
            # prior given as lens->ref A (full res); search uses ref->lens shift
            Wp = np.linalg.inv(priors[lens])
            prior = (Wp[0, 2] * scale, Wp[1, 2] * scale)
        r = align_lens(ref_g, prep(lens), prior, None if max_dev is None else max_dev * scale, affine)
        r.A = Si @ r.A @ S
        out.lenses[lens] = r
    return out


def consensus_check(results: list[FrameLensAlignment], tol_px: float = 12.0,
                    ncc_frac: float = 0.6):
    """
    Robust per-lens series statistics. Returns (median A per lens, list of
    (frame index, lens) that disagree with the series and correlate poorly).
    """
    lenses = sorted({ln for r in results for ln in r.lenses})
    med, bad = {}, []
    for ln in lenses:
        T = np.array([r.lenses[ln].A[:2, 2] for r in results])
        nc = np.array([r.lenses[ln].ncc for r in results])
        mt = np.median(T, axis=0)
        mad = np.median(np.linalg.norm(T - mt, axis=1)) * 1.4826
        lim = max(tol_px, 4 * mad)
        mn = np.median(nc)
        Am = np.eye(3)
        L = np.median(np.array([r.lenses[ln].A[:2, :2] for r in results]), axis=0)
        Am[:2, :2], Am[:2, 2] = L, mt
        med[ln] = Am
        for k, r in enumerate(results):
            d = np.linalg.norm(T[k] - mt)
            if d > lim and nc[k] < max(ncc_frac * mn, mn - 0.15):
                bad.append((k, ln))
            elif nc[k] < 0.35 * mn:
                bad.append((k, ln))
    return med, bad


def choose_reference_band(frames_data: list, prof: CameraProfile, method,
                          match_fn, work_scale: float, log=print) -> tuple[Band, dict]:
    """
    frames_data: list of FrameData for a sample of consecutive frames.
    method: callable(img_u8) -> Features ; match_fn(fa, fb) -> PairResult
    Returns (best band, {band name: median inliers}).
    """
    scores = {}
    for b in prof.bands:
        imgs = []
        for fd in frames_data:
            g = band_image(fd, b).astype(np.float32)
            if work_scale != 1.0:
                g = cv2.resize(g, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA)
            imgs.append(normalise_for_features(g))
        feats = [method(g) for g in imgs]
        inl = []
        for k in range(len(feats) - 1):
            r = match_fn(feats[k], feats[k + 1])
            inl.append(r.n_inliers if r.ok else 0)
        scores[b.name] = float(np.median(inl)) if inl else 0.0
    best = max(prof.bands, key=lambda b: (scores[b.name], -abs((b.wavelength_nm or 550) - 550)))
    log("  reference band scores (median verified inliers): " +
        ", ".join(f"{k}: {v:.0f}" for k, v in sorted(scores.items(), key=lambda kv: -kv[1])))
    return best, scores
