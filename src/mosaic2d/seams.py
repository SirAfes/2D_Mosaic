"""
seams.py

Seamline optimisation and flat-field (vignetting) estimation.

Seamlines
---------
The renderer's default rule gives every pixel to the frame whose centre is
closest (a Voronoi-like layout). That is a good *global* choice -- least lens
distortion, best light, least parallax on average -- but the seam itself may
run straight through a moving object (a floating transect line, a fish) or
through an object with relief, where two frames disagree. Metashape lets the
user edit such seamlines; here they are optimised automatically:

1. The Voronoi labels are computed at a reduced "seam scale".
2. Around every Voronoi seam a band of +-``band_px`` pixels is opened: inside
   it, both neighbouring frames are allowed.
3. A graph cut (OpenCV's ``GraphCutSeamFinder``, colour + gradient cost,
   Kwatra et al. 2003) moves the seam inside that band to where the two frames
   agree best and where the cut is least visible.

Frames are compared after the low-frequency gain correction of the renderer, so
brightness differences between frames (strobes, vignetting) do not attract or
repel the seam; only real disagreement (motion, parallax) does.

Flat field
----------
Vignetting and a fixed illumination pattern (strobes, sunlight from one side)
multiply every frame of a series identically. The per-pixel median over all
frames of (frame / frame median) cancels the scene and leaves that pattern; a
low-order 2D polynomial fitted to it is the flat field. Dividing by it makes
radiometry independent of where in the frame a point was seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------- flat field

def _poly_terms(x, y, degree):
    return np.stack([x ** i * y ** j for i in range(degree + 1) for j in range(degree + 1 - i)], -1)


def estimate_flatfield(thumbs: list, valids: list, degree: int = 4, min_frames: int = 8):
    """
    thumbs: list of h x w x B float arrays (same size, reference-lens coordinates)
    valids: list of h x w bool masks
    Returns (flat h x w x B float32 with mean 1 over the valid area, stats dict)
    or (None, reason).
    """
    if len(thumbs) < min_frames:
        return None, f"only {len(thumbs)} frames (need {min_frames})"
    shape = thumbs[0].shape
    if any(t.shape != shape for t in thumbs):
        return None, "frames have different sizes"
    h, w, B = shape
    stack = np.full((len(thumbs), h, w, B), np.nan, np.float32)
    for k, (t, v) in enumerate(zip(thumbs, valids)):
        t = t.astype(np.float32)
        for b in range(B):
            vals = t[..., b][v]
            med = np.median(vals) if vals.size else 0
            if med > 0:
                layer = t[..., b] / med
                layer[~v] = np.nan
                stack[k, ..., b] = layer
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # pixels never valid -> NaN
        med = np.nanmedian(stack, axis=0)                       # h x w x B
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xn, yn = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
    T = _poly_terms(xn.ravel(), yn.ravel(), degree)
    flat = np.ones((h, w, B), np.float32)
    stats = {}
    ok_any = False
    for b in range(B):
        m = np.isfinite(med[..., b]).ravel()
        if m.sum() < 50:
            continue
        coef, *_ = np.linalg.lstsq(T[m], med[..., b].ravel()[m], rcond=None)
        f = (T @ coef).reshape(h, w)
        mv = np.isfinite(med[..., b])
        f = f / max(float(f[mv].mean()), 1e-6)
        f = np.clip(f, 0.2, 5.0)
        flat[..., b] = f
        ok_any = True
        c = f[h // 2, w // 2]
        corners = [f[2, 2], f[2, -3], f[-3, 2], f[-3, -3]]
        stats[b] = round(float(np.mean(corners) / c), 3)
    if not ok_any:
        return None, "no valid pixels"
    return flat.astype(np.float32), stats


# --------------------------------------------------------------------------- seams

@dataclass
class SeamMap:
    labels: np.ndarray       # Hs x Ws int32 frame index, -1 = empty
    scale: float             # seam-map px per output px
    changed_frac: float = 0.0

    def sample(self, X, Y):
        """Label at output coordinates (nearest)."""
        xs = np.clip((X * self.scale).astype(np.int64), 0, self.labels.shape[1] - 1)
        ys = np.clip((Y * self.scale).astype(np.int64), 0, self.labels.shape[0] - 1)
        return self.labels[ys, xs]


def voronoi_labels(frames, dist, size, scale, score_fn, map_fn):
    """Analytic labels (closest centre with sharpness penalty) at a reduced scale."""
    W, H = size
    Ws, Hs = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    best = np.full((Hs, Ws), np.inf)
    label = np.full((Hs, Ws), -1, np.int32)
    for fr in frames:
        x0, y0, x1, y1 = fr.bbox
        bx0, by0 = int(x0 * scale), int(y0 * scale)
        bx1, by1 = min(Ws, int(np.ceil(x1 * scale)) + 1), min(Hs, int(np.ceil(y1 * scale)) + 1)
        if bx1 <= bx0 or by1 <= by0:
            continue
        X, Y = np.meshgrid((np.arange(bx0, bx1) + 0.5) / scale, (np.arange(by0, by1) + 0.5) / scale)
        u, v = map_fn(fr, X, Y)
        sc, valid, _, _ = score_fn(fr, u, v, dist)
        sub_b = best[by0:by1, bx0:bx1]
        m = valid & (sc < sub_b)
        sub_b[m] = sc[m]
        label[by0:by1, bx0:bx1][m] = fr.index
    return label, (Ws, Hs)


def _consensus_outliers(lum: list, corners: list, valids: list, size, rel_thr=0.25, min_views=3,
                        tile=512):
    """
    Per frame, pixels that disagree with the other frames (moving objects).

    The per-pixel median over all frames that see a pixel is the consensus
    (static scene); a frame deviating from it by more than ``rel_thr`` (relative
    luminance) where at least ``min_views`` frames overlap is an outlier there.
    Computed tile by tile to keep memory small.
    """
    Ws, Hs = size
    out = [np.zeros(v.shape, bool) for v in valids]
    boxes = [(cx, cy, cx + v.shape[1], cy + v.shape[0]) for (cx, cy), v in zip(corners, valids)]
    for ty in range(0, Hs, tile):
        for tx in range(0, Ws, tile):
            tx1, ty1 = min(Ws, tx + tile), min(Hs, ty + tile)
            ks = [k for k, b in enumerate(boxes) if b[0] < tx1 and b[2] > tx and b[1] < ty1 and b[3] > ty]
            if len(ks) < min_views:
                continue
            stack = np.full((len(ks), ty1 - ty, tx1 - tx), np.nan, np.float32)
            for n, k in enumerate(ks):
                cx, cy, ex, ey = boxes[k]
                x0, y0, x1, y1 = max(tx, cx), max(ty, cy), min(tx1, ex), min(ty1, ey)
                if x1 <= x0 or y1 <= y0:
                    continue
                part = lum[k][y0 - cy:y1 - cy, x0 - cx:x1 - cx]
                vm = valids[k][y0 - cy:y1 - cy, x0 - cx:x1 - cx]
                stack[n, y0 - ty:y1 - ty, x0 - tx:x1 - tx] = np.where(vm, part, np.nan)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(stack, axis=0)
            views = np.isfinite(stack).sum(0)
            for n, k in enumerate(ks):
                cx, cy, ex, ey = boxes[k]
                x0, y0, x1, y1 = max(tx, cx), max(ty, cy), min(tx1, ex), min(ty1, ey)
                if x1 <= x0 or y1 <= y0:
                    continue
                sl = (slice(y0 - ty, y1 - ty), slice(x0 - tx, x1 - tx))
                dev = np.abs(stack[n][sl] - med[sl]) / (med[sl] + 0.05 * np.nanmax(med) + 1e-6)
                bad = (dev > rel_thr) & (views[sl] >= min_views)
                out[k][y0 - cy:y1 - cy, x0 - cx:x1 - cx] |= np.nan_to_num(bad).astype(bool)
    # remove speckle, then grow a little so whole objects (and their halo) are excluded
    k3 = np.ones((3, 3), np.uint8)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return [cv2.dilate(cv2.morphologyEx(o.astype(np.uint8), cv2.MORPH_OPEN, k3), k7) > 0 for o in out]


def optimise_seams(frames, dist, size, images: dict, score_fn, map_fn, gain_fn=None,
                   max_side=2500, band_px=40, deghost=True, log=print) -> SeamMap:
    """
    frames : RenderFrame list (bbox set, output coordinates)
    images : frame index -> (h x w x 3 float32 image in reference-lens px * img_scale, img_scale)
    gain_fn: optional callable(frame, X, Y) -> h x w x B gain (renderer low-frequency gains)
    deghost: exclude pixels where a frame disagrees with the consensus of the
             other frames (moving objects), as long as another frame covers them
    """
    W, H = size
    img_scale = min(s for _, s in images.values()) if images else 0.1
    # seam map not finer than the seam images themselves
    s = min(1.0, max_side / max(W, H), img_scale * 1.5)
    label, (Ws, Hs) = voronoi_labels(frames, dist, size, s, score_fn, map_fn)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))

    srcs, corners, valids, scores, idx = [], [], [], [], []
    for fr in frames:
        if fr.index not in images:
            continue
        x0, y0, x1, y1 = fr.bbox
        bx0, by0 = int(x0 * s), int(y0 * s)
        bx1, by1 = min(Ws, int(np.ceil(x1 * s)) + 1), min(Hs, int(np.ceil(y1 * s)) + 1)
        if bx1 <= bx0 or by1 <= by0:
            continue
        X, Y = np.meshgrid((np.arange(bx0, bx1) + 0.5) / s, (np.arange(by0, by1) + 0.5) / s)
        u, v = map_fn(fr, X, Y)
        sc, valid, ud, vd = score_fn(fr, u, v, dist)
        img, isc = images[fr.index]
        warped = cv2.remap(img.astype(np.float32), (ud * isc).astype(np.float32),
                           (vd * isc).astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if warped.ndim == 2:
            warped = np.dstack([warped] * 3)
        if gain_fn is not None:
            g = gain_fn(fr, X, Y)
            warped = warped * (g[..., :3] if g.shape[2] >= 3 else g[..., :1])
        # stay away from the very edge of a frame (interpolation, distortion residuals)
        valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        srcs.append(warped.astype(np.float32))
        corners.append((bx0, by0))
        valids.append(valid)
        scores.append(np.where(valid, sc, np.inf).astype(np.float32))
        idx.append(fr.index)
    if len(srcs) < 2:
        return SeamMap(label, s, 0.0)
    # common perceptual scaling so costs are comparable between frames
    allv = np.concatenate([sv[vm][:: max(1, int(vm.sum()) // 20000)] for sv, vm in zip(srcs, valids)
                           if vm.any()])
    hi = np.percentile(allv, 99.5, axis=0) + 1e-6
    srcs = [np.ascontiguousarray((np.sqrt(np.clip(sv / hi, 0, 1)) * 255.0).astype(np.float32))
            for sv in srcs]

    pos = {fi: k for k, fi in enumerate(idx)}
    usable = [vm.copy() for vm in valids]
    n_ghost = 0
    if deghost:
        lum = [sv.mean(2) / 255.0 for sv in srcs]
        outl = _consensus_outliers(lum, corners, valids, (Ws, Hs))
        # coverage by non-outlier frames, to never exclude the only usable view
        good_cover = np.zeros((Hs, Ws), np.int16)
        for k, (cx, cy) in enumerate(corners):
            h, w = valids[k].shape
            good_cover[cy:cy + h, cx:cx + w] += (valids[k] & ~outl[k])
        for k, (cx, cy) in enumerate(corners):
            h, w = valids[k].shape
            alt = good_cover[cy:cy + h, cx:cx + w] - (valids[k] & ~outl[k]) > 0
            drop = outl[k] & valids[k] & alt
            usable[k] &= ~drop
            n_ghost += int(drop.sum())
        # re-label pixels whose Voronoi frame is not usable there: best usable score wins
        best = np.full((Hs, Ws), np.inf, np.float32)
        relabel = np.full((Hs, Ws), -1, np.int32)
        for k, (cx, cy) in enumerate(corners):
            h, w = valids[k].shape
            sc = np.where(usable[k], scores[k], np.inf)
            sub = best[cy:cy + h, cx:cx + w]
            m = sc < sub
            sub[m] = sc[m]
            relabel[cy:cy + h, cx:cx + w][m] = idx[k]
        bad_lab = np.zeros((Hs, Ws), bool)
        for k, (cx, cy) in enumerate(corners):
            h, w = valids[k].shape
            lab = label[cy:cy + h, cx:cx + w]
            bad_lab[cy:cy + h, cx:cx + w] |= (lab == idx[k]) & ~usable[k]
        fix = bad_lab & (relabel >= 0)
        label = label.copy()
        label[fix] = relabel[fix]

    masks = []
    for k, (cx, cy) in enumerate(corners):
        h, w = valids[k].shape
        own = (label[cy:cy + h, cx:cx + w] == idx[k]).astype(np.uint8)
        band = cv2.dilate(own, kernel) > 0
        masks.append((usable[k] & band).astype(np.uint8) * 255)

    finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    umasks = [cv2.UMat(m) for m in masks]
    out = finder.find(srcs, corners, umasks)
    res = [m.get() if isinstance(m, cv2.UMat) else m for m in out]

    final = np.full((Hs, Ws), -1, np.int32)
    for k, fi in enumerate(idx):
        bx0, by0 = corners[k]
        m = (res[k] > 0) & usable[k]
        h, w = m.shape
        sub = final[by0:by0 + h, bx0:bx0 + w]
        lab = label[by0:by0 + h, bx0:bx0 + w]
        take = m & ((sub == -1) | (lab == fi))
        sub[take] = fi
    holes = (final == -1) & (label >= 0)
    final[holes] = label[holes]
    base, _ = voronoi_labels(frames, dist, size, s, score_fn, map_fn)
    changed = float(((final != base) & (base >= 0)).sum()) / max(1, int((base >= 0).sum()))
    msg = f"  seamlines optimised (graph cut): {changed * 100:.1f} % of the area moved to another frame"
    if deghost:
        msg += f"; {n_ghost * 100 / max(1, int((base >= 0).sum())):.1f} % of the area flagged as moving " \
               "objects in some frame and taken from a consistent one"
    log(msg)
    return SeamMap(final, s, changed)
