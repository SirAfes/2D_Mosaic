"""Seamline optimisation and flat-field estimation."""

import cv2
import numpy as np

from mosaic2d import render as RD
from mosaic2d.seams import estimate_flatfield

from test_pipeline import _texture


def _two_frames(object_in_a=True):
    """Two 600 x 400 frames overlapping by 300 px; frame A shows a 'moving' bar
    across the middle of the overlap (exactly where the Voronoi seam falls)."""
    world = _texture(900, 400, seed=5).astype(np.float32)
    a = world[:, :600].copy()
    b = world[:, 300:900].copy()
    if object_in_a:
        a[60:340, 440:460] = 60000      # bar at world x = 440..460 (seam at x = 450)
    frames = []
    for k, (img, x) in enumerate(((a, 0), (b, 300))):
        H = np.array([[1, 0, x], [0, 1, 0], [0, 0, 1]], np.float64)
        fr = RD.RenderFrame(k, "", H, (600, 400), (0, 0, 600, 400), {1: np.eye(3)},
                            thumb=cv2.resize(img, (150, 100), interpolation=cv2.INTER_AREA).astype(np.uint16),
                            thumb_scale=0.25, seam_img=img.astype(np.uint16), seam_scale=1.0)
        frames.append(fr)
    return frames, {0: a.astype(np.uint16), 1: b.astype(np.uint16)}


def _render(frames, imgs, seams, tmp_path):
    dist = RD.Distortion(np.zeros(2), np.array([300.0, 200.0]), 360.0)
    rr = RD.Renderer(frames, dist, (900, 400), [(1, 0), (1, 1), (1, 2)], 3, np.uint16,
                     blend="disabled", work_dir=tmp_path / seams, log=lambda *a: None,
                     seams=seams, strip=128)
    out = np.array(rr.render(lambda fr: {1: imgs[fr.index]}))
    return out


def test_graphcut_seam_avoids_cutting_moving_object(tmp_path):
    frames, imgs = _two_frames()
    vor = _render(frames, imgs, "voronoi", tmp_path)
    gc = _render(frames, imgs, "graphcut", tmp_path)
    bar = (slice(60, 340), slice(440, 460))
    frac_v = (vor[bar][..., 0] > 59000).mean()
    frac_g = (gc[bar][..., 0] > 59000).mean()
    # the Voronoi seam at x = 450 cuts the bar in half ...
    assert 0.3 < frac_v < 0.7
    # ... the optimised seam takes it completely or not at all
    assert frac_g < 0.1 or frac_g > 0.9, frac_g
    assert (gc[..., 0] > 0).mean() > 0.99          # no holes


def test_flatfield_recovers_vignetting():
    rng = np.random.default_rng(0)
    h, w = 80, 120
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    vign = 1 - 0.25 * r2                              # corners at 0.5 of the centre
    thumbs, valids = [], []
    for k in range(20):
        scene = _texture(w, h, seed=k)[..., :2].astype(np.float32)
        thumbs.append(scene * vign[..., None] * rng.uniform(0.7, 1.3))
        valids.append(np.ones((h, w), bool))
    flat, stats = estimate_flatfield(thumbs, valids)
    assert flat.shape == (h, w, 2)
    ratio = flat[2, 2, 0] / flat[h // 2, w // 2, 0]
    true = vign[2, 2] / vign[h // 2, w // 2]
    assert abs(ratio - true) < 0.08, (ratio, true)
