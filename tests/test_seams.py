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


def test_deghost_removes_object_seen_in_one_frame_only(tmp_path):
    world = _texture(900, 400, seed=7).astype(np.float32)
    frames, imgs = [], {}
    for k, x in enumerate((0, 150, 300)):
        img = world[:, x:x + 600].copy()
        if k == 1:
            img[60:340, 440 - x:460 - x] = 1500           # dark 'moving' object, only in the middle frame
        H = np.array([[1, 0, x], [0, 1, 0], [0, 0, 1]], np.float64)
        frames.append(RD.RenderFrame(k, "", H, (600, 400), (0, 0, 600, 400), {1: np.eye(3)},
                                     thumb=cv2.resize(img, (150, 100), interpolation=cv2.INTER_AREA).astype(np.uint16),
                                     thumb_scale=0.25, seam_img=img.astype(np.uint16), seam_scale=1.0))
        imgs[k] = img.astype(np.uint16)
    dist = RD.Distortion(np.zeros(2), np.array([300.0, 200.0]), 360.0)

    def render(deghost):
        rr = RD.Renderer(frames, dist, (900, 400), [(1, 0), (1, 1), (1, 2)], 3, np.uint16,
                         blend="disabled", work_dir=tmp_path / str(deghost), log=lambda *a: None,
                         seams="graphcut", deghost=deghost, strip=128)
        return np.array(rr.render(lambda fr: {1: imgs[fr.index]}))
    bar = (slice(60, 340), slice(440, 460))
    assert (render(False)[bar][..., 0] < 1600).mean() > 0.9      # middle frame wins without deghosting
    out = render(True)
    assert (out[bar][..., 0] < 1600).mean() < 0.05
    assert (out[..., 0] > 0).mean() > 0.99


def test_feather_weights_sum_to_one_and_ramp():
    from mosaic2d.seams import SeamMap
    H, W = 40, 80
    labels = np.zeros((H, W), np.int32)
    labels[:, 40:] = 1
    soft = {}
    for fi in (0, 1):
        ind = cv2.GaussianBlur((labels == fi).astype(np.float32), (0, 0), 1.0)
        soft[fi] = (0, 0, ind, np.ones((H, W), np.uint8))
    sm = SeamMap(labels, 0.1, 0.0, soft)                 # 1 seam cell = 10 output px
    X, Y = np.meshgrid(np.arange(0, 800, dtype=np.float64), np.full(1, 200.0))
    w0, w1 = sm.weight(0, X, Y)[0], sm.weight(1, X, Y)[0]
    assert np.allclose((w0 + w1)[20:780], 1, atol=1e-5)    # map border excluded (real maps are padded)
    assert w0[20:350].min() == 1 and w0[450:780].max() == 0
    ramp = np.flatnonzero((w0[20:780] > 0) & (w0[20:780] < 1))
    assert 3 <= ramp.size <= 25                          # narrow transition (~1 cell)


def test_seam_accumulator_weighted_mean():
    from mosaic2d.render import _SeamAccumulator
    out = np.full((70, 70, 2), 7, np.uint16)
    acc = _SeamAccumulator(2, tile=32)
    m = np.zeros((70, 70), bool)
    m[30:40, 25:45] = True
    acc.add(0, 0, m, np.full((70, 70, 2), 100.0, np.float32), np.full((70, 70), 0.25, np.float32))
    acc.add(0, 0, m, np.full((70, 70, 2), 200.0, np.float32), np.full((70, 70), 0.75, np.float32))
    acc.flush(out, 65535, np.uint16)
    assert (out[30:40, 25:45] == 175).all()
    assert (out[~m] == 7).all()


def test_seam_accumulator_keeps_directly_written_owner():
    """A lone band contribution (weight 0.25) mixes with the pixel's owner, which
    was written directly (weight >= 0.998), instead of replacing it."""
    from mosaic2d.render import _SeamAccumulator
    out = np.full((40, 40, 1), 1000, np.uint16)
    acc = _SeamAccumulator(1, tile=32)
    m = np.zeros((40, 40), bool)
    m[5:15, 5:35] = True
    acc.add(0, 0, m, np.full((40, 40, 1), 200.0, np.float32), np.full((40, 40), 0.25, np.float32))
    acc.flush(out, 65535, np.uint16)
    assert (out[5:15, 5:35] == 800).all()            # 0.25 * 200 + 0.75 * 1000
    empty = np.zeros((40, 40, 1), np.uint16)         # nothing written: band value alone
    acc.add(0, 0, m, np.full((40, 40, 1), 200.0, np.float32), np.full((40, 40), 0.25, np.float32))
    acc.flush(empty, 65535, np.uint16)
    assert (empty[5:15, 5:35] == 200).all()


def test_veiled_region_taken_from_sharp_frames(tmp_path):
    """A frame veiled in one area (out-of-focus object in front of the lens: low
    contrast, same mean brightness) gives that area to frames that show it sharply."""
    world = _texture(900, 400, seed=11).astype(np.float32)
    frames, imgs = [], {}
    for k, x in enumerate((0, 150, 300)):
        img = world[:, x:x + 600].copy()
        if k == 1:
            reg = img[80:320, 250:450]
            m = reg.mean(axis=(0, 1), keepdims=True)
            img[80:320, 250:450] = m + 0.25 * (reg - m)        # veil: contrast down to 25 %
        H = np.array([[1, 0, x], [0, 1, 0], [0, 0, 1]], np.float64)
        frames.append(RD.RenderFrame(k, "", H, (600, 400), (0, 0, 600, 400), {1: np.eye(3)},
                                     thumb=cv2.resize(img, (150, 100), interpolation=cv2.INTER_AREA).astype(np.uint16),
                                     thumb_scale=0.25, seam_img=img.astype(np.uint16), seam_scale=1.0))
        imgs[k] = img.astype(np.uint16)
    dist = RD.Distortion(np.zeros(2), np.array([300.0, 200.0]), 360.0)

    def render(deghost):
        rr = RD.Renderer(frames, dist, (900, 400), [(1, 0), (1, 1), (1, 2)], 3, np.uint16,
                         blend="disabled", work_dir=tmp_path / f"v{deghost}", log=lambda *a: None,
                         seams="graphcut", deghost=deghost, strip=128)
        return np.array(rr.render(lambda fr: {1: imgs[fr.index]})).astype(np.float32)
    area = (slice(120, 280), slice(440, 560))                  # inside the veil, world coordinates
    truth = world[area][..., 0]

    def detail(a):
        return np.abs(cv2.Laplacian(a[area][..., 0], cv2.CV_32F)).mean()
    assert detail(render(False)) < 0.5 * detail(world)        # middle frame (veiled) wins
    sharp = render(True)
    assert detail(sharp) > 0.85 * detail(world)
    assert np.abs(sharp[area][..., 0] - truth).mean() < 0.05 * truth.mean()


def test_gain_field_ignores_object_seen_in_one_frame(tmp_path):
    """A bright object in one frame only must not darken that frame around it
    (the darkening would stay after deghosting removes the object)."""
    world = _texture(900, 400, seed=3).astype(np.float32)
    frames = []
    for k, x in enumerate((0, 150, 300)):
        img = world[:, x:x + 600].copy()
        if k == 1:
            img[:, 440 - x:470 - x] = 60000                    # bright rope, middle frame only
        H = np.array([[1, 0, x], [0, 1, 0], [0, 0, 1]], np.float64)
        frames.append(RD.RenderFrame(k, "", H, (600, 400), (0, 0, 600, 400), {1: np.eye(3)},
                                     thumb=cv2.resize(img, (150, 100), interpolation=cv2.INTER_AREA).astype(np.uint16),
                                     thumb_scale=0.25))
    dist = RD.Distortion(np.zeros(2), np.array([300.0, 200.0]), 360.0)
    rr = RD.Renderer(frames, dist, (900, 400), [(1, 0), (1, 1), (1, 2)], 3, np.uint16,
                     blend="mosaic", work_dir=tmp_path / "g", log=lambda *a: None, seams="voronoi")
    rr.low_frequency_pass()
    X, Y = np.meshgrid(np.array([400.0, 420.0, 490.0, 510.0]), np.array([150.0, 250.0]))
    g = rr._gain_at(frames[1], X, Y)
    assert np.abs(g - 1).max() < 0.12, g[..., 0]
    RD.cleanup(rr)


def test_gain_field_stable_where_all_frames_disagree(tmp_path):
    """Relief / parallax: every frame shows the overlap differently. Nothing is a
    minority outlier there, so the gains stay close to the plain estimate."""
    frames = []
    for k, x in enumerate((0, 150, 300)):
        img = _texture(600, 400, seed=20 + k).astype(np.float32)       # unrelated content
        H = np.array([[1, 0, x], [0, 1, 0], [0, 0, 1]], np.float64)
        frames.append(RD.RenderFrame(k, "", H, (600, 400), (0, 0, 600, 400), {1: np.eye(3)},
                                     thumb=cv2.resize(img, (150, 100), interpolation=cv2.INTER_AREA).astype(np.uint16),
                                     thumb_scale=0.25))
    dist = RD.Distortion(np.zeros(2), np.array([300.0, 200.0]), 360.0)

    def gains(robust):
        if not robust:
            orig = RD.Renderer._gain_inliers
            RD.Renderer._gain_inliers = staticmethod(lambda own, L, **kw: {fi: v[3] for fi, v in own.items()})
        try:
            rr = RD.Renderer(frames, dist, (900, 400), [(1, 0), (1, 1), (1, 2)], 3, np.uint16,
                             blend="mosaic", work_dir=tmp_path / str(robust), log=lambda *a: None,
                             seams="voronoi")
            rr.low_frequency_pass()
        finally:
            if not robust:
                RD.Renderer._gain_inliers = staticmethod(orig)
        X, Y = np.meshgrid(np.arange(320.0, 580.0, 20.0), np.array([120.0, 200.0, 280.0]))
        return np.array([rr._gain_at(f, X, Y) for f in frames])
    plain, robust = gains(False), gains(True)
    assert np.abs(np.log(robust / plain)).max() < np.log(1.4)      # no blow-up (was 3x on MS rocks)
