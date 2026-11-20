"""Reference band choice: matches between consecutive photos, clipped bands penalised."""

import cv2
import numpy as np

from mosaic2d import features as FT
from mosaic2d import matching as MT
from mosaic2d.bands import band_saturation, choose_reference_band
from mosaic2d.sources import Band, CameraProfile, FrameData


def _world(seed=0, w=1400, h=900):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for s in (2, 6, 18, 50):
        n = rng.normal(0, 1, (h // s + 2, w // s + 2)).astype(np.float32)
        img += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) * s ** 0.5
    return (img - img.min()) / (img.max() - img.min())


def test_band_saturation():
    a = np.full((100, 100), 2000, np.uint16)
    a[:10] = 4095                       # 12-bit data in 16-bit words, 10 % clipped
    a[90:] = 0                          # no data: not counted
    sat, dark = band_saturation(a)
    assert abs(sat - 10 / 90) < 0.01 and dark == 0


def test_textured_unclipped_band_wins():
    world = _world()
    rng = np.random.default_rng(1)
    frames = []
    for k in range(4):
        x0 = 100 + 150 * k
        v = world[100:700, x0:x0 + 800]
        good = (v * 3000 + 500).astype(np.uint16)                       # textured, 12-bit
        clipped = np.minimum(v * 9000 + 500, 4095).astype(np.uint16)    # same texture, 40 % clipped
        noise = (rng.random(v.shape) * 3000 + 500).astype(np.uint16)    # no stable texture
        frames.append(FrameData({1: np.dstack([noise, clipped, good])}, {}, v.shape))
    prof = CameraProfile("t", bands=[Band("405 nm", 1, 0, 405.0), Band("530 nm", 1, 1, 530.0),
                                     Band("560 nm", 1, 2, 560.0)])
    reg = FT.build_registry(3000)
    p = MT.MatchParams(guided=False)
    best, scores = choose_reference_band(frames, prof, reg["sift"].build,
                                         lambda a, b: MT.match_pair(a, b, 0, 1, p), 1.0,
                                         log=lambda *a: None)
    assert best.name == "560 nm", scores
    assert scores["405 nm"] == 0 and scores["530 nm"] < 0.5 * scores["560 nm"], scores
