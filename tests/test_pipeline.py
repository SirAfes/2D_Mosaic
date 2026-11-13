"""End-to-end test on a synthetic survey with known geometry."""

import json

import cv2
import numpy as np
import pytest

from mosaic2d import rawio
from mosaic2d.pipeline import Settings, run


def _texture(w, h, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for s in (2, 6, 18, 50):
        n = rng.normal(0, 1, (h // s + 2, w // s + 2)).astype(np.float32)
        img += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) * s ** 0.5
    img = (img - img.min()) / (img.max() - img.min())
    rgb = np.dstack([img, img ** 1.3, 1 - img * 0.5])
    return (rgb * 50000 + 2000).astype(np.uint16)


def _survey(tmp_path, n_cols=4, n_rows=2, w=900, h=600, seed=1):
    rng = np.random.default_rng(seed)
    world = _texture(3600, 1800, seed)
    truth = {}
    k = 0
    for r in range(n_rows):
        cols = range(n_cols) if r % 2 == 0 else reversed(range(n_cols))  # zig-zag
        for c in cols:
            x, y = 250 + c * 600, 250 + r * 420
            a = np.deg2rad(rng.normal(0, 3))
            s = 1 + rng.normal(0, 0.03)
            # camera -> world
            H = np.array([[s * np.cos(a), -s * np.sin(a), x], [s * np.sin(a), s * np.cos(a), y], [0, 0, 1]])
            H[2, :2] = rng.normal(0, 1.5e-5, 2)
            img = cv2.warpPerspective(world, np.linalg.inv(H), (w, h), flags=cv2.INTER_LINEAR)
            img = img.astype(np.float32) * np.float32(1 - 0.25 * ((np.linspace(-1, 1, w)[None] ** 2
                                                                + np.linspace(-1, 1, h)[:, None] ** 2) / 2))[..., None]
            img = np.clip(img + rng.normal(0, 300, img.shape), 0, 65535).astype(np.uint16)
            fn = tmp_path / f"IMG_{k:03d}.tif"
            cv2.imwrite(str(fn), img[..., ::-1])
            truth[fn.name] = H
            k += 1
    return truth


@pytest.mark.parametrize("blend", ["mosaic", "disabled"])
def test_synthetic_survey(tmp_path, blend):
    src = tmp_path / "src"
    src.mkdir()
    truth = _survey(src)
    s = Settings(methods="sift,orb,fast", accuracy="high", blend=blend, resolution=0.5,
                 distortion=False, n_features=3000, orient="anchor", jobs=2)
    proj, rep = run(src, tmp_path / "out", s)
    assert rep["aligned_images"] == len(truth)
    assert rep["reprojection_median_px"] < 1.0
    # geometry: compare every image's centre relative to the anchor with the truth
    names = [f.path.name for f in proj.frames]
    a = proj.result.anchor
    Ha_true = truth[names[a]]
    errs = []
    for f in proj.frames:
        rel_true = np.linalg.inv(Ha_true) @ truth[f.path.name]   # image -> anchor
        rel_est = proj.result.H[f.index]
        c = np.array([[[450.0, 300.0], [0, 0], [900, 600]]])
        errs.append(np.abs(cv2.perspectiveTransform(c, rel_true) - cv2.perspectiveTransform(c, rel_est)).max())
    assert max(errs) < 3.0, errs
    out = tmp_path / "out" / "mosaic.tif"
    assert out.is_file()
    img = rawio.read_image(out)
    assert img.dtype == np.uint16 and img.shape[2] == 3
    assert (img > 0).mean() > 0.4
    rep2 = json.loads((tmp_path / "out" / "mosaic_report.json").read_text())
    assert rep2["pairs_accepted"] >= len(truth) - 1


def test_project_rerender(tmp_path):
    from mosaic2d.pipeline import Project
    src = tmp_path / "src"
    src.mkdir()
    _survey(src, n_cols=3, n_rows=1)
    s = Settings(methods="orb", accuracy="high", resolution=0.3, distortion=False, n_features=2000, jobs=2)
    run(src, tmp_path / "out", s)
    proj = Project.load(tmp_path / "out" / "mosaic_project.pkl", tmp_path / "out2",
                        blend="disabled", resolution=0.2)
    proj.render("again")
    a = rawio.read_image(tmp_path / "out" / "mosaic.tif")
    b = rawio.read_image(tmp_path / "out2" / "again.tif")
    assert abs(b.shape[1] / a.shape[1] - 2 / 3) < 0.02


def test_blurred_frame_detected_from_overlaps(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _survey(src, n_cols=4, n_rows=1)
    blurred = src / "IMG_002.tif"
    img = cv2.imread(str(blurred), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(blurred), cv2.GaussianBlur(img, (0, 0), 2.5))
    s = Settings(methods="sift", accuracy="high", resolution=0.25, distortion=False,
                 n_features=3000, jobs=2)
    proj, rep = run(src, tmp_path / "out", s)
    q = {f.path.name: f.quality for f in proj.frames}
    assert q["IMG_002.tif"] < 0.5, q
    assert all(v > 0.7 for k, v in q.items() if k != "IMG_002.tif"), q


def test_ingest_cache_reused(tmp_path):
    from mosaic2d.pipeline import Project
    src = tmp_path / "src"
    src.mkdir()
    _survey(src, n_cols=3, n_rows=1)
    s = Settings(methods="orb", accuracy="high", resolution=0.2, distortion=False, n_features=2000, jobs=2)
    p1 = Project(src, tmp_path / "out", s)
    p1.ingest()
    p2 = Project(src, tmp_path / "out", s)
    p2.ingest()
    assert len(list((tmp_path / "out" / "_cache").glob("*.pkl"))) == 3
    assert all(np.array_equal(a.reg, b.reg) for a, b in zip(p1.frames, p2.frames))
    # a changed setting that affects ingest must not hit the cache
    s2 = Settings(methods="orb", accuracy="lowest", resolution=0.2, distortion=False, n_features=2000, jobs=2)
    p3 = Project(src, tmp_path / "out", s2)
    p3.ingest()
    assert p3.frames[0].reg.shape != p1.frames[0].reg.shape
