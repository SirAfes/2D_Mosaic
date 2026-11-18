"""Camera-over-plane adjustment: synthetic surveys with known cameras."""

import cv2
import numpy as np

from mosaic2d import bundle as BA
from mosaic2d import camera as CAM

F_TRUE = 3000.0
SIZE = (1200, 800)


def _survey(n_cols=10, n_rows=2, seed=0, tilt_deg=4.0, height_jitter=0.03, noise=0.3):
    """Cameras over the plane Z = 0 in a lawn-mower pattern; tie points from random plane points."""
    rng = np.random.default_rng(seed)
    h0 = 1.0
    gsd = h0 / F_TRUE                                   # plane units per px at nadir
    step_x, step_y = 0.45 * SIZE[0] * gsd, 0.55 * SIZE[1] * gsd
    poses = []
    for r in range(n_rows):
        for c in range(n_cols):
            x = c * step_x if r % 2 == 0 else (n_cols - 1 - c) * step_x
            t = np.radians(tilt_deg) * rng.uniform(-1, 1, 2)
            poses.append([x, r * step_y, np.log(h0 * (1 + height_jitter * rng.uniform(-1, 1))),
                          t[0], t[1], rng.uniform(-0.15, 0.15)])
    poses = np.array(poses)
    c = (SIZE[0] / 2.0, SIZE[1] / 2.0)
    G = [CAM.plane_to_image(P, F_TRUE, c) for P in poses]
    Hinv = [np.linalg.inv(g) for g in G]
    corr, pairH = [], {}
    n = len(poses)
    for i in range(n):
        for j in range(i + 1, n):
            # random points in image i -> plane -> image j
            pi = np.column_stack([rng.uniform(0, SIZE[0], 600), rng.uniform(0, SIZE[1], 600)])
            X = cv2.perspectiveTransform(pi[None], Hinv[i])[0]
            pj = cv2.perspectiveTransform(X[None], G[j])[0]
            ok = (pj[:, 0] > 0) & (pj[:, 0] < SIZE[0]) & (pj[:, 1] > 0) & (pj[:, 1] < SIZE[1])
            if ok.sum() < 60:
                continue
            p = pi[ok] + rng.normal(0, noise, (ok.sum(), 2))
            q = pj[ok] + rng.normal(0, noise, (ok.sum(), 2))
            corr.append(BA.Correspondences(i, j, p, q))
            Hij, _ = cv2.findHomography(p, q, 0)
            pairH[(i, j)] = Hij
    return poses, corr, pairH, {k: SIZE for k in range(n)}


def _similarity_fit(a, b):
    """Least-squares similarity a -> b (Umeyama); returns the mapped a."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, d])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (A ** 2).sum()
    return (s * (R @ A.T)).T + mb


def _plane_points(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float64)[None], H)[0]


def test_plane_homography_roundtrip():
    P = [0.2, -0.1, np.log(1.3), 0.03, -0.02, 0.4]
    G = CAM.plane_to_image(P, 2500.0, (600, 400))
    H = CAM.image_to_plane(P, 2500.0, (600, 400))
    x = np.array([[100.0, 50.0], [700.0, 300.0]])
    assert np.allclose(_plane_points(G, _plane_points(H, x)), x, atol=1e-6)


def test_camera_adjustment_recovers_survey_geometry():
    poses, corr, pairH, sizes = _survey()
    n = len(poses)
    r = CAM.run_camera_bundle(list(range(n)), corr, pairH, sizes, focal_px=F_TRUE * 1.1,
                              distortion=False, log=lambda *a: None)
    assert len(r.components[0]) == n
    assert r.median_px < 1.0, r.median_px
    # estimated mosaic vs the true plane: one similarity for all photos (no drift, no shear)
    c = (SIZE[0] / 2.0, SIZE[1] / 2.0)
    grid = np.array([[x, y] for x in (0, 600, 1200) for y in (0, 400, 800)], np.float64)
    est, true = [], []
    for k in range(n):
        est.append(_plane_points(r.H[k], grid))
        true.append(_plane_points(CAM.image_to_plane(poses[k], F_TRUE, c), grid))
    est, true = np.concatenate(est), np.concatenate(true)
    fit = _similarity_fit(est, true)
    gsd = 1.0 / F_TRUE
    err_px = np.hypot(*(fit - true).T) / gsd
    assert np.median(err_px) < 2.0 and err_px.max() < 12.0, (np.median(err_px), err_px.max())
    # relative heights (photo scales) recovered
    h_est = np.array([r.camera["height_rel"][k] for k in range(n)])
    h_true = np.exp(poses[:, 2])
    ratio = (h_est / np.median(h_est)) / (h_true / np.median(h_true))
    assert np.abs(ratio - 1).max() < 0.01, ratio


def test_camera_model_does_not_drift_where_homography_does():
    """A long single strip: free homographies pick up scale drift and shear along the
    chain; the camera model keeps the photos at their true size."""
    poses, corr, pairH, sizes = _survey(n_cols=24, n_rows=1, seed=4, noise=0.8)
    n = len(poses)
    c = (SIZE[0] / 2.0, SIZE[1] / 2.0)
    true_scale = [BA_scale(CAM.image_to_plane(P, F_TRUE, c), c) for P in poses]

    def spread(H):
        sc = np.array([BA_scale(H[k], c) for k in range(n)]) / np.array(true_scale)
        return sc.max() / sc.min()
    rc = CAM.run_camera_bundle(list(range(n)), corr, pairH, sizes, focal_px=F_TRUE,
                               distortion=False, log=lambda *a: None)
    rh = BA.run_bundle(list(range(n)), corr, pairH, sizes, model="homography", distortion=False,
                       log=lambda *a: None)
    # measured over seeds: camera 0.7-1.5 %, homography 6-22 %, similarity 9 %
    assert spread(rc.H) < 1.03
    assert spread(rc.H) - 1 < 0.5 * (spread(rh.H) - 1)


def BA_scale(H, c):
    from mosaic2d.pipeline import local_scale
    return local_scale(H, *c)
