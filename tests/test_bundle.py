import numpy as np, cv2
from mosaic2d import bundle as B


def _scene(n=12, model="homography", k=(0.0, 0.0), noise=0.3, seed=1):
    rng = np.random.default_rng(seed)
    w, h = 2000, 1300
    Hs = {}
    for i in range(n):
        row, col = divmod(i, 4)
        x = col * 1100 + (row % 2) * 0 ; y = row * 700
        a = np.deg2rad(rng.normal(0, 4)); s = 1 + rng.normal(0, 0.05)
        H = np.array([[s*np.cos(a), -s*np.sin(a), x], [s*np.sin(a), s*np.cos(a), y], [0, 0, 1.0]])
        if model == "homography":
            H[2, :2] = rng.normal(0, 2e-5, 2)
        Hs[i] = H
    Hs0 = Hs[0].copy()
    Hs = {i: np.linalg.inv(Hs0) @ H for i, H in Hs.items()}
    center = np.array([w/2, h/2]); R = np.hypot(w, h)/2
    def distort(pu):  # inverse of undistort (iterative)
        p = pu.copy()
        for _ in range(20):
            p = p - (B.undistort_points(p, np.array(k), center, R) - pu)
        return p
    corr, pairH = [], {}
    for i in range(n):
        for j in range(i+1, n):
            A = np.linalg.inv(Hs[j]) @ Hs[i]
            pu = rng.uniform([0, 0], [w, h], (400, 2))
            qu = cv2.perspectiveTransform(pu[None], A)[0]
            m = (qu[:, 0] > 0) & (qu[:, 0] < w) & (qu[:, 1] > 0) & (qu[:, 1] < h)
            if m.sum() < 60: continue
            p = distort(pu[m]) + rng.normal(0, noise, (m.sum(), 2))
            q = distort(qu[m]) + rng.normal(0, noise, (m.sum(), 2))
            # 10 % gross outliers
            bad = rng.random(len(p)) < 0.1
            q[bad] += rng.uniform(-200, 200, (bad.sum(), 2))
            corr.append(B.Correspondences(i, j, p, q, 1.0, "synthetic"))
            pairH[(i, j)] = A @ (np.eye(3) + np.r_[rng.normal(0, 1e-3, 6), 0, 0, 0].reshape(3, 3))
    sizes = {i: (w, h) for i in range(n)}
    return Hs, corr, pairH, sizes


def _err(Hs, est, sizes):
    # compare relative to image 0 frame
    w, h = sizes[0]
    c = np.array([[0, 0], [w, 0], [w, h], [0, h], [w/2, h/2]], float)
    G = np.linalg.inv(est[0])
    e = []
    for i in Hs:
        a = cv2.perspectiveTransform(c[None], Hs[i])[0]
        b = cv2.perspectiveTransform(c[None], G @ est[i])[0]
        e.append(np.linalg.norm(a - b, axis=1).max())
    return max(e)


def test_bundle_recovers_homographies():
    Hs, corr, pairH, sizes = _scene()
    r = B.run_bundle(list(range(12)), corr, pairH, sizes, model="homography", distortion=False, log=lambda *a: None)
    assert r.median_px < 1.0
    assert _err(Hs, r.H, sizes) < 3.0


def test_bundle_estimates_radial_distortion():
    Hs, corr, pairH, sizes = _scene(model="similarity", k=(-0.06, 0.01))
    r = B.run_bundle(list(range(12)), corr, pairH, sizes, model="similarity", distortion=True, log=lambda *a: None)
    assert abs(r.k[0] - (-0.06)) < 0.01, r.k
    assert r.median_px < 1.0
    assert _err(Hs, r.H, sizes) < 4.0
