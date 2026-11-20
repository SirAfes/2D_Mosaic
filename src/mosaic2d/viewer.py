"""
Quick viewer for large mosaics, multispectral ones included.

    mosaic2d view mosaic.tif        (or "View a mosaic..." in the window)

Common image viewers cannot show a 14-band 16-bit TIFF, and GIS tools build
pyramids first. This viewer reads only the tiles that are on screen, from the
overview level that matches the zoom (mosaic2d and Metashape/GDAL write these
inside the TIFF), and maps any three bands - or one - to the screen with a
percentile stretch. Files without overviews (older mosaic2d output, strip
TIFFs) get one small overview built in the background when they are opened.

The reading and drawing code (TiffMosaic, View) does not depend on Tk.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


class TiffMosaic:
    """Random access to a (Big)TIFF and its overview levels, tile by tile."""

    def __init__(self, path, cache_mb: int = 512):
        import tifffile
        self.path = Path(path)
        self.tif = tifffile.TiffFile(str(self.path))
        s = self.tif.series[0]
        pages = [lv.keyframe for lv in s.levels]
        p0 = pages[0]
        self.W, self.H = int(p0.imagewidth), int(p0.imagelength)
        self.S = int(p0.samplesperpixel)
        self.dtype = p0.dtype
        if len(s.levels) == 1 and len(self.tif.pages) > 1:
            # overviews as plain reduced-resolution pages (GDAL style) that the series missed
            extra = [p for p in self.tif.pages[1:]
                     if getattr(p, "samplesperpixel", 0) == self.S and p.imagewidth < self.W]
            pages += sorted(extra, key=lambda p: -p.imagewidth)
        self.levels = [self._level(p) for p in pages]
        self._fh = open(self.path, "rb")
        self._lock = threading.Lock()
        self._cache: OrderedDict = OrderedDict()
        self._cache_bytes = 0
        self._cache_max = cache_mb * 2 ** 20
        self.band_names = self._band_names()
        self.nodata = 0
        self.synthetic = None            # overview built here for files without one

    # ------------------------------------------------------------------ structure
    def _level(self, page):
        W, H = int(page.imagewidth), int(page.imagelength)
        mm = None
        try:                       # uncompressed, contiguous image data: read slices directly
            if int(page.compression) == 1 and page.is_contiguous and page.fillorder == 1:
                off = int(page.dataoffsets[0])
                shape = (self.S, H, W) if int(page.planarconfig) == 2 and self.S > 1 else (H, W, self.S)
                mm = np.memmap(self.path, dtype=page.dtype.newbyteorder(page.parent.byteorder),
                               mode="r", offset=off, shape=shape)
        except Exception:                                        # noqa: BLE001
            mm = None
        if page.is_tiled:
            ch, cw = int(page.tilelength), int(page.tilewidth)
        else:
            rps = int(page.rowsperstrip or H)
            ch, cw = min(rps, H), W
        return {"page": page, "W": W, "H": H, "ch": ch, "cw": cw,
                "ny": math.ceil(H / ch), "nx": math.ceil(W / cw),
                "planar": int(page.planarconfig) == 2 and self.S > 1, "factor": self.W / W, "mm": mm}

    def _band_names(self):
        try:
            from .rawio import tiff_band_info
            info = tiff_band_info(self.path)
        except Exception:
            info = {}
        names = []
        for b in range(self.S):
            nm, wl = info.get(b, (None, None)) if isinstance(info, dict) else (None, None)
            names.append((nm or f"band {b + 1}", wl))
        if self.S in (3, 4) and not any(wl for _, wl in names):
            rgb = ["red", "green", "blue", "alpha"]
            names = [(rgb[b], None) for b in range(self.S)]
        return names

    def default_bands(self):
        """Natural-colour guess: bands nearest 650/550/450 nm, else R, G, B."""
        wl = [w for _, w in self.band_names]
        if self.S >= 3 and not any(wl):
            return (0, 1, 2)
        if not any(wl):
            return (0, 0, 0)

        def near(x):
            return int(np.argmin([abs((w or 1e9) - x) for w in wl]))
        return (near(650), near(550), near(450))

    def nir_bands(self):
        """False colour: NIR / red / green when there is a band beyond 700 nm."""
        wl = [w or 0 for _, w in self.band_names]
        if max(wl) < 700:
            return None
        nir = int(np.argmax(wl))
        red = int(np.argmin([abs(w - 650) for w in wl]))
        green = int(np.argmin([abs(w - 550) for w in wl]))
        return (nir, red, green)

    # ------------------------------------------------------------------ reading
    def _chunk(self, lv, k, cy, cx, s):
        """Decoded chunk (rows, cols, samples) for sample plane s (planar) or all samples."""
        key = (k, cy, cx, s if lv["planar"] else -1)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        page = lv["page"]
        idx = cy * lv["nx"] + cx
        if lv["planar"]:
            idx += s * lv["nx"] * lv["ny"]
        off, cnt = page.dataoffsets[idx], page.databytecounts[idx]
        h = min(lv["ch"], lv["H"] - cy * lv["ch"])
        w = min(lv["cw"], lv["W"] - cx * lv["cw"])
        nsamp = 1 if lv["planar"] else self.S
        if cnt == 0 or off == 0:                         # sparse file: empty tile
            arr = np.zeros((h, w, nsamp), page.dtype)
        else:
            with self._lock:
                self._fh.seek(off)
                data = self._fh.read(cnt)
            try:
                seg, _, _ = page.decode(data, idx, jpegtables=page.jpegtables)
            except KeyError as e:          # e.g. LZW / JPEG without the imagecodecs package
                raise RuntimeError(f"cannot decode this TIFF's compression ({e}); "
                                   "install the 'imagecodecs' package") from None
            seg = np.asarray(seg)
            seg = seg.reshape(-1, seg.shape[-2], seg.shape[-1])     # (rows, cols, samples)
            arr = np.ascontiguousarray(seg[:h, :w])
        with self._lock:
            self._cache[key] = arr
            self._cache_bytes += arr.nbytes
            while self._cache_bytes > self._cache_max and len(self._cache) > 1:
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= old.nbytes
        return arr

    def read(self, k, x0, y0, x1, y1, bands):
        """Pixels [y0:y1, x0:x1] of level k (level coordinates, may extend past the
        edges: filled with nodata) for the given bands -> (h, w, len(bands))."""
        if k == "synthetic":
            ov = self.synthetic
            out = np.zeros((y1 - y0, x1 - x0, len(bands)), ov.dtype)
            ys, ye = max(0, y0), min(ov.shape[0], y1)
            xs, xe = max(0, x0), min(ov.shape[1], x1)
            if ye > ys and xe > xs:
                out[ys - y0:ye - y0, xs - x0:xe - x0] = ov[ys:ye, xs:xe][..., list(bands)]
            return out
        lv = self.levels[k]
        out = np.zeros((y1 - y0, x1 - x0, len(bands)), lv["page"].dtype)
        ys, ye = max(0, y0), min(lv["H"], y1)
        xs, xe = max(0, x0), min(lv["W"], x1)
        if ye <= ys or xe <= xs:
            return out
        if lv["mm"] is not None:
            mm = lv["mm"]
            for n, b in enumerate(bands):
                out[ys - y0:ye - y0, xs - x0:xe - x0, n] = (mm[b, ys:ye, xs:xe] if lv["planar"]
                                                            else mm[ys:ye, xs:xe, b])
            return out
        for cy in range(ys // lv["ch"], (ye - 1) // lv["ch"] + 1):
            for cx in range(xs // lv["cw"], (xe - 1) // lv["cw"] + 1):
                ty, tx = cy * lv["ch"], cx * lv["cw"]
                a0, a1 = max(ys, ty), min(ye, ty + lv["ch"])
                b0, b1 = max(xs, tx), min(xe, tx + lv["cw"])
                if lv["planar"]:
                    for n, b in enumerate(bands):
                        c = self._chunk(lv, k, cy, cx, b)
                        out[a0 - y0:a1 - y0, b0 - x0:b1 - x0, n] = c[a0 - ty:a1 - ty, b0 - tx:b1 - tx, 0]
                else:
                    c = self._chunk(lv, k, cy, cx, 0)
                    out[a0 - y0:a1 - y0, b0 - x0:b1 - x0] = c[a0 - ty:a1 - ty, b0 - tx:b1 - tx][..., list(bands)]
        return out

    def value_at(self, x, y):
        """All band values of full-resolution pixel (x, y), or None outside."""
        if not (0 <= x < self.W and 0 <= y < self.H):
            return None
        return self.read(0, int(x), int(y), int(x) + 1, int(y) + 1, list(range(self.S)))[0, 0]

    # ------------------------------------------------------------------ overviews
    def needs_overview(self, max_side=4096):
        return max(self.levels[-1]["W"], self.levels[-1]["H"]) > max_side and self.synthetic is None

    def build_overview(self, max_side=2048, progress=None, stop=None):
        """Small overview from a full pass over the coarsest level (nearest sampling)."""
        k = len(self.levels) - 1
        lv = self.levels[k]
        f = max(1, math.ceil(max(lv["W"], lv["H"]) / max_side))
        oh, ow = math.ceil(lv["H"] / f), math.ceil(lv["W"] / f)
        ov = np.zeros((oh, ow, self.S), lv["page"].dtype)
        total = lv["ny"] * lv["nx"]
        done = 0
        for cy in range(lv["ny"]):
            for cx in range(lv["nx"]):
                if stop is not None and stop.is_set():
                    return None
                ty, tx = cy * lv["ch"], cx * lv["cw"]
                if lv["planar"]:
                    c = np.concatenate([self._chunk(lv, k, cy, cx, s) for s in range(self.S)], axis=2)
                else:
                    c = self._chunk(lv, k, cy, cx, 0)
                oy0, ox0 = -(-ty // f), -(-tx // f)            # first output row/col inside the chunk
                sub = c[oy0 * f - ty::f, ox0 * f - tx::f]
                ov[oy0:oy0 + sub.shape[0], ox0:ox0 + sub.shape[1]] = sub
                done += 1
                if progress and done % 16 == 0:
                    progress(done / total)
        # chunks of the full pass are not useful later
        with self._lock:
            self._cache.clear()
            self._cache_bytes = 0
        self.synthetic = ov
        self.synthetic_factor = self.W / ow
        return ov

    def overview(self):
        """(array, factor) of the smallest available view of the whole mosaic."""
        if self.synthetic is not None:
            return self.synthetic, self.synthetic_factor
        k = len(self.levels) - 1
        lv = self.levels[k]
        return self.read(k, 0, 0, lv["W"], lv["H"], list(range(self.S))), lv["factor"]

    def close(self):
        try:
            self._fh.close()
            self.tif.close()
        except Exception:
            pass


def auto_stretch(img, bands, nodata=0, lo_pct=1.0, hi_pct=99.0):
    """Per-band (low, high) from the valid pixels of an overview."""
    a = img.reshape(-1, img.shape[-1])
    valid = (a != nodata).any(1) if nodata is not None else np.ones(len(a), bool)
    if valid.sum() < 16:
        valid[:] = True
    out = []
    for b in bands:
        v = a[valid, b].astype(np.float64)
        if v.size > 400000:
            v = v[:: v.size // 400000]
        lo, hi = np.percentile(v, [lo_pct, hi_pct])
        if hi <= lo:
            hi = lo + 1
        out.append((float(lo), float(hi)))
    return out


class View:
    """Zoom / pan state and drawing of one screen image (no Tk)."""

    def __init__(self, mosaic: TiffMosaic):
        self.m = mosaic
        self.bands = list(mosaic.default_bands())
        self.stretch = None
        self.gamma = 1.0
        self.cx, self.cy = mosaic.W / 2.0, mosaic.H / 2.0
        self.scale = None                # screen px per full-resolution px

    def fit(self, vw, vh):
        self.cx, self.cy = self.m.W / 2.0, self.m.H / 2.0
        self.scale = min(vw / self.m.W, vh / self.m.H) * 0.98

    def zoom(self, factor, sx, sy, vw, vh):
        """Zoom by factor around screen point (sx, sy)."""
        fx = self.cx + (sx - vw / 2.0) / self.scale
        fy = self.cy + (sy - vh / 2.0) / self.scale
        self.scale = float(np.clip(self.scale * factor, 1e-5, 32.0))
        self.cx = fx - (sx - vw / 2.0) / self.scale
        self.cy = fy - (sy - vh / 2.0) / self.scale

    def pan(self, dx, dy):
        self.cx -= dx / self.scale
        self.cy -= dy / self.scale

    def screen_to_full(self, sx, sy, vw, vh):
        return self.cx + (sx - vw / 2.0) / self.scale, self.cy + (sy - vh / 2.0) / self.scale

    def _source(self):
        """Levels as (key, factor), finest first."""
        src = [(k, lv["factor"]) for k, lv in enumerate(self.m.levels)]
        if self.m.synthetic is not None:
            src.append(("synthetic", self.m.synthetic_factor))
        return src

    def raw(self, vw, vh):
        """Band values for the screen (vh, vw, nbands) at the current zoom, and the mask of valid pixels."""
        need = 1.0 / self.scale                         # full-res px per screen px
        src = self._source()
        k, f = src[0]
        for key, fac in src:
            if fac <= need * 1.001:
                k, f = key, fac
        if k == 0 and self.m.needs_overview() and need > 8 and self.m.synthetic is None:
            raise RuntimeError("overview not ready")
        # screen rectangle in level coordinates
        x0f = (self.cx - vw / 2.0 / self.scale) / f
        y0f = (self.cy - vh / 2.0 / self.scale) / f
        x1f = (self.cx + vw / 2.0 / self.scale) / f
        y1f = (self.cy + vh / 2.0 / self.scale) / f
        x0, y0 = int(math.floor(x0f)), int(math.floor(y0f))
        x1, y1 = int(math.ceil(x1f)) + 1, int(math.ceil(y1f)) + 1
        uniq = sorted(set(self.bands))
        data = self.m.read(k, x0, y0, x1, y1, uniq)
        data = data[..., [uniq.index(b) for b in self.bands]]
        # level pixels -> screen pixels (affine: scale and offset)
        s = self.scale * f
        M = np.array([[s, 0, (x0 - x0f) * s], [0, s, (y0 - y0f) * s]], np.float64)
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
        if s < 1:          # INTER_AREA only works through resize: resize then shift
            rw, rh = max(1, int(round(data.shape[1] * s))), max(1, int(round(data.shape[0] * s)))
            small = cv2.resize(data.astype(np.float32), (rw, rh), interpolation=cv2.INTER_AREA)
            if small.ndim == 2:
                small = small[..., None]
            M2 = np.array([[1, 0, (x0 - x0f) * s], [0, 1, (y0 - y0f) * s]], np.float64)
            out = np.dstack([cv2.warpAffine(small[..., b], M2, (vw, vh), flags=cv2.INTER_LINEAR)
                             for b in range(small.shape[2])])
        else:
            out = np.dstack([cv2.warpAffine(data[..., b].astype(np.float32), M, (vw, vh), flags=interp)
                             for b in range(data.shape[2])])
        valid = (out != self.m.nodata).any(2) if self.m.nodata is not None else np.ones(out.shape[:2], bool)
        return out, valid

    def render(self, vw, vh, background=(40, 40, 40)):
        """Screen image (vh, vw, 3) uint8 RGB."""
        if self.scale is None:
            self.fit(vw, vh)
        if self.stretch is None:
            ov, _ = self.m.overview()
            self.stretch = dict(zip(range(self.m.S), auto_stretch(ov, range(self.m.S), self.m.nodata)))
        out, valid = self.raw(vw, vh)
        rgb = np.empty((vh, vw, 3), np.uint8)
        for c in range(3):
            lo, hi = self.stretch[self.bands[c]]
            v = np.clip((out[..., c] - lo) / (hi - lo), 0, 1)
            if self.gamma != 1.0:
                v = v ** (1.0 / self.gamma)
            rgb[..., c] = (v * 255 + 0.5).astype(np.uint8)
        rgb[~valid] = background
        return rgb

    def stretch_to_view(self, vw, vh):
        out, valid = self.raw(vw, vh)
        if valid.sum() < 16:
            return
        st = auto_stretch(out[valid][None], range(out.shape[2]), nodata=None)
        for c, b in enumerate(self.bands):
            self.stretch[b] = st[c]


# ---------------------------------------------------------------------- window
class ViewerWindow:
    """Tk window around View: band choice, stretch, zoom (wheel), pan (drag),
    band values of the clicked pixel, save the view as PNG."""

    def __init__(self, master, path, standalone=False):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.top = master if standalone else tk.Toplevel(master)
        self.top.title(f"mosaic2d viewer - {Path(path).name}")
        self.top.geometry("1200x800")
        self.m = TiffMosaic(path)
        self.view = View(self.m)
        self._job = None
        self._busy = False
        self._again = False
        self._drag = None
        self._img = None
        self._stop = threading.Event()
        import queue
        self._q = queue.Queue()            # work threads hand results to the Tk thread here
        self._closed = False

        bar = ttk.Frame(self.top, padding=4)
        bar.pack(fill="x")
        names = [f"{i + 1}: {n}" for i, (n, _) in enumerate(self.m.band_names)]
        self.combo = []
        for c, lab in enumerate(("R", "G", "B")):
            ttk.Label(bar, text=lab).pack(side="left", padx=(6 if c else 0, 2))
            cb = ttk.Combobox(bar, values=names, width=14, state="readonly")
            cb.current(self.view.bands[c])
            cb.bind("<<ComboboxSelected>>", self._bands_changed)
            cb.pack(side="left")
            self.combo.append(cb)
        ttk.Button(bar, text="Natural", command=self._natural).pack(side="left", padx=(10, 2))
        if self.m.nir_bands():
            ttk.Button(bar, text="False colour (NIR)", command=self._nir).pack(side="left", padx=2)
        ttk.Button(bar, text="Single band", command=self._single).pack(side="left", padx=2)
        ttk.Button(bar, text="Stretch to view", command=self._stretch_view).pack(side="left", padx=(10, 2))
        ttk.Button(bar, text="Reset stretch", command=self._reset_stretch).pack(side="left", padx=2)
        ttk.Button(bar, text="Fit", command=self._fit).pack(side="left", padx=(10, 2))
        ttk.Button(bar, text="Save view...", command=self._save).pack(side="left", padx=2)

        self.canvas = tk.Canvas(self.top, background="#282828", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.StringVar(value=self._describe())
        ttk.Label(self.top, textvariable=self.status, anchor="w", padding=(6, 2)).pack(fill="x")

        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<MouseWheel>", self._wheel)                      # Windows / macOS
        self.canvas.bind("<Button-4>", lambda e: self._wheel(e, 120))      # X11
        self.canvas.bind("<Button-5>", lambda e: self._wheel(e, -120))
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.top.after(40, self._poll)
        if self.m.needs_overview():
            self.status.set("No overviews in this file: building a small one (only once per opening)...")
            threading.Thread(target=self._build_overview, daemon=True).start()

    def _poll(self):
        if self._closed:
            return
        import queue
        try:
            while True:
                self._q.get_nowait()()
        except queue.Empty:
            pass
        self.top.after(40, self._poll)

    # ---------------------------------------------------------------- helpers
    def _describe(self):
        lv = ", ".join(f"1/{lv['factor']:.0f}" for lv in self.m.levels[1:]) or "none"
        return (f"{self.m.W} x {self.m.H} px, {self.m.S} band(s), {self.m.dtype}; overviews: {lv}. "
                f"Wheel: zoom, drag: pan, click: band values.")

    def _size(self):
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _build_overview(self):
        def prog(f):
            self._q.put(lambda: self.status.set(f"Building an overview... {f * 100:.0f} %"))
        try:
            if self.m.build_overview(progress=prog, stop=self._stop) is not None:
                self._q.put(lambda: (self.status.set(self._describe()), self.redraw()))
        except Exception as e:                                   # noqa: BLE001
            msg = str(e)
            self._q.put(lambda: self.status.set(f"Could not read the file: {msg}"))

    def redraw(self):
        """Draw in a background thread; a redraw requested meanwhile runs afterwards."""
        if self._busy:
            self._again = True
            return
        if self.m.needs_overview() and self.m.synthetic is None and (self.view.scale is None or
                                                                     self.view.scale < 1 / 8):
            return                                               # wait for the overview
        self._busy = True
        vw, vh = self._size()

        def work():
            try:
                rgb = self.view.render(vw, vh)
                ok, png = cv2.imencode(".png", rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
                data = png.tobytes()
                self._q.put(lambda: self._show(data))
            except Exception as e:                               # noqa: BLE001
                msg = str(e)
                self._q.put(lambda: (self.status.set(f"Error: {msg}"), self._done()))
        threading.Thread(target=work, daemon=True).start()

    def _show(self, data):
        import base64
        self._img = self.tk.PhotoImage(data=base64.b64encode(data))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._img)
        self._done()

    def _done(self):
        self._busy = False
        if self._again:
            self._again = False
            self.redraw()

    # ---------------------------------------------------------------- events
    def _bands_changed(self, _e=None):
        self.view.bands = [cb.current() for cb in self.combo]
        self.redraw()

    def _set_bands(self, bands):
        for cb, b in zip(self.combo, bands):
            cb.current(b)
        self._bands_changed()

    def _natural(self):
        self._set_bands(self.m.default_bands())

    def _nir(self):
        self._set_bands(self.m.nir_bands())

    def _single(self):
        b = self.combo[0].current()
        self._set_bands((b, b, b))

    def _stretch_view(self):
        try:
            self.view.stretch_to_view(*self._size())
        except RuntimeError:
            return
        self.redraw()

    def _reset_stretch(self):
        self.view.stretch = None
        self.redraw()

    def _fit(self):
        self.view.fit(*self._size())
        self.redraw()

    def _wheel(self, e, delta=None):
        d = delta if delta is not None else e.delta
        if self.view.scale is None:
            return
        self.view.zoom(1.25 if d > 0 else 0.8, e.x, e.y, *self._size())
        self.redraw()

    def _press(self, e):
        self._drag = (e.x, e.y, e.x, e.y)

    def _motion(self, e):
        if self._drag and self.view.scale:
            x, y = self._drag[2], self._drag[3]
            self.view.pan(e.x - x, e.y - y)
            self.canvas.move("all", e.x - x, e.y - y)
            self._drag = (self._drag[0], self._drag[1], e.x, e.y)

    def _release(self, e):
        if not self._drag:
            return
        moved = abs(e.x - self._drag[0]) + abs(e.y - self._drag[1]) > 3
        self._drag = None
        if moved:
            self.redraw()
            return
        x, y = self.view.screen_to_full(e.x, e.y, *self._size())
        v = self.m.value_at(x, y)
        if v is None:
            self.status.set(f"x {x:.0f}, y {y:.0f}: outside the mosaic")
            return
        vals = "  ".join(f"{n}: {int(a) if np.issubdtype(type(a), np.integer) else float(a):g}"
                         for (n, _), a in zip(self.m.band_names, v))
        self.status.set(f"x {x:.0f}, y {y:.0f}   {vals}")

    def _save(self):
        from tkinter import filedialog
        fn = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")],
                                          initialfile=self.m.path.stem + "_view.png")
        if fn:
            vw, vh = self._size()
            cv2.imwrite(fn, self.view.render(vw, vh)[..., ::-1])
            self.status.set(f"saved {fn}")

    def close(self):
        self._closed = True
        self._stop.set()
        self.m.close()
        self.top.destroy()


def main(path=None, selftest=False) -> int:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    if path is None:
        root.withdraw()
        path = filedialog.askopenfilename(title="Mosaic to view",
                                          filetypes=[("TIFF", "*.tif *.tiff"), ("all files", "*.*")])
        if not path:
            return 1
        root.deiconify()
    w = ViewerWindow(root, path, standalone=True)
    if selftest:                                 # CI: window built, one image drawn and shown
        root.update_idletasks()
        w._busy = True                           # no background redraw
        w.view.fit(400, 300)
        rgb = w.view.render(400, 300)
        ok, png = cv2.imencode(".png", rgb[..., ::-1])
        w._busy = False
        w._show(png.tobytes())
        root.update()
        w.close()
        print("viewer ok")
        return 0
    root.mainloop()
    return 0
