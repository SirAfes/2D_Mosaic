"""Tile-wise reading for the viewer: every TIFF layout gives the same pixels."""

import numpy as np
import pytest
import tifffile

from mosaic2d.tiffwriter import write_tiff
from mosaic2d.viewer import TiffMosaic, View


def _data(h=1300, w=1700, s=5):
    rng = np.random.default_rng(0)
    a = (rng.random((h, w, s)) * 60000).astype(np.uint16)
    a[:100, :200] = 0                                   # nodata corner
    return a


@pytest.mark.parametrize("layout", ["ours", "planar_tiled", "strips", "strips_planar"])
def test_read_regions(tmp_path, layout):
    a = _data()
    f = tmp_path / f"{layout}.tif"
    if layout == "ours":
        write_tiff(f, a, band_names=[f"{w} nm" for w in (450, 550, 650, 730, 850)],
                   wavelengths=[450, 550, 650, 730, 850])
    elif layout == "planar_tiled":
        tifffile.imwrite(f, np.moveaxis(a, 2, 0), tile=(256, 256), planarconfig="separate",
                         photometric="minisblack", compression="zlib")
    elif layout == "strips":
        tifffile.imwrite(f, a, rowsperstrip=37, photometric="minisblack", planarconfig="contig",
                         compression="zlib", predictor=True)
    else:
        tifffile.imwrite(f, np.moveaxis(a, 2, 0), rowsperstrip=50, photometric="minisblack",
                         planarconfig="separate")
    m = TiffMosaic(f)
    assert (m.W, m.H, m.S) == (1700, 1300, 5)
    for (x0, y0, x1, y1) in ((0, 0, 1700, 1300), (500, 333, 901, 777), (-20, 1290, 30, 1310)):
        got = m.read(0, x0, y0, x1, y1, [4, 0, 2])
        ref = np.zeros((y1 - y0, x1 - x0, 3), np.uint16)
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(1300, y1), min(1700, x1)
        ref[ys - y0:ye - y0, xs - x0:xe - x0] = a[ys:ye, xs:xe][..., [4, 0, 2]]
        assert np.array_equal(got, ref), (layout, x0, y0)
    assert np.array_equal(m.value_at(1000, 1001), a[1001, 1000])
    if layout == "ours":
        assert len(m.levels) >= 2 and m.band_names[4] == ("850 nm", 850.0)
        assert m.default_bands() == (2, 1, 0) and m.nir_bands() == (4, 2, 1)
    # overview built from a full pass when the file has none
    ov = m.build_overview(max_side=400)
    f_ = int(np.ceil(1700 / 400)) if len(m.levels) == 1 else None
    if f_:
        assert np.array_equal(ov, a[::f_, ::f_])
    m.close()


def test_view_render_zoom_and_pan(tmp_path):
    a = _data()
    f = tmp_path / "m.tif"
    write_tiff(f, a)
    v = View(TiffMosaic(f))
    rgb = v.render(400, 300)
    assert rgb.shape == (300, 400, 3) and rgb.dtype == np.uint8
    # nodata corner drawn as background, data elsewhere
    assert (rgb[5, 200] == 40).all() or v.scale < 1
    # zoom in on a pixel: 1:1 and beyond read the full-resolution level
    v.zoom(64, 200, 150, 400, 300)
    x, y = v.screen_to_full(200, 150, 400, 300)
    v.render(400, 300)
    raw, valid = v.raw(400, 300)
    sx, sy = 200, 150
    fx, fy = v.screen_to_full(sx, sy, 400, 300)
    assert np.allclose(raw[sy, sx], a[int(fy), int(fx)][list(v.bands)].astype(np.float32), atol=1)
    v.pan(100, 0)
    assert v.render(400, 300).shape == (300, 400, 3)


@pytest.mark.parametrize("planar", [False, True])
def test_uncompressed_read_directly(tmp_path, planar):
    a = _data(300, 400, 3)
    f = tmp_path / "u.tif"
    if planar:
        tifffile.imwrite(f, np.moveaxis(a, 2, 0), photometric="minisblack", planarconfig="separate")
    else:
        tifffile.imwrite(f, a, photometric="rgb")
    m = TiffMosaic(f)
    assert m.levels[0]["mm"] is not None
    assert np.array_equal(m.read(0, 10, 20, 390, 290, [2, 1]), a[20:290, 10:390][..., [2, 1]])


@pytest.mark.parametrize("compression", ["lzw", "jpeg"])
def test_compressions_from_other_software(tmp_path, compression):
    """Metashape and GDAL often write LZW (or JPEG) tiles: needs imagecodecs."""
    pytest.importorskip("imagecodecs")
    yy, xx = np.mgrid[0:600, 0:700]
    a = np.dstack([xx * 255 // 700, yy * 255 // 600, (xx + yy) * 255 // 1300]).astype(np.uint8)
    f = tmp_path / "c.tif"
    tifffile.imwrite(f, a, tile=(256, 256), photometric="rgb", compression=compression)
    m = TiffMosaic(f)
    got = m.read(0, 0, 0, 700, 600, [0, 1, 2])
    if compression == "lzw":
        assert np.array_equal(got, a)
    else:
        assert np.abs(got.astype(int) - a).mean() < 4
