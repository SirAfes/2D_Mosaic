import numpy as np, tifffile, pytest
from mosaic2d.tiffwriter import write_tiff


@pytest.mark.parametrize("big", [False, True])
@pytest.mark.parametrize("shape,dtype,names", [((300, 211, 3), np.uint16, None),
                                                ((257, 190, 14), np.uint16, [f"{w} nm" for w in range(400, 820, 30)]),
                                                ((100, 120, 3), np.uint8, None)])
def test_roundtrip(tmp_path, shape, dtype, names, big):
    rng = np.random.default_rng(0)
    a = rng.integers(0, np.iinfo(dtype).max, shape, dtype=dtype)
    wl = [float(n.split()[0]) for n in names] if names else None
    p = write_tiff(tmp_path / "x.tif", a, band_names=names, wavelengths=wl, nodata=0,
                   description="test", bigtiff=big, tile=64)
    with tifffile.TiffFile(p) as t:
        b = t.asarray()
        assert t.is_bigtiff == big
        assert t.pages[0].tags[42113].value.strip("\x00") == "0"
        if names:
            assert "430 nm" in t.pages[0].tags[42112].value
    assert np.array_equal(a, b.reshape(a.shape))


def test_tiled_vrt(tmp_path):
    from mosaic2d.tiffwriter import write_tiled
    a = np.zeros((300, 500, 3), np.uint16)
    a[10:290, 20:480] = 1000
    a[:, 300:] = 0
    vrt = write_tiled(tmp_path, "m", a, tile=128)
    txt = vrt.read_text()
    assert txt.count("<SimpleSource>") == 3 * len(list((tmp_path / "m_tiles").glob("*.tif")))
    assert len(list((tmp_path / "m_tiles").glob("*.tif"))) == 9  # 3 x 4 tiles, last column empty


def test_metric_scale_tags_and_world_file(tmp_path):
    a = np.ones((50, 80, 3), np.uint16)
    p = write_tiff(tmp_path / "s.tif", a, pixel_size=0.0005, origin=(0.0, 0.025))
    with tifffile.TiffFile(p) as t:
        tags = t.pages[0].tags
        assert tags[33550].value[:2] == (0.0005, 0.0005)
        assert tags[33922].value[3:5] == (0.0, 0.025)
    tfw = (tmp_path / "s.tfw").read_text().split()
    assert float(tfw[0]) == 0.0005 and float(tfw[3]) == -0.0005


def test_tiled_vrt_carries_band_wavelengths_and_metadata(tmp_path):
    import xml.etree.ElementTree as ET
    from mosaic2d.tiffwriter import write_tiled
    a = np.full((200, 300, 3), 500, np.uint16)
    vrt = write_tiled(tmp_path, "ms", a, band_names=["450 nm", "560 nm", "850 nm"],
                      wavelengths=[450, 560, 850], tile=128,
                      metadata={"PROFILE": "agrowing6", "REFERENCE_BAND": "525 nm"})
    root = ET.parse(vrt).getroot()
    md = {m.get("key"): m.text for m in root.find("Metadata")}
    assert md == {"PROFILE": "agrowing6", "REFERENCE_BAND": "525 nm"}
    bands = root.findall("VRTRasterBand")
    assert [b.find("Description").text for b in bands] == ["450 nm", "560 nm", "850 nm"]
    um = [b.find("Metadata[@domain='IMAGERY']/MDI").text for b in bands]
    assert um == ["0.4500", "0.5600", "0.8500"]


def test_tiled_with_internal_overviews(tmp_path):
    a = np.zeros((1000, 1500, 3), np.uint16)
    a[100:900, 200:1300] = 4000
    a[100:900, 200:700, 1] = 8000
    p = write_tiff(tmp_path / "o.tif", a, nodata=0, tile=256)
    with tifffile.TiffFile(p) as t:
        assert t.pages[0].is_tiled and t.pages[0].tilewidth == 256
        levels = t.series[0].levels
        assert [lv.shape[:2] for lv in levels] == [(1000, 1500), (500, 750), (250, 375), (125, 188)]
        assert all(t.pages[k].tags[254].value == 1 for k in range(1, 4))     # reduced-resolution
        assert np.array_equal(levels[0].asarray(), a)
        o1 = levels[1].asarray()
        assert np.array_equal(o1, a[::2, ::2])                               # piecewise constant input
        # empty areas do not darken the edge of the image in overviews
        b = np.zeros((10, 10, 1), np.uint16)
        b[:, :5] = 1000
        b[:, 5] = 1000
        from mosaic2d.tiffwriter import _downsample2
        d = _downsample2(b)
        assert d[0, 2] == 1000 and d[0, 3] == 0


def test_large_overview_levels_use_disk(tmp_path, monkeypatch):
    """Overviews bigger than the in-memory limit go to a temporary file that is removed."""
    import mosaic2d.tiffwriter as TW
    monkeypatch.setattr(TW, "_OVR_RAM_LIMIT", 0)
    seen = []
    orig = np.lib.format.open_memmap
    monkeypatch.setattr(np.lib.format, "open_memmap", lambda *a, **k: seen.append(a[0]) or orig(*a, **k))
    a = np.full((600, 700, 2), 7, np.uint16)
    p = write_tiff(tmp_path / "d.tif", a, tile=128, work_dir=tmp_path)
    assert len(seen) == 3 and not list(tmp_path.glob("m2d_ovr_*"))
    with tifffile.TiffFile(p) as t:
        assert np.array_equal(t.series[0].levels[1].asarray(), a[::2, ::2])
