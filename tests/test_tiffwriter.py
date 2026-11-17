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
                   description="test", bigtiff=big, rows_per_strip=37)
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
