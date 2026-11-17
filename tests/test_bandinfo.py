"""Band names and wavelengths from multiband TIFFs (msalign / GDAL metadata)."""

import json

import numpy as np
import tifffile

from mosaic2d import rawio
from mosaic2d.sources import get_profile

WL = [405, 430, 450, 490, 525, 550, 560, 570, 630, 650, 685, 710, 735, 850]


def _msalign_like(path, gdal=True, desc=True):
    items = ['<Item name="wavelength_units">nm</Item>']
    for k, w in enumerate(WL):
        items += [f'<Item name="DESCRIPTION" sample="{k}" role="description">{w} nm</Item>',
                  f'<Item name="wavelength" sample="{k}">{w}</Item>',
                  f'<Item name="CENTRAL_WAVELENGTH_UM" sample="{k}" domain="IMAGERY">{w / 1000:.4f}</Item>']
    extratags = [(42112, "s", 0, "<GDALMetadata>\n" + "\n".join(items) + "\n</GDALMetadata>", True)] if gdal else []
    d = json.dumps({"msalign_version": "1.2.0", "bands": [
        {"band": k + 1, "name": f"{w} nm", "center_wavelength_nm": float(w)} for k, w in enumerate(WL)]})
    tifffile.imwrite(path, np.ones((len(WL), 64, 48), np.uint16), photometric="minisblack",
                     planarconfig="separate", description=d if desc else None, metadata=None,
                     extratags=extratags)


def test_gdal_metadata_band_info(tmp_path):
    p = tmp_path / "a.tif"
    _msalign_like(p, desc=False)
    info = rawio.tiff_band_info(p)
    assert [info[k][0] for k in range(14)] == [f"{w} nm" for w in WL]
    assert [round(info[k][1]) for k in range(14)] == WL
    prof = get_profile("multiband", p)
    assert [b.name for b in prof.bands] == [f"{w} nm" for w in WL]
    assert [b.wavelength_nm for b in prof.bands] == [float(w) for w in WL]


def test_msalign_json_band_info(tmp_path):
    p = tmp_path / "b.tif"
    _msalign_like(p, gdal=False)
    info = rawio.tiff_band_info(p)
    assert info[13] == ("850 nm", 850.0)


def test_no_metadata_falls_back_to_numbers(tmp_path):
    p = tmp_path / "c.tif"
    tifffile.imwrite(p, np.ones((5, 32, 32), np.uint16), photometric="minisblack",
                     planarconfig="separate", metadata=None)
    prof = get_profile("multiband", p)
    assert [b.name for b in prof.bands] == [f"band {k}" for k in range(1, 6)]
