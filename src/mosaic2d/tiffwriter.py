"""
tiffwriter.py

Streaming (Big)TIFF writer with no dependencies beyond NumPy and zlib.

Writes an H x W x C array (any object supporting row slicing, e.g. a NumPy
memmap) strip by strip, so mosaics larger than memory can be saved. Pixel
interleaved, uint8 or uint16, Deflate compression with horizontal
differencing (predictor 2). Band names, wavelengths and nodata are stored in
GDAL's metadata tags, so QGIS / GDAL / rasterio show them.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

# tag codes
_NEW_SUBFILE, _W, _H, _BPS, _COMP, _PHOTO, _DESC, _OFFS, _SPP, _RPS, _COUNTS = \
    254, 256, 257, 258, 259, 262, 270, 273, 277, 278, 279
_PLANAR, _PRED, _SOFTWARE, _EXTRA, _SFMT = 284, 317, 305, 338, 339
_GDAL_META, _GDAL_NODATA = 42112, 42113
_PIXEL_SCALE, _TIEPOINT, _GEOKEYS = 33550, 33922, 34735


def _gdal_metadata(band_names, wavelengths, extra):
    items = []
    for k, v in (extra or {}).items():
        items.append(f'<Item name="{escape(str(k))}">{escape(str(v))}</Item>')
    for i, name in enumerate(band_names or []):
        items.append(f'<Item name="DESCRIPTION" sample="{i}" role="description">{escape(name)}</Item>')
        if wavelengths and wavelengths[i] is not None:
            um = wavelengths[i] / 1000.0
            items.append(f'<Item name="CENTRAL_WAVELENGTH_UM" sample="{i}" domain="IMAGERY">{um:.4f}</Item>')
            items.append(f'<Item name="wavelength" sample="{i}">{wavelengths[i]:g}</Item>')
    return "<GDALMetadata>" + "".join(items) + "</GDALMetadata>"


def write_tiff(path, arr, band_names=None, wavelengths=None, nodata=None, description="",
               metadata=None, rows_per_strip=64, level=6, bigtiff=None, progress=None,
               pixel_size=None, origin=(0.0, 0.0)):
    """
    pixel_size: metres per pixel -> GeoTIFF scale/tie-point tags in a local
    (non-georeferenced) metric frame, plus a world file (.tfw). origin is the
    model coordinate of the top-left corner (x right, y up).
    """
    path = Path(path)
    H, W = arr.shape[:2]
    C = arr.shape[2] if arr.ndim == 3 else 1
    dtype = np.dtype(arr.dtype)
    if dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise ValueError("write_tiff supports uint8 and uint16")
    bits = dtype.itemsize * 8
    raw_size = H * W * C * dtype.itemsize
    if bigtiff is None:
        bigtiff = raw_size > 3_500_000_000
    E = "<"
    off_fmt, cnt_fmt = ("Q", "Q") if bigtiff else ("I", "I")
    n_strips = (H + rows_per_strip - 1) // rows_per_strip

    with open(path, "wb") as fh:
        if bigtiff:
            fh.write(b"II" + struct.pack(E + "HHHQ", 43, 8, 0, 0))
            ifd_ptr_pos = 8
        else:
            fh.write(b"II" + struct.pack(E + "HI", 42, 0))
            ifd_ptr_pos = 4
        offsets, counts = [], []
        for s in range(n_strips):
            r0 = s * rows_per_strip
            block = np.asarray(arr[r0:r0 + rows_per_strip])
            block = block.reshape(block.shape[0], W, C)
            # predictor 2: horizontal differencing per sample (modular arithmetic)
            diff = block.copy()
            diff[:, 1:, :] = block[:, 1:, :] - block[:, :-1, :]
            data = zlib.compress(diff.astype(dtype.newbyteorder("<"), copy=False).tobytes(), level)
            offsets.append(fh.tell())
            counts.append(len(data))
            fh.write(data)
            if progress and s % 50 == 0:
                progress(s / n_strips)

        SHORT, LONG, ASCII, LONG8 = 3, 4, 2, 16
        offtype = LONG8 if bigtiff else LONG
        rgb = C == 3 and not band_names_is_ms(band_names)
        n_extra = 0 if rgb else C - 1
        meta = _gdal_metadata(band_names, wavelengths, metadata)
        entries = [
            (_NEW_SUBFILE, LONG, [0]),
            (_W, LONG, [W]),
            (_H, LONG, [H]),
            (_BPS, SHORT, [bits] * C),
            (_COMP, SHORT, [8]),
            (_PHOTO, SHORT, [2 if rgb else 1]),
            (_OFFS, offtype, offsets),
            (_SPP, SHORT, [C]),
            (_RPS, LONG, [rows_per_strip]),
            (_COUNTS, offtype, counts),
            (_PLANAR, SHORT, [1]),
            (_SOFTWARE, ASCII, "mosaic2d"),
            (_PRED, SHORT, [2]),
            (_SFMT, SHORT, [1] * C),
            (_GDAL_META, ASCII, meta),
        ]
        if description:
            entries.append((_DESC, ASCII, description))
        if n_extra:
            entries.append((_EXTRA, SHORT, [0] * n_extra))
        if nodata is not None:
            entries.append((_GDAL_NODATA, ASCII, str(nodata)))
        if pixel_size:
            DOUBLE = 12
            entries.append((_PIXEL_SCALE, DOUBLE, [float(pixel_size), float(pixel_size), 0.0]))
            entries.append((_TIEPOINT, DOUBLE, [0.0, 0.0, 0.0, float(origin[0]), float(origin[1]), 0.0]))
            # GeoKeyDirectory: version 1.1.0, 2 keys; model type user-defined (local
            # engineering frame), raster type PixelIsArea
            entries.append((_GEOKEYS, SHORT, [1, 1, 0, 2, 1024, 0, 1, 32767, 1025, 0, 1, 1]))
        entries.sort(key=lambda e: e[0])

        inline_bytes = 8 if bigtiff else 4
        fmt_of = {SHORT: "H", LONG: "I", LONG8: "Q", 12: "d"}
        encoded = []
        for tag, typ, val in entries:
            if typ == ASCII:
                data = val.encode("utf-8") + b"\0"
                count = len(data)
            else:
                data = struct.pack(E + fmt_of[typ] * len(val), *val)
                count = len(val)
            if len(data) > inline_bytes:
                if fh.tell() % 2:
                    fh.write(b"\0")
                pos = fh.tell()
                fh.write(data)
                field = struct.pack(E + ("Q" if bigtiff else "I"), pos)
            else:
                field = data.ljust(inline_bytes, b"\0")
            encoded.append((tag, typ, count, field))
        if fh.tell() % 2:
            fh.write(b"\0")
        ifd_pos = fh.tell()
        if bigtiff:
            fh.write(struct.pack(E + "Q", len(encoded)))
        else:
            fh.write(struct.pack(E + "H", len(encoded)))
        for tag, typ, count, field in encoded:
            fh.write(struct.pack(E + ("HHQ" if bigtiff else "HHI"), tag, typ, count) + field)
        fh.write(struct.pack(E + ("Q" if bigtiff else "I"), 0))
        fh.seek(ifd_ptr_pos)
        fh.write(struct.pack(E + ("Q" if bigtiff else "I"), ifd_pos))
    if pixel_size:
        write_world_file(path.with_suffix(".tfw"), pixel_size, origin)
    return path


def write_world_file(path, pixel_size, origin):
    """ESRI world file: pixel size, rotation terms, centre of the top-left pixel."""
    ps = float(pixel_size)
    Path(path).write_text("\n".join(f"{v:.10f}" for v in
                                     (ps, 0.0, 0.0, -ps, origin[0] + ps / 2, origin[1] - ps / 2)) + "\n")


def band_names_is_ms(names) -> bool:
    return bool(names) and any("nm" in n for n in names)


def write_tiled(out_dir, name, arr, band_names=None, wavelengths=None, nodata=0, tile=16384,
                metadata=None, log=None, pixel_size=None):
    """
    Write a large H x W x C array as tiles <name>_tiles/<name>_rRR_cCC.tif plus a
    GDAL VRT (<name>.vrt) that presents them as one raster. Empty tiles are skipped.
    """
    out_dir = Path(out_dir)
    tdir = out_dir / f"{name}_tiles"
    tdir.mkdir(parents=True, exist_ok=True)
    H, W = arr.shape[:2]
    C = arr.shape[2] if arr.ndim == 3 else 1
    dt = "Byte" if np.dtype(arr.dtype) == np.uint8 else "UInt16"
    sources = []
    for r, y in enumerate(range(0, H, tile)):
        for c, x in enumerate(range(0, W, tile)):
            blk = arr[y:y + tile, x:x + tile]
            if not np.asarray(blk[::16, ::16]).any() and not np.asarray(blk).any():
                continue
            fn = f"{name}_r{r:02d}_c{c:02d}.tif"
            org = (x * pixel_size, (H - y) * pixel_size) if pixel_size else (0.0, 0.0)
            write_tiff(tdir / fn, blk, band_names, wavelengths, nodata, metadata=metadata,
                       pixel_size=pixel_size, origin=org)
            sources.append((fn, x, y, blk.shape[1], blk.shape[0]))
            if log:
                log(f"  tile {fn} ({blk.shape[1]} x {blk.shape[0]})")
    rgb = C == 3 and not band_names_is_ms(band_names)
    interp = ["Red", "Green", "Blue"] if rgb else ["Gray"] * C
    lines = [f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">']
    if pixel_size:
        lines.append(f"  <GeoTransform>0, {pixel_size!r}, 0, {H * pixel_size!r}, 0, {-pixel_size!r}</GeoTransform>")
    if metadata:                     # same processing information as the tiles carry
        lines.append("  <Metadata>")
        lines += [f'    <MDI key="{escape(str(k))}">{escape(str(v))}</MDI>' for k, v in metadata.items()]
        lines.append("  </Metadata>")
    for b in range(C):
        lines.append(f'  <VRTRasterBand dataType="{dt}" band="{b + 1}">')
        if band_names:
            lines.append(f"    <Description>{escape(band_names[b])}</Description>")
        if wavelengths and wavelengths[b] is not None:
            lines += ["    <Metadata>",
                      f'      <MDI key="wavelength">{wavelengths[b]:g}</MDI>',
                      "    </Metadata>",
                      '    <Metadata domain="IMAGERY">',
                      f'      <MDI key="CENTRAL_WAVELENGTH_UM">{wavelengths[b] / 1000.0:.4f}</MDI>',
                      "    </Metadata>"]
        lines.append(f"    <NoDataValue>{nodata}</NoDataValue>")
        lines.append(f"    <ColorInterp>{interp[b]}</ColorInterp>")
        for fn, x, y, w, h in sources:
            lines += ["    <SimpleSource>",
                      f'      <SourceFilename relativeToVRT="1">{tdir.name}/{fn}</SourceFilename>',
                      f"      <SourceBand>{b + 1}</SourceBand>",
                      f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>',
                      f'      <DstRect xOff="{x}" yOff="{y}" xSize="{w}" ySize="{h}"/>',
                      "    </SimpleSource>"]
        lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")
    vrt = out_dir / f"{name}.vrt"
    vrt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return vrt
