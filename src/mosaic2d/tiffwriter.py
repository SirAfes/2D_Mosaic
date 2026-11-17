"""
tiffwriter.py

Streaming (Big)TIFF writer with no dependencies beyond NumPy and zlib.

Writes an H x W x C array (any object supporting slicing, e.g. a NumPy
memmap) tile by tile, so mosaics larger than memory can be saved: 512 x 512
tiles, internal overviews (pyramids), pixel interleaved, uint8 or uint16,
Deflate compression with horizontal differencing (predictor 2). Band names, wavelengths and nodata are stored in
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
_TILE_W, _TILE_H, _TILE_OFFS, _TILE_COUNTS = 322, 323, 324, 325
_GDAL_META, _GDAL_NODATA = 42112, 42113
_OVR_RAM_LIMIT = 256 * 2 ** 20      # larger overview levels are built in a temporary file
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


def _downsample2(block: np.ndarray) -> np.ndarray:
    """2 x 2 mean that ignores empty (all-zero) pixels, so mosaic edges do not darken."""
    h, w, c = block.shape
    b = block
    if h % 2 or w % 2:
        b = np.zeros((h + h % 2, w + w % 2, c), block.dtype)
        b[:h, :w] = block
    b = b.astype(np.float32).reshape(b.shape[0] // 2, 2, b.shape[1] // 2, 2, c)
    valid = (b > 0).any(axis=4, keepdims=True).astype(np.float32)
    s = (b * valid).sum(axis=(1, 3))
    n = valid.sum(axis=(1, 3))
    return np.where(n > 0, s / np.maximum(n, 1), 0)


def _encode_tile(t: np.ndarray, tile: int, dtype, level: int) -> bytes:
    """Pad to a full tile, horizontal differencing (predictor 2), Deflate."""
    h, w, c = t.shape
    if h != tile or w != tile:
        full = np.zeros((tile, tile, c), dtype)
        full[:h, :w] = t
        t = full
    t = np.ascontiguousarray(t, dtype=dtype)
    diff = t.copy()
    diff[:, 1:, :] = t[:, 1:, :] - t[:, :-1, :]
    return zlib.compress(diff.astype(np.dtype(dtype).newbyteorder("<"), copy=False).tobytes(), level)


def _write_ifd(fh, entries, bigtiff):
    """Write one IFD (out-of-line values first). Returns (IFD position, position of its next-IFD field)."""
    E = "<"
    SHORT, LONG, ASCII, LONG8, DOUBLE = 3, 4, 2, 16, 12
    inline_bytes = 8 if bigtiff else 4
    fmt_of = {SHORT: "H", LONG: "I", LONG8: "Q", DOUBLE: "d"}
    encoded = []
    for tag, typ, val in sorted(entries, key=lambda e: e[0]):
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
    fh.write(struct.pack(E + ("Q" if bigtiff else "H"), len(encoded)))
    for tag, typ, count, field in encoded:
        fh.write(struct.pack(E + ("HHQ" if bigtiff else "HHI"), tag, typ, count) + field)
    next_pos = fh.tell()
    fh.write(struct.pack(E + ("Q" if bigtiff else "I"), 0))
    return ifd_pos, next_pos


def write_tiff(path, arr, band_names=None, wavelengths=None, nodata=None, description="",
               metadata=None, tile=512, level=6, bigtiff=None, progress=None,
               pixel_size=None, origin=(0.0, 0.0), overviews=True, work_dir=None):
    """
    Tiled (512 x 512) Deflate TIFF with internal overviews (reduced-resolution
    pyramids at 1/2, 1/4, ... down to one tile), the layout GIS software and
    Metashape exports use: a viewer reads only the tiles on screen, and a
    zoomed-out view reads a small overview instead of the full image.

    arr: H x W x C (or H x W) uint8/uint16, anything sliceable (e.g. a NumPy
    memmap larger than memory). Overviews are built on disk next to the output
    (or in work_dir) and removed afterwards.

    pixel_size: metres per pixel -> GeoTIFF scale/tie-point tags in a local
    (non-georeferenced) metric frame, plus a world file (.tfw). origin is the
    model coordinate of the top-left corner (x right, y up).
    """
    import shutil
    import tempfile

    path = Path(path)
    H, W = arr.shape[:2]
    C = arr.shape[2] if arr.ndim == 3 else 1
    dtype = np.dtype(arr.dtype)
    if dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise ValueError("write_tiff supports uint8 and uint16")
    bits = dtype.itemsize * 8
    raw_size = H * W * C * dtype.itemsize
    if bigtiff is None:
        bigtiff = raw_size * 1.4 > 3_500_000_000        # base + overviews, before compression
    E = "<"
    SHORT, LONG, ASCII, LONG8, DOUBLE = 3, 4, 2, 16, 12
    offtype = LONG8 if bigtiff else LONG

    # pyramid sizes: halve until the level fits in one tile
    sizes = [(H, W)]
    if overviews:
        while max(sizes[-1]) > tile:
            h, w = sizes[-1]
            sizes.append(((h + 1) // 2, (w + 1) // 2))
    tmp = Path(tempfile.mkdtemp(prefix="m2d_ovr_", dir=work_dir or path.parent)) if len(sizes) > 1 else None

    def level_array(k):
        h, w = sizes[k]
        if h * w * C * dtype.itemsize < _OVR_RAM_LIMIT:
            return np.zeros((h, w, C), dtype)
        return np.lib.format.open_memmap(tmp / f"ovr{k}.npy", mode="w+", dtype=dtype, shape=(h, w, C))

    levels_data = []                                     # per level: (offsets, counts)
    total_tiles = sum(((h + tile - 1) // tile) * ((w + tile - 1) // tile) for h, w in sizes)
    done = 0
    try:
        with open(path, "wb") as fh:
            if bigtiff:
                fh.write(b"II" + struct.pack(E + "HHHQ", 43, 8, 0, 0))
                first_ptr = 8
            else:
                fh.write(b"II" + struct.pack(E + "HI", 42, 0))
                first_ptr = 4
            src = arr
            for k, (h, w) in enumerate(sizes):
                nxt = level_array(k + 1) if k + 1 < len(sizes) else None
                offsets, counts = [], []
                for y in range(0, h, tile):
                    for x in range(0, w, tile):
                        t = np.array(src[y:y + tile, x:x + tile])     # copy: no view keeps a memmap open
                        t = t.reshape(t.shape[0], t.shape[1], C)
                        data = _encode_tile(t, tile, dtype, level)
                        offsets.append(fh.tell())
                        counts.append(len(data))
                        fh.write(data)
                        if nxt is not None:
                            d = _downsample2(t)
                            nxt[y // 2:y // 2 + d.shape[0], x // 2:x // 2 + d.shape[1]] = \
                                np.clip(np.rint(d), 0, np.iinfo(dtype).max).astype(dtype)
                        done += 1
                        if progress and done % 200 == 0:
                            progress(done / total_tiles)
                levels_data.append((offsets, counts))
                if isinstance(nxt, np.memmap):
                    nxt.flush()
                src = nxt
            src = nxt = t = None

            rgb = C == 3 and not band_names_is_ms(band_names)
            n_extra = 0 if rgb else C - 1

            def common(k):
                h, w = sizes[k]
                offs, cnts = levels_data[k]
                e = [(_NEW_SUBFILE, LONG, [0 if k == 0 else 1]),
                     (_W, LONG, [w]), (_H, LONG, [h]),
                     (_BPS, SHORT, [bits] * C), (_COMP, SHORT, [8]),
                     (_PHOTO, SHORT, [2 if rgb else 1]), (_SPP, SHORT, [C]),
                     (_PLANAR, SHORT, [1]), (_PRED, SHORT, [2]), (_SFMT, SHORT, [1] * C),
                     (_TILE_W, LONG, [tile]), (_TILE_H, LONG, [tile]),
                     (_TILE_OFFS, offtype, offs), (_TILE_COUNTS, offtype, cnts)]
                if n_extra:
                    e.append((_EXTRA, SHORT, [0] * n_extra))
                if nodata is not None:
                    e.append((_GDAL_NODATA, ASCII, str(nodata)))
                return e

            entries = common(0) + [(_SOFTWARE, ASCII, "mosaic2d"),
                                   (_GDAL_META, ASCII, _gdal_metadata(band_names, wavelengths, metadata))]
            if description:
                entries.append((_DESC, ASCII, description))
            if pixel_size:
                entries.append((_PIXEL_SCALE, DOUBLE, [float(pixel_size), float(pixel_size), 0.0]))
                entries.append((_TIEPOINT, DOUBLE, [0.0, 0.0, 0.0, float(origin[0]), float(origin[1]), 0.0]))
                # GeoKeyDirectory: version 1.1.0, 2 keys; model type user-defined (local
                # engineering frame), raster type PixelIsArea
                entries.append((_GEOKEYS, SHORT, [1, 1, 0, 2, 1024, 0, 1, 32767, 1025, 0, 1, 1]))
            prev_next = first_ptr
            for k in range(len(sizes)):
                ifd, nxt_field = _write_ifd(fh, entries if k == 0 else common(k), bigtiff)
                here = fh.tell()
                fh.seek(prev_next)
                fh.write(struct.pack(E + ("Q" if bigtiff else "I"), ifd))
                fh.seek(here)
                prev_next = nxt_field
    finally:
        if tmp is not None:
            import gc
            src = nxt = None                     # Windows cannot delete a file that is still mapped
            gc.collect()
            shutil.rmtree(tmp, ignore_errors=True)
    if progress:
        progress(1.0)
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
