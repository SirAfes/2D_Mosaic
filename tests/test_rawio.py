import numpy as np

from mosaic2d import rawio


def _encode_block(vals):
    """Encode 16 11-bit values (max - min < 128 -> exact) as a Sony cRAW block."""
    vals = [int(v) for v in vals]
    imax, imin = int(np.argmax(vals)), int(np.argmin(vals))
    if imax == imin:
        imin = (imax + 1) % 16
    vmax, vmin = vals[imax], vals[imin]
    bits = vmax | (vmin << 11) | (imax << 22) | (imin << 26)
    pos = 30
    for i in range(16):
        if i in (imax, imin):
            continue
        bits |= int(vals[i] - vmin) << pos
        pos += 7
    return np.frombuffer(bits.to_bytes(16, "little"), np.uint8)


def test_sony_craw_decoder_roundtrip():
    rng = np.random.default_rng(0)
    width, rows = 64, 3
    curve = rawio._sony_curve((8000, 10400, 12900, 14100))
    data = np.zeros((rows, width), np.uint8)
    expect = np.zeros((rows, width), np.int64)
    for r in range(rows):
        for b in range(width // 16):
            base = rng.integers(0, 1900)
            vals = base + rng.integers(0, 127, 16)
            data[r, b * 16:(b + 1) * 16] = _encode_block(vals)
            pair, second = divmod(b, 2)
            cols = pair * 32 + second + 2 * np.arange(16)
            expect[r, cols] = curve[vals << 1]
    out = rawio.decode_sony_craw_rows(data, width, curve)
    assert np.array_equal(out, expect)


def test_superpixel_demosaic_rggb():
    cfa = np.array([[10, 20, 11, 21], [30, 40, 31, 41]], np.uint16)
    raw = rawio.RawFrame(cfa, "RGGB", 16000, 512, "test")
    sp = rawio.demosaic(raw, "superpixel")
    assert sp.shape == (1, 2, 3)
    assert sp[0, 0].tolist() == [10, 25, 40]
    assert sp[0, 1].tolist() == [11, 26, 41]
