# mosaic2d

**mosaic2d** builds large 2D mosaics from overlapping photographs — RGB or multispectral, including multi-lens snapshot cameras such as the Agrowing Sextuple — without GPS, ground control or a 3D reconstruction. It follows the alignment and orthomosaic workflow of photogrammetry software such as Agisoft Metashape, reduced to a planar scene model, and it does not rely on one feature detector: several methods are tried per image pair, and a benchmark tells you which one works best on your data.

```text
photos (RAW / JPG / TIFF)
   ↓ decode (Sony ARW without extra libraries; other RAW via rawpy)
   ↓ multispectral: register every lens to the reference lens, pick the best band
   ↓ features: SIFT · RootSIFT · AKAZE · ORB · BRISK · KAZE · FAST · GFTT · BEBLID …
   ↓ pair preselection: sequential + generic (thumbnails) + estimated (footprints)
   ↓ matching: ratio test → MAGSAC++ homography → guided matching → quality checks
   ↓ global bundle adjustment (+ radial distortion self-calibration, outlier rounds)
   ↓ image quality from overlaps · flat field (vignetting) from the whole series
   ↓ seamlines: graph cut + removal of moving objects (rope, fish)
   ↓ blending (mosaic / average / disabled), out-of-core rendering
mosaic (Big)TIFF or tiles + VRT · preview · seam map · reports
```

## Install

Python ≥ 3.10:

```bash
pip install "git+https://github.com/SirAfes/mosaic2d.git"
pip install rawpy          # optional: RAW formats other than Sony ARW
```

mosaic2d depends on `opencv-contrib-python` (OpenCV 5 moved AKAZE, KAZE, BRISK and BEBLID to contrib). If a plain `opencv-python` is already installed, uninstall it first — the two packages must not be installed together. With plain OpenCV the tool still runs; it skips the methods that are missing. `mosaic2d methods` lists what your build provides.

## Use

```bash
mosaic2d run D:/survey/RGB                       # RGB mosaic, all defaults
mosaic2d run D:/survey/MS -o D:/out/ms           # Agrowing multispectral (auto-detected)
mosaic2d run D:/survey/MS --ref-band "560 nm"    # force the registration band
mosaic2d run D:/survey/RGB --blend disabled --resolution 0.5
mosaic2d run D:/survey/RGB --tiles 16384         # tiles + .vrt for very large mosaics
mosaic2d run D:/survey/RGB --gsd 0.0002          # 0.2 mm per pixel -> measurements in metres
mosaic2d render D:/out/mosaic_project.pkl --resolution 0.5 --blend disabled
                                                 # render again without re-aligning
mosaic2d benchmark D:/survey/RGB                 # compare all feature methods
mosaic2d methods                                 # available methods
```

Important options (`mosaic2d run -h` lists all):

| Option | Default | Meaning |
|---|---|---|
| `--methods` | all available | methods in cascade order, e.g. `sift,akaze,orb,fast` |
| `--strategy` | `cascade` | `cascade`: next method only if the previous failed for that pair; `union`: pool matches of all methods; `single` |
| `--model` | `homography` | per-image transform: `translation`, `similarity`, `affine`, `homography` |
| `--accuracy` | `medium` | matching resolution (max. side 6000/3600/2400/1200/700 px for highest…lowest) |
| `--preselection` | `auto` | `auto` = sequential + generic + estimated; or `sequential`, `generic`, `exhaustive` |
| `--blend` | `mosaic` | `mosaic` (two-band), `average` (feathered), `disabled` (no radiometric change) |
| `--resolution` | 1.0 | output pixel size relative to the native pixels of the reference image |
| `--no-distortion` | — | do not self-calibrate radial distortion |
| `--seams` | `graphcut` | `graphcut`: seams move to where frames agree; `voronoi`: closest frame centre only |
| `--no-deghost` | — | keep moving objects (by default areas where one frame disagrees with the others are taken from a consistent frame) |
| `--flatfield` | `auto` | estimate and remove vignetting / fixed illumination from the series; `none` to keep raw radiometry |
| `--gsd` | — | metres per pixel at resolution 1.0: GeoTIFF scale tags + `.tfw` world file |
| `--no-cache` | — | decode every image again instead of reusing `<out>/_cache` |
| `--profile` | `auto` | `rgb`, `agrowing6`, `multiband` (pre-aligned TIFFs, e.g. msalign output) or a JSON file |

## Outputs

```text
out/
├── mosaic.tif              all bands, uint16 for RAW input, 0 = no data (BigTIFF when > 3.5 GB)
│   (or mosaic.vrt + mosaic_tiles/ with --tiles)
├── mosaic_preview.jpg      downsampled preview (MS: 650/550/450 nm)
├── mosaic_seams.jpg        which photo supplies each part of the mosaic
├── mosaic_report.md/.json  images aligned, pairs, methods used, reprojection error, distortion, timing
├── mosaic_pairs.csv        every tried pair: method cascade, matches, inliers, coverage, residuals
├── mosaic_images.csv       per image: pairs, residual, quality; MS: lens offsets per lens
├── mosaic_project.pkl      aligned project for `mosaic2d render` (other resolution, blending, seams)
├── _cache/                 prepared frames, reused by the next run on the same images
└── mosaic2d.log
```

Band names and wavelengths are written to GDAL metadata, so QGIS shows "405 nm" … "850 nm".

## How it works

**Decoding.** Sony ARW (lossy *compressed RAW* and uncompressed) is decoded with NumPy following the public dcraw/LibRaw description: black level subtracted, no white balance, no colour matrix, linear. Multispectral profiles use *super-pixel* demosaicing (each 2 × 2 Bayer quad → one pixel; no interpolation between spectral channels); RGB uses bilinear demosaicing at full resolution.

**Multispectral bands.** Each lens image is registered to the reference lens of the same frame by phase correlation on gradient-magnitude images (robust across bands) and refined with ECC (small affine part). Lens offsets of a rigid lens array change only with distance, so offsets far from the series median that also correlate poorly are re-estimated around the median. The **reference band** for frame-to-frame registration is chosen by matching every band between neighbouring frames and taking the one with the most verified inliers (or set it with `--ref-band`).

**Features and matching.** Every method in `features.py` produces key points and descriptors; FAST and Shi-Tomasi corners (which have no descriptor of their own) are paired with rBRIEF, SIFT or BEBLID descriptors. Matching uses Lowe's ratio test with one-to-one matches, a MAGSAC++ homography, and Metashape-style guided matching around the predicted position. A pair is accepted only with enough inliers, a sane inlier ratio, inliers spread over an area (not on one line — e.g. a transect rope) and a plausible transform. Phase correlation is the last resort for low-texture neighbours.

**Pair preselection.** Neighbours in capture order, plus "generic" preselection (all pairs matched on small thumbnails — finds overlaps between survey lanes), plus "estimated" preselection after a first adjustment (pairs whose footprints overlap). Images left outside the main block get a further attempt.

**Bundle adjustment.** All transforms are optimised together (Levenberg–Marquardt, Cauchy loss), minimising reprojection error in the pixels of the target image. A radial distortion (k1, k2) shared by all images is estimated with them — important for wide lenses and flat-port underwater housings. Pairs and points inconsistent with the solution are removed and the adjustment repeated (like gradual selection).

**Image quality.** Blur (motion, defocus close to rocks) is measured where frames overlap: each accepted pair is compared on its common area only, in both directions, and the log ratios are solved for one sharpness per frame. Scene content (smooth rock vs. textured sand) therefore does not count as blur. Blurred frames give way to sharper overlapping frames.

**Flat field.** Vignetting and a fixed illumination pattern (strobes, light from one side) multiply every frame of a series the same way. The per-pixel median of all frames (each divided by its own median) cancels the scene; a 2D polynomial fitted to it is divided out of every frame. On the underwater RGB test series the corners were at 0.63 of the centre brightness before correction.

**Seamlines.** The closest-centre rule gives a good global layout but can put a seam straight through a moving object or an object with relief. Around each of those seams a band is opened and a graph cut (colour + gradient cost) moves the seam to where the frames agree. With *deghosting*, every pixel is also compared with the median of all frames that see it: a frame that disagrees (a floating rope, a fish, a diver's shadow) is not used there if another frame shows the pixel consistently.

**Rendering.** The canvas is a disk-backed memmap; frames are decoded one at a time. Each output pixel comes from the frame chosen by the seamlines (closest to the image centre where nothing else matters, Metashape's rule). `mosaic` blending keeps the high frequencies of that frame and harmonises low frequencies across overlaps (a smooth per-frame gain field), which removes brightness steps from vignetting or strobes without ghosting. Use `--blend disabled` for quantitative multispectral analysis: pixel values are then exactly those of the source frames.

## Benchmark

On an underwater RGB survey (57 × 61 MP Sony ARW) every method aligned all images; RootSIFT/SIFT were the most accurate, FAST variants the weakest across survey lanes. Details: [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Limits

A 2D mosaic models the scene as a plane. Relief (rocks, plants) causes parallax that no planar transform can remove — seams are placed near frame centres to keep it small; for true orthomosaics of rough terrain use SfM software with a dense surface model. Moving objects are removed only where at least three frames see the area and another frame shows it consistently; an object present in most frames stays. The geometry is relative: `--gsd` gives a metric scale but no georeferencing.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests: synthetic bundle adjustment with known transforms and distortion, an end-to-end synthetic zig-zag survey with known geometry, blur detection, seamlines around and removal of a moving object, flat-field recovery, Sony cRAW decoding, cache reuse and TIFF/BigTIFF/VRT/world-file round trips.

## License

MIT
