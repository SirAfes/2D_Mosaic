# Changelog

## 0.3.0

- New default adjustment, `--model camera`: every photo is a camera over one ground plane
  with a shared lens (position, height, heading, two tilts per photo; focal length and
  radial distortion refined from EXIF / profile), soft priors for similar height and
  near-nadir view. Photos can no longer change size along the survey without a real
  change of height, shear or fold. 476-photo RGB dive: one continuous mosaic of all 301
  connected photos with a straight transect tape (homographies: two pieces, one sheared);
  synthetic strips: scale error 0.7-1.5 % vs 6-22 %. Estimated focal length 1.31-1.34 x
  EXIF underwater (flat-port refraction). Separate groups reuse the main block's lens.
- `--height` (m): metric pixel size from camera height / focal length; also in the window.
  `--focal-px`, `--height-sigma`. Report and images.csv list focal length, heights, tilts.
- Agrowing profile carries the lens focal length (21.8 mm, 3.76 um pixels).
- Output TIFFs are tiled (512 x 512) with internal overviews (pyramids), like Metashape
  and GDAL exports: viewers read only the tiles on screen and use the small overviews
  when zoomed out, instead of decoding image-wide strips and building pyramids first.
- Main block: frames with degenerate geometry are dropped, the main mosaic keeps only
  the largest piece still connected through good pairs (pieces that hung on through
  dropped frames become separate groups instead of being drawn beside it), and that
  piece is adjusted again on its own. The affine fallback is used only if it is clearly
  better (on a 476-image RGB dive it was 3x worse and had sheared part of the mosaic).
- Projects saved on Windows can be loaded on Linux/macOS and the other way round.
- Gaps in the coverage: every other group of connected images (`--min-block`, default 5)
  is adjusted and rendered on its own as `<name>_blockNN`; `<name>_blocks.csv` lists all
  groups. Before, only the largest group was rendered.
- Multiband TIFFs (msalign output and other GDAL-style files): band names and centre
  wavelengths are read from the file (GDAL metadata or msalign JSON) instead of
  "band 1 ... band N"; preview colours chosen by wavelength. `mosaic2d render` refreshes
  the names in projects saved before this change.
- Seam maps label frames by image number. tifffile is a regular dependency.
- Windows executable (`mosaic2d.exe`, built by GitHub Actions on every version tag)
  with a small window: choose folders and options, follow the log, open the results.
  `mosaic2d gui` opens the same window from a Python installation.
- Seamline feathering: a narrow transition (about one seam cell) across every seamline
  instead of a hard cut, so the short straight seams next to removed objects no longer
  show as steps (`--no-feather` restores hard seams).
- Moving-object removal keeps thin objects: small specks are rejected by area instead
  of a morphological opening that erased thin ropes.
- Veil detection: areas where one frame shows clearly less detail than the other frames
  (out-of-focus rope or hand close to the lens) are taken from frames that show them sharply.
- Image quality is measured on log-intensity images instead of contrast-equalised (CLAHE)
  images; CLAHE had inflated the detail of hazy frames, so the hazy views were preferred.
- Gain fields (two-band blending) ignore cells where a frame is a minority outlier
  against the per-cell median of all frames: a rope seen by one frame no longer darkens
  that frame around it (dark blocks next to removed objects). Where most views disagree
  (relief, parallax) nothing is left out, so the gains stay stable.
- Feathering keeps the weight of the frame that owns a pixel outright; a lone minor band
  contribution can no longer replace it.
- `--resolution` (and `--gsd`) refer to the native pixel of the median frame, not of the
  anchor frame, so the output scale no longer depends on which frame becomes the anchor
  (1.34x difference on the RGB test series, whose camera height varies).
- Seam analysis resolution relative to the frame size (same behaviour at every `--resolution`).
- The empty temporary canvas folder is removed after rendering.

## 0.2.0

- Seamlines optimised with a graph cut inside a band around the closest-centre seams.
- Deghosting: areas where a frame disagrees with the consensus of the other frames
  (moving rope, fish) are taken from a consistent frame.
- Flat field (vignetting / fixed illumination) estimated from the whole series and removed.
- Image quality (blur) estimated from overlaps instead of per image, so scene content
  does not count as blur.
- Low-frequency gain fields defined relative to the frame size.
- Ingest cache (`<out>/_cache`) and cached reference-band selection.
- `--gsd`: metric pixel size in GeoTIFF tags, `.tfw` world files and the VRT.
- `mosaic2d render`: render a saved project again (resolution, blending, seams, scale).
- Each render uses a private temporary folder (parallel runs no longer collide).

## 0.1.0

- Sony ARW decoder, Agrowing Sextuple profile with per-frame lens alignment and
  automatic reference band.
- Eleven feature methods with per-pair cascade / union, sequential + generic +
  estimated pair preselection, MAGSAC++ verification, guided matching.
- Global bundle adjustment with radial distortion self-calibration.
- Out-of-core rendering, mosaic/average/disabled blending, (Big)TIFF and tiled VRT output.
- Method benchmark.
