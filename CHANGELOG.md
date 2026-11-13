# Changelog

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
