# Changelog

## 0.3.0

- Seamline feathering: a narrow transition (about one seam cell) across every seamline
  instead of a hard cut, so the short straight seams next to removed objects no longer
  show as steps (`--no-feather` restores hard seams).
- Moving-object removal keeps thin objects: small specks are rejected by area instead
  of a morphological opening that erased thin ropes.
- Veil detection: areas where one frame shows clearly less detail than the other frames
  (out-of-focus rope or hand close to the lens) are taken from frames that show them sharply.
- Image quality is measured on log-intensity images instead of contrast-equalised (CLAHE)
  images; CLAHE had inflated the detail of hazy frames, so the hazy views were preferred.
- Gain fields (two-band blending) ignore cells where a frame disagrees with the per-cell
  median of all frames: a rope seen by one frame no longer darkens that frame around it
  (dark blocks next to removed objects).
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
