# mosaic2d method benchmark — RGB (57 images)

Same images, same 565–567 candidate pairs, homography model, medium accuracy (2400 px working images), 8000 key points. Errors are reprojection errors after global adjustment, in native pixels (9504 x 6336 image). Times: 2 CPU cores, all 57 images.

| method | aligned | pairs accepted | median inliers | inlier ratio | median error px | RMSE px | features s | matching s |
|---|---|---|---|---|---|---|---|---|
| rootsift | 57/57 | 542 | 394 | 0.675 | 6.467 | 10.598 | 42.5 | 96.6 |
| sift | 57/57 | 534 | 403 | 0.678 | 6.54 | 10.682 | 40.3 | 90.2 |
| fast_sift | 57/57 | 357 | 457 | 0.516 | 7.005 | 11.578 | 165.3 | 87.7 |
| orb_beblid | 57/57 | 538 | 230 | 0.613 | 7.309 | 11.211 | 9.5 | 143.3 |
| brisk | 57/57 | 555 | 330 | 0.589 | 7.458 | 11.905 | 122.4 | 178.5 |
| akaze | 57/57 | 553 | 498 | 0.615 | 7.545 | 12.279 | 44.2 | 217.5 |
| orb | 57/57 | 535 | 241 | 0.59 | 7.567 | 11.691 | 8.0 | 126.7 |
| gftt_sift | 57/57 | 354 | 479 | 0.529 | 7.705 | 11.994 | 148.6 | 73.0 |
| fast_beblid | 57/57 | 471 | 506 | 0.57 | 7.8 | 12.554 | 9.5 | 173.6 |
| kaze | 57/57 | 556 | 440 | 0.607 | 7.803 | 12.484 | 241.0 | 98.9 |
| fast | 57/57 | 457 | 510 | 0.582 | 7.895 | 12.454 | 9.2 | 145.2 |

## Reading the table

- **All eleven methods connected all 57 images** — pair preselection (sequential + thumbnail "generic" + footprint "estimated") and the global adjustment matter more for completeness than the detector.
- **RootSIFT and SIFT give the lowest reprojection error** and the most consistent cross-lane pairs; they are first in the default cascade.
- **FAST-based methods** (fast, fast_sift, fast_beblid) find many inliers between consecutive frames but fewer cross-lane pairs (357–471 vs. 534–556): FAST has no scale or rotation invariance, and lanes of a zig-zag survey are rotated/at a different height relative to each other. They are fast and good as a cascade fallback.
- **ORB / ORB+BEBLID** are the fastest overall at slightly higher error.
- A median error of ~6.5 px on 9504 px images (0.07 %) is dominated by relief parallax on the rocky seabed, which a planar 2D mosaic cannot model.

## Multispectral (Agrowing Sextuple, 62 frames)

Reference band selection (median verified SIFT inliers between neighbouring frames):
525 nm 1525 · 550 nm 1522 · 560 nm 1473 · 630 nm 1368 · 570 nm 1329 · 650 nm 1268 · 450 nm 1170 · 710 nm 1048 · 490 nm 940 · 430 nm 885 · 685 nm 725 · 850 nm 179 · 735 nm 159 · 405 nm 51.
Green bands win under water; NIR (735/850 nm) and 405 nm are attenuated or noisy. With 525 nm, 62/62 frames aligned, median reprojection error 1.8 px (super-pixel resolution).
