"""
Command line interface.

    mosaic2d run FOLDER [-o OUT] [options]        build a mosaic
    mosaic2d benchmark FOLDER [-o OUT] [options]  compare feature methods
    mosaic2d methods                              list methods and availability
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

from . import __version__
from . import features as FT
from .pipeline import ACCURACY, Settings, benchmark, run


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("inputs", nargs="+", help="image folder(s) or files (RAW, JPG, TIFF)")
    p.add_argument("-o", "--out", help="output folder (default: <first input>/mosaic2d)")
    p.add_argument("--profile", default="auto",
                   help="camera profile: auto, rgb, agrowing6, multiband or a JSON file")
    p.add_argument("--methods", default="auto",
                   help="feature methods in cascade order, e.g. sift,akaze,orb,fast (default: all available)")
    p.add_argument("--strategy", default="cascade", choices=["cascade", "union", "single"],
                   help="cascade: per pair, next method only if the previous failed; "
                        "union: pool matches of all methods; single: first method only")
    p.add_argument("--model", default="homography",
                   choices=["translation", "similarity", "affine", "homography"])
    p.add_argument("--accuracy", default="medium", choices=list(ACCURACY),
                   help="working resolution for matching (like Metashape's alignment accuracy)")
    p.add_argument("--features", type=int, default=8000, dest="n_features", help="key point limit per image")
    p.add_argument("--preselection", default="auto",
                   choices=["auto", "sequential", "generic", "exhaustive"])
    p.add_argument("--window", type=int, default=3, help="sequential neighbours matched per image")
    p.add_argument("--no-distortion", dest="distortion", action="store_false",
                   help="do not self-calibrate radial lens distortion")
    p.add_argument("--ref-band", default=None, help='reference band for multispectral, e.g. "560 nm"')
    p.add_argument("--demosaic", default=None, choices=["superpixel", "bilinear"])
    p.add_argument("--no-phasecorr", dest="phasecorr", action="store_false")
    p.add_argument("--min-inliers", type=int, default=25)
    p.add_argument("--ratio", type=float, default=0.8)
    p.add_argument("--jobs", type=int, default=0, help="worker threads (default: up to 4)")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--work-dir", default=None, help="folder for the temporary canvas and the cache")
    p.add_argument("--no-cache", dest="cache", action="store_false",
                   help="do not reuse decoded frames from earlier runs (<out>/_cache)")


def build_parser():
    ap = argparse.ArgumentParser(prog="mosaic2d", description=f"mosaic2d {__version__}: 2D mosaics "
                                 "from overlapping RGB or multispectral photographs")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="build a mosaic")
    _add_common(r)
    r.add_argument("--blend", default="mosaic", choices=["mosaic", "average", "disabled"])
    r.add_argument("--seams", default="graphcut", choices=["graphcut", "voronoi"],
                   help="graphcut: move seams away from moving objects and parallax; voronoi: closest centre")
    r.add_argument("--no-deghost", dest="deghost", action="store_false",
                   help="keep moving objects (rope, fish) instead of taking those areas from consistent frames")
    r.add_argument("--no-feather", dest="feather", action="store_false",
                   help="hard seamlines (no narrow transition) in mosaic blending")
    r.add_argument("--flatfield", default="auto", choices=["auto", "none"],
                   help="estimate and remove vignetting / fixed illumination pattern from the series")
    r.add_argument("--resolution", type=float, default=1.0,
                   help="output pixel size relative to the native pixels of the median image "
                        "(0.5 = half resolution)")
    r.add_argument("--orient", default="pca", choices=["pca", "anchor"])
    r.add_argument("--name", default="mosaic")
    r.add_argument("--gsd", type=float, default=None,
                   help="metres per pixel at --resolution 1 (e.g. from a scale bar); writes GeoTIFF "
                        "scale tags and a .tfw so GIS measurements are in metres")
    r.add_argument("--no-render", action="store_true", help="align only; write a preview from thumbnails")
    r.add_argument("--keep-canvas", action="store_true")
    r.add_argument("--tiles", type=int, default=0, dest="tile_size",
                   help="split the output into tiles of this size (px) plus a .vrt that joins them")
    rr = sub.add_parser("render", help="render again from a saved project (<name>_project.pkl)")
    rr.add_argument("project")
    rr.add_argument("-o", "--out", default=None)
    rr.add_argument("--blend", default=None, choices=["mosaic", "average", "disabled"])
    rr.add_argument("--resolution", type=float, default=None)
    rr.add_argument("--orient", default=None, choices=["pca", "anchor"])
    rr.add_argument("--tiles", type=int, default=None, dest="tile_size")
    rr.add_argument("--seams", default=None, choices=["graphcut", "voronoi"])
    rr.add_argument("--gsd", type=float, default=None)
    rr.add_argument("--no-deghost", dest="deghost", action="store_const", const=False, default=None)
    rr.add_argument("--no-feather", dest="feather", action="store_const", const=False, default=None)
    rr.add_argument("--work-dir", default=None)
    rr.add_argument("--name", default="mosaic")
    b = sub.add_parser("benchmark", help="compare feature methods on the same images")
    _add_common(b)
    sub.add_parser("methods", help="list feature methods")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "methods":
        reg = FT.build_registry()
        for m in reg.values():
            print(f"{m.name:12s} {'available ' if m.available else 'MISSING   '} {m.family:6s}  {m.note}")
        return 0
    if args.cmd == "render":
        from .pipeline import Project
        proj = Project.load(args.project, args.out, blend=args.blend, resolution=args.resolution,
                            orient=args.orient, tile_size=args.tile_size, work_dir=args.work_dir,
                            seams=args.seams, deghost=args.deghost, gsd=args.gsd,
                            feather=args.feather)
        proj.render(args.name)
        proj.report(args.name)
        return 0
    known = {f.name for f in fields(Settings)}
    s = Settings(**{k: v for k, v in vars(args).items() if k in known and v is not None})
    out = Path(args.out) if args.out else Path(args.inputs[0]) / "mosaic2d"
    if args.cmd == "benchmark":
        benchmark(args.inputs, out, s, methods=args.methods)
        return 0
    proj, rep = run(args.inputs, out, s, name=args.name, render=not args.no_render)
    return 0 if rep["aligned_images"] == rep["images"] else 1


if __name__ == "__main__":
    sys.exit(main())
