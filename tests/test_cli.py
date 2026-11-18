"""Command line and window argument handling (no display needed)."""

from mosaic2d.cli import build_parser
from mosaic2d.gui import build_argv


def test_gui_builds_valid_run_arguments(tmp_path):
    argv = build_argv(tmp_path, tmp_path / "out", 0.5, "agrowing6", "high", True, name="MS")
    args = build_parser().parse_args(argv)
    assert args.cmd == "run" and args.inputs == [str(tmp_path)]
    assert args.out == str(tmp_path / "out")
    assert args.resolution == 0.5 and args.profile == "agrowing6" and args.accuracy == "high"
    assert args.tile_size == 8192 and args.name == "MS"
    args = build_parser().parse_args(build_argv(tmp_path, "", 1.0, "auto", "medium", False))
    assert args.out is None and args.tile_size == 0 and args.height_m is None and args.model == "camera"
    args = build_parser().parse_args(build_argv(tmp_path, "", 1.0, "auto", "medium", False, height=1.5))
    assert args.height_m == 1.5


def test_gui_subcommand_parses():
    args = build_parser().parse_args(["gui", "--selftest"])
    assert args.cmd == "gui" and args.selftest


def test_render_readjust_parses():
    args = build_parser().parse_args(["render", "p.pkl", "--readjust", "--min-block", "0"])
    assert args.readjust and args.min_block == 0 and args.model is None
