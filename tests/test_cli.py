"""The command line interface (specification section 14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from heightmap_prep import cli
from heightmap_prep.config import PrepareOptions
from heightmap_prep.manifest import MANIFEST_FILENAME

from conftest import SyntheticRgbSource, SyntheticSource


def args(*argv: str):
    return cli.build_parser().parse_args(list(argv))


# --- argument shapes (section 5) -----------------------------------------


def test_several_world_files_then_the_output_directory() -> None:
    parsed = args("a.yaml", "b.yaml", "out")
    assert parsed.paths[:-1] == ["a.yaml", "b.yaml"]
    assert parsed.paths[-1] == "out"


def test_a_worlds_directory_then_the_output_directory() -> None:
    parsed = args("worlds", "out")
    assert parsed.paths == ["worlds", "out"]


def test_defaults_match_the_specification() -> None:
    parsed = args("w.yaml", "out")
    assert parsed.vertical_datum == "egm96"
    assert parsed.include_rgb is False
    assert parsed.rgb_resolution == 0.25
    assert parsed.rgb_jpeg_quality == 90
    assert parsed.resolution == 2.0
    assert parsed.tile_size == 4096
    assert parsed.workers == 1
    assert parsed.overwrite is False
    assert parsed.validate_only is False
    assert parsed.log_level == "info"
    assert parsed.grid_origin is None


def test_every_documented_option_is_accepted(tmp_path: Path) -> None:
    parsed = args(
        "w.yaml",
        "out",
        "--vertical-datum",
        "wgs84-ellipsoid",
        "--include-rgb",
        "--resolution",
        "5",
        "--tile-size",
        "2048",
        "--proj-data-dir",
        str(tmp_path),
        "--cache-dir",
        str(tmp_path),
        "--workers",
        "4",
        "--overwrite",
        "--log-level",
        "debug",
    )
    options = cli.options_from_args(parsed)
    assert options == PrepareOptions(
        vertical_datum="wgs84-ellipsoid",
        include_rgb=True,
        resolution_m=5.0,
        tile_size_px=2048,
        proj_data_dir=tmp_path,
        cache_dir=tmp_path,
        workers=4,
        overwrite=True,
    )


def test_an_unknown_datum_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        args("w.yaml", "out", "--vertical-datum", "egm2008")


def test_an_explicit_grid_origin_overrides_the_derived_one() -> None:
    options = cli.options_from_args(args("w.yaml", "out", "--grid-origin", "-5", "-6"))
    assert (options.grid_origin_x, options.grid_origin_y) == (-5.0, -6.0)


# --- behaviour ------------------------------------------------------------


def test_a_full_run_exits_zero(
    world_file: Path, tmp_path: Path, proj_dir: Path, monkeypatch
) -> None:
    real = cli.prepare_worlds
    monkeypatch.setattr(
        cli,
        "prepare_worlds",
        lambda inputs, out, options: real(inputs, out, options, source=SyntheticSource()),
    )
    out = tmp_path / "out"
    code = cli.main(
        [
            str(world_file),
            str(out),
            "--proj-data-dir",
            str(proj_dir),
            "--tile-size",
            "128",
            "--block-size",
            "16",
            "--log-level",
            "error",
        ]
    )
    assert code == 0
    assert (out / MANIFEST_FILENAME).is_file()
    assert list((out / "height").glob("tile_*.tif"))


def test_validate_only_needs_no_world_input(
    world_file: Path, tmp_path: Path, proj_dir: Path, monkeypatch
) -> None:
    real = cli.prepare_worlds
    monkeypatch.setattr(
        cli,
        "prepare_worlds",
        lambda inputs, out, options: real(inputs, out, options, source=SyntheticSource()),
    )
    out = tmp_path / "out"
    cli.main(
        [str(world_file), str(out), "--proj-data-dir", str(proj_dir),
         "--tile-size", "128", "--block-size", "16", "--log-level", "error"]
    )
    assert cli.main([str(out), "--validate-only", "--log-level", "error"]) == 0


def test_validate_only_reports_a_broken_dataset(tmp_path: Path) -> None:
    # Exit 1 means "the dataset is not valid", as opposed to exit 2 for a
    # library error such as a bad configuration file.
    assert cli.main([str(tmp_path), "--validate-only", "--log-level", "error"]) == 1


def test_a_missing_world_input_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "out")])


def test_library_errors_become_exit_code_two(tmp_path: Path) -> None:
    code = cli.main([str(tmp_path / "absent.yaml"), str(tmp_path / "out"), "--log-level", "error"])
    assert code == 2


def test_include_rgb_prepares_rgb_tiles(
    world_file: Path, tmp_path: Path, proj_dir: Path, monkeypatch
) -> None:
    real = cli.prepare_worlds
    seen = {}

    def fake(inputs, out, options):
        seen["options"] = options
        return real(
            inputs, out, options, source=SyntheticSource(), rgb_source=SyntheticRgbSource()
        )

    monkeypatch.setattr(cli, "prepare_worlds", fake)
    out = tmp_path / "out"
    code = cli.main(
        [str(world_file), str(out), "--proj-data-dir", str(proj_dir), "--tile-size", "128",
         "--block-size", "16", "--include-rgb", "--rgb-resolution", "1.0",
         "--rgb-jpeg-quality", "80"]
    )
    assert code == 0
    assert seen["options"].include_rgb and seen["options"].rgb_jpeg_quality == 80
    assert list((out / "rgb").glob("tile_*.tif"))
    assert (out / "worlds.sqlite").is_file()
