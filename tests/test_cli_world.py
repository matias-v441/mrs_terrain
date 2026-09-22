"""The heightmap-prep-world CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from heightmap_prep import cli_world
from heightmap_prep.config import load_world

from test_mrs_world import LATLON_WORLD, UTM_WORLD


@pytest.fixture
def mrs_world(tmp_path: Path) -> Path:
    path = tmp_path / "world_bechovice.yaml"
    path.write_text(LATLON_WORLD, encoding="utf-8")
    return path


# --- output ---------------------------------------------------------------


def test_the_output_is_a_loadable_heightmap_prep_world(mrs_world: Path, tmp_path: Path) -> None:
    out = tmp_path / "bechovice.yaml"
    assert cli_world.main([str(mrs_world), "-o", str(out)]) == 0

    world = load_world(out)
    assert world.name == "bechovice"
    assert world.source == "cuzk-dmr5g"
    assert world.resolution_m == 2.0
    assert world.rgb_enabled is False
    assert world.bounds.west < 14.6327381
    assert world.bounds.east > 14.6348664
    assert world.bounds.south < 50.0896023
    assert world.bounds.north > 50.0910495


def test_the_output_matches_the_example_layout(mrs_world: Path, tmp_path: Path) -> None:
    out = tmp_path / "w.yaml"
    cli_world.main([str(mrs_world), "-o", str(out)])
    document = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert set(document) == {"name", "bounds", "heightmap", "rgb"}
    assert document["bounds"]["type"] == "wgs84"
    assert set(document["bounds"]) == {"type", "west", "south", "east", "north"}
    assert document["heightmap"] == {"source": "cuzk-dmr5g", "resolution_m": 2.0}
    assert document["rgb"] == {"enabled": False}
    # resolution_m must stay a float, not become an integer.
    assert isinstance(document["heightmap"]["resolution_m"], float)


def test_the_header_records_where_the_bounds_came_from(
    mrs_world: Path, tmp_path: Path
) -> None:
    out = tmp_path / "w.yaml"
    cli_world.main([str(mrs_world), "-o", str(out), "--margin-m", "250"])
    header = out.read_text(encoding="utf-8").splitlines()[0:3]
    text = "\n".join(header)
    assert "world_bechovice.yaml" in text
    assert "4 vertices" in text
    assert "250 m" in text


def test_without_output_the_config_goes_to_stdout(mrs_world: Path, capsys) -> None:
    assert cli_world.main([str(mrs_world)]) == 0
    captured = capsys.readouterr().out
    assert yaml.safe_load(captured)["name"] == "bechovice"


# --- options --------------------------------------------------------------


def test_the_margin_changes_the_bounds(mrs_world: Path, tmp_path: Path) -> None:
    small = tmp_path / "small.yaml"
    large = tmp_path / "large.yaml"
    cli_world.main([str(mrs_world), "-o", str(small), "--margin-m", "10"])
    cli_world.main([str(mrs_world), "-o", str(large), "--margin-m", "500"])
    assert load_world(large).bounds.west < load_world(small).bounds.west
    assert load_world(large).bounds.north > load_world(small).bounds.north


def test_a_zero_margin_is_allowed(mrs_world: Path, tmp_path: Path) -> None:
    out = tmp_path / "w.yaml"
    assert cli_world.main([str(mrs_world), "-o", str(out), "--margin-m", "0"]) == 0
    assert load_world(out).bounds.west == pytest.approx(14.6327381)


def test_name_source_and_resolution_can_be_overridden(
    mrs_world: Path, tmp_path: Path
) -> None:
    out = tmp_path / "w.yaml"
    cli_world.main(
        [str(mrs_world), "-o", str(out), "--name", "custom",
         "--resolution", "5", "--source", "local-geotiff"]
    )
    world = load_world(out)
    assert (world.name, world.resolution_m, world.source) == ("custom", 5.0, "local-geotiff")


def test_the_utm_zone_reaches_the_parser(tmp_path: Path) -> None:
    source = tmp_path / "world_cisar.yaml"
    source.write_text(UTM_WORLD, encoding="utf-8")
    out = tmp_path / "w.yaml"
    cli_world.main([str(source), "-o", str(out), "--utm-zone", "34N"])
    assert load_world(out).bounds.west == pytest.approx(20.4185, abs=1e-2)


def test_defaults_match_the_documented_ones() -> None:
    parsed = cli_world.build_parser().parse_args(["w.yaml"])
    assert parsed.margin_m == cli_world.DEFAULT_MARGIN_M == 100.0
    assert parsed.resolution == 2.0
    assert parsed.source == "cuzk-dmr5g"
    assert parsed.utm_zone == "33N"
    assert parsed.output is None


# --- several inputs -------------------------------------------------------


def test_several_inputs_are_written_into_a_directory(tmp_path: Path) -> None:
    first = tmp_path / "world_one.yaml"
    second = tmp_path / "world_two.yaml"
    first.write_text(LATLON_WORLD, encoding="utf-8")
    second.write_text(LATLON_WORLD, encoding="utf-8")
    out = tmp_path / "generated"
    out.mkdir()

    assert cli_world.main([str(first), str(second), "-o", str(out)]) == 0
    assert load_world(out / "one.yaml").name == "one"
    assert load_world(out / "two.yaml").name == "two"


def test_several_inputs_cannot_share_one_output_file(tmp_path: Path) -> None:
    first = tmp_path / "world_one.yaml"
    second = tmp_path / "world_two.yaml"
    first.write_text(LATLON_WORLD, encoding="utf-8")
    second.write_text(LATLON_WORLD, encoding="utf-8")
    existing = tmp_path / "out.yaml"
    existing.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli_world.main([str(first), str(second), "-o", str(existing)])


def test_name_is_refused_for_several_inputs(tmp_path: Path) -> None:
    first = tmp_path / "world_one.yaml"
    second = tmp_path / "world_two.yaml"
    first.write_text(LATLON_WORLD, encoding="utf-8")
    second.write_text(LATLON_WORLD, encoding="utf-8")
    with pytest.raises(SystemExit):
        cli_world.main([str(first), str(second), "--name", "x"])


# --- failures -------------------------------------------------------------


def test_an_existing_output_is_not_clobbered(mrs_world: Path, tmp_path: Path) -> None:
    out = tmp_path / "w.yaml"
    out.write_text("keep me\n", encoding="utf-8")
    assert cli_world.main([str(mrs_world), "-o", str(out), "--log-level", "error"]) == 2
    assert out.read_text(encoding="utf-8") == "keep me\n"
    assert cli_world.main([str(mrs_world), "-o", str(out), "--force"]) == 0
    assert load_world(out).name == "bechovice"


def test_an_unusable_world_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "world_local.yaml"
    path.write_text(
        LATLON_WORLD.replace('frame_name: "latlon_origin"', 'frame_name: "local_origin"'),
        encoding="utf-8",
    )
    assert cli_world.main([str(path), "--log-level", "error"]) == 2


def test_a_missing_input_exits_two(tmp_path: Path) -> None:
    assert cli_world.main([str(tmp_path / "absent.yaml"), "--log-level", "error"]) == 2


def test_a_negative_margin_is_refused(mrs_world: Path) -> None:
    with pytest.raises(SystemExit):
        cli_world.main([str(mrs_world), "--margin-m", "-5"])
