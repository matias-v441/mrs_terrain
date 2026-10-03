"""World configs (MRS world files), input discovery and options."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from heightmap_prep.config import (
    NOMINAL_GRID_ORIGIN,
    PrepareOptions,
    align_origin,
    discover_world_paths,
    load_world,
    load_worlds,
    world_name_from_path,
)
from heightmap_prep.errors import ConfigError

from conftest import TEST_WORLD_POINTS, mrs_world


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


LATLON = mrs_world(TEST_WORLD_POINTS)


# --- parsing --------------------------------------------------------------


def test_latlon_points_are_read_as_lat_lon(tmp_path: Path) -> None:
    world = load_world(write(tmp_path / "world_bechovice.yaml", LATLON))
    assert world.name == "bechovice"
    assert len(world.points) == 4
    # Stored as (lon, lat); the file lists latitude first.
    assert world.points[0] == (14.4166195, 50.0800925)


def test_points_may_be_a_yaml_sequence(tmp_path: Path) -> None:
    text = LATLON.replace(
        f"points: [{TEST_WORLD_POINTS}]",
        "points:\n" + "".join(f"          - {v.strip()}\n" for v in TEST_WORLD_POINTS.split(",")),
    )
    assert load_world(write(tmp_path / "w.yaml", text)).points == load_world(
        write(tmp_path / "v.yaml", LATLON)
    ).points


def test_the_enabled_flag_is_irrelevant(tmp_path: Path) -> None:
    text = LATLON.replace("enabled: true", "enabled: false")
    assert len(load_world(write(tmp_path / "w.yaml", text)).points) == 4


def test_world_names_come_from_the_filename() -> None:
    assert world_name_from_path(Path("a/world_kn_yard.yaml")) == "kn_yard"
    assert world_name_from_path(Path("ricany.yaml")) == "ricany"
    assert world_name_from_path(Path("world_.yaml")) == "world_"


@pytest.mark.parametrize("frame", ["world_origin", "local_origin", "fcu", ""])
def test_a_safety_area_in_another_frame_is_rejected(tmp_path: Path, frame: str) -> None:
    text = mrs_world("0.0, 0.0, 10.0, 0.0, 10.0, 10.0", frame=frame)
    with pytest.raises(ConfigError, match="not 'latlon_origin'"):
        load_world(write(tmp_path / "w.yaml", text))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "just: a mapping\n",
        "- a\n- list\n",
        "mrs_uav_managers:\n  world_origin: {units: LATLON}\n",
        "mrs_uav_managers:\n  safety_area_manager:\n    safety_area:\n      enabled: true\n",
    ],
)
def test_a_file_without_a_safety_area_is_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match="has no"):
        load_world(write(tmp_path / "w.yaml", text))


@pytest.mark.parametrize(
    "points,message",
    [
        ("50.08, 14.41, 50.07, 14.42, 50.07", "not a whole number"),
        ("50.08, 14.41, 50.07, 14.42", "at least 3 vertices"),
        ('"north", 14.41, 50.07, 14.42, 50.07, 14.43', "not a number"),
        ("500.0, 14.41, 50.07, 14.42, 50.07, 14.43", "not a valid latitude/longitude"),
        (".nan, 14.41, 50.07, 14.42, 50.07, 14.43", "not finite"),
    ],
)
def test_a_malformed_latlon_safety_area_is_an_error(
    tmp_path: Path, points: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_world(write(tmp_path / "w.yaml", mrs_world(points)))


def test_missing_points_are_an_error(tmp_path: Path) -> None:
    text = LATLON.replace(f"        points: [{TEST_WORLD_POINTS}]\n", "")
    with pytest.raises(ConfigError, match="no 'points'"):
        load_world(write(tmp_path / "w.yaml", text))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_world(tmp_path / "absent.yaml")


def test_malformed_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_world(write(tmp_path / "w.yaml", "mrs_uav_managers: [unclosed\n"))


# --- loading several worlds -----------------------------------------------


def test_unusable_world_configs_are_skipped_with_a_warning(tmp_path: Path, caplog) -> None:
    kept = write(tmp_path / "world_kept.yaml", LATLON)
    metric = write(tmp_path / "world_metric.yaml", mrs_world("1, 2, 3, 4, 5, 6", "world_origin"))
    unrelated = write(tmp_path / "unrelated.yaml", "some: other config\n")
    malformed = write(tmp_path / "world_bad.yaml", mrs_world("50.08, 14.41, 50.07, 14.42"))
    broken = write(tmp_path / "world_broken.yaml", "mrs_uav_managers: [unclosed\n")
    missing = tmp_path / "world_absent.yaml"
    with caplog.at_level(logging.WARNING):
        worlds, skipped = load_worlds([kept, metric, unrelated, malformed, broken, missing])
    assert [w.name for w in worlds] == ["kept"]
    assert list(skipped) == [metric, unrelated, malformed, broken, missing]
    assert "not 'latlon_origin'" in skipped[metric]
    assert "at least 3 vertices" in skipped[malformed]
    assert "invalid YAML" in skipped[broken]
    assert "not found" in skipped[missing]
    for path in skipped:
        assert path.name in caplog.text


def test_no_usable_world_leaves_nothing_but_reasons(tmp_path: Path) -> None:
    metric = write(tmp_path / "world_metric.yaml", mrs_world("1, 2, 3, 4, 5, 6", "world_origin"))
    worlds, skipped = load_worlds([metric])
    assert worlds == [] and list(skipped) == [metric]


def test_a_world_name_already_taken_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = write(tmp_path / "a" / "world_same.yaml", LATLON)
    b = write(tmp_path / "b" / "world_same.yaml", LATLON)
    worlds, skipped = load_worlds([a, b])
    assert [w.path for w in worlds] == [a]
    assert "already taken" in skipped[b]


# --- input discovery (section 5) -----------------------------------------


def test_directory_input_is_expanded_and_sorted(tmp_path: Path) -> None:
    worlds = tmp_path / "worlds"
    worlds.mkdir()
    write(worlds / "b.yaml", LATLON)
    write(worlds / "a.yml", LATLON)
    write(worlds / "notes.txt", "ignored")
    assert [p.name for p in discover_world_paths([worlds])] == ["a.yml", "b.yaml"]


def test_file_inputs_keep_their_order(tmp_path: Path) -> None:
    first = write(tmp_path / "one.yaml", LATLON)
    second = write(tmp_path / "two.yaml", LATLON)
    assert discover_world_paths([second, first]) == [second, first]


def test_mixing_a_directory_with_files_is_rejected(tmp_path: Path) -> None:
    worlds = tmp_path / "worlds"
    worlds.mkdir()
    write(worlds / "a.yaml", LATLON)
    extra = write(tmp_path / "extra.yaml", LATLON)
    with pytest.raises(ConfigError, match="either a single worlds directory"):
        discover_world_paths([worlds, extra])


def test_empty_directory_is_rejected(tmp_path: Path) -> None:
    worlds = tmp_path / "worlds"
    worlds.mkdir()
    with pytest.raises(ConfigError, match="no .* files"):
        discover_world_paths([worlds])


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "worlds").is_dir(), reason="no worlds/"
)
def test_the_repository_worlds_load() -> None:
    """The shipped worlds with a latlon safety area load; the others are skipped."""
    worlds, _ = load_worlds([Path(__file__).resolve().parents[1] / "worlds"])
    names = {w.name for w in worlds}
    assert {"bechovice", "kn_yard", "ricany"} <= names
    assert "cisar" not in names and "local" not in names


# --- options --------------------------------------------------------------


def test_option_defaults_match_the_specification() -> None:
    options = PrepareOptions()
    assert options.vertical_datum == "egm96"
    assert options.include_rgb is False
    assert options.rgb_resolution_m == 0.25
    assert options.rgb_jpeg_quality == 90
    assert options.resolution_m == 2.0
    assert options.tile_size_px == 4096
    assert options.block_size_px == 256
    assert options.nodata == -9999.0
    assert options.workers == 1
    assert options.overwrite is False


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"vertical_datum": "egm2008"}, "unsupported vertical datum"),
        ({"resolution_m": 0.0}, "resolution_m must be positive"),
        ({"tile_size_px": 0}, "tile_size_px must be positive"),
        ({"block_size_px": 100}, "multiple of 16"),
        ({"workers": 0}, "workers must be at least 1"),
        ({"rgb_resolution_m": -1.0}, "rgb_resolution_m must be positive"),
        ({"rgb_jpeg_quality": 0}, "rgb_jpeg_quality must be between"),
        ({"rgb_jpeg_quality": 101}, "rgb_jpeg_quality must be between"),
    ],
)
def test_invalid_options_are_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        PrepareOptions(**kwargs)


# --- grid phase alignment -------------------------------------------------


def test_align_origin_matches_the_native_pixel_phase() -> None:
    native = (-904_703.6, -935_118.12)
    origin_x, origin_y = align_origin(NOMINAL_GRID_ORIGIN, native, 2.0)
    assert origin_x == pytest.approx(-999_999.6)
    assert origin_y == pytest.approx(-800_000.12)
    # Every tile edge now falls on a native pixel boundary.
    assert (origin_x - native[0]) % 2.0 == pytest.approx(0.0, abs=1e-9)
    assert (origin_y - native[1]) % 2.0 == pytest.approx(0.0, abs=1e-9)


def test_align_origin_stays_west_and_north_of_the_data() -> None:
    origin_x, origin_y = align_origin(NOMINAL_GRID_ORIGIN, (-904_703.6, -935_118.12), 2.0)
    assert origin_x <= -904_703.6
    assert origin_y >= -935_118.12
    assert abs(origin_x - NOMINAL_GRID_ORIGIN[0]) < 2.0
    assert abs(origin_y - NOMINAL_GRID_ORIGIN[1]) < 2.0


def test_align_origin_without_a_native_grid_is_a_no_op() -> None:
    assert align_origin(NOMINAL_GRID_ORIGIN, None, 2.0) == NOMINAL_GRID_ORIGIN
