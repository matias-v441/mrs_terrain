"""World configuration loading and option validation (specification sections 5, 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from heightmap_prep.config import (
    NOMINAL_GRID_ORIGIN,
    PrepareOptions,
    Wgs84Bounds,
    align_origin,
    discover_world_paths,
    load_world,
    load_worlds,
    resolve_dataset_settings,
)
from heightmap_prep.errors import ConfigError, InvalidBoundsError


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


MINIMAL = """\
name: prague
bounds:
  type: wgs84
  west: 14.20
  south: 49.95
  east: 14.70
  north: 50.25
heightmap:
  source: cuzk-dmr5g
  resolution_m: 2.0
rgb:
  enabled: false
"""


def test_loads_the_specification_example(tmp_path: Path) -> None:
    world = load_world(write(tmp_path / "prague.yaml", MINIMAL))
    assert world.name == "prague"
    assert world.source == "cuzk-dmr5g"
    assert world.resolution_m == 2.0
    assert world.rgb_enabled is False
    assert world.bounds == Wgs84Bounds(14.20, 49.95, 14.70, 50.25)


def test_name_defaults_to_the_filename(tmp_path: Path) -> None:
    text = MINIMAL.replace("name: prague\n", "")
    assert load_world(write(tmp_path / "brno.yaml", text)).name == "brno"


def test_heightmap_section_is_optional(tmp_path: Path) -> None:
    text = "bounds:\n  west: 14.2\n  south: 49.95\n  east: 14.7\n  north: 50.25\n"
    world = load_world(write(tmp_path / "w.yaml", text))
    assert world.resolution_m is None
    assert world.source == "cuzk-dmr5g"


@pytest.mark.parametrize(
    "bounds,message",
    [
        ({"west": 14.7, "south": 49.9, "east": 14.2, "north": 50.2}, "east"),
        ({"west": 14.2, "south": 50.3, "east": 14.7, "north": 50.2}, "north"),
        ({"west": -200.0, "south": 49.9, "east": 14.7, "north": 50.2}, "longitudes"),
        ({"west": 14.2, "south": -95.0, "east": 14.7, "north": 50.2}, "latitudes"),
    ],
)
def test_invalid_bounds_are_rejected(bounds: dict, message: str) -> None:
    with pytest.raises(InvalidBoundsError, match=message):
        Wgs84Bounds(**bounds)


def test_missing_bounds_key_names_the_file(tmp_path: Path) -> None:
    path = write(tmp_path / "w.yaml", "name: w\nbounds:\n  west: 14.2\n  south: 49.9\n")
    with pytest.raises(ConfigError, match="east"):
        load_world(path)


def test_unsupported_bounds_type(tmp_path: Path) -> None:
    text = MINIMAL.replace("type: wgs84", "type: polygon")
    with pytest.raises(ConfigError, match="unsupported bounds type"):
        load_world(write(tmp_path / "w.yaml", text))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_world(tmp_path / "absent.yaml")


def test_empty_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="empty"):
        load_world(write(tmp_path / "w.yaml", ""))


def test_malformed_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_world(write(tmp_path / "w.yaml", "name: [unclosed\n"))


# --- input discovery (section 5) -----------------------------------------


def test_directory_input_is_expanded_and_sorted(tmp_path: Path) -> None:
    worlds = tmp_path / "worlds"
    worlds.mkdir()
    write(worlds / "b.yaml", MINIMAL.replace("prague", "b"))
    write(worlds / "a.yml", MINIMAL.replace("prague", "a"))
    write(worlds / "notes.txt", "ignored")
    assert [p.name for p in discover_world_paths([worlds])] == ["a.yml", "b.yaml"]


def test_file_inputs_keep_their_order(tmp_path: Path) -> None:
    first = write(tmp_path / "one.yaml", MINIMAL.replace("prague", "one"))
    second = write(tmp_path / "two.yaml", MINIMAL.replace("prague", "two"))
    assert discover_world_paths([second, first]) == [second, first]


def test_mixing_a_directory_with_files_is_rejected(tmp_path: Path) -> None:
    worlds = tmp_path / "worlds"
    worlds.mkdir()
    write(worlds / "a.yaml", MINIMAL)
    extra = write(tmp_path / "extra.yaml", MINIMAL)
    with pytest.raises(ConfigError, match="either a single worlds directory"):
        discover_world_paths([worlds, extra])


def test_empty_directory_is_rejected(tmp_path: Path) -> None:
    worlds = tmp_path / "worlds"
    worlds.mkdir()
    with pytest.raises(ConfigError, match="no .* files"):
        discover_world_paths([worlds])


def test_duplicate_world_names_are_rejected(tmp_path: Path) -> None:
    a = write(tmp_path / "a.yaml", MINIMAL)
    b = write(tmp_path / "b.yaml", MINIMAL)
    with pytest.raises(ConfigError, match="duplicate world name"):
        load_worlds([a, b])


# --- dataset-wide settings ------------------------------------------------


def test_resolution_is_inherited_from_options(tmp_path: Path) -> None:
    text = "name: w\nbounds:\n  west: 14.2\n  south: 49.9\n  east: 14.7\n  north: 50.2\n"
    world = load_world(write(tmp_path / "w.yaml", text))
    assert resolve_dataset_settings([world], PrepareOptions(resolution_m=5.0)) == (
        "cuzk-dmr5g",
        5.0,
    )


def test_conflicting_resolutions_are_rejected(tmp_path: Path) -> None:
    a = load_world(write(tmp_path / "a.yaml", MINIMAL.replace("prague", "a")))
    b = load_world(
        write(
            tmp_path / "b.yaml",
            MINIMAL.replace("prague", "b").replace("resolution_m: 2.0", "resolution_m: 5.0"),
        )
    )
    with pytest.raises(ConfigError, match="same heightmap resolution"):
        resolve_dataset_settings([a, b], PrepareOptions())


def test_conflicting_sources_are_rejected(tmp_path: Path) -> None:
    a = load_world(write(tmp_path / "a.yaml", MINIMAL.replace("prague", "a")))
    b = load_world(
        write(
            tmp_path / "b.yaml",
            MINIMAL.replace("prague", "b").replace("cuzk-dmr5g", "local-geotiff"),
        )
    )
    with pytest.raises(ConfigError, match="same heightmap source"):
        resolve_dataset_settings([a, b], PrepareOptions())


# --- options --------------------------------------------------------------


def test_option_defaults_match_the_specification() -> None:
    options = PrepareOptions()
    assert options.vertical_datum == "egm96"
    assert options.include_rgb is False
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
