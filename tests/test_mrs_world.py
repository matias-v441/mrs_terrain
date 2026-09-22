"""Reading MRS UAV system world files and deriving heightmap bounds."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyproj import Geod

from heightmap_prep.errors import ConfigError
from heightmap_prep.mrs_world import (
    DEFAULT_UTM_ZONE,
    expand_bounds,
    load_mrs_world,
    parse_mrs_world,
    safety_area_bounds,
    world_name_from_path,
    _utm_epsg,
)

GEOD = Geod(ellps="WGS84")

REPO_WORLDS = Path(__file__).resolve().parents[1] / "worlds"

LATLON_WORLD = """\
mrs_uav_managers:
  world_origin:
    units: "LATLON"
    origin_x: 50.090278
    origin_y: 14.634639
  safety_area_manager:
    safety_area:
      enabled: true
      horizontal:
        frame_name: "latlon_origin"
        points: [
          50.0905258, 14.6327381,
          50.0896023, 14.6330607,
          50.0902173, 14.6348664,
          50.0910495, 14.6346838,
        ]
      vertical:
        frame_name: "world_origin"
        max_z: 35.0
        min_z: 1.0
"""


def write(tmp_path: Path, text: str, name: str = "world_test.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- parsing --------------------------------------------------------------


def test_latlon_points_are_read_as_lat_lon(tmp_path: Path) -> None:
    world = load_mrs_world(write(tmp_path, LATLON_WORLD))
    assert world.units == "LATLON"
    assert (world.origin_lat, world.origin_lon) == (50.090278, 14.634639)
    # Stored as (lon, lat); the file lists latitude first.
    assert world.safety_area.points[0] == (14.6327381, 50.0905258)
    assert len(world.safety_area.points) == 4
    assert world.safety_area.enabled is True
    assert world.safety_area.max_z == 35.0


def test_the_bounding_box_spans_the_polygon(tmp_path: Path) -> None:
    area = load_mrs_world(write(tmp_path, LATLON_WORLD)).safety_area
    west, south, east, north = area.bounding_box()
    assert (west, south, east, north) == (14.6327381, 50.0896023, 14.6348664, 50.0910495)


def test_name_is_derived_from_the_filename(tmp_path: Path) -> None:
    assert load_mrs_world(write(tmp_path, LATLON_WORLD, "world_bechovice.yaml")).name == "bechovice"
    assert load_mrs_world(write(tmp_path, LATLON_WORLD, "ricany.yaml")).name == "ricany"


def test_an_explicit_name_wins(tmp_path: Path) -> None:
    assert load_mrs_world(write(tmp_path, LATLON_WORLD), name="custom").name == "custom"


def test_world_name_helper() -> None:
    assert world_name_from_path(Path("a/world_kn_yard.yaml")) == "kn_yard"
    assert world_name_from_path(Path("world.yaml")) == "world"  # no bare prefix strip


def test_a_disabled_safety_area_is_still_usable(tmp_path: Path) -> None:
    world = load_mrs_world(write(tmp_path, LATLON_WORLD.replace("enabled: true", "enabled: false")))
    assert world.safety_area.enabled is False
    assert len(world.safety_area.points) == 4


def test_points_may_be_a_yaml_sequence(tmp_path: Path) -> None:
    text = LATLON_WORLD.replace(
        """points: [
          50.0905258, 14.6327381,
          50.0896023, 14.6330607,
          50.0902173, 14.6348664,
          50.0910495, 14.6346838,
        ]""",
        """points:
          - 50.0905258
          - 14.6327381
          - 50.0896023
          - 14.6330607
          - 50.0902173
          - 14.6348664
          - 50.0910495
          - 14.6346838""",
    )
    assert len(load_mrs_world(write(tmp_path, text)).safety_area.points) == 4


# --- metric frames --------------------------------------------------------

UTM_WORLD = """\
mrs_uav_managers:
  world_origin:
    units: "UTM"
    origin_x: 458422.2
    origin_y: 5551241.4
  safety_area_manager:
    safety_area:
      enabled: true
      horizontal:
        frame_name: "world_origin"
        points: [
          30.9, 26.4,
          -45.0, 7.27,
          -31.5, -38.0,
          44.5, -23.5,
        ]
      vertical:
        frame_name: "world_origin"
        max_z: 25.0
        min_z: 1.0
"""


def test_a_utm_origin_is_projected_to_lat_lon(tmp_path: Path) -> None:
    world = load_mrs_world(write(tmp_path, UTM_WORLD))
    assert world.utm_zone == DEFAULT_UTM_ZONE
    # Zone 33N puts this origin in Prague.
    assert world.origin_lat == pytest.approx(50.111964, abs=1e-5)
    assert world.origin_lon == pytest.approx(14.418495, abs=1e-5)


def test_metric_points_keep_their_metre_offsets(tmp_path: Path) -> None:
    """x is east and y is north, so the polygon keeps its real dimensions."""
    world = load_mrs_world(write(tmp_path, UTM_WORLD))
    west, south, east, north = world.safety_area.bounding_box()
    mid_lat, mid_lon = (south + north) / 2, (west + east) / 2
    width = GEOD.inv(west, mid_lat, east, mid_lat)[2]
    height = GEOD.inv(mid_lon, south, mid_lon, north)[2]
    assert width == pytest.approx(44.5 - -45.0, abs=1.0)
    assert height == pytest.approx(26.4 - -38.0, abs=1.0)


def test_the_utm_zone_can_be_overridden(tmp_path: Path) -> None:
    path = write(tmp_path, UTM_WORLD)
    assert load_mrs_world(path, utm_zone="34N").origin_lon == pytest.approx(20.4185, abs=1e-3)


def test_a_bad_utm_zone_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="malformed UTM zone"):
        load_mrs_world(write(tmp_path, UTM_WORLD), utm_zone="33")
    with pytest.raises(ConfigError, match="1..60"):
        load_mrs_world(write(tmp_path, UTM_WORLD), utm_zone="99N")


def test_utm_epsg_codes() -> None:
    assert _utm_epsg("33N") == 32633
    assert _utm_epsg("18s") == 32718


def test_metric_points_around_a_geographic_origin(tmp_path: Path) -> None:
    """world_origin metres with a LATLON origin step along the ellipsoid."""
    text = LATLON_WORLD.replace('frame_name: "latlon_origin"', 'frame_name: "world_origin"').replace(
        """points: [
          50.0905258, 14.6327381,
          50.0896023, 14.6330607,
          50.0902173, 14.6348664,
          50.0910495, 14.6346838,
        ]""",
        "points: [100.0, 0.0, 0.0, 100.0, -100.0, 0.0, 0.0, -100.0]",
    )
    world = load_mrs_world(write(tmp_path, text))
    # (100, 0) is 100 m due east of the origin.
    lon, lat = world.safety_area.points[0]
    azimuth, _, distance = GEOD.inv(world.origin_lon, world.origin_lat, lon, lat)
    assert distance == pytest.approx(100.0, abs=0.01)
    assert azimuth == pytest.approx(90.0, abs=0.01)


# --- rejections -----------------------------------------------------------


def test_an_unreferenced_frame_is_rejected(tmp_path: Path) -> None:
    text = LATLON_WORLD.replace('frame_name: "latlon_origin"', 'frame_name: "local_origin"')
    with pytest.raises(ConfigError, match="not georeferenced"):
        load_mrs_world(write(tmp_path, text))


def test_an_unknown_frame_is_rejected(tmp_path: Path) -> None:
    text = LATLON_WORLD.replace('frame_name: "latlon_origin"', 'frame_name: "moon_origin"')
    with pytest.raises(ConfigError, match="unsupported safety area frame_name"):
        load_mrs_world(write(tmp_path, text))


def test_unknown_units_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unsupported world_origin units"):
        load_mrs_world(write(tmp_path, LATLON_WORLD.replace('"LATLON"', '"MGRS"')))


def test_an_odd_number_of_coordinates_is_rejected(tmp_path: Path) -> None:
    text = LATLON_WORLD.replace("50.0910495, 14.6346838,", "50.0910495,")
    with pytest.raises(ConfigError, match="not a whole number of x/y pairs"):
        load_mrs_world(write(tmp_path, text))


def test_too_few_vertices_are_rejected(tmp_path: Path) -> None:
    text = LATLON_WORLD.replace(
        """points: [
          50.0905258, 14.6327381,
          50.0896023, 14.6330607,
          50.0902173, 14.6348664,
          50.0910495, 14.6346838,
        ]""",
        "points: [50.09, 14.63, 50.10, 14.64]",
    )
    with pytest.raises(ConfigError, match="at least 3 vertices"):
        load_mrs_world(write(tmp_path, text))


def test_non_numeric_points_are_rejected(tmp_path: Path) -> None:
    text = LATLON_WORLD.replace("50.0905258,", '"north",')
    with pytest.raises(ConfigError, match="not a number"):
        load_mrs_world(write(tmp_path, text))


def test_lat_lon_points_out_of_range_are_rejected(tmp_path: Path) -> None:
    """A metric polygon left tagged as latlon_origin is caught here."""
    text = LATLON_WORLD.replace("50.0905258, 14.6327381,", "500.0, 14.6327381,")
    with pytest.raises(ConfigError, match="not a valid latitude/longitude"):
        load_mrs_world(write(tmp_path, text))


@pytest.mark.parametrize(
    "removed,message",
    [
        ("mrs_uav_managers", "mrs_uav_managers"),
        ("world_origin", "world_origin"),
        ("safety_area_manager", "safety_area_manager"),
        ("horizontal", "horizontal"),
    ],
)
def test_missing_sections_are_reported(removed: str, message: str) -> None:
    document = {
        "mrs_uav_managers": {
            "world_origin": {"units": "LATLON", "origin_x": 50.0, "origin_y": 14.0},
            "safety_area_manager": {
                "safety_area": {
                    "horizontal": {"frame_name": "latlon_origin", "points": [50.0, 14.0] * 3}
                }
            },
        }
    }
    if removed == "mrs_uav_managers":
        document = {}
    elif removed == "world_origin":
        del document["mrs_uav_managers"]["world_origin"]
    elif removed == "safety_area_manager":
        del document["mrs_uav_managers"]["safety_area_manager"]
    else:
        del document["mrs_uav_managers"]["safety_area_manager"]["safety_area"]["horizontal"]
    with pytest.raises(ConfigError, match=message):
        parse_mrs_world(document, Path("w.yaml"))


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_mrs_world(tmp_path / "absent.yaml")


def test_malformed_yaml_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_mrs_world(write(tmp_path, "mrs_uav_managers: [unclosed\n"))


# --- margins --------------------------------------------------------------


def test_the_margin_is_the_requested_distance_on_every_side(tmp_path: Path) -> None:
    world = load_mrs_world(write(tmp_path, LATLON_WORLD))
    raw_w, raw_s, raw_e, raw_n = world.safety_area.bounding_box()
    bounds = safety_area_bounds(world, 100.0)

    mid_lon = (raw_w + raw_e) / 2
    assert GEOD.inv(mid_lon, bounds.south, mid_lon, raw_s)[2] == pytest.approx(100.0, abs=0.5)
    assert GEOD.inv(mid_lon, raw_n, mid_lon, bounds.north)[2] == pytest.approx(100.0, abs=0.5)
    assert GEOD.inv(bounds.west, raw_n, raw_w, raw_n)[2] == pytest.approx(100.0, abs=0.5)
    assert GEOD.inv(raw_e, raw_n, bounds.east, raw_n)[2] == pytest.approx(100.0, abs=0.5)


def test_the_margined_box_contains_every_vertex(tmp_path: Path) -> None:
    world = load_mrs_world(write(tmp_path, LATLON_WORLD))
    bounds = safety_area_bounds(world, 50.0)
    for lon, lat in world.safety_area.points:
        assert bounds.west < lon < bounds.east
        assert bounds.south < lat < bounds.north


def test_a_zero_margin_is_the_bare_bounding_box(tmp_path: Path) -> None:
    world = load_mrs_world(write(tmp_path, LATLON_WORLD))
    bounds = safety_area_bounds(world, 0.0)
    assert (bounds.west, bounds.south, bounds.east, bounds.north) == world.safety_area.bounding_box()


def test_the_east_west_margin_is_sufficient_at_the_worst_latitude() -> None:
    """A tall northern box must still be wide enough at its poleward edge."""
    bounds = expand_bounds(14.0, 60.0, 14.1, 69.0, 100.0)
    for lat in (60.0, 64.5, 69.0):
        assert GEOD.inv(bounds.west, lat, 14.0, lat)[2] >= 100.0 - 0.5
        assert GEOD.inv(14.1, lat, bounds.east, lat)[2] >= 100.0 - 0.5


def test_a_negative_margin_is_rejected(tmp_path: Path) -> None:
    world = load_mrs_world(write(tmp_path, LATLON_WORLD))
    with pytest.raises(ConfigError, match="must not be negative"):
        safety_area_bounds(world, -10.0)


# --- the real files in this repository ------------------------------------


@pytest.mark.skipif(not REPO_WORLDS.is_dir(), reason="no worlds/ directory")
def test_every_repository_world_is_handled() -> None:
    """Each shipped world either converts cleanly or fails for a stated reason."""
    georeferenced = 0
    for path in sorted(REPO_WORLDS.glob("*.yaml")):
        try:
            world = load_mrs_world(path)
        except ConfigError as exc:
            # The only acceptable refusal is a world that is not on the map.
            assert "not georeferenced" in str(exc), f"{path}: {exc}"
            continue
        bounds = safety_area_bounds(world, 100.0)
        assert bounds.east > bounds.west and bounds.north > bounds.south
        for lon, lat in world.safety_area.points:
            assert bounds.west <= lon <= bounds.east
            assert bounds.south <= lat <= bounds.north
        georeferenced += 1
    assert georeferenced >= 10
