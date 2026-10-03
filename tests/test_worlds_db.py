"""World metadata: parsing it from world files and the worlds database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from heightmap_prep.config import WorldConfig, WorldOrigin, load_world
from heightmap_prep.manifest import Manifest
from heightmap_prep.tiling import TileIndex
from heightmap_prep.validate import ValidationReport, validate_worlds_db
from heightmap_prep.worlds_db import (
    SCHEMA_VERSION,
    WORLDS_DB_FILENAME,
    origin_latlon,
    polygon_wkt,
    write_worlds_db,
)

from conftest import TEST_WORLD_POINTS, mrs_world

SQUARE = ((14.0, 50.0), (14.1, 50.0), (14.1, 50.1), (14.0, 50.1))


def world_text(origin: str | None, vertical: bool = True) -> str:
    text = mrs_world(TEST_WORLD_POINTS)
    head, rest = text.split("  safety_area_manager:\n", 1)
    head = "mrs_uav_managers:\n" + (origin or "")
    if not vertical:
        rest = rest.split("      vertical:\n")[0]
    return head + "  safety_area_manager:\n" + rest


# --- parsing -------------------------------------------------------------


def test_a_latlon_origin_and_the_vertical_limits_are_read(world_file: Path) -> None:
    world = load_world(world_file)
    assert world.origin == WorldOrigin("LATLON", 50.0785, 14.4205)
    assert (world.vertical_frame, world.min_z, world.max_z) == ("world_origin", 1.0, 30.0)


def test_a_utm_origin_is_read(tmp_path: Path) -> None:
    path = tmp_path / "world_utm.yaml"
    path.write_text(
        world_text("  world_origin:\n    units: UTM\n    origin_x: 458422.2\n    origin_y: 5551241.4\n"),
        encoding="utf-8",
    )
    assert load_world(path).origin == WorldOrigin("UTM", 458422.2, 5551241.4)


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "  world_origin:\n    units: FEET\n    origin_x: 1.0\n    origin_y: 2.0\n",
        "  world_origin:\n    units: LATLON\n    origin_x: north\n    origin_y: 2.0\n",
        "  world_origin:\n    units: LATLON\n    origin_x: 95.0\n    origin_y: 2.0\n",
    ],
)
def test_an_unusable_origin_does_not_make_the_world_unusable(tmp_path: Path, origin) -> None:
    path = tmp_path / "world_odd.yaml"
    path.write_text(world_text(origin, vertical=False), encoding="utf-8")
    world = load_world(path)
    assert world.origin is None
    assert world.min_z is None and world.max_z is None and world.vertical_frame is None


# --- origins and geometry ------------------------------------------------


def test_a_latlon_origin_is_already_in_degrees() -> None:
    world = WorldConfig("w", SQUARE, origin=WorldOrigin("LATLON", 50.05, 14.05))
    assert origin_latlon(world) == (50.05, 14.05)


def test_a_utm_origin_is_placed_in_the_safety_areas_zone() -> None:
    # world_cisar's origin, in UTM 33N, with a safety area near it.
    world = WorldConfig(
        "cisar",
        ((14.42, 50.10), (14.43, 50.10), (14.43, 50.11)),
        origin=WorldOrigin("UTM", 458422.2, 5551241.4),
    )
    lat, lon = origin_latlon(world)
    assert lat == pytest.approx(50.106, abs=0.01)
    assert lon == pytest.approx(14.418, abs=0.01)


def test_no_origin_has_no_position() -> None:
    assert origin_latlon(WorldConfig("w", SQUARE)) is None


def test_the_wkt_polygon_is_closed_and_lon_lat() -> None:
    wkt = polygon_wkt(WorldConfig("w", SQUARE))
    assert wkt.startswith("POLYGON((14.0 50.0, ") and wkt.endswith(", 14.0 50.0))")


# --- the database --------------------------------------------------------


def manifest_with(tiles, worlds) -> Manifest:
    manifest = Manifest(horizontal_crs="EPSG:5514", vertical_crs="EPSG:5773",
                        vertical_datum="egm96", nodata=-9999.0)
    manifest.set_tiles(tiles)
    manifest.worlds = list(worlds)
    return manifest


def build(tmp_path: Path) -> tuple[Path, Manifest]:
    worlds = [
        WorldConfig("b", SQUARE, Path("worlds/world_b.yaml"), WorldOrigin("LATLON", 50.05, 14.05),
                    "world_origin", 1.0, 30.0),
        WorldConfig("a", SQUARE[:3]),
    ]
    tiles = [TileIndex(1, 2), TileIndex(2, 2)]
    manifest = manifest_with(tiles, ["a", "b"])
    path = write_worlds_db(
        tmp_path, manifest, worlds, {"a": [TileIndex(1, 2)], "b": tiles},
    )
    return path, manifest


def test_the_database_records_every_world(tmp_path: Path) -> None:
    path, manifest = build(tmp_path)
    assert path == tmp_path / WORLDS_DB_FILENAME
    assert manifest.worlds_db_path == WORLDS_DB_FILENAME
    assert manifest.worlds_db_schema_version == SCHEMA_VERSION

    db = sqlite3.connect(path)
    assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    rows = db.execute(
        "SELECT name, source_file, origin_units, origin_lat, origin_lon, min_z, max_z "
        "FROM worlds ORDER BY name"
    ).fetchall()
    assert rows == [
        ("a", None, None, None, None, None, None),
        ("b", "world_b.yaml", "LATLON", 50.05, 14.05, 1.0, 30.0),
    ]
    vertices = db.execute(
        "SELECT lat, lon FROM safety_area_vertices WHERE world = 'b' ORDER BY seq"
    ).fetchall()
    assert vertices == [(lat, lon) for lon, lat in SQUARE]
    assert db.execute("SELECT ix, iy FROM world_tiles WHERE world = 'a'").fetchall() == [(1, 2)]
    assert db.execute(
        "SELECT height_path, rgb_path FROM tiles ORDER BY ix"
    ).fetchall() == [("height/tile_1_2.tif", None), ("height/tile_2_2.tif", None)]
    db.close()
    assert validate_worlds_db(tmp_path, manifest).ok


def test_a_rebuild_is_byte_identical(tmp_path: Path) -> None:
    """The database is committed with the dataset, so rebuilds must not churn it."""
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    assert build(tmp_path / "one")[0].read_bytes() == build(tmp_path / "two")[0].read_bytes()


def test_a_database_that_disagrees_with_the_manifest_fails_validation(tmp_path: Path) -> None:
    _, manifest = build(tmp_path)
    manifest.worlds = ["a", "b", "c"]
    manifest.set_tiles([TileIndex(1, 2)])
    report = validate_worlds_db(tmp_path, manifest, ValidationReport())
    assert any("holds worlds" in e for e in report.errors)
    assert any("does not declare" in e for e in report.errors)


def test_a_missing_database_fails_validation(tmp_path: Path) -> None:
    path, manifest = build(tmp_path)
    path.unlink()
    assert not validate_worlds_db(tmp_path, manifest).ok
