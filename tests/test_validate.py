"""Dataset validation (specification section 17)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from affine import Affine
from pyproj import Transformer

from heightmap_prep.manifest import MANIFEST_FILENAME, Manifest
from heightmap_prep.raster import write_raster
from heightmap_prep.sampling import ReferencePoint, write_test_points
from heightmap_prep.tiling import TileIndex
from heightmap_prep.validate import (
    PLAUSIBLE_MAX_M,
    TileFacts,
    ValidationReport,
    validate_dataset,
    validate_manifest,
    validate_spatial_consistency,
    validate_test_points,
    validate_tile,
)

TILE_PX = 64


def make_manifest(**overrides) -> Manifest:
    defaults = dict(
        horizontal_crs="EPSG:5514",
        vertical_crs="EPSG:5773",
        vertical_datum="egm96",
        nodata=-9999.0,
        tile_width_px=TILE_PX,
        tile_height_px=TILE_PX,
        origin_x=-1_000_000.0,
        origin_y=-800_000.0,
        block_width_px=16,
        block_height_px=16,
    )
    defaults.update(overrides)
    return Manifest(**defaults)


def write_tile(
    root: Path,
    manifest: Manifest,
    tile: TileIndex,
    *,
    value: float = 300.0,
    transform: Affine | None = None,
    crs: str = "EPSG:5514",
    nodata: float = -9999.0,
    shape: tuple[int, int] | None = None,
) -> Path:
    grid = manifest.grid()
    path = manifest.tile_path(root, tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = shape or (manifest.tile_height_px, manifest.tile_width_px)
    return write_raster(
        path,
        np.full((height, width), value, dtype=np.float32),
        transform=transform or grid.tile_transform(tile),
        crs=crs,
        nodata=nodata,
        block_size_px=16,
    )


def build_dataset(root: Path, tiles=((10, 10), (11, 10)), **overrides) -> Manifest:
    manifest = make_manifest(**overrides)
    indices = [TileIndex(*t) for t in tiles]
    manifest.set_tiles(indices)
    for tile in indices:
        write_tile(root, manifest, tile)
    manifest.mark_complete()
    manifest.write(root)
    return manifest


# --- 17.1 manifest --------------------------------------------------------


def test_a_good_manifest_passes() -> None:
    assert validate_manifest(make_manifest()).ok


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"format_version": 99}, "format_version"),
        ({"horizontal_crs": ""}, "horizontal_crs is missing"),
        ({"horizontal_crs": "not a crs"}, "not a resolvable CRS"),
        ({"vertical_crs": ""}, "vertical_crs is missing"),
        ({"unit": ""}, "unit is missing"),
        ({"unit": "ft"}, "must be 'm'"),
        ({"dtype": "float64"}, "dtype"),
        ({"resolution_x": 0.0}, "resolution must be positive"),
        ({"tile_width_px": 0}, "tile dimensions"),
        ({"axis_order": "north_east"}, "axis_order"),
        ({"tile_pattern": "height/tile.tif"}, "placeholders"),
        ({"tile_pattern": "/abs/tile_{ix}_{iy}.tif"}, "relative path"),
        ({"tile_pattern": "../tile_{ix}_{iy}.tif"}, "relative path"),
        ({"block_width_px": 0}, "block dimensions"),
    ],
)
def test_bad_manifests_are_rejected(overrides: dict, message: str) -> None:
    report = validate_manifest(make_manifest(**overrides))
    assert not report.ok
    assert any(message in error for error in report.errors), report.errors


# --- 17.2 per-tile --------------------------------------------------------


def test_a_good_tile_passes(tmp_path: Path) -> None:
    manifest = make_manifest()
    path = write_tile(tmp_path, manifest, TileIndex(10, 10))
    report, facts = validate_tile(path, manifest, TileIndex(10, 10))
    assert report.ok, report.errors
    assert facts is not None
    assert facts.width == facts.height == TILE_PX
    assert facts.valid_min == facts.valid_max == 300.0


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    report, facts = validate_tile(tmp_path / "absent.tif", make_manifest(), TileIndex(0, 0))
    assert not report.ok and facts is None
    assert "cannot open" in report.errors[0]


def test_a_misaligned_tile_is_rejected(tmp_path: Path) -> None:
    manifest = make_manifest()
    tile = TileIndex(10, 10)
    shifted = manifest.grid().tile_transform(tile) @ Affine.translation(0.5, 0.0)
    path = write_tile(tmp_path, manifest, tile, transform=shifted)
    report, _ = validate_tile(path, manifest, tile)
    assert any("west edge" in e for e in report.errors), report.errors


def test_a_wrong_crs_is_rejected(tmp_path: Path) -> None:
    manifest = make_manifest()
    tile = TileIndex(10, 10)
    path = write_tile(tmp_path, manifest, tile, crs="EPSG:3857")
    report, _ = validate_tile(path, manifest, tile)
    assert any("CRS is EPSG:3857" in e for e in report.errors), report.errors


def test_a_wrong_nodata_is_rejected(tmp_path: Path) -> None:
    manifest = make_manifest()
    tile = TileIndex(10, 10)
    path = write_tile(tmp_path, manifest, tile, nodata=-32768.0)
    report, _ = validate_tile(path, manifest, tile)
    assert any("NoData is -32768" in e for e in report.errors), report.errors


def test_a_wrong_resolution_is_rejected(tmp_path: Path) -> None:
    manifest = make_manifest()
    tile = TileIndex(10, 10)
    bounds = manifest.grid().tile_bounds(tile)
    coarse = Affine(4.0, 0.0, bounds.west, 0.0, -4.0, bounds.north)
    path = write_tile(tmp_path, manifest, tile, transform=coarse)
    report, _ = validate_tile(path, manifest, tile)
    assert any("x resolution" in e for e in report.errors), report.errors


def test_an_oversized_tile_is_rejected(tmp_path: Path) -> None:
    manifest = make_manifest()
    tile = TileIndex(10, 10)
    path = write_tile(tmp_path, manifest, tile, shape=(TILE_PX + 8, TILE_PX))
    report, _ = validate_tile(path, manifest, tile)
    assert any("exceeds the declared tile size" in e for e in report.errors), report.errors


def test_a_clipped_edge_tile_only_warns(tmp_path: Path) -> None:
    manifest = make_manifest()
    tile = TileIndex(10, 10)
    path = write_tile(tmp_path, manifest, tile, shape=(TILE_PX // 2, TILE_PX))
    report, _ = validate_tile(path, manifest, tile)
    assert report.ok, report.errors
    assert any("clipped edge tile" in w for w in report.warnings)


def test_non_finite_values_are_rejected(tmp_path: Path) -> None:
    manifest = make_manifest()
    tile = TileIndex(10, 10)
    path = manifest.tile_path(tmp_path, tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((TILE_PX, TILE_PX), 300.0, dtype=np.float32)
    data[0, 0] = np.inf
    data[1, 1] = np.nan
    write_raster(
        path, data, transform=manifest.grid().tile_transform(tile), crs="EPSG:5514", block_size_px=16
    )
    report, _ = validate_tile(path, manifest, tile)
    assert any("not finite" in e for e in report.errors), report.errors


# --- 17.5 plausibility ----------------------------------------------------


def test_implausible_elevations_only_warn(tmp_path: Path) -> None:
    manifest = make_manifest()
    tile = TileIndex(10, 10)
    path = write_tile(tmp_path, manifest, tile, value=PLAUSIBLE_MAX_M + 5_000.0)
    report, _ = validate_tile(path, manifest, tile, plausibility=True)
    assert report.ok, report.errors
    assert any("plausible range" in w for w in report.warnings)


def test_all_zero_tiles_warn(tmp_path: Path) -> None:
    manifest = make_manifest()
    tile = TileIndex(10, 10)
    path = write_tile(tmp_path, manifest, tile, value=0.0)
    report, _ = validate_tile(path, manifest, tile, plausibility=True)
    assert any("exactly 0.0" in w for w in report.warnings)


def test_an_all_nodata_tile_warns(tmp_path: Path) -> None:
    manifest = make_manifest()
    tile = TileIndex(10, 10)
    path = write_tile(tmp_path, manifest, tile, value=-9999.0)
    report, facts = validate_tile(path, manifest, tile, plausibility=True)
    assert report.ok, report.errors
    assert facts is not None and facts.nodata_fraction == 1.0
    assert any("no valid pixel" in w for w in report.warnings)


def test_a_mostly_nodata_tile_does_not_warn(tmp_path: Path) -> None:
    """Tiles only hold the pixels around safety areas, so this is normal."""
    manifest = make_manifest()
    tile = TileIndex(10, 10)
    data = np.full((TILE_PX, TILE_PX), -9999.0, dtype=np.float32)
    data[3:5, 3:5] = 300.0
    path = manifest.tile_path(tmp_path, tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_raster(
        path, data, transform=manifest.grid().tile_transform(tile), crs="EPSG:5514",
        nodata=-9999.0, block_size_px=16,
    )
    report, _ = validate_tile(path, manifest, tile, plausibility=True)
    assert report.ok and not report.warnings


def test_plausibility_can_be_switched_off(tmp_path: Path) -> None:
    manifest = make_manifest()
    tile = TileIndex(10, 10)
    path = write_tile(tmp_path, manifest, tile, value=99_999.0)
    report, _ = validate_tile(path, manifest, tile, plausibility=False)
    assert report.warnings == []


# --- 17.3 spatial consistency --------------------------------------------


def facts_for(manifest: Manifest, tile: TileIndex, **overrides) -> TileFacts:
    bounds = manifest.grid().tile_bounds(tile)
    values = dict(
        tile=tile,
        path=Path(f"tile_{tile.ix}_{tile.iy}.tif"),
        width=manifest.tile_width_px,
        height=manifest.tile_height_px,
        resolution_x=manifest.resolution_x,
        resolution_y=manifest.resolution_y,
        west=bounds.west,
        north=bounds.north,
    )
    values.update(overrides)
    return TileFacts(**values)


def test_neighbouring_tiles_that_abut_exactly_pass() -> None:
    manifest = make_manifest()
    facts = [
        facts_for(manifest, TileIndex(10, 10)),
        facts_for(manifest, TileIndex(11, 10)),
        facts_for(manifest, TileIndex(10, 11)),
    ]
    assert validate_spatial_consistency(facts, manifest).ok


def test_a_horizontal_gap_is_detected() -> None:
    manifest = make_manifest()
    right = facts_for(manifest, TileIndex(11, 10))
    facts = [facts_for(manifest, TileIndex(10, 10)), TileFacts(**{**right.__dict__, "west": right.west + 2.0})]
    report = validate_spatial_consistency(facts, manifest)
    assert any("gap of 2.0" in e for e in report.errors), report.errors


def test_a_vertical_overlap_is_detected() -> None:
    manifest = make_manifest()
    below = facts_for(manifest, TileIndex(10, 11))
    facts = [facts_for(manifest, TileIndex(10, 10)), TileFacts(**{**below.__dict__, "north": below.north + 4.0})]
    report = validate_spatial_consistency(facts, manifest)
    assert any("overlap of 4.0" in e for e in report.errors), report.errors


def test_mixed_resolutions_are_detected() -> None:
    manifest = make_manifest()
    facts = [
        facts_for(manifest, TileIndex(10, 10)),
        facts_for(manifest, TileIndex(20, 20), resolution_x=5.0, resolution_y=5.0),
    ]
    report = validate_spatial_consistency(facts, manifest)
    assert any("differing resolutions" in e for e in report.errors), report.errors


# --- whole dataset --------------------------------------------------------


def test_a_complete_dataset_validates(tmp_path: Path) -> None:
    build_dataset(tmp_path)
    report = validate_dataset(tmp_path, require_complete=True)
    assert report.ok, report.errors
    assert report.tiles_checked == 2


def test_a_declared_but_absent_tile_is_an_error(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path)
    manifest.tile_path(tmp_path, TileIndex(11, 10)).unlink()
    report = validate_dataset(tmp_path)
    assert not report.ok
    assert report.tiles_missing == [TileIndex(11, 10)]


def test_an_undeclared_tile_only_warns(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path)
    write_tile(tmp_path, manifest, TileIndex(12, 10))
    report = validate_dataset(tmp_path)
    assert report.ok, report.errors
    assert report.unexpected_files == ["height/tile_12_10.tif"]


def test_a_leftover_temporary_warns(tmp_path: Path) -> None:
    build_dataset(tmp_path)
    (tmp_path / "height" / "tile_12_10.tif.tmp").write_bytes(b"partial")
    report = validate_dataset(tmp_path)
    assert report.ok, report.errors
    assert any("leftover" in w for w in report.warnings)


def test_a_building_dataset_fails_a_completeness_check(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path)
    manifest.status = "building"
    manifest.write(tmp_path)
    assert validate_dataset(tmp_path, require_complete=True).ok is False
    assert validate_dataset(tmp_path, require_complete=False).ok is True


def test_a_missing_manifest_is_an_error(tmp_path: Path) -> None:
    report = validate_dataset(tmp_path)
    assert not report.ok
    assert "manifest not found" in report.errors[0]


def test_manifest_errors_short_circuit_tile_checks(tmp_path: Path) -> None:
    build_dataset(tmp_path)
    document = yaml.safe_load((tmp_path / MANIFEST_FILENAME).read_text())
    document["heightmap"]["unit"] = "ft"
    (tmp_path / MANIFEST_FILENAME).write_text(yaml.safe_dump(document))
    report = validate_dataset(tmp_path)
    assert not report.ok
    assert report.tiles_checked == 0


def test_report_helpers() -> None:
    report = ValidationReport()
    assert report.ok
    report.warn("careful")
    assert report.ok
    report.error("broken")
    assert not report.ok
    with pytest.raises(Exception, match="broken"):
        report.raise_for_errors()


# --- test points ----------------------------------------------------------


def tile_centre_lonlat(manifest: Manifest, tile: TileIndex) -> tuple[float, float]:
    bounds = manifest.grid().tile_bounds(tile)
    to_wgs84 = Transformer.from_crs("EPSG:5514", "EPSG:4326", always_xy=True)
    return to_wgs84.transform((bounds.west + bounds.east) / 2, (bounds.south + bounds.north) / 2)


def with_test_points(root: Path, manifest: Manifest, heights: list[float]) -> None:
    """Record one test point per height, all at the centre of tile (10, 10)."""
    lon, lat = tile_centre_lonlat(manifest, TileIndex(10, 10))
    write_test_points(
        root / "test_points.csv", [ReferencePoint(lat, lon, h) for h in heights]
    )
    manifest.test_points_path = "test_points.csv"
    manifest.test_points_count = len(heights)
    manifest.write(root)


def test_matching_test_points_pass(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path)
    with_test_points(tmp_path, manifest, [300.0, 300.0])
    report = validate_dataset(tmp_path, require_complete=True)
    assert report.ok, report.errors
    assert not any("test points" in w for w in report.warnings)


def test_a_wrong_test_point_height_is_an_error(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path)
    with_test_points(tmp_path, manifest, [300.001])
    report = validate_test_points(tmp_path, manifest)
    assert any("the tiles give 300.0" in e for e in report.errors)


def test_a_test_point_off_the_tiles_is_an_error(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path)
    write_test_points(tmp_path / "test_points.csv", [ReferencePoint(50.0, 14.4, 300.0)])
    manifest.test_points_path = "test_points.csv"
    manifest.test_points_count = 1
    report = validate_test_points(tmp_path, manifest)
    assert any("the tiles give nan" in e for e in report.errors)


def test_a_missing_test_points_file_is_an_error(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path)
    manifest.test_points_path = "test_points.csv"
    assert any("is missing" in e for e in validate_test_points(tmp_path, manifest).errors)


def test_a_test_point_count_mismatch_is_an_error(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path)
    with_test_points(tmp_path, manifest, [300.0])
    manifest.test_points_count = 2
    assert any("manifest declares 2" in e for e in validate_test_points(tmp_path, manifest).errors)


def test_a_malformed_test_point_is_an_error(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path)
    (tmp_path / "test_points.csv").write_text("50.0,14.4\n", encoding="utf-8")
    manifest.test_points_path = "test_points.csv"
    manifest.test_points_count = 1
    report = validate_test_points(tmp_path, manifest)
    assert any("test_points.csv:1: malformed" in e for e in report.errors)


def test_a_test_points_path_outside_the_dataset_is_an_error(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path)
    manifest.test_points_path = "../elsewhere.csv"
    assert any("stays inside" in e for e in validate_test_points(tmp_path, manifest).errors)


def test_a_dataset_without_test_points_warns(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path)
    report = validate_test_points(tmp_path, manifest)
    assert report.ok and any("no test points" in w for w in report.warnings)
