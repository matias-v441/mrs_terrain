"""End-to-end preparation (specification sections 18, 21 and 28).

Acquisition is served by the synthetic source from ``conftest``, so these tests
exercise the whole pipeline without touching the network.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.windows
import yaml
from pyproj import Transformer

from heightmap_prep import PrepareOptions, prepare_worlds
from heightmap_prep.config import WorldConfig, load_world
from heightmap_prep.crs import build_converter
from heightmap_prep.errors import (
    ConfigError,
    OutOfCoverageError,
    OutputConflictError,
    ValidationError,
)
from heightmap_prep.manifest import STATUS_COMPLETE, Manifest
from heightmap_prep.pipeline import (
    merge_tile_requirements,
    merge_wgs_requirements,
    plan_world,
    validate_only,
)
from heightmap_prep.raster import (
    TAG_RGB_FILLED,
    TAG_SOURCE_ID,
    TAG_VERTICAL_DATUM,
    TAG_WGS_FILLED,
    read_tags,
)
from heightmap_prep.sampling import DatasetSampler, read_test_points
from heightmap_prep.tiling import ProjectedBounds, TileGrid, TileIndex
from heightmap_prep.wgs import LatticeWindow, WgsGrid

from conftest import (
    REPO_ROOT,
    SECOND_WORLD_POINTS,
    TEST_AREA,
    SyntheticRgbSource,
    SyntheticSource,
    mrs_world,
)

TILE_PX = 128

#: The grid prepare_worlds builds for these tests (phase-aligned, 256 m tiles).
GRID = TileGrid.from_options(2.0, TILE_PX, -999_999.6, -800_000.12)

#: The WGS tiling prepare_worlds builds with the default options.
WGS = WgsGrid.from_metres(0.25)

_TO_WGS84 = Transformer.from_crs("EPSG:5514", "EPSG:4326", always_xy=True)


def projected_world(name: str, vertices: list[tuple[float, float]]) -> WorldConfig:
    """A world whose safety area has the given EPSG:5514 vertices."""
    return WorldConfig(name=name, points=tuple(_TO_WGS84.transform(x, y) for x, y in vertices))


def write_world(path: Path, world: WorldConfig) -> Path:
    points = ", ".join(f"{lat!r}, {lon!r}" for lon, lat in world.points)
    path.write_text(mrs_world(points), encoding="utf-8")
    return path


def reference_sampler(dataset: Path):
    """examples/sampler.py, the sampler the dataset is meant for."""
    sys.path.insert(0, str(REPO_ROOT / "examples"))
    try:
        from sampler import HeightSampler
    finally:
        sys.path.pop(0)
    return HeightSampler(dataset)


def polygon_samples(world: WorldConfig, per_edge: int = 25) -> list[tuple[float, float]]:
    """Points along every edge of the safety area, plus its centroid."""
    points = list(world.points)
    samples = []
    for (lon0, lat0), (lon1, lat1) in zip(points, points[1:] + points[:1]):
        for k in range(per_edge):
            t = k / per_edge
            samples.append((lon0 + (lon1 - lon0) * t, lat0 + (lat1 - lat0) * t))
    samples.append(
        (sum(lon for lon, _ in points) / len(points), sum(lat for _, lat in points) / len(points))
    )
    return samples


def options(proj_dir: Path, **overrides) -> PrepareOptions:
    defaults = dict(
        vertical_datum="egm96",
        resolution_m=2.0,
        tile_size_px=TILE_PX,
        block_size_px=16,
        proj_data_dir=proj_dir,
    )
    defaults.update(overrides)
    return PrepareOptions(**defaults)


def run(world_file: Path, out: Path, proj_dir: Path, source=None, **overrides):
    return prepare_worlds(
        [world_file], out, options(proj_dir, **overrides), source=source or SyntheticSource()
    )


# --- acceptance criteria (section 28) ------------------------------------


def test_a_world_becomes_a_validated_tiled_dataset(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    result = run(world_file, tmp_path / "out", proj_dir)

    assert result.report.ok, result.report.errors
    assert result.manifest.status == STATUS_COMPLETE
    assert result.tiles_written
    assert not result.tiles_reused

    manifest = Manifest.read(result.output_dir)
    assert manifest.horizontal_crs == "EPSG:5514"
    assert manifest.vertical_crs == "EPSG:5773"
    assert manifest.vertical_datum == "egm96"
    assert manifest.dtype == "float32"
    assert manifest.nodata == -9999.0
    assert manifest.worlds == ["testworld"]
    assert manifest.proj_grids  # a real PROJ operation was recorded
    assert manifest.network_enabled is False

    for tile in manifest.tiles:
        path = manifest.tile_path(result.output_dir, tile)
        with rasterio.open(path) as dataset:
            assert dataset.dtypes == ("float32",)
            assert dataset.width == dataset.height == TILE_PX
            assert dataset.transform == manifest.grid().tile_transform(tile)


def test_stored_heights_are_the_converted_source_heights(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    source = SyntheticSource()
    result = run(world_file, tmp_path / "out", proj_dir, source=source)
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    manifest = result.manifest
    grid = manifest.grid()

    checked = 0
    for tile in manifest.tiles:
        with rasterio.open(manifest.tile_path(result.output_dir, tile)) as dataset:
            data = dataset.read(1)
        transform = grid.tile_transform(tile)
        for row, col in ((5, 5), (TILE_PX // 2, TILE_PX // 3)):
            stored = float(data[row, col])
            if stored == manifest.nodata:
                continue
            x, y = transform @ (col + 0.5, row + 0.5)
            expected = float(
                converter.transform_heights(
                    np.array([x]),
                    np.array([y]),
                    np.array([float(SyntheticSource.height_at(np.array(x), np.array(y)))]),
                )[0]
            )
            assert stored == pytest.approx(expected, abs=1e-3)
            checked += 1
    assert checked > 0


def test_the_horizontal_grid_is_preserved(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    """A vertical-only conversion must not move a single pixel (section 3.2)."""
    source = SyntheticSource()
    result = run(world_file, tmp_path / "out", proj_dir, source=source)
    native_x, native_y = source.native_grid_origin()
    grid = result.manifest.grid()

    for tile in result.manifest.tiles:
        transform = grid.tile_transform(tile)
        assert (transform.c - native_x) % 2.0 == pytest.approx(0.0, abs=1e-9)
        assert (transform.f - native_y) % 2.0 == pytest.approx(0.0, abs=1e-9)
        assert transform.b == 0.0 and transform.d == 0.0

    # Every fetched window also sat on the native grid, so nothing was resampled.
    for bounds, width, height in source.requests:
        assert (bounds.west - native_x) % 2.0 == pytest.approx(0.0, abs=1e-9)
        assert (bounds.north - native_y) % 2.0 == pytest.approx(0.0, abs=1e-9)
        assert bounds.width / width == pytest.approx(2.0)
        assert bounds.height / height == pytest.approx(2.0)


def test_the_dataset_answers_a_wgs84_query(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    """The runtime flow of section 22, with no vertical transformation."""
    from pyproj import Transformer

    result = run(world_file, tmp_path / "out", proj_dir)
    manifest = result.manifest
    world = load_world(world_file)

    lon = sum(lon for lon, _ in world.points) / len(world.points)
    lat = sum(lat for _, lat in world.points) / len(world.points)
    x, y = Transformer.from_crs("EPSG:4326", manifest.horizontal_crs, always_xy=True).transform(
        lon, lat
    )
    tile = TileIndex(
        ix=math.floor((x - manifest.origin_x) / manifest.tile_span_x),
        iy=math.floor((manifest.origin_y - y) / manifest.tile_span_y),
    )
    assert tile in manifest.tiles

    with rasterio.open(manifest.tile_path(result.output_dir, tile)) as dataset:
        col, row = (~dataset.transform) @ (x, y)
        value = float(dataset.read(1)[int(row), int(col)])
    assert value != manifest.nodata
    assert 250.0 < value < 350.0


# --- tiling and planning --------------------------------------------------


def test_worlds_are_planned_onto_the_global_grid(world_file: Path) -> None:
    plan = plan_world(load_world(world_file), GRID, WGS, TEST_AREA)
    assert plan.tiles
    assert set(plan.tiles) <= set(GRID.tiles_for_pixels(plan.pixels))
    assert plan.wgs_tiles == WGS.tiles_for_window(plan.wgs_fill)


def test_a_world_outside_coverage_is_rejected() -> None:
    atlantic = WorldConfig("atlantic", ((-30.0, 40.0), (-29.0, 40.0), (-29.0, 41.0)))
    with pytest.raises(OutOfCoverageError, match="not inside the source coverage"):
        plan_world(atlantic, GRID, WGS, TEST_AREA)


def test_a_world_partly_outside_coverage_is_rejected() -> None:
    """Every point of the safety area must be samplable, so no clipping."""
    world = projected_world(
        "straddling",
        [(-743_600.0, -1_044_000.0), (-743_300.0, -1_044_000.0), (-743_300.0, -1_043_800.0)],
    )
    with pytest.raises(OutOfCoverageError, match="not inside the source coverage"):
        plan_world(world, GRID, WGS, TEST_AREA)


# A tile corner inside TEST_AREA, where the seams x = SEAM_X and y = SEAM_Y meet.
SEAM_X = GRID.tile_bounds(TileIndex(1003, 953)).east
SEAM_Y = GRID.tile_bounds(TileIndex(1003, 953)).south
NW, NE, SW, SE = (
    TileIndex(1003, 953),
    TileIndex(1004, 953),
    TileIndex(1003, 954),
    TileIndex(1004, 954),
)


def seam_triangle(name: str, west_x: float) -> WorldConfig:
    """A triangle north of SEAM_Y whose westmost vertex is at ``west_x``."""
    return projected_world(
        name,
        [(west_x, SEAM_Y + 50.0), (SEAM_X + 80.0, SEAM_Y + 20.0), (SEAM_X + 80.0, SEAM_Y + 90.0)],
    )


DIAGONAL = projected_world(
    "diagonal",
    # A triangle over the corner whose hypotenuse passes 14 m north-east of it.
    [(SEAM_X - 80.0, SEAM_Y + 100.0), (SEAM_X + 120.0, SEAM_Y + 100.0), (SEAM_X + 120.0, SEAM_Y - 100.0)],
)


def assert_fill_is_samplable(plan) -> None:
    """Every WGS height pixel centre of the fill reads only planned pixels and tiles."""
    lattice = WGS.height
    fill = plan.wgs_fill
    res = lattice.resolution_deg
    lons = lattice.origin_lon + (np.arange(fill.col_min, fill.col_end) + 0.5) * res
    lats = lattice.origin_lat - (np.arange(fill.row_min, fill.row_end) + 0.5) * res
    lon, lat = np.meshgrid(lons, lats)
    edge = np.zeros(lon.shape, dtype=bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    xs, ys = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True).transform(
        lon[edge], lat[edge]
    )
    cols = np.floor((xs - GRID.origin_x) / GRID.resolution_x - 0.5).astype(int)
    rows = np.floor((GRID.origin_y - ys) / GRID.resolution_y - 0.5).astype(int)
    pixels = plan.pixels
    assert cols.min() >= pixels.col_min and cols.max() + 1 <= pixels.col_max
    assert rows.min() >= pixels.row_min and rows.max() + 1 <= pixels.row_max
    for col, row in {(c + dc, r + dr) for c, r in zip(cols, rows) for dc in (0, 1) for dr in (0, 1)}:
        tile = TileIndex(col // GRID.tile_width_px, row // GRID.tile_height_px)
        assert tile in plan.tiles, (col, row, tile)


@pytest.mark.parametrize(
    "world",
    [seam_triangle("on", SEAM_X), seam_triangle("clear", SEAM_X + 30.0), DIAGONAL],
    ids=lambda world: world.name,
)
def test_every_pixel_centre_of_the_lat_lon_box_is_samplable(world: WorldConfig) -> None:
    assert_fill_is_samplable(plan_world(world, GRID, WGS, TEST_AREA))


def test_the_lat_lon_box_of_a_real_world_is_samplable(world_file: Path) -> None:
    assert_fill_is_samplable(plan_world(load_world(world_file), GRID, WGS, TEST_AREA))


def test_the_fill_is_the_lat_lon_box_on_the_height_lattice(world_file: Path) -> None:
    world = load_world(world_file)
    plan = plan_world(world, GRID, WGS, TEST_AREA)
    assert plan.wgs_fill == WGS.height.window_for_lonlat(world.bounds)


def test_a_safety_area_touching_a_tile_edge_pulls_in_the_adjacent_tile() -> None:
    assert set(plan_world(seam_triangle("on", SEAM_X), GRID, WGS, TEST_AREA).tiles) == {NW, NE}


def test_a_safety_area_clear_of_a_tile_edge_stays_in_its_tile() -> None:
    # Clear by more than its lat/lon box reaches beyond it in EPSG:5514.
    assert plan_world(seam_triangle("clear", SEAM_X + 30.0), GRID, WGS, TEST_AREA).tiles == [NE]


def test_a_diagonal_safety_area_pulls_in_every_tile_its_lat_lon_box_covers() -> None:
    assert set(plan_world(DIAGONAL, GRID, WGS, TEST_AREA).tiles) == {NW, NE, SW, SE}


def test_overlapping_worlds_share_one_fetch_per_tile(world_file: Path) -> None:
    plan = plan_world(load_world(world_file), GRID, WGS, TEST_AREA)
    required = merge_tile_requirements([plan, plan], GRID)
    assert set(required) == set(plan.tiles)
    for tile, pixels in required.items():
        # Merging the same world twice must not enlarge the area to fetch.
        assert pixels == plan.pixels.intersection(GRID.tile_pixels(tile))
    wgs = merge_wgs_requirements([plan, plan], WGS)
    assert set(wgs) == set(plan.wgs_tiles)
    for tile, fill in wgs.items():
        assert fill == plan.wgs_fill.intersection(WGS.height_tile_window(tile))


def test_worlds_in_one_wgs_tile_share_the_window_holding_both(
    world_file: Path, tmp_path: Path
) -> None:
    second_file = tmp_path / "world_second.yaml"
    second_file.write_text(mrs_world(SECOND_WORLD_POINTS), encoding="utf-8")
    first = plan_world(load_world(world_file), GRID, WGS, TEST_AREA)
    second = plan_world(load_world(second_file), GRID, WGS, TEST_AREA)
    merged = merge_wgs_requirements([first, second], WGS)
    assert set(merged) == set(first.wgs_tiles) | set(second.wgs_tiles)
    shared = set(first.wgs_tiles) & set(second.wgs_tiles)
    assert shared
    both = first.wgs_fill.union(second.wgs_fill)
    for tile in shared:
        assert merged[tile] == both.intersection(WGS.height_tile_window(tile))


def test_only_the_requested_part_of_a_tile_is_fetched(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    source = SyntheticSource()
    run(world_file, tmp_path / "out", proj_dir, source=source, tile_size_px=2048)
    fetched = sum(w * h for _, w, h in source.requests)
    # Far less than a full 2048 x 2048 tile: the world is a small sliver of it.
    assert 0 < fetched < 2048 * 2048


def test_nodata_fills_the_part_of_a_tile_no_world_asked_for(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    result = run(world_file, tmp_path / "out", proj_dir, tile_size_px=1024)
    manifest = result.manifest
    with rasterio.open(manifest.tile_path(result.output_dir, manifest.tiles[0])) as dataset:
        data = dataset.read(1)
    assert np.any(data == manifest.nodata)
    assert np.any(data != manifest.nodata)


def test_source_nodata_survives_into_the_output(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    # A hole in the middle of the safety area, away from its corners.
    hole = ProjectedBounds(-743_050.0, -1_044_050.0, -742_950.0, -1_043_950.0)
    source = SyntheticSource(nodata_region=hole)
    result = run(world_file, tmp_path / "out", proj_dir, source=source)
    with DatasetSampler(result.output_dir) as sampler:
        lon, lat = _TO_WGS84.transform(-743_000.0, -1_044_000.0)
        assert math.isnan(sampler.sample(lon, lat))


def test_a_run_whose_only_world_has_no_corner_data_fails(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    source = SyntheticSource(nodata_region=TEST_AREA.buffered(1_000.0))
    with pytest.raises(ValidationError, match="nothing to publish"):
        run(world_file, tmp_path / "out", proj_dir, source=source)


# --- resume and atomicity (section 18) -----------------------------------


def test_a_second_run_reuses_every_tile(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    first = run(world_file, out, proj_dir)
    source = SyntheticSource()
    second = run(world_file, out, proj_dir, source=source)

    assert second.tiles_reused == first.tiles_written
    assert not second.tiles_written
    assert source.requests == []  # nothing was fetched at all


def test_an_interrupted_run_resumes_without_redoing_valid_tiles(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    first = run(world_file, out, proj_dir)
    assert len(first.tiles_written) >= 2

    victim = first.tiles_written[0]
    first.manifest.tile_path(out, victim).unlink()

    source = SyntheticSource()
    second = run(world_file, out, proj_dir, source=source)
    assert second.tiles_written == [victim]
    assert set(second.tiles_reused) == set(first.tiles_written) - {victim}
    assert source.requests  # only the missing tile was fetched


def test_a_truncated_tile_is_not_mistaken_for_valid(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    first = run(world_file, out, proj_dir)
    victim = first.tiles_written[0]
    first.manifest.tile_path(out, victim).write_bytes(b"truncated")

    with pytest.raises(OutputConflictError, match="cannot be reused"):
        run(world_file, out, proj_dir)

    second = run(world_file, out, proj_dir, overwrite=True)
    assert victim in second.tiles_written


def test_changing_the_datum_without_overwrite_is_refused(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    run(world_file, out, proj_dir)
    with pytest.raises(OutputConflictError, match="VERTICAL_DATUM"):
        run(world_file, out, proj_dir, vertical_datum="wgs84-ellipsoid")


def test_a_failed_run_leaves_a_complete_dataset_complete(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    run(world_file, out, proj_dir)
    assert Manifest.read(out).status == STATUS_COMPLETE

    with pytest.raises(OutputConflictError):
        run(world_file, out, proj_dir, vertical_datum="wgs84-ellipsoid")

    assert Manifest.read(out).status == STATUS_COMPLETE
    assert validate_only(out).ok


def test_overwrite_regenerates_onto_a_new_datum(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    egm96 = run(world_file, out, proj_dir)
    tile = egm96.manifest.tiles[0]
    with rasterio.open(egm96.manifest.tile_path(out, tile)) as dataset:
        before = dataset.read(1)

    ellipsoid = run(world_file, out, proj_dir, vertical_datum="wgs84-ellipsoid", overwrite=True)
    assert ellipsoid.manifest.vertical_crs == "EPSG:4979"
    with rasterio.open(ellipsoid.manifest.tile_path(out, tile)) as dataset:
        after = dataset.read(1)

    valid = before != egm96.manifest.nodata
    assert np.all((after[valid] - before[valid]) > 40.0)  # the geoid undulation


def test_tiles_carry_the_provenance_used_for_resume(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    result = run(world_file, tmp_path / "out", proj_dir)
    tags = read_tags(result.manifest.tile_path(result.output_dir, result.manifest.tiles[0]))
    assert tags[TAG_SOURCE_ID] == "synthetic"
    assert tags[TAG_VERTICAL_DATUM] == "egm96"


def test_stale_temporaries_are_cleaned_up(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    run(world_file, out, proj_dir)
    leftover = out / "height" / "tile_1_1.tif.tmp"
    leftover.write_bytes(b"partial")
    run(world_file, out, proj_dir)
    assert not leftover.exists()


# --- parallelism (section 21) --------------------------------------------


def test_workers_produce_identical_output(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    serial = run(world_file, tmp_path / "serial", proj_dir, workers=1)
    parallel = run(world_file, tmp_path / "parallel", proj_dir, workers=4)

    assert [o.tile for o in serial.outcomes] == [o.tile for o in parallel.outcomes]
    for tile in serial.manifest.tiles:
        with rasterio.open(serial.manifest.tile_path(serial.output_dir, tile)) as a:
            with rasterio.open(parallel.manifest.tile_path(parallel.output_dir, tile)) as b:
                np.testing.assert_array_equal(a.read(1), b.read(1))


# --- multiple worlds ------------------------------------------------------


def test_two_worlds_share_one_dataset(tmp_path: Path, world_file: Path, proj_dir: Path) -> None:
    second = tmp_path / "world_second.yaml"
    second.write_text(mrs_world(SECOND_WORLD_POINTS), encoding="utf-8")
    result = prepare_worlds(
        [world_file, second],
        tmp_path / "out",
        options(proj_dir),
        source=SyntheticSource(),
    )
    assert result.report.ok, result.report.errors
    assert result.manifest.worlds == ["testworld", "second"]
    assert len(result.plans) == 2
    assert len(result.test_points) == 8
    # The dataset is exactly the union of what each world needs.
    assert set(result.manifest.tiles) == {t for plan in result.plans for t in plan.tiles}
    # Tiles are shared, never duplicated.
    assert len(result.manifest.tiles) == len(set(result.manifest.tiles))


def test_validate_only_reads_an_existing_dataset(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    run(world_file, out, proj_dir)
    assert validate_only(out).ok


def test_cog_output_is_a_valid_resumable_dataset(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    result = run(world_file, out, proj_dir, cog=True, tile_size_px=512)
    assert result.report.ok, result.report.errors
    assert result.manifest.storage_format == "cog"

    with rasterio.open(result.manifest.tile_path(out, result.manifest.tiles[0])) as dataset:
        assert dataset.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        assert dataset.overviews(1)

    # Provenance survives the COG driver, so the dataset still resumes.
    second = run(world_file, out, proj_dir, cog=True, tile_size_px=512)
    assert second.tiles_reused == result.tiles_written


def test_a_source_on_the_wrong_vertical_datum_is_refused(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    from heightmap_prep.errors import UnexpectedVerticalDatumError

    source = SyntheticSource()
    source.vertical_crs = "EPSG:5705"  # Baltic 1977, not Bpv
    with pytest.raises(UnexpectedVerticalDatumError, match="EPSG:8357"):
        run(world_file, tmp_path / "out", proj_dir, source=source)


# --- coverage of the safety areas ----------------------------------------


def assert_samplable_everywhere(dataset: Path, world: WorldConfig) -> None:
    sampler = reference_sampler(dataset)
    try:
        for lon, lat in polygon_samples(world):
            assert math.isfinite(sampler.sample(lon, lat)), (lon, lat)
    finally:
        sampler.close()


def test_every_point_of_the_safety_area_is_samplable(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    result = run(world_file, tmp_path / "out", proj_dir)
    assert_samplable_everywhere(result.output_dir, load_world(world_file))


def test_a_safety_area_on_a_tile_edge_is_samplable_along_it(
    tmp_path: Path, proj_dir: Path
) -> None:
    path = write_world(tmp_path / "world_on_seam.yaml", seam_triangle("on_seam", SEAM_X))
    result = run(path, tmp_path / "out", proj_dir)
    assert set(result.manifest.tiles) == {NW, NE}
    assert_samplable_everywhere(result.output_dir, load_world(path))


def test_the_dataset_holds_only_the_needed_tiles(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    result = run(world_file, tmp_path / "out", proj_dir)
    plan = plan_world(load_world(world_file), GRID, WGS, TEST_AREA)
    assert result.manifest.tiles == sorted(plan.tiles)
    assert sorted(p.name for p in (result.output_dir / "height").iterdir()) == sorted(
        f"tile_{t.ix}_{t.iy}.tif" for t in plan.tiles
    )


def test_worlds_without_a_latlon_safety_area_are_skipped(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    metric = tmp_path / "world_metric.yaml"
    metric.write_text(
        mrs_world("10.0, 10.0, -10.0, 10.0, 0.0, -10.0", "world_origin"), encoding="utf-8"
    )
    result = prepare_worlds(
        [world_file, metric], tmp_path / "out", options(proj_dir), source=SyntheticSource()
    )
    assert result.manifest.worlds == ["testworld"]

    with pytest.raises(ConfigError, match="nothing to prepare"):
        prepare_worlds([metric], tmp_path / "none", options(proj_dir), source=SyntheticSource())


def test_the_manifest_carries_no_bounds(world_file: Path, tmp_path: Path, proj_dir: Path) -> None:
    result = run(world_file, tmp_path / "out", proj_dir)
    document = yaml.safe_load((result.output_dir / "dataset.yaml").read_text(encoding="utf-8"))
    assert "world_bounds_wgs84" not in document
    assert "dataset_bounds" not in document["grid"]


# --- test points ----------------------------------------------------------


def test_test_points_are_the_safety_area_corners(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    result = run(world_file, tmp_path / "out", proj_dir)
    manifest = Manifest.read(result.output_dir)
    assert manifest.test_points_path == "test_points.csv"
    assert manifest.test_points_count == 4

    lines = (result.output_dir / "test_points.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("50.0800925,14.4166195,")
    points = read_test_points(result.output_dir / "test_points.csv")
    assert [(p.lon, p.lat) for p in points] == list(load_world(world_file).points)
    assert all(250.0 < p.height < 350.0 for p in points)


def test_test_points_match_the_reference_sampler_exactly(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    result = run(world_file, tmp_path / "out", proj_dir)
    sampler = reference_sampler(result.output_dir)
    try:
        for point in read_test_points(result.output_dir / "test_points.csv"):
            assert sampler.sample(point.lon, point.lat) == point.height
    finally:
        sampler.close()


def test_shared_corners_are_listed_once(tmp_path: Path, world_file: Path, proj_dir: Path) -> None:
    twin = tmp_path / "world_twin.yaml"
    twin.write_text(world_file.read_text(encoding="utf-8"), encoding="utf-8")
    result = prepare_worlds(
        [world_file, twin], tmp_path / "out", options(proj_dir), source=SyntheticSource()
    )
    assert len(result.test_points) == 4


def test_a_tampered_test_point_fails_validation(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    run(world_file, out, proj_dir)
    path = out / "test_points.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lat, lon, height = lines[0].split(",")
    lines[0] = f"{lat},{lon},{float(height) + 0.01!r}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = validate_only(out)
    assert any("test point" in e for e in report.errors)


# --- skipping world configs -----------------------------------------------

#: A small triangle in the south-east of TEST_AREA, on a tile testworld does not use.
LONELY = projected_world(
    "lonely",
    [(-742_650.0, -1_044_350.0), (-742_550.0, -1_044_450.0), (-742_650.0, -1_044_450.0)],
)
LONELY_TILE = TileIndex(1005, 954)


def test_the_lonely_world_has_a_tile_of_its_own(world_file: Path) -> None:
    assert plan_world(LONELY, GRID, WGS, TEST_AREA).tiles == [LONELY_TILE]
    assert LONELY_TILE not in plan_world(load_world(world_file), GRID, WGS, TEST_AREA).tiles


def test_a_world_outside_coverage_is_skipped(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    atlantic = tmp_path / "world_atlantic.yaml"
    atlantic.write_text(mrs_world("40.0, -30.0, 40.0, -29.0, 41.0, -29.0"), encoding="utf-8")
    result = prepare_worlds(
        [world_file, atlantic], tmp_path / "out", options(proj_dir), source=SyntheticSource()
    )
    assert result.report.ok, result.report.errors
    assert [w.name for w in result.worlds] == ["testworld"]
    assert result.manifest.worlds == ["testworld"]
    assert "not inside the source coverage" in result.skipped[atlantic]


def test_a_world_without_corner_data_is_skipped_with_its_tiles(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    lonely = write_world(tmp_path / "world_lonely.yaml", LONELY)
    hole = ProjectedBounds(-742_700.0, -1_044_480.0, -742_500.0, -1_044_320.0)
    out = tmp_path / "out"
    result = prepare_worlds(
        [world_file, lonely], out, options(proj_dir), source=SyntheticSource(nodata_region=hole)
    )
    assert result.report.ok, result.report.errors
    assert "no height around safety area corner" in result.skipped[lonely]
    assert result.manifest.worlds == ["testworld"]
    assert LONELY_TILE not in result.manifest.tiles
    assert not result.manifest.tile_path(out, LONELY_TILE).exists()
    assert all(o.tile != LONELY_TILE for o in result.outcomes)
    assert len(result.test_points) == 4

    manifest = Manifest.read(out)
    assert manifest.worlds == ["testworld"] and manifest.test_points_count == 4
    report = validate_only(out)
    assert report.ok and not report.unexpected_files


def test_a_bad_world_config_is_skipped(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    broken = tmp_path / "world_broken.yaml"
    broken.write_text(mrs_world("50.08, 14.41, 50.07"), encoding="utf-8")
    missing = tmp_path / "world_absent.yaml"
    result = prepare_worlds(
        [world_file, broken, missing],
        tmp_path / "out",
        options(proj_dir),
        source=SyntheticSource(),
    )
    assert result.manifest.worlds == ["testworld"]
    assert set(result.skipped) == {broken, missing}


# --- RGB imagery and the worlds database -------------------------------------


def run_rgb(world_files, out: Path, proj_dir: Path, rgb_source=None, **overrides):
    overrides.setdefault("rgb_resolution_m", 1.0)
    return prepare_worlds(
        list(world_files),
        out,
        options(proj_dir, include_rgb=True, **overrides),
        source=SyntheticSource(),
        rgb_source=rgb_source or SyntheticRgbSource(),
    )


def wgs_tile_bounds(manifest: Manifest, tile: TileIndex) -> tuple[float, ...]:
    return manifest.wgs.grid().tile_bounds(tile)


def test_every_wgs_tile_gets_validated_height_and_rgb_tiles(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    result = run_rgb([world_file], out, proj_dir)
    assert result.report.ok, result.report.errors
    wgs = result.manifest.wgs
    assert wgs.tiles
    assert {o.tile for o in result.wgs_outcomes} == set(wgs.tiles)
    assert {o.tile for o in result.rgb_outcomes} == set(wgs.tiles)
    assert result.report.tiles_checked == len(result.manifest.tiles) + 2 * len(wgs.tiles)

    manifest = Manifest.read(out)
    grid = manifest.wgs.grid()
    assert manifest.wgs.crs == "EPSG:4326" and manifest.wgs.resolution_m == 1.0
    for tile in manifest.wgs.tiles:
        with rasterio.open(manifest.wgs_height_path(out, tile)) as heights:
            assert heights.crs.to_epsg() == 4326
            assert (heights.count, heights.dtypes[0]) == (1, "float32")
            assert grid.height.window_of_transform(
                heights.transform, heights.width, heights.height
            ) == grid.height_tile_window(tile)
        with rasterio.open(manifest.wgs_rgb_path(out, tile)) as rgb:
            assert (rgb.count, rgb.dtypes[0]) == (3, "uint8")
            assert grid.rgb.window_of_transform(
                rgb.transform, rgb.width, rgb.height
            ) == grid.rgb_tile_window(tile)
            assert tuple(rgb.bounds) == pytest.approx(tuple(heights.bounds), abs=1e-12)
    assert validate_only(out).ok


def test_wgs_tiles_do_not_overlap(world_file: Path, tmp_path: Path, proj_dir: Path) -> None:
    second = tmp_path / "world_second.yaml"
    second.write_text(mrs_world(SECOND_WORLD_POINTS), encoding="utf-8")
    # Tiles small enough that the two worlds spread over several.
    result = run_rgb([world_file, second], tmp_path / "out", proj_dir, wgs_tile_size_px=256)
    tiles = result.manifest.wgs.tiles
    assert len(tiles) > 4
    boxes = [wgs_tile_bounds(result.manifest, tile) for tile in tiles]
    for i, (w0, s0, e0, n0) in enumerate(boxes):
        for w1, s1, e1, n1 in boxes[i + 1 :]:
            assert min(e0, e1) <= max(w0, w1) + 1e-12 or min(n0, n1) <= max(s0, s1) + 1e-12


def test_wgs_heights_are_the_sampler_at_their_pixel_centres(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    # Tiles small enough that the world spreads over several.
    result = run(world_file, out, proj_dir, wgs_tile_size_px=2048)
    manifest = result.manifest
    grid = manifest.wgs.grid()
    assert len(manifest.wgs.tiles) > 1
    fill = grid.fill_for_lonlat(load_world(world_file).bounds)
    seen = 0
    with DatasetSampler(out, manifest) as sampler:
        for tile in manifest.wgs.tiles:
            with rasterio.open(manifest.wgs_height_path(out, tile)) as dataset:
                stored = dataset.read(1).astype(np.float64)
                window = grid.height_tile_window(tile)
            inside = fill.intersection(window)
            col, row = inside.offset_in(window)
            block = stored[row : row + inside.height, col : col + inside.width]
            res = grid.height.resolution_deg
            lons = grid.height.origin_lon + (np.arange(inside.col_min, inside.col_end) + 0.5) * res
            lats = grid.height.origin_lat - (np.arange(inside.row_min, inside.row_end) + 0.5) * res
            expected = sampler.sample_many(lons[np.newaxis, :], lats[:, np.newaxis])
            # The whole lat/lon box has heights, and they are the sampler's.
            assert np.isfinite(expected).all()
            np.testing.assert_array_equal(block, expected.astype(np.float32))
            # Nothing outside the box.
            outside = np.ones(stored.shape, dtype=bool)
            outside[row : row + inside.height, col : col + inside.width] = False
            assert (stored[outside] == manifest.nodata).all()
            seen += inside.width * inside.height
    assert seen == fill.width * fill.height


def test_worlds_sharing_a_wgs_tile_are_both_fully_covered(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    second = tmp_path / "world_second.yaml"
    second.write_text(mrs_world(SECOND_WORLD_POINTS), encoding="utf-8")
    result = prepare_worlds(
        [world_file, second], tmp_path / "out", options(proj_dir), source=SyntheticSource()
    )
    assert result.report.ok, result.report.errors
    assert not any("WGS heights" in warning for warning in result.report.warnings)


def test_a_hole_inside_a_lat_lon_box_is_warned_about(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    # Inside the safety area, away from its corners.
    hole = ProjectedBounds(-743_100.0, -1_043_950.0, -742_950.0, -1_043_850.0)
    result = run(world_file, tmp_path / "out", proj_dir, source=SyntheticSource(nodata_region=hole))
    assert result.report.ok, result.report.errors
    assert any(
        "world testworld" in warning and "has WGS heights" in warning
        for warning in result.report.warnings
    )


def test_rgb_imagery_covers_the_whole_safety_area(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    run_rgb([world_file], out, proj_dir, wgs_tile_size_px=256)
    manifest = Manifest.read(out)
    grid = manifest.wgs.grid()
    world = load_world(world_file)
    for lon, lat in polygon_samples(world):
        tile = grid.tile_index(lon, lat)
        with rasterio.open(manifest.wgs_rgb_path(out, tile)) as dataset:
            row, col = dataset.index(lon, lat)
            window = rasterio.windows.Window(col, row, 1, 1)
            assert dataset.read_masks(1, window=window)[0, 0] == 255
            pixel = dataset.read(window=window)[:, 0, 0].astype(int)
        expected = SyntheticRgbSource.colour_at(*dataset.xy(row, col)).astype(int)
        assert np.abs(pixel - expected).max() <= 6  # JPEG


def test_a_second_rgb_run_reuses_every_rgb_tile(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    first = run_rgb([world_file], out, proj_dir)
    rgb = SyntheticRgbSource()
    second = run_rgb([world_file], out, proj_dir, rgb_source=rgb)
    assert {o.action for o in second.rgb_outcomes} == {"reused"}
    assert len(second.rgb_outcomes) == len(first.rgb_outcomes)
    assert rgb.requests == []


def test_a_new_world_in_an_rgb_tile_needs_overwrite(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    run_rgb([world_file], out, proj_dir)
    second = tmp_path / "world_second.yaml"
    second.write_text(mrs_world(SECOND_WORLD_POINTS), encoding="utf-8")
    # A new world is new to the height tiles too, so they need replacing first.
    with pytest.raises(OutputConflictError, match="heights for less"):
        run_rgb([world_file, second], out, proj_dir)
    prepare_worlds(
        [world_file, second],
        out,
        options(proj_dir, overwrite=True, rgb_resolution_m=1.0),
        source=SyntheticSource(),
    )
    with pytest.raises(OutputConflictError, match="imagery for less than the worlds now need"):
        run_rgb([world_file, second], out, proj_dir)
    result = run_rgb([world_file, second], out, proj_dir, overwrite=True)
    assert result.report.ok, result.report.errors
    assert {o.action for o in result.rgb_outcomes} == {"written"}


def test_a_different_jpeg_quality_without_overwrite_is_refused(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    run_rgb([world_file], out, proj_dir)
    with pytest.raises(OutputConflictError, match="RGB_JPEG_QUALITY"):
        run_rgb([world_file], out, proj_dir, rgb_jpeg_quality=50)


def test_a_different_rgb_resolution_cuts_other_tiles(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    first = run_rgb([world_file], out, proj_dir).manifest.wgs.tiles
    # The coarser WGS lattice snaps the lat/lon box differently, so the
    # EPSG:5514 tiles need other pixels too.
    with pytest.raises(OutputConflictError, match="heights for less"):
        run_rgb([world_file], out, proj_dir, rgb_resolution_m=2.0)
    second = run_rgb([world_file], out, proj_dir, rgb_resolution_m=2.0, overwrite=True)
    assert not set(first) & set(second.manifest.wgs.tiles)
    # The first run's tiles are left behind, and reported.
    assert validate_only(out).unexpected_files


def test_a_new_world_in_a_height_tile_needs_overwrite(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    run(world_file, out, proj_dir)
    second = tmp_path / "world_second.yaml"
    second.write_text(mrs_world(SECOND_WORLD_POINTS), encoding="utf-8")
    worlds = [world_file, second]
    with pytest.raises(OutputConflictError, match="heights for less than the worlds now need"):
        prepare_worlds(worlds, out, options(proj_dir), source=SyntheticSource())
    result = prepare_worlds(worlds, out, options(proj_dir, overwrite=True), source=SyntheticSource())
    assert result.report.ok, result.report.errors
    assert sorted(result.manifest.worlds) == ["second", "testworld"]


def test_a_missing_rgb_tile_fails_validation(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    result = run_rgb([world_file], out, proj_dir)
    result.manifest.wgs_rgb_path(out, result.manifest.wgs.tiles[0]).unlink()
    report = validate_only(out)
    assert not report.ok and any("rgb tile" in e for e in report.errors)


def test_wgs_tiles_hold_only_the_worlds_kept(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    lonely = write_world(tmp_path / "world_lonely.yaml", LONELY)
    hole = ProjectedBounds(-742_700.0, -1_044_480.0, -742_500.0, -1_044_320.0)
    out = tmp_path / "out"
    result = prepare_worlds(
        [world_file, lonely],
        out,
        options(proj_dir, include_rgb=True, rgb_resolution_m=1.0),
        source=SyntheticSource(nodata_region=hole),
        rgb_source=SyntheticRgbSource(),
    )
    assert result.report.ok, result.report.errors
    manifest = result.manifest
    grid = manifest.wgs.grid()
    plan = plan_world(load_world(world_file), GRID, grid, TEST_AREA)
    assert manifest.wgs.tiles == sorted(plan.wgs_tiles)
    for tile in manifest.wgs.tiles:
        fill = plan.wgs_fill.intersection(grid.height_tile_window(tile))
        heights = read_tags(manifest.wgs_height_path(out, tile))[TAG_WGS_FILLED]
        rgb = read_tags(manifest.wgs_rgb_path(out, tile))[TAG_RGB_FILLED]
        assert LatticeWindow.from_tag(heights) == fill
        assert LatticeWindow.from_tag(rgb) == fill.scaled(grid.height_factor)
    assert not validate_only(out).unexpected_files


def test_a_world_outside_the_imagery_coverage_is_skipped(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    lonely = write_world(tmp_path / "world_lonely.yaml", LONELY)
    # Imagery only for the north-west of the test area: not the lonely world.
    coverage = ProjectedBounds(TEST_AREA.west, -1_044_300.0, -742_700.0, TEST_AREA.north)
    result = run_rgb(
        [world_file, lonely], tmp_path / "out", proj_dir,
        rgb_source=SyntheticRgbSource(coverage=coverage),
    )
    assert result.manifest.worlds == ["testworld"]
    assert "synthetic-rgb coverage" in result.skipped[lonely]


def test_imagery_holes_are_masked_and_warned_about(
    world_file: Path, tmp_path: Path, proj_dir: Path, caplog
) -> None:
    hole = (14.418, 50.077, 14.420, 50.079)
    with caplog.at_level("WARNING"):
        result = run_rgb(
            [world_file], tmp_path / "out", proj_dir, rgb_source=SyntheticRgbSource(hole=hole)
        )
    assert result.report.ok
    assert any("has imagery" in r.message for r in caplog.records)


def test_the_worlds_database_is_written_without_rgb(
    world_file: Path, tmp_path: Path, proj_dir: Path
) -> None:
    out = tmp_path / "out"
    result = run(world_file, out, proj_dir)
    assert result.manifest.wgs is not None and result.manifest.wgs.rgb is None
    assert result.manifest.worlds_db_path == "worlds.sqlite"
    assert (out / "worlds.sqlite").is_file()
    assert (out / "wgs" / "height").is_dir()
    assert not (out / "wgs" / "rgb").exists()
