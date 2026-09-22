"""End-to-end preparation (specification sections 18, 21 and 28).

Acquisition is served by the synthetic source from ``conftest``, so these tests
exercise the whole pipeline without touching the network.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio

from heightmap_prep import PrepareOptions, prepare_worlds
from heightmap_prep.config import load_world
from heightmap_prep.crs import build_converter
from heightmap_prep.errors import OutOfCoverageError, OutputConflictError
from heightmap_prep.manifest import STATUS_COMPLETE, Manifest
from heightmap_prep.pipeline import merge_tile_requirements, plan_world, validate_only
from heightmap_prep.raster import TAG_SOURCE_ID, TAG_VERTICAL_DATUM, read_tags
from heightmap_prep.tiling import TileGrid, TileIndex

from conftest import TEST_AREA, SyntheticSource

TILE_PX = 128


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

    lon = (world.bounds.west + world.bounds.east) / 2
    lat = (world.bounds.south + world.bounds.north) / 2
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
    grid = TileGrid.from_options(2.0, TILE_PX, -999_999.6, -800_000.12)
    world = load_world(world_file)
    plan = plan_world(world, grid, TEST_AREA)
    assert plan.tiles
    assert set(plan.tiles) == set(grid.tiles_for_bounds(plan.projected_bounds))


def test_a_world_outside_coverage_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "atlantic.yaml"
    path.write_text(
        "name: atlantic\nbounds:\n  west: -30.0\n  south: 40.0\n  east: -29.0\n  north: 41.0\n",
        encoding="utf-8",
    )
    grid = TileGrid.from_options(2.0, TILE_PX, -999_999.6, -800_000.12)
    with pytest.raises(OutOfCoverageError, match="does not intersect"):
        plan_world(load_world(path), grid, TEST_AREA)


def test_overlapping_worlds_share_one_fetch_per_tile(world_file: Path) -> None:
    grid = TileGrid.from_options(2.0, TILE_PX, -999_999.6, -800_000.12)
    world = load_world(world_file)
    plan = plan_world(world, grid, TEST_AREA)
    required = merge_tile_requirements([plan, plan], grid)
    assert set(required) == set(plan.tiles)
    for tile, bounds in required.items():
        # Merging the same world twice must not enlarge the area to fetch.
        assert bounds.as_tuple() == grid.tile_bounds(tile).intersection(
            plan.projected_bounds
        ).as_tuple()


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
    # Buffered so that snapping the fetch window outward to whole pixels cannot
    # reach past the hole and pick up real values at the edges.
    source = SyntheticSource(nodata_region=TEST_AREA.buffered(1_000.0))
    result = run(world_file, tmp_path / "out", proj_dir, source=source)
    for tile in result.manifest.tiles:
        with rasterio.open(result.manifest.tile_path(result.output_dir, tile)) as dataset:
            assert np.all(dataset.read(1) == result.manifest.nodata)


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
    second = tmp_path / "second.yaml"
    second.write_text(
        "name: second\nbounds:\n"
        "  west: 14.4240\n  south: 50.0800\n  east: 14.4300\n  north: 50.0845\n",
        encoding="utf-8",
    )
    result = prepare_worlds(
        [world_file, second],
        tmp_path / "out",
        options(proj_dir),
        source=SyntheticSource(),
    )
    assert result.report.ok, result.report.errors
    assert result.manifest.worlds == ["testworld", "second"]
    assert set(result.manifest.world_bounds_wgs84) == {"testworld", "second"}
    assert len(result.plans) == 2
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
