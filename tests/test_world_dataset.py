"""Reading worlds and their imagery back from a prepared dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from heightmap_prep import PrepareOptions, prepare_worlds
from heightmap_prep.config import load_world
from heightmap_prep.sampling import DatasetSampler
from heightmap_prep.world_dataset import WorldDataset, WorldNotFoundError, main

from conftest import (
    SECOND_WORLD_POINTS,
    TEST_WORLD_POINTS,
    SyntheticRgbSource,
    SyntheticSource,
    mrs_world,
)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, proj_dir: Path) -> Path:
    """A dataset of two overlapping worlds, with RGB at 1 m on WGS tiles of 256 px."""
    root = tmp_path_factory.mktemp("prepared")
    first = root / "world_testworld.yaml"
    first.write_text(mrs_world(TEST_WORLD_POINTS), encoding="utf-8")
    second = root / "world_second.yaml"
    second.write_text(mrs_world(SECOND_WORLD_POINTS), encoding="utf-8")
    out = root / "dataset"
    prepare_worlds(
        [first, second],
        out,
        PrepareOptions(
            resolution_m=2.0,
            tile_size_px=128,
            block_size_px=16,
            proj_data_dir=proj_dir,
            include_rgb=True,
            rgb_resolution_m=1.0,
            wgs_tile_size_px=256,
        ),
        source=SyntheticSource(),
        rgb_source=SyntheticRgbSource(),
    )
    return out


@pytest.fixture
def dataset(prepared: Path):
    with WorldDataset(prepared) as opened:
        yield opened


def test_the_worlds_are_listed(dataset: WorldDataset) -> None:
    assert dataset.world_names() == ["second", "testworld"]
    assert [w.name for w in dataset.worlds()] == ["second", "testworld"]


def test_a_world_comes_back_as_its_world_file_gave_it(
    dataset: WorldDataset, prepared: Path
) -> None:
    source = load_world(prepared.parent / "world_testworld.yaml")
    world = dataset.world("testworld")
    assert world.safety_area == tuple((lat, lon) for lon, lat in source.points)
    assert dataset.safety_area("testworld") == world.safety_area
    assert world.origin is not None
    assert (world.origin.units, world.origin.lat, world.origin.lon) == ("LATLON", 50.0785, 14.4205)
    assert (world.min_z, world.max_z, world.vertical_frame) == (1.0, 30.0, "world_origin")
    assert world.source_file == "world_testworld.yaml"
    assert set(world.height_tiles) <= set(dataset.manifest.tiles) and len(world.height_tiles) >= 2
    assert set(world.wgs_tiles) <= set(dataset.manifest.wgs.tiles) and len(world.wgs_tiles) >= 2


def test_an_unknown_world_is_reported(dataset: WorldDataset) -> None:
    with pytest.raises(WorldNotFoundError, match="testworld"):
        dataset.world("atlantis")


def test_a_worlds_wgs_tiles_are_its_tiles(dataset: WorldDataset) -> None:
    world = dataset.world("testworld")
    tiles = dataset.wgs_tiles("testworld")
    assert [t.tile for t in tiles] == list(world.wgs_tiles)
    for tile in tiles:
        for path in (tile.height_path, tile.rgb_path):
            assert path.is_file()
            with rasterio.open(path) as opened:
                assert tuple(opened.bounds) == pytest.approx(tile.bounds, abs=1e-12)


def test_the_images_cover_the_lat_lon_box_on_the_height_lattice(dataset: WorldDataset) -> None:
    world = dataset.world("testworld")
    heights = dataset.read_height("testworld")
    image = dataset.read_rgb("testworld")
    lattice = dataset.manifest.wgs.grid().height
    assert heights.bounds == pytest.approx(
        lattice.window_bounds(lattice.window_for_lonlat(world.bounds)), abs=1e-12
    )
    assert image.bounds == pytest.approx(heights.bounds, abs=1e-12)
    assert image.mask.shape == tuple(8 * n for n in heights.data.shape)
    west, south, east, north = heights.bounds
    w_west, w_south, w_east, w_north = world.bounds
    assert west <= w_west and south <= w_south and east >= w_east and north >= w_north


def test_the_heights_are_the_samplers_at_pixel_centres(dataset: WorldDataset) -> None:
    heights = dataset.read_height("testworld")
    rows, cols = heights.data.shape
    t = heights.transform
    lons = t.c + (np.arange(cols) + 0.5) * t.a
    lats = t.f + (np.arange(rows) + 0.5) * t.e
    with DatasetSampler(dataset.root, dataset.manifest) as sampler:
        expected = sampler.sample_many(lons[np.newaxis, :], lats[:, np.newaxis])
    assert np.isfinite(heights.data).all()
    np.testing.assert_array_equal(heights.data, expected.astype(np.float32))


def test_the_imagery_mosaic_covers_the_safety_area(dataset: WorldDataset) -> None:
    world = dataset.world("testworld")
    image = dataset.read_rgb("testworld")
    w_west, w_south, w_east, w_north = world.bounds

    # Every vertex, and the centre, has the synthetic colour of its pixel.
    inverse = ~image.transform
    samples = list(world.safety_area) + [((w_south + w_north) / 2, (w_west + w_east) / 2)]
    for lat, lon in samples:
        col, row = (int(v) for v in inverse @ (lon, lat))
        assert image.mask[row, col]
        centre_lon, centre_lat = image.transform @ (col + 0.5, row + 0.5)
        expected = SyntheticRgbSource.colour_at(np.array(centre_lon), np.array(centre_lat))
        assert np.abs(image.data[:, row, col].astype(int) - expected.astype(int)).max() <= 6
    assert image.mask.all()


def test_the_images_can_be_saved(dataset: WorldDataset, tmp_path: Path) -> None:
    image = dataset.read_rgb("testworld")
    path = image.write(tmp_path / "world.tif")
    with rasterio.open(path) as saved:
        assert saved.crs.to_epsg() == 4326
        np.testing.assert_array_equal(saved.read(), image.data)
        np.testing.assert_array_equal(saved.read_masks(1) > 0, image.mask)
    heights = dataset.read_height("testworld")
    with rasterio.open(heights.write(tmp_path / "heights.tif")) as saved:
        assert saved.crs.to_epsg() == 4326
        np.testing.assert_array_equal(saved.read(1), heights.data)


def test_a_dataset_without_rgb_still_has_its_worlds(
    tmp_path: Path, proj_dir: Path, world_file: Path
) -> None:
    out = tmp_path / "out"
    prepare_worlds(
        [world_file],
        out,
        PrepareOptions(tile_size_px=128, block_size_px=16, proj_data_dir=proj_dir),
        source=SyntheticSource(),
    )
    with WorldDataset(out) as plain:
        assert plain.world_names() == ["testworld"]
        assert not plain.has_rgb
        assert all(t.rgb_path is None for t in plain.wgs_tiles("testworld"))
        assert np.isfinite(plain.read_height("testworld").data).all()
        with pytest.raises(ValueError, match="without RGB"):
            plain.read_rgb("testworld")


def test_the_command_line_lists_describes_and_exports(
    prepared: Path, tmp_path: Path, capsys
) -> None:
    assert main([str(prepared)]) == 0
    assert capsys.readouterr().out.split() == ["second", "testworld"]

    export = tmp_path / "testworld.tif"
    heights = tmp_path / "testworld_heights.tif"
    assert main(
        [str(prepared), "testworld", "--export", str(export), "--export-height", str(heights)]
    ) == 0
    out = capsys.readouterr().out
    assert "origin: LATLON" in out and "wgs/rgb/tile_" in out and "wgs/height/tile_" in out
    assert export.is_file() and heights.is_file()

    assert main([str(prepared), "atlantis"]) == 1
