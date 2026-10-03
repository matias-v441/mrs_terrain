"""Reading worlds and their imagery back from a prepared dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from heightmap_prep import PrepareOptions, prepare_worlds
from heightmap_prep.config import load_world
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
    """A dataset of two overlapping worlds, with RGB at 1 m."""
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
    assert set(world.tiles) <= set(dataset.manifest.tiles) and len(world.tiles) >= 2


def test_an_unknown_world_is_reported(dataset: WorldDataset) -> None:
    with pytest.raises(WorldNotFoundError, match="testworld"):
        dataset.world("atlantis")


def test_a_worlds_rgb_tiles_are_its_tiles(dataset: WorldDataset) -> None:
    world = dataset.world("testworld")
    tiles = dataset.rgb_tiles("testworld")
    assert [t.tile for t in tiles] == list(world.tiles)
    for tile in tiles:
        assert tile.path.is_file()
        with rasterio.open(tile.path) as opened:
            assert tuple(opened.bounds) == pytest.approx(tile.bounds)


def test_the_imagery_mosaic_covers_the_safety_area(dataset: WorldDataset) -> None:
    world = dataset.world("testworld")
    image = dataset.read_rgb("testworld")
    west, south, east, north = image.bounds
    w_west, w_south, w_east, w_north = world.bounds
    res = dataset.manifest.rgb.resolution_deg
    assert west <= w_west < west + res and east - res < w_east <= east
    assert south <= w_south < south + res and north - res < w_north <= north

    # Every vertex, and the centre, has the synthetic colour of its pixel.
    inverse = ~image.transform
    samples = list(world.safety_area) + [((w_south + w_north) / 2, (w_west + w_east) / 2)]
    for lat, lon in samples:
        col, row = (int(v) for v in inverse @ (lon, lat))
        col, row = min(col, image.mask.shape[1] - 1), min(row, image.mask.shape[0] - 1)
        assert image.mask[row, col]
        centre_lon, centre_lat = image.transform @ (col + 0.5, row + 0.5)
        expected = SyntheticRgbSource.colour_at(np.array(centre_lon), np.array(centre_lat))
        assert np.abs(image.data[:, row, col].astype(int) - expected.astype(int)).max() <= 6


def test_a_margin_grows_the_mosaic(dataset: WorldDataset) -> None:
    plain = dataset.read_rgb("testworld")
    wide = dataset.read_rgb("testworld", margin_m=50.0)
    assert wide.mask.shape[0] > plain.mask.shape[0] + 90
    assert wide.mask.shape[1] > plain.mask.shape[1] + 90
    # Imagery stops a few metres beyond the safety area.
    assert not wide.mask[0, 0]


def test_the_mosaic_can_be_saved(dataset: WorldDataset, tmp_path: Path) -> None:
    image = dataset.read_rgb("testworld")
    path = image.write(tmp_path / "world.tif")
    with rasterio.open(path) as saved:
        assert saved.crs.to_epsg() == 4326
        np.testing.assert_array_equal(saved.read(), image.data)
        np.testing.assert_array_equal(saved.read_masks(1) > 0, image.mask)


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
        with pytest.raises(ValueError, match="without RGB"):
            plain.rgb_tiles("testworld")


def test_the_command_line_lists_describes_and_exports(
    prepared: Path, tmp_path: Path, capsys
) -> None:
    assert main([str(prepared)]) == 0
    assert capsys.readouterr().out.split() == ["second", "testworld"]

    export = tmp_path / "testworld.tif"
    assert main([str(prepared), "testworld", "--export", str(export), "--margin", "5"]) == 0
    out = capsys.readouterr().out
    assert "origin: LATLON" in out and "rgb/tile_" in out
    assert export.is_file()

    assert main([str(prepared), "atlantis"]) == 1
