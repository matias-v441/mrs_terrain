"""The deterministic global tile grid (specification sections 9.4 and 12)."""

from __future__ import annotations

import math

import pytest

from heightmap_prep.errors import ConfigError
from heightmap_prep.tiling import (
    ProjectedBounds,
    TileGrid,
    TileIndex,
    iter_blocks,
    parse_tile_filename,
    tile_filename,
)

GRID = TileGrid.from_options(
    resolution_m=2.0, tile_size_px=4096, origin_x=-1_000_000.0, origin_y=-800_000.0
)


def test_tile_span_matches_the_specification() -> None:
    # 4096 px at 2 m is 8.192 km on a side.
    assert GRID.tile_span_x == 8192.0
    assert GRID.tile_span_y == 8192.0


def test_index_convention_matches_the_documented_formula() -> None:
    x, y = -743_011.72, -1_043_823.18
    expected = TileIndex(
        ix=math.floor((x - GRID.origin_x) / GRID.tile_span_x),
        iy=math.floor((GRID.origin_y - y) / GRID.tile_span_y),
    )
    assert GRID.index_for_point(x, y) == expected


def test_ix_increases_eastward_and_iy_southward() -> None:
    base = GRID.index_for_point(-743_000.0, -1_043_000.0)
    east = GRID.index_for_point(-743_000.0 + GRID.tile_span_x, -1_043_000.0)
    south = GRID.index_for_point(-743_000.0, -1_043_000.0 - GRID.tile_span_y)
    assert east.ix == base.ix + 1 and east.iy == base.iy
    assert south.iy == base.iy + 1 and south.ix == base.ix


def test_origin_is_the_west_north_corner_of_tile_zero() -> None:
    bounds = GRID.tile_bounds(TileIndex(0, 0))
    assert bounds.west == GRID.origin_x
    assert bounds.north == GRID.origin_y
    assert bounds.east == GRID.origin_x + GRID.tile_span_x
    assert bounds.south == GRID.origin_y - GRID.tile_span_y


def test_tile_bounds_and_index_round_trip() -> None:
    for tile in (TileIndex(0, 0), TileIndex(31, 29), TileIndex(-3, 7)):
        bounds = GRID.tile_bounds(tile)
        centre_x = (bounds.west + bounds.east) / 2
        centre_y = (bounds.south + bounds.north) / 2
        assert GRID.index_for_point(centre_x, centre_y) == tile


def test_tile_transform_is_north_up_and_anchored_at_the_corner() -> None:
    tile = TileIndex(31, 29)
    transform = GRID.tile_transform(tile)
    bounds = GRID.tile_bounds(tile)
    assert transform.a == 2.0 and transform.e == -2.0
    assert transform.b == 0.0 and transform.d == 0.0
    assert transform.c == bounds.west
    assert transform.f == bounds.north
    # The far corner lands exactly on the next tile's origin: no gap, no overlap.
    assert transform @ (GRID.tile_width_px, GRID.tile_height_px) == (
        bounds.east,
        bounds.south,
    )


def test_adjacent_tiles_abut_exactly() -> None:
    left = GRID.tile_bounds(TileIndex(10, 5))
    right = GRID.tile_bounds(TileIndex(11, 5))
    below = GRID.tile_bounds(TileIndex(10, 6))
    assert left.east == right.west
    assert left.south == below.north


def test_tiles_for_bounds_covers_the_area() -> None:
    bounds = ProjectedBounds(-743_200.0, -1_044_000.0, -742_800.0, -1_043_600.0)
    tiles = GRID.tiles_for_bounds(bounds)
    assert tiles == [TileIndex(31, 29)]


def test_tiles_for_bounds_spans_several_tiles_in_reading_order() -> None:
    origin = GRID.tile_bounds(TileIndex(4, 4))
    bounds = ProjectedBounds(
        origin.west + 1.0,
        origin.south - GRID.tile_span_y + 1.0,
        origin.east + 1.0,
        origin.north - 1.0,
    )
    tiles = GRID.tiles_for_bounds(bounds)
    assert tiles == [
        TileIndex(4, 4),
        TileIndex(5, 4),
        TileIndex(4, 5),
        TileIndex(5, 5),
    ]


def test_bounds_ending_exactly_on_a_seam_do_not_pull_in_the_next_tile() -> None:
    tile = GRID.tile_bounds(TileIndex(7, 7))
    assert GRID.tiles_for_bounds(tile) == [TileIndex(7, 7)]


def test_pixel_window_snaps_outward_and_clips_to_the_tile() -> None:
    tile = TileIndex(31, 29)
    bounds = GRID.tile_bounds(tile)
    # A one-metre sliver inside the first pixel still selects that whole pixel.
    window = GRID.pixel_window_for_bounds(
        tile, ProjectedBounds(bounds.west + 0.5, bounds.north - 1.5, bounds.west + 1.5, bounds.north - 0.5)
    )
    assert window == (0, 0, 1, 1)

    # An area larger than the tile is clipped to the tile.
    window = GRID.pixel_window_for_bounds(tile, bounds.buffered(10_000.0))
    assert window == (0, 0, GRID.tile_width_px, GRID.tile_height_px)


def test_pixel_window_is_none_when_disjoint() -> None:
    far = ProjectedBounds(0.0, 0.0, 100.0, 100.0)
    assert GRID.pixel_window_for_bounds(TileIndex(31, 29), far) is None


def test_window_bounds_inverts_pixel_window() -> None:
    tile = TileIndex(31, 29)
    window = (100, 200, 300, 400)
    bounds = GRID.window_bounds(tile, window)
    assert GRID.pixel_window_for_bounds(tile, bounds) == window


def test_filenames_are_deterministic_and_parseable() -> None:
    assert tile_filename(TileIndex(31, 29)) == "height/tile_31_29.tif"
    assert parse_tile_filename("height/tile_31_29.tif") == TileIndex(31, 29)
    assert parse_tile_filename("tile_-3_7.tif") == TileIndex(-3, 7)
    assert parse_tile_filename("dataset.yaml") is None
    assert parse_tile_filename("tile_a_b.tif") is None


def test_projected_bounds_operations() -> None:
    a = ProjectedBounds(0.0, 0.0, 10.0, 10.0)
    b = ProjectedBounds(5.0, 5.0, 15.0, 15.0)
    assert a.intersection(b) == ProjectedBounds(5.0, 5.0, 10.0, 10.0)
    assert a.union(b) == ProjectedBounds(0.0, 0.0, 15.0, 15.0)
    assert a.intersection(ProjectedBounds(20.0, 20.0, 30.0, 30.0)) is None
    # Touching rectangles share no area.
    assert a.intersection(ProjectedBounds(10.0, 0.0, 20.0, 10.0)) is None


def test_degenerate_bounds_are_rejected() -> None:
    with pytest.raises(ConfigError, match="degenerate"):
        ProjectedBounds(10.0, 0.0, 10.0, 10.0)


def test_invalid_grid_parameters_are_rejected() -> None:
    with pytest.raises(ConfigError, match="tile dimensions"):
        TileGrid(0.0, 0.0, 0, 4096, 2.0, 2.0)
    with pytest.raises(ConfigError, match="resolution"):
        TileGrid(0.0, 0.0, 4096, 4096, 0.0, 2.0)


def test_iter_blocks_tiles_an_array_without_gaps_or_overlaps() -> None:
    covered = 0
    seen = set()
    for col, row, width, height in iter_blocks(300, 200, 128, 128):
        covered += width * height
        for r in range(row, row + height):
            for c in range(col, col + width):
                assert (r, c) not in seen
                seen.add((r, c))
    assert covered == 300 * 200
