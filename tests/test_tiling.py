"""The deterministic global tile grid (specification sections 9.4 and 12)."""

from __future__ import annotations

import math

import pytest

from heightmap_prep.errors import ConfigError
from heightmap_prep.tiling import (
    PixelRange,
    ProjectedBounds,
    TileGrid,
    TileIndex,
    iter_blocks,
    parse_tile_filename,
    polygon_intersects_bounds,
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


# --- pixels ---------------------------------------------------------------

#: 10 px tiles at 2 m: tile (ix, iy) holds global columns 10*ix .. 10*ix + 9.
SMALL = TileGrid.from_options(resolution_m=2.0, tile_size_px=10, origin_x=0.0, origin_y=0.0)


def neighbours(grid: TileGrid, x: float, y: float) -> set[tuple[int, int]]:
    """The four global pixels a bilinear sampler reads for (x, y)."""
    col0 = math.floor((x - grid.origin_x) / grid.resolution_x - 0.5)
    row0 = math.floor((grid.origin_y - y) / grid.resolution_y - 0.5)
    return {(col0 + dc, row0 + dr) for dc in (0, 1) for dr in (0, 1)}


def tiles_of(grid: TileGrid, pixels: set[tuple[int, int]]) -> set[TileIndex]:
    return {
        TileIndex(col // grid.tile_width_px, row // grid.tile_height_px) for col, row in pixels
    }


def test_the_tile_pixels_follow_the_index_convention() -> None:
    assert SMALL.tile_pixels(TileIndex(2, 3)) == PixelRange(20, 30, 29, 39)
    assert SMALL.tile_pixels(TileIndex(-1, 0)) == PixelRange(-10, 0, -1, 9)


def test_sampling_pixels_are_the_bilinear_neighbours_of_a_point() -> None:
    x, y = 13.3, -27.7
    pixels = SMALL.sampling_pixels([x], [y])
    block = {
        (c, r)
        for c in range(pixels.col_min, pixels.col_max + 1)
        for r in range(pixels.row_min, pixels.row_max + 1)
    }
    assert block == neighbours(SMALL, x, y)


def test_sampling_pixels_cover_every_point_of_a_box() -> None:
    xs, ys = [3.1, 41.7, 20.0], [-5.2, -33.9, -12.5]
    pixels = SMALL.sampling_pixels(xs, ys)
    for i in range(50):
        for j in range(50):
            x = 3.1 + (41.7 - 3.1) * i / 49
            y = -33.9 + (-5.2 + 33.9) * j / 49
            for col, row in neighbours(SMALL, x, y):
                assert pixels.col_min <= col <= pixels.col_max
                assert pixels.row_min <= row <= pixels.row_max


def test_a_query_on_a_pixel_centre_line_keeps_both_neighbourhoods() -> None:
    # x = 5.0 is the centre of column 2: a sampler may read columns 1-2 or 2-3.
    pixels = SMALL.sampling_pixels([5.0], [-5.5])
    assert (pixels.col_min, pixels.col_max) == (1, 3)


@pytest.mark.parametrize("x", [19.1, 20.0, 20.9])
def test_a_query_near_a_tile_edge_needs_both_tiles(x: float) -> None:
    """Within half a pixel of the seam at x = 20, pixels on both sides are read."""
    assert tiles_of(SMALL, neighbours(SMALL, x, -5.5)) == {TileIndex(0, 0), TileIndex(1, 0)}
    for tile in (TileIndex(0, 0), TileIndex(1, 0)):
        region = SMALL.sampling_region(tile)
        assert region.west <= x <= region.east


def test_sampling_regions_match_what_a_sampler_reads() -> None:
    """A tile's sampling region holds exactly the queries that read one of its pixels."""
    step = 0.37
    for i in range(160):
        for j in range(160):
            x, y = -10.0 + i * step, 10.0 - j * step
            read = tiles_of(SMALL, neighbours(SMALL, x, y))
            for iy in range(-1, 4):
                for ix in range(-1, 4):
                    tile = TileIndex(ix, iy)
                    region = SMALL.sampling_region(tile)
                    inside = region.west <= x <= region.east and region.south <= y <= region.north
                    if tile in read:
                        assert inside, (x, y, tile)
                    elif inside:
                        # Only the slack band may include a tile nobody reads.
                        assert (
                            min(abs(x - region.west), abs(x - region.east),
                                abs(y - region.south), abs(y - region.north))
                            < 0.01
                        ), (x, y, tile)


def test_tiles_for_pixels_is_in_reading_order() -> None:
    assert SMALL.tiles_for_pixels(PixelRange(8, 9, 12, 10)) == [
        TileIndex(0, 0),
        TileIndex(1, 0),
        TileIndex(0, 1),
        TileIndex(1, 1),
    ]
    assert SMALL.tiles_for_pixels(PixelRange(-1, 0, 0, 0)) == [TileIndex(-1, 0), TileIndex(0, 0)]


def test_tile_window_and_pixel_bounds() -> None:
    tile = TileIndex(1, 2)
    pixels = PixelRange(12, 23, 15, 29)
    assert SMALL.tile_window(tile, pixels) == (2, 3, 4, 7)
    assert SMALL.pixels_bounds(pixels) == ProjectedBounds(24.0, -60.0, 32.0, -46.0)
    with pytest.raises(ValueError, match="does not lie inside"):
        SMALL.tile_window(tile, PixelRange(9, 23, 15, 29))


def test_pixel_range_operations() -> None:
    a = PixelRange(0, 0, 9, 9)
    b = PixelRange(5, 5, 14, 14)
    assert a.intersection(b) == PixelRange(5, 5, 9, 9)
    assert a.union(b) == PixelRange(0, 0, 14, 14)
    assert a.intersection(PixelRange(10, 0, 19, 9)) is None
    # Ranges are inclusive, so sharing one column is an overlap.
    assert a.intersection(PixelRange(9, 0, 19, 9)) == PixelRange(9, 0, 9, 9)


# --- polygons -------------------------------------------------------------

BOX = ProjectedBounds(0.0, 0.0, 10.0, 10.0)


@pytest.mark.parametrize(
    "polygon,expected",
    [
        ([(5, 5), (20, 5), (20, 20)], True),  # a vertex inside
        ([(-10, -10), (30, -10), (30, 30), (-10, 30)], True),  # box inside polygon
        ([(-5, 4), (15, 4), (15, 6), (-5, 6)], True),  # a strip through, no vertex inside
        ([(10, 12), (20, 12), (20, 20)], False),  # disjoint
        ([(12, -20), (30, 30), (12, 30)], False),  # disjoint, although the boxes overlap
        ([(10, 10), (20, 10), (20, 20)], True),  # touches a corner
        ([(-20, 30), (30, -20), (30, 30)], True),  # hypotenuse cuts the box
        ([(-20, 15), (15, -20), (-30, -30)], False),  # hypotenuse passes it by
    ],
)
def test_polygon_intersects_bounds(polygon, expected: bool) -> None:
    xs = [float(x) for x, _ in polygon]
    ys = [float(y) for _, y in polygon]
    assert polygon_intersects_bounds(xs, ys, BOX) is expected


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
    assert a.contains(ProjectedBounds(1.0, 1.0, 10.0, 10.0))
    assert not a.contains(b)
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
