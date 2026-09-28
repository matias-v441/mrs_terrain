// The grid math is the part a sampler cannot get subtly wrong without
// returning plausible heights from the wrong place, so these mirror the
// assertions of tests/test_tiling.py.

#include <gtest/gtest.h>

#include "manifest.hpp"
#include "tile_grid.hpp"

using heightmap_sampler::ProjectedBounds;
using heightmap_sampler::TileGrid;
using heightmap_sampler::TileIndex;

namespace
{

/// The grid of a real dataset: 4096 px tiles at 2 m, phase-shifted origin.
TileGrid real_grid()
{
  TileGrid grid;
  grid.origin_x = -999999.6;
  grid.origin_y = -800000.12;
  grid.resolution_x = 2.0;
  grid.resolution_y = 2.0;
  grid.tile_width_px = 4096;
  grid.tile_height_px = 4096;
  return grid;
}

}  // namespace

TEST(TileGrid, SpanIsPixelsTimesResolution) {
  const TileGrid grid = real_grid();
  EXPECT_DOUBLE_EQ(grid.tile_span_x(), 8192.0);
  EXPECT_DOUBLE_EQ(grid.tile_span_y(), 8192.0);
}

TEST(TileGrid, OriginIsTheWestNorthCornerOfTileZero) {
  const TileGrid grid = real_grid();
  const ProjectedBounds bounds = grid.tile_bounds(TileIndex{0, 0});
  EXPECT_DOUBLE_EQ(bounds.west, grid.origin_x);
  EXPECT_DOUBLE_EQ(bounds.north, grid.origin_y);
  EXPECT_DOUBLE_EQ(bounds.east, grid.origin_x + 8192.0);
  EXPECT_DOUBLE_EQ(bounds.south, grid.origin_y - 8192.0);
}

TEST(TileGrid, IxIncreasesEastwardAndIyIncreasesSouthward) {
  const TileGrid grid = real_grid();
  const TileIndex here = grid.index_for_point(grid.origin_x + 100.0, grid.origin_y - 100.0);
  const TileIndex east = grid.index_for_point(grid.origin_x + 8292.0, grid.origin_y - 100.0);
  const TileIndex south = grid.index_for_point(grid.origin_x + 100.0, grid.origin_y - 8292.0);

  EXPECT_EQ(here.ix, 0);
  EXPECT_EQ(here.iy, 0);
  EXPECT_EQ(east.ix, here.ix + 1);
  EXPECT_EQ(east.iy, here.iy);
  EXPECT_EQ(south.ix, here.ix);
  EXPECT_EQ(south.iy, here.iy + 1);
}

TEST(TileGrid, IndexOfTheRealDatasetTile) {
  // The sole tile of the prepared Temesvar dataset.
  const TileGrid grid = real_grid();
  const TileIndex tile = grid.index_for_point(-765263.5372587888, -1121289.1259690863);
  EXPECT_EQ(tile.ix, 28);
  EXPECT_EQ(tile.iy, 39);
}

TEST(TileGrid, NegativeIndicesWorkWestAndNorthOfTheOrigin) {
  const TileGrid grid = real_grid();
  const TileIndex tile = grid.index_for_point(grid.origin_x - 1.0, grid.origin_y + 1.0);
  EXPECT_EQ(tile.ix, -1);
  EXPECT_EQ(tile.iy, -1);
}

TEST(TileGrid, APointOnATileEdgeBelongsToTheTileItOpens) {
  const TileGrid grid = real_grid();
  const TileIndex tile = grid.index_for_point(grid.origin_x + 8192.0, grid.origin_y - 8192.0);
  EXPECT_EQ(tile.ix, 1);
  EXPECT_EQ(tile.iy, 1);
}

TEST(TileGrid, BoundsAndIndexRoundTrip) {
  const TileGrid grid = real_grid();
  for (const TileIndex tile : {TileIndex{0, 0}, TileIndex{28, 39}, TileIndex{-3, 7}}) {
    const ProjectedBounds bounds = grid.tile_bounds(tile);
    const TileIndex recovered = grid.index_for_point(
      (bounds.west + bounds.east) / 2.0, (bounds.south + bounds.north) / 2.0);
    EXPECT_EQ(recovered.ix, tile.ix);
    EXPECT_EQ(recovered.iy, tile.iy);
  }
}

TEST(TileGrid, AdjacentTilesAbutExactly) {
  const TileGrid grid = real_grid();
  const ProjectedBounds here = grid.tile_bounds(TileIndex{5, 5});
  const ProjectedBounds east = grid.tile_bounds(TileIndex{6, 5});
  const ProjectedBounds south = grid.tile_bounds(TileIndex{5, 6});
  EXPECT_DOUBLE_EQ(here.east, east.west);
  EXPECT_DOUBLE_EQ(here.south, south.north);
}

TEST(TilePattern, RendersTheDeterministicRelativePath) {
  EXPECT_EQ(
    heightmap_sampler::render_tile_path("height/tile_{ix}_{iy}.tif", TileIndex{28, 39}),
    "height/tile_28_39.tif");
  EXPECT_EQ(
    heightmap_sampler::render_tile_path("height/tile_{ix}_{iy}.tif", TileIndex{-3, 7}),
    "height/tile_-3_7.tif");
}

TEST(TilePattern, RejectsAnUnknownPlaceholder) {
  EXPECT_THROW(
    heightmap_sampler::render_tile_path("height/{zoom}/tile_{ix}_{iy}.tif", TileIndex{0, 0}),
    heightmap_sampler::DatasetError);
}
