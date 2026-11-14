// The region-of-interest window is the contract of height_grid and rgb_grid:
// it must hold every pixel bilinear interpolation reads for a point of the
// region, and no other.  These check that rule on its own, away from tiles.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "wgs_lattice.hpp"

using heightmap_sampler::floor_div;
using heightmap_sampler::intersection;
using heightmap_sampler::Lattice;
using heightmap_sampler::PixelWindow;
using heightmap_sampler::tile_window;
using heightmap_sampler::tiles_for_window;
using heightmap_sampler::TileIndex;

namespace
{

/// A lattice whose pixel edges and centres are all exact in binary.
constexpr Lattice kLattice{-180.0, 90.0, 0.25};

/// Longitude of fractional pixel-centre column \p col (column c's centre is c).
double lon_at(double col) {return kLattice.origin_lon + (col + 0.5) * kLattice.resolution_deg;}

/// Latitude of fractional pixel-centre row \p row.
double lat_at(double row) {return kLattice.origin_lat - (row + 0.5) * kLattice.resolution_deg;}

PixelWindow window_for(double col_west, double row_north, double col_east, double row_south)
{
  const std::optional<PixelWindow> window = kLattice.bilinear_window(
    lon_at(col_west), lat_at(row_south), lon_at(col_east), lat_at(row_north));
  EXPECT_TRUE(window.has_value());
  return window.value_or(PixelWindow{});
}

/// The pixels bilinear interpolation reads at a point.
PixelWindow pixels_read_at(double longitude, double latitude)
{
  const auto col = static_cast<std::int64_t>(std::floor(kLattice.centre_col(longitude)));
  const auto row = static_cast<std::int64_t>(std::floor(kLattice.centre_row(latitude)));
  return PixelWindow{col, row, 2, 2};
}

}  // namespace

TEST(BilinearWindow, ACornerInsideAPixelReadsItsNeighbours) {
  // Fractional centre columns 10.3 .. 12.6: the west corner reads 10 and 11,
  // the east corner 12 and 13.
  const PixelWindow window = window_for(10.3, 20.2, 12.6, 21.9);
  EXPECT_EQ(window, (PixelWindow{10, 20, 4, 3}));
}

TEST(BilinearWindow, ACornerOnAPixelCentreStillReadsTheNextPixel) {
  // The sampler reads floor(c) and floor(c) + 1 even when c is integral.
  const PixelWindow window = window_for(10.0, 20.0, 12.0, 21.0);
  EXPECT_EQ(window, (PixelWindow{10, 20, 4, 3}));
}

TEST(BilinearWindow, ACornerOnAPixelEdgeReadsBothPixelsBesideIt) {
  // A pixel edge is half-way between two centres.
  const PixelWindow window = window_for(9.5, 19.5, 12.5, 21.5);
  EXPECT_EQ(window, (PixelWindow{9, 19, 5, 4}));
}

TEST(BilinearWindow, APointIsTwoByTwo) {
  const PixelWindow window = window_for(10.3, 20.7, 10.3, 20.7);
  EXPECT_EQ(window, (PixelWindow{10, 20, 2, 2}));
}

TEST(BilinearWindow, HoldsWhatEveryPointReadsAndNothingElse) {
  std::mt19937 random(7);
  std::uniform_real_distribution<double> centre(-5.0, 5.0);
  std::uniform_real_distribution<double> extent(0.0, 4.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  for (int trial = 0; trial < 2000; ++trial) {
    const double west = 14.0 + centre(random);
    const double north = 50.0 + centre(random);
    const double east = west + extent(random);
    const double south = north - extent(random);
    const auto window = kLattice.bilinear_window(west, south, east, north);
    ASSERT_TRUE(window.has_value());

    // The corners need the window's outermost pixels, so it is minimal...
    const PixelWindow north_west = pixels_read_at(west, north);
    const PixelWindow south_east = pixels_read_at(east, south);
    EXPECT_EQ(window->col_min, north_west.col_min);
    EXPECT_EQ(window->row_min, north_west.row_min);
    EXPECT_EQ(window->col_end(), south_east.col_end());
    EXPECT_EQ(window->row_end(), south_east.row_end());

    // ...and every point of the region reads only pixels inside it.
    for (const auto & [lon, lat] : std::vector<std::pair<double, double>>{
      {west, north}, {east, north}, {west, south}, {east, south}})
    {
      EXPECT_TRUE(window->contains(pixels_read_at(lon, lat)));
    }
    for (int point = 0; point < 20; ++point) {
      const double lon = west + unit(random) * (east - west);
      const double lat = south + unit(random) * (north - south);
      EXPECT_TRUE(window->contains(pixels_read_at(lon, lat)))
        << "point " << lat << ", " << lon;
    }
  }
}

TEST(BilinearWindow, RejectsAnInvertedRegion) {
  EXPECT_FALSE(kLattice.bilinear_window(14.1, 50.0, 14.0, 50.1).has_value());
  EXPECT_FALSE(kLattice.bilinear_window(14.0, 50.1, 14.1, 50.0).has_value());
}

TEST(BilinearWindow, RejectsANonFiniteRegion) {
  EXPECT_FALSE(kLattice.bilinear_window(std::nan(""), 50.0, 14.1, 50.1).has_value());
  EXPECT_FALSE(kLattice.bilinear_window(14.0, 50.0, INFINITY, 50.1).has_value());
}

TEST(Lattice, EdgesOfAWindowAreItsOuterPixelEdges) {
  EXPECT_DOUBLE_EQ(kLattice.west_of(4), -179.0);
  EXPECT_DOUBLE_EQ(kLattice.north_of(4), 89.0);
  EXPECT_DOUBLE_EQ(kLattice.centre_col(kLattice.west_of(4)), 3.5);
  EXPECT_DOUBLE_EQ(kLattice.centre_row(kLattice.north_of(4)), 3.5);
}

TEST(Lattice, PlacesARasterOnTheLattice) {
  const double transform[6] = {kLattice.west_of(16), 0.25, 0.0, kLattice.north_of(8), 0.0, -0.25};
  const auto window = kLattice.window_of_geotransform(transform, 8, 4);
  ASSERT_TRUE(window.has_value());
  EXPECT_EQ(*window, (PixelWindow{16, 8, 8, 4}));
}

TEST(Lattice, RejectsARasterOffTheLattice) {
  const double shifted[6] = {kLattice.west_of(16) + 0.1, 0.25, 0.0, 89.0, 0.0, -0.25};
  const double coarser[6] = {kLattice.west_of(16), 0.5, 0.0, 89.0, 0.0, -0.5};
  const double south_up[6] = {kLattice.west_of(16), 0.25, 0.0, 89.0, 0.0, 0.25};
  const double rotated[6] = {kLattice.west_of(16), 0.25, 0.01, 89.0, 0.0, -0.25};
  EXPECT_FALSE(kLattice.window_of_geotransform(shifted, 8, 8).has_value());
  EXPECT_FALSE(kLattice.window_of_geotransform(coarser, 8, 8).has_value());
  EXPECT_FALSE(kLattice.window_of_geotransform(south_up, 8, 8).has_value());
  EXPECT_FALSE(kLattice.window_of_geotransform(rotated, 8, 8).has_value());
}

TEST(Tiles, AWindowInsideOneTileTouchesOnlyIt) {
  const std::vector<TileIndex> tiles = tiles_for_window(PixelWindow{10, 20, 4, 4}, 8, 8);
  ASSERT_EQ(tiles.size(), 1U);
  EXPECT_EQ(tiles[0], (TileIndex{1, 2}));
}

TEST(Tiles, AWindowAcrossSeamsTouchesEveryTileInReadingOrder) {
  const std::vector<TileIndex> tiles = tiles_for_window(PixelWindow{6, 6, 4, 4}, 8, 8);
  const std::vector<TileIndex> expected{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  EXPECT_EQ(tiles, expected);
}

TEST(Tiles, AWindowEndingOnASeamDoesNotTouchTheNextTile) {
  const std::vector<TileIndex> tiles = tiles_for_window(PixelWindow{0, 0, 8, 8}, 8, 8);
  ASSERT_EQ(tiles.size(), 1U);
  EXPECT_EQ(tiles[0], (TileIndex{0, 0}));
}

TEST(Tiles, NegativePixelsBelongToNegativeTiles) {
  EXPECT_EQ(floor_div(-1, 8), -1);
  EXPECT_EQ(floor_div(-8, 8), -1);
  EXPECT_EQ(floor_div(-9, 8), -2);
  EXPECT_EQ(floor_div(7, 8), 0);
  const std::vector<TileIndex> tiles = tiles_for_window(PixelWindow{-2, 0, 4, 1}, 8, 8);
  const std::vector<TileIndex> expected{{-1, 0}, {0, 0}};
  EXPECT_EQ(tiles, expected);
}

TEST(Tiles, ATileWindowAndItsIndexAgree) {
  const PixelWindow window = tile_window(TileIndex{3, 5}, 8, 16);
  EXPECT_EQ(window, (PixelWindow{24, 80, 8, 16}));
  const std::vector<TileIndex> tiles = tiles_for_window(window, 8, 16);
  ASSERT_EQ(tiles.size(), 1U);
  EXPECT_EQ(tiles[0], (TileIndex{3, 5}));
}

TEST(Windows, IntersectionIsTheOverlap) {
  const auto overlap = intersection(PixelWindow{0, 0, 10, 10}, PixelWindow{5, 8, 10, 10});
  ASSERT_TRUE(overlap.has_value());
  EXPECT_EQ(*overlap, (PixelWindow{5, 8, 5, 2}));
  EXPECT_FALSE(intersection(PixelWindow{0, 0, 5, 5}, PixelWindow{5, 0, 5, 5}).has_value());
}
