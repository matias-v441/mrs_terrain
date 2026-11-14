// Region-of-interest grids against the WGS tiles of the synthetic fixture.
//
// test/make_fixture.py stores a plane on the WGS height lattice and a known
// pattern on the RGB lattice, on power-of-two degree lattices where every pixel
// edge and centre is exact.  Bilinear interpolation of a plane is exact too, so
// every expectation here is closed form.  The constants repeat the fixture's.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "heightmap_sampler/height_sampler.hpp"

using heightmap_sampler::HeightGrid;
using heightmap_sampler::HeightSampler;
using heightmap_sampler::LatLon;
using heightmap_sampler::RgbGrid;
using heightmap_sampler::Roi;

namespace
{

// --- the fixture, as test/make_fixture.py writes it --------------------------

constexpr double kOriginLon = -180.0;
constexpr double kOriginLat = 90.0;
constexpr double kRgbResolution = 1.0 / (1 << 18);
constexpr int kHeightFactor = 4;
constexpr double kHeightResolution = kRgbResolution * kHeightFactor;
constexpr std::int64_t kHeightTilePx = 8;
constexpr std::int64_t kRgbTilePx = 32;

/// The north-west tile of the fixture's 2x2 block.
constexpr std::int64_t kIx0 = 1591393;
constexpr std::int64_t kIy0 = 332901;
/// Its first height pixel.
constexpr std::int64_t kCol0 = kIx0 * kHeightTilePx;
constexpr std::int64_t kRow0 = kIy0 * kHeightTilePx;

constexpr double kPlaneA = 0.25;
constexpr double kPlaneB = -0.125;
constexpr double kPlaneC = 200.0;
constexpr int kBlue = 77;

std::filesystem::path fixture_dir()
{
  return std::filesystem::path(HEIGHTMAP_SAMPLER_TEST_DATA) / "synthetic";
}

// --- positions, relative to the fixture's first height pixel -----------------

/// Longitude of fractional height-lattice centre column kCol0 + \p col.
double lon_at(double col) {return kOriginLon + (kCol0 + col + 0.5) * kHeightResolution;}

/// Latitude of fractional height-lattice centre row kRow0 + \p row.
double lat_at(double row) {return kOriginLat - (kRow0 + row + 0.5) * kHeightResolution;}

/// A region between fractional centre columns and rows, relative to kCol0/kRow0.
Roi roi_between(double col_west, double row_north, double col_east, double row_south)
{
  return Roi{
    LatLon{lat_at(row_north), lon_at(col_west)},
    LatLon{lat_at(row_south), lon_at(col_east)}};
}

/// The fixture's height at a point, from the plane it stores.
double plane_at(double latitude, double longitude)
{
  const double col = (longitude - kOriginLon) / kHeightResolution - 0.5;
  const double row = (kOriginLat - latitude) / kHeightResolution - 0.5;
  return kPlaneA * (col - kCol0) + kPlaneB * (row - kRow0) + kPlaneC;
}

/// The grid's first pixel, on the height lattice, relative to kCol0/kRow0.
std::pair<std::int64_t, std::int64_t> first_pixel(const HeightGrid & grid)
{
  return {
    std::llround((grid.geo.west - kOriginLon) / kHeightResolution) - kCol0,
    std::llround((kOriginLat - grid.geo.north) / kHeightResolution) - kRow0};
}

/// Bilinear interpolation of \p grid alone at a point; NaN, with a test
/// failure, if the point needs a pixel the grid does not hold.
double interpolate(const HeightGrid & grid, double latitude, double longitude)
{
  const double col_f = (longitude - grid.geo.west) / grid.geo.resolution_deg - 0.5;
  const double row_f = (grid.geo.north - latitude) / grid.geo.resolution_deg - 0.5;
  const double col0 = std::floor(col_f);
  const double row0 = std::floor(row_f);
  const auto width = static_cast<double>(grid.geo.width);
  const auto height = static_cast<double>(grid.geo.height);
  if (col0 < 0.0 || col0 + 1.0 >= width || row0 < 0.0 || row0 + 1.0 >= height) {
    ADD_FAILURE() << "the grid does not hold the pixels around " << latitude << ", "
                  << longitude << " (column " << col_f << ", row " << row_f << " of "
                  << grid.geo.width << "x" << grid.geo.height << ")";
    return std::nan("");
  }
  const auto at = [&](double row, double col) -> double {
      return grid.heights[static_cast<std::size_t>(row) * grid.geo.width +
               static_cast<std::size_t>(col)];
    };
  const double fx = col_f - col0;
  const double fy = row_f - row0;
  const double top = at(row0, col0) * (1.0 - fx) + at(row0, col0 + 1) * fx;
  const double bottom = at(row0 + 1, col0) * (1.0 - fx) + at(row0 + 1, col0 + 1) * fx;
  return top * (1.0 - fy) + bottom * fy;
}

void expect_corners_and_interior_interpolate(const HeightSampler & sampler, const Roi & roi)
{
  const std::optional<HeightGrid> grid = sampler.height_grid(roi);
  ASSERT_TRUE(grid.has_value());

  const double north = roi.top_left.latitude;
  const double west = roi.top_left.longitude;
  const double south = roi.bottom_right.latitude;
  const double east = roi.bottom_right.longitude;

  std::vector<std::pair<double, double>> points{
    {north, west}, {north, east}, {south, west}, {south, east}};
  std::mt19937 random(3);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int point = 0; point < 50; ++point) {
    points.emplace_back(
      south + unit(random) * (north - south), west + unit(random) * (east - west));
  }
  for (const auto & [lat, lon] : points) {
    EXPECT_NEAR(interpolate(*grid, lat, lon), plane_at(lat, lon), 1e-9)
      << "at " << lat << ", " << lon;
  }
}

}  // namespace

TEST(HeightGrid, TheFixtureHasGrids) {
  const HeightSampler sampler{fixture_dir()};
  EXPECT_TRUE(sampler.has_height_grids());
  EXPECT_TRUE(sampler.has_rgb_grids());
}

TEST(HeightGrid, StitchesAcrossTheVerticalSeam) {
  // Columns 5.3 .. 10.2 cross the seam between tile columns at column 8.
  const HeightSampler sampler{fixture_dir()};
  const std::optional<HeightGrid> grid = sampler.height_grid(roi_between(5.3, 1.4, 10.2, 3.7));
  ASSERT_TRUE(grid.has_value());

  EXPECT_EQ(grid->geo.width, 7U);    // columns 5 .. 11
  EXPECT_EQ(grid->geo.height, 4U);   // rows 1 .. 4
  EXPECT_DOUBLE_EQ(grid->geo.resolution_deg, kHeightResolution);
  EXPECT_EQ(first_pixel(*grid), (std::pair<std::int64_t, std::int64_t>{5, 1}));
  ASSERT_EQ(grid->heights.size(), grid->geo.width * grid->geo.height);

  for (std::size_t row = 0; row < grid->geo.height; ++row) {
    for (std::size_t col = 0; col < grid->geo.width; ++col) {
      const double expected = kPlaneA * static_cast<double>(5 + col) +
        kPlaneB * static_cast<double>(1 + row) + kPlaneC;
      EXPECT_EQ(grid->heights[row * grid->geo.width + col], expected)
        << "pixel " << col << ", " << row;
    }
  }
}

TEST(HeightGrid, TheCornersAndEveryPointInsideInterpolateFromTheGridAlone) {
  const HeightSampler sampler{fixture_dir()};
  // Across the vertical seam.
  expect_corners_and_interior_interpolate(sampler, roi_between(5.3, 1.4, 10.2, 3.7));
  // Across the horizontal seam, clear of the NoData patch.
  expect_corners_and_interior_interpolate(sampler, roi_between(4.1, 6.5, 6.9, 9.25));
  // Corners on pixel centres, and on pixel edges.
  expect_corners_and_interior_interpolate(sampler, roi_between(2.0, 1.0, 6.0, 5.0));
  expect_corners_and_interior_interpolate(sampler, roi_between(1.5, 0.5, 9.5, 5.5));
}

TEST(HeightGrid, APointIsATwoByTwoGrid) {
  const HeightSampler sampler{fixture_dir()};
  const std::optional<HeightGrid> grid = sampler.height_grid(roi_between(3.6, 2.2, 3.6, 2.2));
  ASSERT_TRUE(grid.has_value());
  EXPECT_EQ(grid->geo.width, 2U);
  EXPECT_EQ(grid->geo.height, 2U);
  EXPECT_NEAR(
    interpolate(*grid, lat_at(2.2), lon_at(3.6)), plane_at(lat_at(2.2), lon_at(3.6)), 1e-9);
}

TEST(HeightGrid, NoDataIsNaN) {
  // The NoData patch: rows 10..11, columns 2..3 (tile (kIx0, kIy0 + 1)).
  const HeightSampler sampler{fixture_dir()};
  const std::optional<HeightGrid> grid = sampler.height_grid(roi_between(0.5, 8.5, 5.5, 13.5));
  ASSERT_TRUE(grid.has_value());
  ASSERT_EQ(first_pixel(*grid), (std::pair<std::int64_t, std::int64_t>{0, 8}));

  for (std::size_t row = 0; row < grid->geo.height; ++row) {
    for (std::size_t col = 0; col < grid->geo.width; ++col) {
      const std::size_t lattice_row = 8 + row;
      const float height = grid->heights[row * grid->geo.width + col];
      const bool nodata = lattice_row >= 10 && lattice_row < 12 && col >= 2 && col < 4;
      EXPECT_EQ(std::isnan(height), nodata) << "pixel " << col << ", " << row;
    }
  }
}

TEST(HeightGrid, AnAbsentTileIsNoGrid) {
  // Columns and rows 6 .. 9 reach into the absent south-east tile.
  const HeightSampler sampler{fixture_dir()};
  EXPECT_FALSE(sampler.height_grid(roi_between(6.0, 6.0, 8.5, 8.5)).has_value());
  EXPECT_FALSE(sampler.rgb_grid(roi_between(6.0, 6.0, 8.5, 8.5)).has_value());
  // Just short of it, the same region is available.
  EXPECT_TRUE(sampler.height_grid(roi_between(6.0, 6.0, 6.4, 8.5)).has_value());
}

TEST(HeightGrid, ARegionOutsideTheDatasetIsNoGrid) {
  const HeightSampler sampler{fixture_dir()};
  EXPECT_FALSE(sampler.height_grid(Roi{{50.1, 14.4}, {50.0, 14.5}}).has_value());
}

TEST(HeightGrid, AnInvalidRegionIsNoGrid) {
  const HeightSampler sampler{fixture_dir()};
  const Roi valid = roi_between(1.0, 1.0, 3.0, 3.0);
  const Roi inverted_east_west{
    LatLon{valid.top_left.latitude, valid.bottom_right.longitude},
    LatLon{valid.bottom_right.latitude, valid.top_left.longitude}};
  const Roi inverted_north_south{
    LatLon{valid.bottom_right.latitude, valid.top_left.longitude},
    LatLon{valid.top_left.latitude, valid.bottom_right.longitude}};
  const Roi not_a_number{LatLon{std::nan(""), 14.26}, valid.bottom_right};
  const Roi off_the_globe{LatLon{95.0, 14.26}, valid.bottom_right};

  ASSERT_TRUE(sampler.height_grid(valid).has_value());
  for (const Roi & roi : {inverted_east_west, inverted_north_south, not_a_number, off_the_globe}) {
    EXPECT_FALSE(sampler.height_grid(roi).has_value());
    EXPECT_FALSE(sampler.rgb_grid(roi).has_value());
  }
}

TEST(HeightGrid, SurvivesATileCacheTooSmallForOneGrid) {
  heightmap_sampler::Options options;
  options.dataset_dir = fixture_dir();
  options.tile_cache_size = 1;
  const HeightSampler tight{options};
  const HeightSampler roomy{fixture_dir()};

  const Roi roi = roi_between(4.1, 6.5, 6.9, 9.25);   // two tiles
  const std::optional<HeightGrid> a = tight.height_grid(roi);
  const std::optional<HeightGrid> b = roomy.height_grid(roi);
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  EXPECT_EQ(a->heights, b->heights);
}

TEST(RgbGrid, StitchesTheImageryOnItsOwnLattice) {
  // A height-lattice centre coordinate c is RGB-lattice centre coordinate
  // 4 * (c + 0.5) - 0.5, relative to 4 * kCol0 / 4 * kRow0.  Columns 5.3 .. 10.2
  // are RGB columns 22.7 .. 42.3, and rows 1.4 .. 3.7 are RGB rows 7.1 .. 16.3.
  const HeightSampler sampler{fixture_dir()};
  const Roi roi = roi_between(5.3, 1.4, 10.2, 3.7);
  const std::optional<RgbGrid> grid = sampler.rgb_grid(roi);
  ASSERT_TRUE(grid.has_value());

  EXPECT_DOUBLE_EQ(grid->geo.resolution_deg, kRgbResolution);
  const std::int64_t col_first =
    std::llround((grid->geo.west - kOriginLon) / kRgbResolution) - kHeightFactor * kCol0;
  const std::int64_t row_first =
    std::llround((kOriginLat - grid->geo.north) / kRgbResolution) - kHeightFactor * kRow0;
  EXPECT_EQ(col_first, 22);
  EXPECT_EQ(row_first, 7);
  EXPECT_EQ(grid->geo.width, 22U);    // columns 22 .. 43
  EXPECT_EQ(grid->geo.height, 11U);   // rows 7 .. 17
  ASSERT_EQ(grid->rgb.size(), 3 * grid->geo.width * grid->geo.height);
  ASSERT_EQ(grid->valid.size(), grid->geo.width * grid->geo.height);

  for (std::size_t row = 0; row < grid->geo.height; ++row) {
    for (std::size_t col = 0; col < grid->geo.width; ++col) {
      const std::int64_t lattice_col = col_first + static_cast<std::int64_t>(col);
      const std::int64_t lattice_row = row_first + static_cast<std::int64_t>(row);
      // The masked patch: rows 8..15, columns 4..11 of tile (kIx0 + 1, kIy0).
      const bool masked = lattice_row >= 8 && lattice_row < 16 &&
        lattice_col >= kRgbTilePx + 4 && lattice_col < kRgbTilePx + 12;
      const std::size_t at = row * grid->geo.width + col;
      EXPECT_EQ(grid->valid[at], masked ? 0 : 255) << "pixel " << col << ", " << row;
      if (masked) {
        EXPECT_EQ(grid->rgb[3 * at], 0);
        EXPECT_EQ(grid->rgb[3 * at + 1], 0);
        EXPECT_EQ(grid->rgb[3 * at + 2], 0);
      } else {
        EXPECT_EQ(grid->rgb[3 * at], lattice_col) << "pixel " << col << ", " << row;
        EXPECT_EQ(grid->rgb[3 * at + 1], lattice_row) << "pixel " << col << ", " << row;
        EXPECT_EQ(grid->rgb[3 * at + 2], kBlue) << "pixel " << col << ", " << row;
      }
    }
  }
}

TEST(RgbGrid, CoversWhatTheHeightGridCovers) {
  // Both grids hold every point of the region; the RGB one is finer.
  const HeightSampler sampler{fixture_dir()};
  const Roi roi = roi_between(4.1, 6.5, 6.9, 9.25);
  const std::optional<HeightGrid> heights = sampler.height_grid(roi);
  const std::optional<RgbGrid> imagery = sampler.rgb_grid(roi);
  ASSERT_TRUE(heights.has_value());
  ASSERT_TRUE(imagery.has_value());

  const auto east_of = [](const auto & geo) {
      return geo.west + static_cast<double>(geo.width) * geo.resolution_deg;
    };
  const auto south_of = [](const auto & geo) {
      return geo.north - static_cast<double>(geo.height) * geo.resolution_deg;
    };
  for (const auto & geo : {heights->geo, imagery->geo}) {
    EXPECT_LE(geo.west, roi.top_left.longitude);
    EXPECT_GE(geo.north, roi.top_left.latitude);
    EXPECT_GE(east_of(geo), roi.bottom_right.longitude);
    EXPECT_LE(south_of(geo), roi.bottom_right.latitude);
  }
}
