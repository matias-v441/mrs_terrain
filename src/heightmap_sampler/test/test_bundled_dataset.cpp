// Sampling the dataset bundled into this package.
//
// heightmap-prep writes test_points.csv next to the tiles: every safety area
// corner, with the height its reference sampler gives there.  This library
// must reproduce each one.  Regenerating the dataset with regenerate.sh
// refreshes these expectations along with it.
//
// The dataset's EPSG:4326 tiles hold heights and imagery over each world's
// bounding box, so a region around any of those corners must be readable.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "heightmap_sampler/height_sampler.hpp"

using heightmap_sampler::GridGeoreference;
using heightmap_sampler::HeightGrid;
using heightmap_sampler::HeightSampler;
using heightmap_sampler::LatLon;
using heightmap_sampler::RgbGrid;
using heightmap_sampler::Roi;

namespace
{

/// C++ and Python transform and interpolate in slightly different order.
constexpr double kTolerance = 1e-6;

struct TestPoint
{
  double latitude;
  double longitude;
  double height;
};

std::filesystem::path bundled_dir()
{
  return std::filesystem::path(HEIGHTMAP_SAMPLER_BUNDLED_DATASET);
}

std::vector<TestPoint> load_test_points()
{
  const std::filesystem::path path = bundled_dir() / "test_points.csv";
  std::ifstream in(path);
  EXPECT_TRUE(in.is_open()) << "missing " << path << "; regenerate the dataset";

  std::vector<TestPoint> points;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream fields(line);
    std::string latitude;
    std::string longitude;
    std::string height;
    std::getline(fields, latitude, ',');
    std::getline(fields, longitude, ',');
    std::getline(fields, height);
    points.push_back(TestPoint{std::stod(latitude), std::stod(longitude), std::stod(height)});
  }
  return points;
}

void expect_reproduces(const HeightSampler & sampler, const std::vector<TestPoint> & points)
{
  ASSERT_FALSE(points.empty());
  for (const TestPoint & point : points) {
    const std::optional<double> height = sampler.sample(point.latitude, point.longitude);
    ASSERT_TRUE(height.has_value())
      << "no height at " << point.latitude << ", " << point.longitude;
    EXPECT_NEAR(*height, point.height, kTolerance)
      << "at " << point.latitude << ", " << point.longitude;
  }
}

/// Row-major index of the pixel of \p geo containing a point.
std::size_t pixel_of(const GridGeoreference & geo, double latitude, double longitude)
{
  const auto col = static_cast<std::size_t>((longitude - geo.west) / geo.resolution_deg);
  const auto row = static_cast<std::size_t>((geo.north - latitude) / geo.resolution_deg);
  EXPECT_LT(col, geo.width);
  EXPECT_LT(row, geo.height);
  return row * geo.width + col;
}

}  // namespace

TEST(BundledDataset, ReproducesEveryTestPoint) {
  expect_reproduces(HeightSampler{bundled_dir()}, load_test_points());
}

TEST(BundledDataset, TheBatchFormReproducesEveryTestPoint) {
  const HeightSampler sampler{bundled_dir()};
  const std::vector<TestPoint> points = load_test_points();

  std::vector<LatLon> queries;
  for (const TestPoint & point : points) {
    queries.push_back(LatLon{point.latitude, point.longitude});
  }
  const std::vector<std::optional<double>> heights = sampler.sample(queries);
  ASSERT_EQ(heights.size(), points.size());
  for (std::size_t at = 0; at < points.size(); ++at) {
    ASSERT_TRUE(heights[at].has_value()) << "test point " << at;
    EXPECT_NEAR(*heights[at], points[at].height, kTolerance) << "test point " << at;
  }
}

TEST(BundledDataset, IsWhatADefaultConstructedSamplerReads) {
  // The installed copy, found through the ament index, must be this dataset.
  expect_reproduces(HeightSampler{}, load_test_points());
}

TEST(BundledDataset, HasGridsAroundEveryTestPoint) {
  const HeightSampler sampler{bundled_dir()};
  ASSERT_TRUE(sampler.has_height_grids());
  ASSERT_TRUE(sampler.has_rgb_grids());

  for (const TestPoint & point : load_test_points()) {
    // A point-sized region: the pixels around the safety area corner.
    const Roi roi{
      LatLon{point.latitude, point.longitude}, LatLon{point.latitude, point.longitude}};

    const std::optional<HeightGrid> heights = sampler.height_grid(roi);
    ASSERT_TRUE(heights.has_value()) << "no heights at " << point.latitude << ", "
                                     << point.longitude;
    EXPECT_EQ(heights->geo.width, 2U);
    EXPECT_EQ(heights->geo.height, 2U);
    // The pixel holding a corner is inside the world's bounding box.
    EXPECT_FALSE(
      std::isnan(heights->heights[pixel_of(heights->geo, point.latitude, point.longitude)]))
      << "at " << point.latitude << ", " << point.longitude;

    const std::optional<RgbGrid> imagery = sampler.rgb_grid(roi);
    ASSERT_TRUE(imagery.has_value()) << "no imagery at " << point.latitude << ", "
                                     << point.longitude;
    EXPECT_EQ(imagery->valid[pixel_of(imagery->geo, point.latitude, point.longitude)], 255)
      << "at " << point.latitude << ", " << point.longitude;
  }
}
