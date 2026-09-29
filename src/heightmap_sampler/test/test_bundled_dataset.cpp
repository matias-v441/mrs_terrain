// Sampling the dataset bundled into this package.
//
// heightmap-prep writes test_points.csv next to the tiles: every safety area
// corner, with the height its reference sampler gives there.  This library
// must reproduce each one.  Regenerating the dataset with regenerate.sh
// refreshes these expectations along with it.

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "heightmap_sampler/height_sampler.hpp"

using heightmap_sampler::HeightSampler;
using heightmap_sampler::LatLon;

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
