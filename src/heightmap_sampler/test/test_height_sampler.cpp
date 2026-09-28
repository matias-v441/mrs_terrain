// End-to-end sampling against the synthetic fixture in test/data/synthetic.
//
// The fixture stores a plane, which bilinear interpolation reproduces exactly,
// so the expected values in expected.csv are closed form.  They were computed
// by test/make_fixture.py with pyproj, so agreeing with them also confirms
// that GDAL's WGS84 -> EPSG:5514 transformation matches the one the Python
// reference sampler uses.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "heightmap_sampler/height_sampler.hpp"

using heightmap_sampler::DatasetError;
using heightmap_sampler::HeightSampler;
using heightmap_sampler::LatLon;

namespace
{

struct Expectation
{
  double longitude;
  double latitude;
  double expected;      ///< NaN where the sampler must report no height
  std::string description;
};

std::filesystem::path fixture_dir()
{
  return std::filesystem::path(HEIGHTMAP_SAMPLER_TEST_DATA) / "synthetic";
}

std::vector<Expectation> load_expectations()
{
  const std::filesystem::path path = fixture_dir() / "expected.csv";
  std::ifstream in(path);
  EXPECT_TRUE(in.is_open()) << "missing fixture " << path
                            << "; regenerate it with test/make_fixture.py";

  std::vector<Expectation> expectations;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream fields(line);
    std::string longitude;
    std::string latitude;
    std::string expected;
    std::string description;
    std::getline(fields, longitude, ',');
    std::getline(fields, latitude, ',');
    std::getline(fields, expected, ',');
    std::getline(fields, description);
    expectations.push_back(
      Expectation{
        std::stod(longitude), std::stod(latitude),
        expected == "nan" ? std::nan("") : std::stod(expected), description});
  }
  return expectations;
}

}  // namespace

TEST(HeightSampler, ReportsWhatTheManifestSays) {
  const HeightSampler sampler{fixture_dir()};
  EXPECT_EQ(sampler.info().horizontal_crs, "EPSG:5514");
  EXPECT_EQ(sampler.info().vertical_crs, "EPSG:5773");
  EXPECT_EQ(sampler.info().vertical_datum, "egm96");
  EXPECT_EQ(sampler.info().unit, "m");
  EXPECT_DOUBLE_EQ(sampler.info().resolution_x, 1.0);
  EXPECT_EQ(sampler.dataset_dir(), fixture_dir());
}

TEST(HeightSampler, MatchesTheClosedFormExpectations) {
  const HeightSampler sampler{fixture_dir()};
  const std::vector<Expectation> expectations = load_expectations();
  ASSERT_FALSE(expectations.empty());

  for (const Expectation & expectation : expectations) {
    const auto height = sampler.sample(expectation.latitude, expectation.longitude);
    if (std::isnan(expectation.expected)) {
      EXPECT_FALSE(height.has_value())
        << expectation.description << ": expected no height, got " << *height;
    } else {
      ASSERT_TRUE(height.has_value()) << expectation.description << ": expected a height";
      EXPECT_NEAR(*height, expectation.expected, 1e-6) << expectation.description;
    }
  }
}

TEST(HeightSampler, TheBatchFormAgreesWithTheSingleForm) {
  const HeightSampler sampler{fixture_dir()};
  const std::vector<Expectation> expectations = load_expectations();

  std::vector<LatLon> points;
  points.reserve(expectations.size());
  for (const Expectation & expectation : expectations) {
    points.push_back(LatLon{expectation.latitude, expectation.longitude});
  }

  const std::vector<std::optional<double>> heights = sampler.sample(points);
  ASSERT_EQ(heights.size(), expectations.size());
  for (std::size_t at = 0; at < heights.size(); ++at) {
    const auto one = sampler.sample(points[at].latitude, points[at].longitude);
    EXPECT_EQ(heights[at].has_value(), one.has_value()) << expectations[at].description;
    if (one.has_value()) {
      EXPECT_DOUBLE_EQ(*heights[at], *one) << expectations[at].description;
    }
  }
}

TEST(HeightSampler, AnEmptyBatchIsAnEmptyResult) {
  const HeightSampler sampler{fixture_dir()};
  EXPECT_TRUE(sampler.sample(std::vector<LatLon>{}).empty());
}

TEST(HeightSampler, SurvivesATileCacheTooSmallForOneSample) {
  // A sample touching four tiles with a one-tile cache must still be correct,
  // just slower: each corner reopens its tile.
  heightmap_sampler::Options options;
  options.dataset_dir = fixture_dir();
  options.tile_cache_size = 1;
  const HeightSampler sampler{options};

  const HeightSampler roomy{fixture_dir()};
  for (const Expectation & expectation : load_expectations()) {
    const auto tight = sampler.sample(expectation.latitude, expectation.longitude);
    const auto spacious = roomy.sample(expectation.latitude, expectation.longitude);
    EXPECT_EQ(tight.has_value(), spacious.has_value()) << expectation.description;
    if (spacious.has_value()) {
      EXPECT_DOUBLE_EQ(*tight, *spacious) << expectation.description;
    }
  }
}

TEST(HeightSampler, RefusesADatasetThatIsNotThere) {
  EXPECT_THROW(
    HeightSampler{std::filesystem::path("/nonexistent/heightmap/dataset")}, DatasetError);
}
