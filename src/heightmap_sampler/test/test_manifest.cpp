// A manifest this sampler misreads would produce heights from the wrong place
// rather than an error, so every assumption it makes is checked up front.

#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "manifest.hpp"

using heightmap_sampler::DatasetError;
using heightmap_sampler::load_manifest;

namespace
{

const char * kValidManifest =
  R"(format_version: 1
status: complete
heightmap:
  horizontal_crs: EPSG:5514
  vertical_crs: EPSG:5773
  vertical_datum: egm96
  unit: m
  dtype: float32
  nodata: -9999.0
grid:
  resolution_x: 2.0
  resolution_y: 2.0
  tile_width_px: 4096
  tile_height_px: 4096
  origin_x: -999999.6
  origin_y: -800000.12
  axis_order: east_north
storage:
  tile_pattern: height/tile_{ix}_{iy}.tif
)";

/// A dataset directory holding nothing but a manifest, cleaned up on scope exit.
class TempDataset
{
public:
  explicit TempDataset(const std::string & manifest)
  {
    root_ = std::filesystem::temp_directory_path() /
      ("heightmap_sampler_test_" + std::to_string(++counter_) + "_" +
      std::to_string(::getpid()));
    std::filesystem::create_directories(root_);
    std::ofstream out(root_ / "dataset.yaml");
    out << manifest;
  }

  ~TempDataset()
  {
    std::error_code code;
    std::filesystem::remove_all(root_, code);
  }

  const std::filesystem::path & path() const {return root_;}

private:
  static int counter_;
  std::filesystem::path root_;
};

int TempDataset::counter_ = 0;

/// The valid manifest with one line replaced, for the rejection cases.
std::string manifest_with(const std::string & from, const std::string & to)
{
  std::string text = kValidManifest;
  const std::size_t at = text.find(from);
  EXPECT_NE(at, std::string::npos) << "test needs to patch '" << from << "'";
  text.replace(at, from.size(), to);
  return text;
}

}  // namespace

TEST(Manifest, ReadsWhatASamplerNeeds) {
  const TempDataset dataset{kValidManifest};
  const auto manifest = load_manifest(dataset.path());

  EXPECT_EQ(manifest.info.horizontal_crs, "EPSG:5514");
  EXPECT_EQ(manifest.info.vertical_crs, "EPSG:5773");
  EXPECT_EQ(manifest.info.vertical_datum, "egm96");
  EXPECT_EQ(manifest.info.unit, "m");
  EXPECT_DOUBLE_EQ(manifest.info.resolution_x, 2.0);
  EXPECT_DOUBLE_EQ(manifest.nodata, -9999.0);
  EXPECT_EQ(manifest.tile_pattern, "height/tile_{ix}_{iy}.tif");

  EXPECT_DOUBLE_EQ(manifest.grid.origin_x, -999999.6);
  EXPECT_DOUBLE_EQ(manifest.grid.origin_y, -800000.12);
  EXPECT_EQ(manifest.grid.tile_width_px, 4096);
  EXPECT_DOUBLE_EQ(manifest.grid.tile_span_x(), 8192.0);
}

TEST(Manifest, IgnoresTheProvenanceSections) {
  // source/, transform/, software/ and processing/ are absent from
  // kValidManifest entirely; loading must not care.
  const TempDataset dataset{kValidManifest};
  EXPECT_NO_THROW(load_manifest(dataset.path()));
}

TEST(Manifest, RejectsAnIncompleteDataset) {
  // A dataset still being built has tiles on disk that are not finished.
  const TempDataset dataset{manifest_with("status: complete", "status: building")};
  EXPECT_THROW(load_manifest(dataset.path()), DatasetError);
}

TEST(Manifest, RejectsAnUnknownFormatVersion) {
  const TempDataset dataset{manifest_with("format_version: 1", "format_version: 2")};
  EXPECT_THROW(load_manifest(dataset.path()), DatasetError);
}

TEST(Manifest, RejectsAMissingGridKey) {
  const TempDataset dataset{manifest_with("  origin_x: -999999.6\n", "")};
  EXPECT_THROW(load_manifest(dataset.path()), DatasetError);
}

TEST(Manifest, RejectsAnUnknownAxisOrder) {
  const TempDataset dataset{
    manifest_with("axis_order: east_north", "axis_order: north_east")};
  EXPECT_THROW(load_manifest(dataset.path()), DatasetError);
}

TEST(Manifest, RejectsANonPositiveResolution) {
  const TempDataset dataset{manifest_with("resolution_x: 2.0", "resolution_x: 0.0")};
  EXPECT_THROW(load_manifest(dataset.path()), DatasetError);
}

TEST(Manifest, RejectsAnUnusableTilePattern) {
  const TempDataset dataset{
    manifest_with("tile_pattern: height/tile_{ix}_{iy}.tif",
    "tile_pattern: height/{zoom}/tile_{ix}_{iy}.tif")};
  EXPECT_THROW(load_manifest(dataset.path()), DatasetError);
}

TEST(Manifest, RejectsAMissingManifest) {
  EXPECT_THROW(
    load_manifest(std::filesystem::temp_directory_path() / "heightmap_sampler_no_such_dir"),
    DatasetError);
}
