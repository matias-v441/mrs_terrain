// In-memory view of dataset.yaml, the contract between heightmap-prep and this
// sampler (specification section 11).
//
// Only the keys a sampler actually needs are required.  Everything under
// source/, transform/, sampling_contract/, software/ and processing/ is
// provenance and is deliberately ignored, as is test_points (reference
// heights for checking samplers, see test/parity_check.py).
//
// The optional wgs/ section describes the EPSG:4326 tiles of heights and
// imagery that region-of-interest queries read (see wgs_lattice.hpp).

#ifndef MANIFEST_HPP_
#define MANIFEST_HPP_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "heightmap_sampler/height_sampler.hpp"
#include "tile_grid.hpp"

namespace heightmap_sampler
{

/// Only this manifest layout is understood.
inline constexpr int kFormatVersion = 2;

inline constexpr const char * kManifestFilename = "dataset.yaml";

/// The only CRS WGS tiles are stored in.
inline constexpr const char * kWgsCrs = "EPSG:4326";

/// One layer of WGS tiles (heights or imagery) on its own global lattice.
struct WgsLayer
{
  /// Pixel size along both axes, in degrees; lattice pixels are square.
  double resolution_deg = 0.0;
  std::int64_t tile_width_px = 0;
  std::int64_t tile_height_px = 0;
  std::string tile_pattern;
};

/// The wgs/ section: disjoint EPSG:4326 tiles anchored on one lattice origin.
struct WgsManifest
{
  /// West/north corner of lattice pixel (0, 0), in degrees.
  double origin_lon = 0.0;
  double origin_lat = 0.0;
  WgsLayer height;
  double height_nodata = 0.0;
  /// Absent when the dataset was prepared without imagery.
  std::optional<WgsLayer> rgb;
};

struct Manifest
{
  DatasetInfo info;
  TileGrid grid;
  double nodata = 0.0;
  std::string tile_pattern;
  /// Absent when the dataset holds no WGS tiles.
  std::optional<WgsManifest> wgs;
};

/// Parse \p dataset_dir/dataset.yaml.  Throws DatasetError if the file is
/// missing, malformed, of an unknown format version, not marked complete, or
/// missing a key the sampler needs.
Manifest load_manifest(const std::filesystem::path & dataset_dir);

/// Render a tile's path relative to the dataset root from a manifest pattern
/// such as "height/tile_{ix}_{iy}.tif".  Throws DatasetError if the pattern
/// still contains a placeholder afterwards.
std::string render_tile_path(const std::string & pattern, TileIndex tile);

}  // namespace heightmap_sampler

#endif  // MANIFEST_HPP_
