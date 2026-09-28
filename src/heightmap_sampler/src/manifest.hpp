// In-memory view of dataset.yaml, the contract between heightmap-prep and this
// sampler (specification section 11).
//
// Only the keys a sampler actually needs are required.  Everything under
// source/, transform/, sampling_contract/, software/ and processing/ is
// provenance and is deliberately ignored, as are the optional
// grid.dataset_bounds and world_bounds_wgs84.

#ifndef MANIFEST_HPP_
#define MANIFEST_HPP_

#include <filesystem>
#include <string>

#include "heightmap_sampler/height_sampler.hpp"
#include "tile_grid.hpp"

namespace heightmap_sampler
{

/// Only this manifest layout is understood.
inline constexpr int kFormatVersion = 1;

inline constexpr const char * kManifestFilename = "dataset.yaml";

struct Manifest
{
  DatasetInfo info;
  TileGrid grid;
  double nodata = 0.0;
  std::string tile_pattern;
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
