#include "manifest.hpp"

#include <yaml-cpp/yaml.h>

#include <string>

namespace heightmap_sampler
{
namespace
{

/// Look up a required child node, naming the full path in the error message so
/// a malformed manifest is diagnosable without reading this source.
const YAML::Node require(const YAML::Node & parent, const char * key, const std::string & path)
{
  const YAML::Node node = parent[key];
  if (!node) {
    throw DatasetError("dataset.yaml is missing required key '" + path + "'");
  }
  return node;
}

template<typename T>
T require_as(const YAML::Node & parent, const char * key, const std::string & path)
{
  try {
    return require(parent, key, path).as<T>();
  } catch (const YAML::Exception & error) {
    throw DatasetError("dataset.yaml key '" + path + "' has an unexpected value: " + error.what());
  }
}

void replace_all(std::string & text, const std::string & needle, const std::string & value)
{
  for (std::size_t at = text.find(needle); at != std::string::npos;
    at = text.find(needle, at + value.size()))
  {
    text.replace(at, needle.size(), value);
  }
}

}  // namespace

std::string render_tile_path(const std::string & pattern, TileIndex tile)
{
  std::string path = pattern;
  replace_all(path, "{ix}", std::to_string(tile.ix));
  replace_all(path, "{iy}", std::to_string(tile.iy));
  if (path.find('{') != std::string::npos) {
    throw DatasetError(
            "storage.tile_pattern '" + pattern +
            "' uses a placeholder this sampler does not understand");
  }
  return path;
}

Manifest load_manifest(const std::filesystem::path & dataset_dir)
{
  const std::filesystem::path path = dataset_dir / kManifestFilename;

  std::error_code code;
  if (!std::filesystem::is_regular_file(path, code)) {
    throw DatasetError("no dataset manifest at " + path.string());
  }

  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (const YAML::Exception & error) {
    throw DatasetError("could not parse " + path.string() + ": " + error.what());
  }
  if (!root.IsMap()) {
    throw DatasetError(path.string() + " is not a YAML mapping");
  }

  const auto format_version = require_as<int>(root, "format_version", "format_version");
  if (format_version != kFormatVersion) {
    throw DatasetError(
            "dataset.yaml format_version is " + std::to_string(format_version) + ", this sampler "
            "understands " + std::to_string(kFormatVersion));
  }

  const auto status = require_as<std::string>(root, "status", "status");
  if (status != "complete") {
    throw DatasetError("dataset status is '" + status + "', expected 'complete'");
  }

  const YAML::Node heightmap = require(root, "heightmap", "heightmap");
  const YAML::Node grid = require(root, "grid", "grid");
  const YAML::Node storage = require(root, "storage", "storage");

  const auto axis_order = require_as<std::string>(grid, "axis_order", "grid.axis_order");
  if (axis_order != "east_north") {
    throw DatasetError(
            "grid.axis_order is '" + axis_order + "', this sampler only handles 'east_north'");
  }

  Manifest manifest;

  manifest.info.horizontal_crs =
    require_as<std::string>(heightmap, "horizontal_crs", "heightmap.horizontal_crs");
  manifest.info.vertical_crs =
    require_as<std::string>(heightmap, "vertical_crs", "heightmap.vertical_crs");
  manifest.info.vertical_datum =
    require_as<std::string>(heightmap, "vertical_datum", "heightmap.vertical_datum");
  manifest.info.unit = heightmap["unit"] ? heightmap["unit"].as<std::string>() : "m";
  manifest.nodata = require_as<double>(heightmap, "nodata", "heightmap.nodata");

  manifest.grid.origin_x = require_as<double>(grid, "origin_x", "grid.origin_x");
  manifest.grid.origin_y = require_as<double>(grid, "origin_y", "grid.origin_y");
  manifest.grid.resolution_x = require_as<double>(grid, "resolution_x", "grid.resolution_x");
  manifest.grid.resolution_y = require_as<double>(grid, "resolution_y", "grid.resolution_y");
  manifest.grid.tile_width_px =
    require_as<std::int64_t>(grid, "tile_width_px", "grid.tile_width_px");
  manifest.grid.tile_height_px =
    require_as<std::int64_t>(grid, "tile_height_px", "grid.tile_height_px");

  if (manifest.grid.resolution_x <= 0.0 || manifest.grid.resolution_y <= 0.0) {
    throw DatasetError("grid resolution must be positive");
  }
  if (manifest.grid.tile_width_px <= 0 || manifest.grid.tile_height_px <= 0) {
    throw DatasetError("grid tile dimensions must be positive");
  }

  manifest.info.resolution_x = manifest.grid.resolution_x;
  manifest.info.resolution_y = manifest.grid.resolution_y;

  manifest.tile_pattern =
    require_as<std::string>(storage, "tile_pattern", "storage.tile_pattern");
  // Fail now rather than on the first sample if the pattern is unusable.
  (void)render_tile_path(manifest.tile_pattern, TileIndex{0, 0});

  return manifest;
}

}  // namespace heightmap_sampler
