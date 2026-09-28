#include "heightmap_sampler/height_sampler.hpp"

#include <ogr_spatialref.h>

#include <array>
#include <cmath>
#include <mutex>
#include <utility>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "manifest.hpp"
#include "tile_cache.hpp"
#include "tile_grid.hpp"

namespace heightmap_sampler
{
namespace
{

/// The CRS every query arrives in (specification section 22).
constexpr const char * kQueryCrs = "EPSG:4326";

/// OGR hands out transformations that must be released through DestroyCT.
struct DestroyCoordinateTransformation
{
  void operator()(OGRCoordinateTransformation * transformation) const
  {
    OGRCoordinateTransformation::DestroyCT(transformation);
  }
};

using CoordinateTransformationPtr =
  std::unique_ptr<OGRCoordinateTransformation, DestroyCoordinateTransformation>;

OGRSpatialReference make_crs(const std::string & definition)
{
  OGRSpatialReference crs;
  if (crs.SetFromUserInput(definition.c_str()) != OGRERR_NONE) {
    throw DatasetError("could not resolve CRS '" + definition + "'");
  }
  // The equivalent of pyproj's always_xy=True: coordinates are (longitude,
  // latitude) and (easting, northing) regardless of what the CRS authority
  // declares as its axis order.
  crs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  return crs;
}

}  // namespace

std::filesystem::path default_dataset_dir()
{
  try {
    return std::filesystem::path(
      ament_index_cpp::get_package_share_directory("heightmap_sampler")) / "dataset";
  } catch (const std::exception & error) {
    throw DatasetError(
            std::string("could not locate the heightmap_sampler share directory: ") +
            error.what());
  }
}

struct HeightSampler::Impl
{
  explicit Impl(Options options)
  : dataset_dir(
      options.dataset_dir.empty() ? default_dataset_dir() : std::move(options.dataset_dir)),
    manifest(load_manifest(dataset_dir)),
    tiles(dataset_dir, manifest.tile_pattern, options.tile_cache_size),
    nodata(static_cast<float>(manifest.nodata))
  {
    OGRSpatialReference query_crs = make_crs(kQueryCrs);
    OGRSpatialReference stored_crs = make_crs(manifest.info.horizontal_crs);
    to_stored.reset(OGRCreateCoordinateTransformation(&query_crs, &stored_crs));
    if (to_stored == nullptr) {
      throw DatasetError(
              "no coordinate transformation from " + std::string(kQueryCrs) + " to " +
              manifest.info.horizontal_crs);
    }
  }

  /// Nearest stored sample value at projected (x, y), or nothing.
  std::optional<double> pixel(double x, double y)
  {
    const TileIndex index = manifest.grid.index_for_point(x, y);
    const Tile * tile = tiles.get(index);
    if (tile == nullptr) {
      return std::nullopt;
    }
    const std::optional<float> value = tile->value_at(x, y);
    if (!value.has_value() || *value == nodata) {
      return std::nullopt;
    }
    return static_cast<double>(*value);
  }

  std::optional<double> sample(double latitude, double longitude)
  {
    double x = longitude;
    double y = latitude;
    if (!to_stored->Transform(1, &x, &y)) {
      return std::nullopt;
    }

    const TileGrid & grid = manifest.grid;

    // Pixel values sit at pixel centres, so shift by half a pixel before taking
    // the fractional part (specification section 23).
    const double col_f = (x - grid.origin_x) / grid.resolution_x - 0.5;
    const double row_f = (grid.origin_y - y) / grid.resolution_y - 0.5;
    const double col0 = std::floor(col_f);
    const double row0 = std::floor(row_f);
    const double fx = col_f - col0;
    const double fy = row_f - row0;

    // Each corner is resolved independently from its own pixel-centre
    // coordinate, which is what makes interpolation work across a tile seam:
    // tiles carry no halo, so the four corners may live in up to four tiles.
    std::array<double, 4> corners{};
    std::size_t at = 0;
    for (int drow = 0; drow < 2; ++drow) {
      for (int dcol = 0; dcol < 2; ++dcol) {
        const double cx = grid.origin_x + (col0 + dcol + 0.5) * grid.resolution_x;
        const double cy = grid.origin_y - (row0 + drow + 0.5) * grid.resolution_y;
        const std::optional<double> value = pixel(cx, cy);
        if (!value.has_value()) {
          return std::nullopt;
        }
        corners[at++] = *value;
      }
    }

    const double v00 = corners[0];
    const double v10 = corners[1];
    const double v01 = corners[2];
    const double v11 = corners[3];
    const double top = v00 * (1.0 - fx) + v10 * fx;
    const double bottom = v01 * (1.0 - fx) + v11 * fx;
    return top * (1.0 - fy) + bottom * fy;
  }

  std::filesystem::path dataset_dir;
  Manifest manifest;
  TileCache tiles;
  float nodata;
  CoordinateTransformationPtr to_stored;
  // Neither OGRCoordinateTransformation nor the tile cache is thread-safe, and
  // a sample touches both.
  std::mutex mutex;
};

HeightSampler::HeightSampler()
: HeightSampler(Options{}) {}

namespace
{
Options options_for(std::filesystem::path dataset_dir)
{
  Options options;
  options.dataset_dir = std::move(dataset_dir);
  return options;
}
}  // namespace

HeightSampler::HeightSampler(std::filesystem::path dataset_dir)
: HeightSampler(options_for(std::move(dataset_dir))) {}

HeightSampler::HeightSampler(Options options)
: impl_(std::make_unique<Impl>(std::move(options))) {}

HeightSampler::~HeightSampler() = default;
HeightSampler::HeightSampler(HeightSampler &&) noexcept = default;
HeightSampler & HeightSampler::operator=(HeightSampler &&) noexcept = default;

std::optional<double> HeightSampler::sample(double latitude, double longitude) const
{
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  try {
    return impl_->sample(latitude, longitude);
  } catch (const std::exception &) {
    // A tile that exists but cannot be opened is indistinguishable, to a
    // caller, from one that is missing.  sample() promises not to throw.
    return std::nullopt;
  }
}

std::vector<std::optional<double>> HeightSampler::sample(const std::vector<LatLon> & points) const
{
  std::vector<std::optional<double>> heights;
  heights.reserve(points.size());
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  for (const LatLon & point : points) {
    try {
      heights.push_back(impl_->sample(point.latitude, point.longitude));
    } catch (const std::exception &) {
      heights.push_back(std::nullopt);
    }
  }
  return heights;
}

const DatasetInfo & HeightSampler::info() const noexcept
{
  return impl_->manifest.info;
}

const std::filesystem::path & HeightSampler::dataset_dir() const noexcept
{
  return impl_->dataset_dir;
}

}  // namespace heightmap_sampler
