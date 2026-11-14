#include "heightmap_sampler/height_sampler.hpp"

#include <ogr_spatialref.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "manifest.hpp"
#include "tile_cache.hpp"
#include "tile_grid.hpp"
#include "wgs_lattice.hpp"

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

/// The window of \p lattice that \p roi needs, or nothing when \p roi is
/// inverted, non-finite or off the globe.
std::optional<PixelWindow> roi_window(const Lattice & lattice, const Roi & roi)
{
  const double west = roi.top_left.longitude;
  const double north = roi.top_left.latitude;
  const double east = roi.bottom_right.longitude;
  const double south = roi.bottom_right.latitude;
  // Written so that NaN fails too.
  if (!(west >= -180.0 && east <= 180.0 && south >= -90.0 && north <= 90.0)) {
    return std::nullopt;
  }
  return lattice.bilinear_window(west, south, east, north);
}

GridGeoreference georeference(const Lattice & lattice, const PixelWindow & window)
{
  GridGeoreference geo;
  geo.west = lattice.west_of(window.col_min);
  geo.north = lattice.north_of(window.row_min);
  geo.resolution_deg = lattice.resolution_deg;
  geo.width = static_cast<std::size_t>(window.width);
  geo.height = static_cast<std::size_t>(window.height);
  return geo;
}

/// Where one tile's contribution to a stitched window lies: \p width x
/// \p height pixels at (\p tile_col, \p tile_row) of the tile's raster go to
/// (\p out_col, \p out_row) of the output.
struct TilePiece
{
  int tile_col;
  int tile_row;
  int width;
  int height;
  std::size_t out_col;
  std::size_t out_row;
};

/// Hand every piece of \p window to \p read, one tile at a time.  Returns
/// false as soon as a tile is absent, does not cover its part of the window,
/// or cannot be read.  Tiles never overlap, so every pixel is read once.
template<typename Read>
bool stitch(
  TileCache & tiles, const Lattice & lattice, const WgsLayer & layer,
  const PixelWindow & window, Read && read)
{
  for (const TileIndex index :
    tiles_for_window(window, layer.tile_width_px, layer.tile_height_px))
  {
    const Tile * tile = tiles.get(index);
    if (tile == nullptr) {
      return false;
    }
    const std::optional<PixelWindow> placed =
      lattice.window_of_geotransform(tile->geotransform(), tile->width(), tile->height());
    // Never empty: the tile was chosen because the window touches it.
    const std::optional<PixelWindow> needed =
      intersection(window, tile_window(index, layer.tile_width_px, layer.tile_height_px));
    if (!placed.has_value() || !needed.has_value() || !placed->contains(*needed)) {
      return false;
    }
    const TilePiece piece{
      static_cast<int>(needed->col_min - placed->col_min),
      static_cast<int>(needed->row_min - placed->row_min),
      static_cast<int>(needed->width),
      static_cast<int>(needed->height),
      static_cast<std::size_t>(needed->col_min - window.col_min),
      static_cast<std::size_t>(needed->row_min - window.row_min)};
    if (!read(*tile, piece)) {
      return false;
    }
  }
  return true;
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
    if (manifest.wgs.has_value()) {
      wgs_height_tiles.emplace(dataset_dir, manifest.wgs->height.tile_pattern,
        options.tile_cache_size);
      if (manifest.wgs->rgb.has_value()) {
        wgs_rgb_tiles.emplace(dataset_dir, manifest.wgs->rgb->tile_pattern,
          options.tile_cache_size);
      }
    }

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

  std::optional<HeightGrid> height_grid(const Roi & roi)
  {
    if (!manifest.wgs.has_value()) {
      return std::nullopt;
    }
    const WgsManifest & wgs = *manifest.wgs;
    const Lattice lattice{wgs.origin_lon, wgs.origin_lat, wgs.height.resolution_deg};
    const std::optional<PixelWindow> window = roi_window(lattice, roi);
    if (!window.has_value()) {
      return std::nullopt;
    }

    HeightGrid grid;
    grid.geo = georeference(lattice, *window);
    const std::size_t width = grid.geo.width;
    grid.heights.resize(width * grid.geo.height);

    const bool complete = stitch(
      *wgs_height_tiles, lattice, wgs.height, *window,
      [&](const Tile & tile, const TilePiece & piece) {
        float * out = grid.heights.data() + piece.out_row * width + piece.out_col;
        return tile.read_floats(
          piece.tile_col, piece.tile_row, piece.width, piece.height, out,
          static_cast<std::ptrdiff_t>(width));
      });
    if (!complete) {
      return std::nullopt;
    }

    const float wgs_nodata = static_cast<float>(wgs.height_nodata);
    for (float & height : grid.heights) {
      if (height == wgs_nodata) {
        height = std::numeric_limits<float>::quiet_NaN();
      }
    }
    return grid;
  }

  std::optional<RgbGrid> rgb_grid(const Roi & roi)
  {
    if (!manifest.wgs.has_value() || !manifest.wgs->rgb.has_value()) {
      return std::nullopt;
    }
    const WgsManifest & wgs = *manifest.wgs;
    const Lattice lattice{wgs.origin_lon, wgs.origin_lat, wgs.rgb->resolution_deg};
    const std::optional<PixelWindow> window = roi_window(lattice, roi);
    if (!window.has_value()) {
      return std::nullopt;
    }

    RgbGrid grid;
    grid.geo = georeference(lattice, *window);
    const std::size_t width = grid.geo.width;
    grid.rgb.resize(3 * width * grid.geo.height);
    grid.valid.resize(width * grid.geo.height);

    const bool complete = stitch(
      *wgs_rgb_tiles, lattice, *wgs.rgb, *window,
      [&](const Tile & tile, const TilePiece & piece) {
        const std::size_t at = piece.out_row * width + piece.out_col;
        return tile.read_rgb(
          piece.tile_col, piece.tile_row, piece.width, piece.height, grid.rgb.data() + 3 * at,
          static_cast<std::ptrdiff_t>(3 * width)) &&
               tile.read_mask(
          piece.tile_col, piece.tile_row, piece.width, piece.height, grid.valid.data() + at,
          static_cast<std::ptrdiff_t>(width));
      });
    if (!complete) {
      return std::nullopt;
    }

    // Lossy compression leaves arbitrary values under the mask; report black.
    for (std::size_t at = 0; at < grid.valid.size(); ++at) {
      if (grid.valid[at] == 0) {
        grid.rgb[3 * at] = grid.rgb[3 * at + 1] = grid.rgb[3 * at + 2] = 0;
      }
    }
    return grid;
  }

  std::filesystem::path dataset_dir;
  Manifest manifest;
  TileCache tiles;
  /// Set when the dataset has EPSG:4326 height (and RGB) tiles.
  std::optional<TileCache> wgs_height_tiles;
  std::optional<TileCache> wgs_rgb_tiles;
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

std::optional<HeightGrid> HeightSampler::height_grid(const Roi & roi) const
{
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  try {
    return impl_->height_grid(roi);
  } catch (const std::exception &) {
    // An unreadable tile, or a grid too large to allocate, is no grid.
    return std::nullopt;
  }
}

std::optional<RgbGrid> HeightSampler::rgb_grid(const Roi & roi) const
{
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  try {
    return impl_->rgb_grid(roi);
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

bool HeightSampler::has_height_grids() const noexcept
{
  return impl_->manifest.wgs.has_value();
}

bool HeightSampler::has_rgb_grids() const noexcept
{
  return impl_->manifest.wgs.has_value() && impl_->manifest.wgs->rgb.has_value();
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
