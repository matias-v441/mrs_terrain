// Runtime sampling of a dataset prepared by heightmap-prep.
//
// This is the C++ counterpart of examples/sampler.py: one horizontal
// transformation from WGS84, a tile index computed from the manifest grid, and
// bilinear interpolation of the stored heights.  No vertical datum
// transformation happens at runtime -- the stored values are already in the
// dataset's target vertical datum (EGM96 for a dataset prepared with
// --vertical-datum egm96).
//
// The dataset location is resolved by the library, not by the caller:
//
//     heightmap_sampler::HeightSampler sampler;
//     if (auto height = sampler.sample(49.3625695, 14.2619165)) {
//       use(*height);
//     }
//
// finds the dataset installed into this package's share directory.  Which
// dataset that is, is decided when this package is built (see README.md).

#ifndef HEIGHTMAP_SAMPLER__HEIGHT_SAMPLER_HPP_
#define HEIGHTMAP_SAMPLER__HEIGHT_SAMPLER_HPP_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace heightmap_sampler
{

/// A WGS84 geographic coordinate, in degrees.
struct LatLon
{
  double latitude;
  double longitude;
};

/// What the dataset manifest says about the values being returned.
struct DatasetInfo
{
  std::string horizontal_crs;   ///< CRS the tiles are stored in, e.g. "EPSG:5514".
  std::string vertical_crs;     ///< CRS of the returned heights, e.g. "EPSG:5773".
  std::string vertical_datum;   ///< e.g. "egm96".
  std::string unit;             ///< e.g. "m".
  double resolution_x = 0.0;    ///< Ground sample distance, projected units.
  double resolution_y = 0.0;
};

struct Options
{
  /// Where the dataset lives.  Empty means this package's share directory,
  /// which is what almost every caller wants.
  std::filesystem::path dataset_dir;

  /// How many tiles to keep open at once.  Each open tile costs a file
  /// descriptor plus GDAL block cache; sampling a trajectory rarely touches
  /// more than a handful.
  std::size_t tile_cache_size = 16;
};

/// Thrown by HeightSampler's constructors when the dataset cannot be used.
class DatasetError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

class HeightSampler
{
public:
  /// Sample the dataset installed into this package's share directory.
  HeightSampler();

  /// Sample a dataset at an explicit location, for tests and for callers that
  /// really do manage datasets themselves.
  explicit HeightSampler(std::filesystem::path dataset_dir);

  explicit HeightSampler(Options options);

  ~HeightSampler();

  HeightSampler(HeightSampler &&) noexcept;
  HeightSampler & operator=(HeightSampler &&) noexcept;
  HeightSampler(const HeightSampler &) = delete;
  HeightSampler & operator=(const HeightSampler &) = delete;

  /// Bilinearly interpolated height at a WGS84 coordinate, in the dataset's
  /// vertical datum.
  ///
  /// Returns nothing when the covering tile is absent, a pixel involved in the
  /// interpolation is NoData, or the point falls outside the raster.  Never
  /// throws.  Safe to call from several threads.
  std::optional<double> sample(double latitude, double longitude) const;

  /// Batch form; the result is parallel to \p points.
  std::vector<std::optional<double>> sample(const std::vector<LatLon> & points) const;

  const DatasetInfo & info() const noexcept;

  /// The directory the dataset was actually loaded from.
  const std::filesystem::path & dataset_dir() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// The dataset directory a default-constructed HeightSampler uses.
/// Throws DatasetError if this package's share directory cannot be located.
std::filesystem::path default_dataset_dir();

}  // namespace heightmap_sampler

#endif  // HEIGHTMAP_SAMPLER__HEIGHT_SAMPLER_HPP_
