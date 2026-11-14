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
//
// A region of interest -- an axis-aligned latitude/longitude rectangle -- can
// also be read whole, as heights or imagery stitched from the dataset's
// EPSG:4326 tiles:
//
//     heightmap_sampler::Roi roi{{49.3630, 14.2610}, {49.3620, 14.2625}};
//     if (auto grid = sampler.height_grid(roi)) {
//       use(grid->geo, grid->heights);
//     }

#ifndef HEIGHTMAP_SAMPLER__HEIGHT_SAMPLER_HPP_
#define HEIGHTMAP_SAMPLER__HEIGHT_SAMPLER_HPP_

#include <cstddef>
#include <cstdint>
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

/// An axis-aligned WGS84 rectangle.  Valid when \p top_left is no further
/// south or east than \p bottom_right; a single point is a valid rectangle.
struct Roi
{
  LatLon top_left;       ///< north-west corner
  LatLon bottom_right;   ///< south-east corner
};

/// Where a returned grid sits: a north-up block of square-degree pixels in
/// EPSG:4326.  Pixel (col, row) spans longitudes
/// [west + col * resolution_deg, west + (col + 1) * resolution_deg] and
/// latitudes [north - (row + 1) * resolution_deg, north - row * resolution_deg];
/// its value belongs to its centre.
struct GridGeoreference
{
  double west = 0.0;             ///< West edge of the grid, degrees.
  double north = 0.0;            ///< North edge of the grid, degrees.
  double resolution_deg = 0.0;   ///< Pixel size along both axes, degrees.
  std::size_t width = 0;         ///< Columns, west to east.
  std::size_t height = 0;        ///< Rows, north to south.
};

/// Heights over a region of interest.
struct HeightGrid
{
  GridGeoreference geo;
  /// Row-major, row 0 northmost; NaN where the dataset has no height.  In the
  /// dataset's vertical datum, like HeightSampler::sample.
  std::vector<float> heights;
};

/// Orthophoto imagery over a region of interest.
struct RgbGrid
{
  GridGeoreference geo;
  /// Row-major, row 0 northmost, three bytes (red, green, blue) per pixel.
  std::vector<std::uint8_t> rgb;
  /// One byte per pixel: 255 where there is imagery, 0 where there is none.
  std::vector<std::uint8_t> valid;
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

  /// The heights over \p roi, stitched from the dataset's EPSG:4326 height
  /// tiles.
  ///
  /// The grid holds exactly the pixels that bilinear interpolation between
  /// pixel centres reads for some point of \p roi, so any point of it -- the
  /// two corners included -- can be interpolated from the grid alone.  A pixel
  /// without a height is NaN.
  ///
  /// Returns nothing when \p roi is invalid or not on the globe, when the
  /// dataset has no EPSG:4326 tiles, or when a tile the grid needs is absent or
  /// does not cover its part of the grid.  Never throws.  Safe to call from
  /// several threads.
  std::optional<HeightGrid> height_grid(const Roi & roi) const;

  /// The orthophoto over \p roi, stitched from the dataset's EPSG:4326 RGB
  /// tiles, on the RGB lattice and by the same rule as height_grid.  Returns
  /// nothing in the same cases, and when the dataset has no imagery.
  std::optional<RgbGrid> rgb_grid(const Roi & roi) const;

  /// Whether the dataset has the EPSG:4326 tiles height_grid reads.
  bool has_height_grids() const noexcept;

  /// Whether the dataset has the EPSG:4326 imagery rgb_grid reads.
  bool has_rgb_grids() const noexcept;

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
