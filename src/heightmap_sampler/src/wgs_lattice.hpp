// The WGS lattices, ported from src/heightmap_prep/wgs.py.
//
// Besides the EPSG:5514 tiles point sampling reads, a dataset holds EPSG:4326
// tiles of heights and imagery.  Each layer is cut from one global lattice of
// square-degree pixels anchored at (origin_lon, origin_lat), the west/north
// corner of lattice pixel (0, 0):
//
//   * columns increase eastward, rows increase southward
//   * tile (ix, iy) holds lattice pixels [ix * tile_w, (ix + 1) * tile_w) by
//     [iy * tile_h, (iy + 1) * tile_h), so tiles never overlap
//
// A region of interest given in latitude and longitude is therefore an
// axis-aligned block of lattice pixels.

#ifndef WGS_LATTICE_HPP_
#define WGS_LATTICE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "tile_grid.hpp"

namespace heightmap_sampler
{

/// A block of lattice pixels; col_min/row_min count from the lattice origin.
struct PixelWindow
{
  std::int64_t col_min = 0;
  std::int64_t row_min = 0;
  std::int64_t width = 0;
  std::int64_t height = 0;

  /// One past the last column.
  std::int64_t col_end() const {return col_min + width;}
  /// One past the last row.
  std::int64_t row_end() const {return row_min + height;}

  bool contains(const PixelWindow & other) const
  {
    return col_min <= other.col_min && row_min <= other.row_min &&
           other.col_end() <= col_end() && other.row_end() <= row_end();
  }

  friend bool operator==(const PixelWindow & a, const PixelWindow & b)
  {
    return a.col_min == b.col_min && a.row_min == b.row_min && a.width == b.width &&
           a.height == b.height;
  }
};

/// The overlap of two windows, or nothing when they do not overlap.
inline std::optional<PixelWindow> intersection(const PixelWindow & a, const PixelWindow & b)
{
  const std::int64_t col_min = std::max(a.col_min, b.col_min);
  const std::int64_t row_min = std::max(a.row_min, b.row_min);
  const std::int64_t col_end = std::min(a.col_end(), b.col_end());
  const std::int64_t row_end = std::min(a.row_end(), b.row_end());
  if (col_end <= col_min || row_end <= row_min) {
    return std::nullopt;
  }
  return PixelWindow{col_min, row_min, col_end - col_min, row_end - row_min};
}

/// Floor division, correct for negative numerators.
inline std::int64_t floor_div(std::int64_t numerator, std::int64_t denominator)
{
  const std::int64_t quotient = numerator / denominator;
  const bool inexact = quotient * denominator != numerator;
  return inexact && ((numerator < 0) != (denominator < 0)) ? quotient - 1 : quotient;
}

/// A global EPSG:4326 lattice of square-degree pixels.
struct Lattice
{
  double origin_lon = 0.0;
  double origin_lat = 0.0;
  double resolution_deg = 0.0;

  /// Fractional column of \p longitude, measured in pixel-centre coordinates:
  /// column c's centre sits at exactly c.
  double centre_col(double longitude) const
  {
    return (longitude - origin_lon) / resolution_deg - 0.5;
  }

  /// Fractional row of \p latitude, in pixel-centre coordinates.
  double centre_row(double latitude) const
  {
    return (origin_lat - latitude) / resolution_deg - 0.5;
  }

  /// Every pixel that pixel-centre bilinear interpolation reads for some point
  /// of the rectangle [west, east] x [south, north], and no other.
  ///
  /// A point reads the pixels floor(c) and floor(c) + 1 along each axis, where
  /// c is its centre_col / centre_row -- the rule HeightSampler::sample uses on
  /// the EPSG:5514 grid.  Both are monotonic in the coordinate, so the corners
  /// of the rectangle bound the window.  Returns nothing for an inverted or
  /// non-finite rectangle.
  std::optional<PixelWindow> bilinear_window(
    double west, double south, double east, double north) const
  {
    if (!(std::isfinite(west) && std::isfinite(south) && std::isfinite(east) &&
      std::isfinite(north)) || west > east || south > north)
    {
      return std::nullopt;
    }
    const double col_first = std::floor(centre_col(west));
    const double col_last = std::floor(centre_col(east)) + 1.0;
    const double row_first = std::floor(centre_row(north));
    const double row_last = std::floor(centre_row(south)) + 1.0;
    return PixelWindow{
      static_cast<std::int64_t>(col_first), static_cast<std::int64_t>(row_first),
      static_cast<std::int64_t>(col_last - col_first) + 1,
      static_cast<std::int64_t>(row_last - row_first) + 1};
  }

  /// Longitude of the west edge of column \p col.
  double west_of(std::int64_t col) const
  {
    return origin_lon + static_cast<double>(col) * resolution_deg;
  }

  /// Latitude of the north edge of row \p row.
  double north_of(std::int64_t row) const
  {
    return origin_lat - static_cast<double>(row) * resolution_deg;
  }

  /// The window a raster of \p width x \p height px with GDAL geotransform
  /// \p transform (six coefficients) occupies, or nothing if it is not on this
  /// lattice.
  /// The port of Lattice.window_of_transform, with its tolerances.
  std::optional<PixelWindow> window_of_geotransform(
    const double * transform, std::int64_t width, std::int64_t height) const
  {
    const auto close = [](double a, double b) {
        return std::abs(a - b) <= 1e-9 * std::max(std::abs(a), std::abs(b));
      };
    if (!close(transform[1], resolution_deg) || !close(transform[5], -resolution_deg) ||
      transform[2] != 0.0 || transform[4] != 0.0)
    {
      return std::nullopt;
    }
    const double col = (transform[0] - origin_lon) / resolution_deg;
    const double row = (origin_lat - transform[3]) / resolution_deg;
    if (std::abs(col - std::round(col)) > 1e-4 || std::abs(row - std::round(row)) > 1e-4) {
      return std::nullopt;
    }
    return PixelWindow{
      static_cast<std::int64_t>(std::llround(col)), static_cast<std::int64_t>(std::llround(row)),
      width, height};
  }
};

/// The window tile \p tile occupies, for tiles of \p tile_width x \p tile_height px.
inline PixelWindow tile_window(TileIndex tile, std::int64_t tile_width, std::int64_t tile_height)
{
  return PixelWindow{tile.ix * tile_width, tile.iy * tile_height, tile_width, tile_height};
}

/// The tiles \p window touches, in reading order.
inline std::vector<TileIndex> tiles_for_window(
  const PixelWindow & window, std::int64_t tile_width, std::int64_t tile_height)
{
  std::vector<TileIndex> tiles;
  for (std::int64_t iy = floor_div(window.row_min, tile_height);
    iy <= floor_div(window.row_end() - 1, tile_height); ++iy)
  {
    for (std::int64_t ix = floor_div(window.col_min, tile_width);
      ix <= floor_div(window.col_end() - 1, tile_width); ++ix)
    {
      tiles.push_back(TileIndex{ix, iy});
    }
  }
  return tiles;
}

}  // namespace heightmap_sampler

#endif  // WGS_LATTICE_HPP_
