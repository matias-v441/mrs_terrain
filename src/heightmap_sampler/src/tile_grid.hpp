// The deterministic global tile grid, ported from src/heightmap_prep/tiling.py.
//
// Tiles sit on one fixed grid in the stored horizontal CRS rather than on a
// per-world grid, so a tile index is computable directly from projected
// coordinates (specification sections 9.4 and 12):
//
//   * origin_x is the *west* edge of tile (0, 0)
//   * origin_y is the *north* edge of tile (0, 0)
//   * ix increases eastward, iy increases southward

#ifndef TILE_GRID_HPP_
#define TILE_GRID_HPP_

#include <cmath>
#include <cstdint>
#include <functional>

namespace heightmap_sampler
{

struct TileIndex
{
  std::int64_t ix = 0;
  std::int64_t iy = 0;

  friend bool operator==(TileIndex a, TileIndex b) {return a.ix == b.ix && a.iy == b.iy;}
  friend bool operator!=(TileIndex a, TileIndex b) {return !(a == b);}
};

struct TileIndexHash
{
  std::size_t operator()(TileIndex tile) const noexcept
  {
    const std::size_t h1 = std::hash<std::int64_t>{}(tile.ix);
    const std::size_t h2 = std::hash<std::int64_t>{}(tile.iy);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

/// An axis-aligned rectangle in the stored horizontal CRS, in projected units.
struct ProjectedBounds
{
  double west = 0.0;
  double south = 0.0;
  double east = 0.0;
  double north = 0.0;
};

struct TileGrid
{
  double origin_x = 0.0;
  double origin_y = 0.0;
  double resolution_x = 0.0;
  double resolution_y = 0.0;
  std::int64_t tile_width_px = 0;
  std::int64_t tile_height_px = 0;

  /// Tile width in projected units.
  double tile_span_x() const {return static_cast<double>(tile_width_px) * resolution_x;}

  /// Tile height in projected units.
  double tile_span_y() const {return static_cast<double>(tile_height_px) * resolution_y;}

  /// The tile containing projected point (x, y).
  TileIndex index_for_point(double x, double y) const
  {
    return TileIndex{
      static_cast<std::int64_t>(std::floor((x - origin_x) / tile_span_x())),
      static_cast<std::int64_t>(std::floor((origin_y - y) / tile_span_y()))};
  }

  /// The projected extent covered by \p tile.
  ProjectedBounds tile_bounds(TileIndex tile) const
  {
    const double west = origin_x + static_cast<double>(tile.ix) * tile_span_x();
    const double north = origin_y - static_cast<double>(tile.iy) * tile_span_y();
    return ProjectedBounds{west, north - tile_span_y(), west + tile_span_x(), north};
  }
};

}  // namespace heightmap_sampler

#endif  // TILE_GRID_HPP_
