// A bounded cache of open tile rasters.
//
// examples/sampler.py caches every tile it has ever touched, including the
// misses, and never lets go.  That is fine for a one-shot script but not for a
// long-lived node over a region-sized dataset, so this cache keeps the miss
// caching (a missing tile is a normal, permanent fact about the dataset) and
// adds LRU eviction of the open rasters.
//
// Not thread-safe on its own; HeightSampler serialises access.

#ifndef TILE_CACHE_HPP_
#define TILE_CACHE_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "tile_grid.hpp"

class GDALDataset;

namespace heightmap_sampler
{

/// One open tile, with everything a pixel lookup needs.
class Tile
{
public:
  Tile(GDALDataset * dataset, const std::string & path);
  ~Tile();

  Tile(const Tile &) = delete;
  Tile & operator=(const Tile &) = delete;

  /// Value of the pixel containing projected (x, y), or nothing when the point
  /// falls outside this raster.  The NoData check is the caller's.
  std::optional<float> value_at(double x, double y) const;

  int width() const {return width_;}
  int height() const {return height_;}

  /// The tile's GDAL geotransform, six coefficients.
  const double * geotransform() const {return transform_;}

  // Block reads for region queries.  Each reads the \p width x \p height
  // pixels at (\p col, \p row) of this raster into \p out, whose rows are
  // \p line_stride elements apart, and returns false if GDAL fails.

  /// Band 1 as float32.
  bool read_floats(
    int col, int row, int width, int height, float * out,
    std::ptrdiff_t line_stride) const;

  /// Bands 1-3, pixel-interleaved RGB; false if the raster has fewer bands.
  bool read_rgb(
    int col, int row, int width, int height, std::uint8_t * out,
    std::ptrdiff_t line_stride) const;

  /// Band 1's mask: 255 where there is data, 0 where there is none.
  bool read_mask(
    int col, int row, int width, int height, std::uint8_t * out,
    std::ptrdiff_t line_stride) const;

private:
  GDALDataset * dataset_;
  std::string path_;
  int width_ = 0;
  int height_ = 0;
  double transform_[6] = {0, 0, 0, 0, 0, 0};
  // Inverse of the tile's own geotransform.  Taken from the file rather than
  // recomputed from the grid because heightmap-prep permits clipped edge tiles
  // that are smaller than a full tile while keeping the same origin.
  double inv_transform_[6] = {0, 0, 0, 0, 0, 0};
};

class TileCache
{
public:
  TileCache(std::filesystem::path root, std::string tile_pattern, std::size_t capacity);

  /// The tile covering \p index, or nullptr when the dataset has no such tile.
  /// Both outcomes are remembered.
  const Tile * get(TileIndex index);

private:
  struct Entry
  {
    std::unique_ptr<Tile> tile;                 ///< null for a known-absent tile
    std::list<TileIndex>::iterator recency;     ///< only valid when tile is set
  };

  void touch(Entry & entry, TileIndex index);
  void evict_if_needed();

  std::filesystem::path root_;
  std::string tile_pattern_;
  std::size_t capacity_;
  std::unordered_map<TileIndex, Entry, TileIndexHash> entries_;
  std::list<TileIndex> recency_;  ///< front = most recently used
};

}  // namespace heightmap_sampler

#endif  // TILE_CACHE_HPP_
