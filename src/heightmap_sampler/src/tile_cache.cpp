#include "tile_cache.hpp"

#include <gdal.h>
#include <gdal_priv.h>

#include <cmath>
#include <mutex>
#include <utility>

#include "heightmap_sampler/height_sampler.hpp"
#include "manifest.hpp"

namespace heightmap_sampler
{
namespace
{

void register_gdal_once()
{
  static std::once_flag flag;
  std::call_once(flag, [] {GDALAllRegister();});
}

}  // namespace

Tile::Tile(GDALDataset * dataset, const std::string & path)
: dataset_(dataset), path_(path)
{
  width_ = dataset_->GetRasterXSize();
  height_ = dataset_->GetRasterYSize();

  double transform[6];
  if (dataset_->GetGeoTransform(transform) != CE_None) {
    GDALClose(dataset_);
    dataset_ = nullptr;
    throw DatasetError("tile " + path_ + " has no geotransform");
  }
  if (!GDALInvGeoTransform(transform, inv_transform_)) {
    GDALClose(dataset_);
    dataset_ = nullptr;
    throw DatasetError("tile " + path_ + " has a non-invertible geotransform");
  }
  if (dataset_->GetRasterCount() < 1) {
    GDALClose(dataset_);
    dataset_ = nullptr;
    throw DatasetError("tile " + path_ + " has no raster band");
  }
}

Tile::~Tile()
{
  if (dataset_ != nullptr) {
    GDALClose(dataset_);
  }
}

std::optional<float> Tile::value_at(double x, double y) const
{
  // The inverse geotransform yields pixel-corner coordinates, so flooring gives
  // the pixel containing the point (specification section 23).
  double col_corner = 0.0;
  double row_corner = 0.0;
  GDALApplyGeoTransform(const_cast<double *>(inv_transform_), x, y, &col_corner, &row_corner);

  const double col = std::floor(col_corner);
  const double row = std::floor(row_corner);
  if (col < 0.0 || col >= width_ || row < 0.0 || row >= height_) {
    return std::nullopt;
  }

  // One pixel at a time, as the reference sampler does.  GDAL's block cache
  // turns the four corners of a sample into a single block decompression.
  float value = 0.0F;
  GDALRasterBand * band = dataset_->GetRasterBand(1);
  const CPLErr status = band->RasterIO(
    GF_Read, static_cast<int>(col), static_cast<int>(row), 1, 1,
    &value, 1, 1, GDT_Float32, 0, 0, nullptr);
  if (status != CE_None) {
    return std::nullopt;
  }
  return value;
}

TileCache::TileCache(std::filesystem::path root, std::string tile_pattern, std::size_t capacity)
: root_(std::move(root)),
  tile_pattern_(std::move(tile_pattern)),
  capacity_(capacity > 0 ? capacity : 1)
{
  register_gdal_once();
}

void TileCache::touch(Entry & entry, TileIndex index)
{
  if (entry.tile == nullptr) {
    return;
  }
  recency_.erase(entry.recency);
  recency_.push_front(index);
  entry.recency = recency_.begin();
}

void TileCache::evict_if_needed()
{
  while (recency_.size() > capacity_) {
    const TileIndex victim = recency_.back();
    recency_.pop_back();
    // Drop the whole entry rather than leaving a null one behind: a null entry
    // means "this tile does not exist", which is not what happened here.
    entries_.erase(victim);
  }
}

const Tile * TileCache::get(TileIndex index)
{
  const auto found = entries_.find(index);
  if (found != entries_.end()) {
    touch(found->second, index);
    return found->second.tile.get();
  }

  const std::filesystem::path path = root_ / render_tile_path(tile_pattern_, index);

  std::error_code code;
  if (!std::filesystem::is_regular_file(path, code)) {
    // Remember the miss; a dataset's coverage does not change under us.
    entries_.emplace(index, Entry{nullptr, recency_.end()});
    return nullptr;
  }

  GDALDataset * dataset = static_cast<GDALDataset *>(
    GDALOpenEx(path.string().c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr,
    nullptr));
  if (dataset == nullptr) {
    throw DatasetError("could not open tile " + path.string() + ": " + CPLGetLastErrorMsg());
  }

  auto tile = std::make_unique<Tile>(dataset, path.string());  // takes ownership
  recency_.push_front(index);
  Entry entry{std::move(tile), recency_.begin()};
  const Tile * raw = entry.tile.get();
  entries_.emplace(index, std::move(entry));
  evict_if_needed();
  return raw;
}

}  // namespace heightmap_sampler
