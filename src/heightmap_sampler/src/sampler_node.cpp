#include "sampler_node.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

namespace heightmap_sampler
{
namespace
{

/// Grids are georeferenced in EPSG:4326 by the response's own fields; the
/// frame only names that.
constexpr const char * kGridFrame = "wgs84";

Roi to_roi(const geographic_msgs::msg::BoundingBox & box)
{
  return Roi{
    LatLon{box.max_pt.latitude, box.min_pt.longitude},
    LatLon{box.min_pt.latitude, box.max_pt.longitude}};
}

/// Why a valid-looking request got no grid: the region itself, or the dataset.
std::string no_grid_reason(const Roi & roi, const char * tiles)
{
  const bool ordered = roi.top_left.latitude >= roi.bottom_right.latitude &&
    roi.top_left.longitude <= roi.bottom_right.longitude;
  const bool on_globe = roi.top_left.latitude <= 90.0 && roi.bottom_right.latitude >= -90.0 &&
    roi.top_left.longitude >= -180.0 && roi.bottom_right.longitude <= 180.0;
  if (!ordered || !on_globe) {
    return "invalid region: min_pt must be south-west of max_pt, within the globe";
  }
  return std::string("the dataset's ") + tiles + " tiles do not cover the region";
}

bool host_is_big_endian()
{
  const std::uint16_t probe = 1;
  std::uint8_t first = 0;
  std::memcpy(&first, &probe, 1);
  return first == 0;
}

template<typename Response>
void set_georeference(Response & response, const GridGeoreference & geo)
{
  response.west = geo.west;
  response.north = geo.north;
  response.resolution_deg = geo.resolution_deg;
}

}  // namespace

SamplerNode::SamplerNode(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("heightmap_sampler", node_options)
{
  // Empty means "the dataset installed into this package's share directory",
  // which is the normal case; an override exists mostly for development.
  const auto dataset_dir = declare_parameter<std::string>("dataset_dir", "");
  const auto service_name = declare_parameter<std::string>("service_name", "~/sample_height");
  const auto height_grid_service_name =
    declare_parameter<std::string>("height_grid_service_name", "~/get_height_grid");
  const auto rgb_grid_service_name =
    declare_parameter<std::string>("rgb_grid_service_name", "~/get_rgb_grid");
  const auto tile_cache_size = declare_parameter<int>("tile_cache_size", 16);

  Options options;
  options.dataset_dir = dataset_dir;
  options.tile_cache_size = tile_cache_size > 0 ? static_cast<std::size_t>(tile_cache_size) : 1;

  try {
    sampler_ = std::make_unique<HeightSampler>(std::move(options));
  } catch (const DatasetError & error) {
    RCLCPP_FATAL(get_logger(), "could not load the heightmap dataset: %s", error.what());
    throw;
  }

  const DatasetInfo & info = sampler_->info();
  RCLCPP_INFO(
    get_logger(),
    "dataset %s: %.3g x %.3g %s, stored in %s, heights in %s (%s)",
    sampler_->dataset_dir().c_str(), info.resolution_x, info.resolution_y, info.unit.c_str(),
    info.horizontal_crs.c_str(), info.vertical_crs.c_str(), info.vertical_datum.c_str());

  service_ = create_service<SampleHeight>(
    service_name,
    [this](
      const std::shared_ptr<SampleHeight::Request> request,
      std::shared_ptr<SampleHeight::Response> response) {
      handle_sample(request, response);
    });

  RCLCPP_INFO(get_logger(), "serving height samples on %s", service_->get_service_name());

  height_grid_service_ = create_service<GetHeightGrid>(
    height_grid_service_name,
    [this](
      const std::shared_ptr<GetHeightGrid::Request> request,
      std::shared_ptr<GetHeightGrid::Response> response) {
      handle_height_grid(request, response);
    });
  rgb_grid_service_ = create_service<GetRgbGrid>(
    rgb_grid_service_name,
    [this](
      const std::shared_ptr<GetRgbGrid::Request> request,
      std::shared_ptr<GetRgbGrid::Response> response) {
      handle_rgb_grid(request, response);
    });
  RCLCPP_INFO(
    get_logger(), "serving region heights on %s and imagery on %s",
    height_grid_service_->get_service_name(), rgb_grid_service_->get_service_name());
}

void SamplerNode::handle_sample(
  const std::shared_ptr<SampleHeight::Request> request,
  std::shared_ptr<SampleHeight::Response> response)
{
  std::vector<LatLon> points;
  points.reserve(request->points.size());
  for (const auto & point : request->points) {
    points.push_back(LatLon{point.latitude, point.longitude});
  }

  const std::vector<std::optional<double>> heights = sampler_->sample(points);

  response->heights.reserve(heights.size());
  response->valid.reserve(heights.size());
  std::size_t unavailable = 0;
  for (const std::optional<double> & height : heights) {
    response->valid.push_back(height.has_value());
    response->heights.push_back(
      height.value_or(std::numeric_limits<double>::quiet_NaN()));
    unavailable += height.has_value() ? 0 : 1;
  }

  // Points outside the dataset are a normal answer, not a request failure.
  response->success = true;
  response->message = unavailable == 0 ?
    "" :
    std::to_string(unavailable) + " of " + std::to_string(heights.size()) +
    " points have no prepared height";
}

void SamplerNode::handle_height_grid(
  const std::shared_ptr<GetHeightGrid::Request> request,
  std::shared_ptr<GetHeightGrid::Response> response)
{
  const Roi roi = to_roi(request->roi);
  std::optional<HeightGrid> grid = sampler_->height_grid(roi);
  if (!grid.has_value()) {
    response->success = false;
    response->message = sampler_->has_height_grids() ?
      no_grid_reason(roi, "height") :
      "the dataset holds no EPSG:4326 height tiles";
    return;
  }

  set_georeference(*response, grid->geo);
  sensor_msgs::msg::Image & image = response->grid;
  image.header.stamp = now();
  image.header.frame_id = kGridFrame;
  image.width = static_cast<std::uint32_t>(grid->geo.width);
  image.height = static_cast<std::uint32_t>(grid->geo.height);
  image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  image.is_bigendian = host_is_big_endian();   // float32 in host order
  image.step = static_cast<std::uint32_t>(grid->geo.width * sizeof(float));
  image.data.resize(grid->heights.size() * sizeof(float));
  std::memcpy(image.data.data(), grid->heights.data(), image.data.size());

  response->success = true;
  std::size_t missing = 0;
  for (const float height : grid->heights) {
    missing += std::isnan(height) ? 1 : 0;
  }
  response->message = missing == 0 ?
    "" :
    std::to_string(missing) + " of " + std::to_string(grid->heights.size()) +
    " pixels have no prepared height";
}

void SamplerNode::handle_rgb_grid(
  const std::shared_ptr<GetRgbGrid::Request> request,
  std::shared_ptr<GetRgbGrid::Response> response)
{
  const Roi roi = to_roi(request->roi);
  std::optional<RgbGrid> grid = sampler_->rgb_grid(roi);
  if (!grid.has_value()) {
    response->success = false;
    response->message = sampler_->has_rgb_grids() ?
      no_grid_reason(roi, "RGB") :
      "the dataset holds no EPSG:4326 imagery";
    return;
  }

  set_georeference(*response, grid->geo);
  sensor_msgs::msg::Image & image = response->grid;
  image.header.stamp = now();
  image.header.frame_id = kGridFrame;
  image.width = static_cast<std::uint32_t>(grid->geo.width);
  image.height = static_cast<std::uint32_t>(grid->geo.height);
  image.encoding = sensor_msgs::image_encodings::RGBA8;
  image.is_bigendian = false;
  image.step = static_cast<std::uint32_t>(grid->geo.width * 4);
  const std::size_t pixels = grid->valid.size();
  image.data.resize(4 * pixels);
  std::size_t missing = 0;
  for (std::size_t at = 0; at < pixels; ++at) {
    image.data[4 * at] = grid->rgb[3 * at];
    image.data[4 * at + 1] = grid->rgb[3 * at + 1];
    image.data[4 * at + 2] = grid->rgb[3 * at + 2];
    image.data[4 * at + 3] = grid->valid[at];
    missing += grid->valid[at] == 0 ? 1 : 0;
  }

  response->success = true;
  response->message = missing == 0 ?
    "" :
    std::to_string(missing) + " of " + std::to_string(pixels) + " pixels have no imagery";
}

}  // namespace heightmap_sampler

RCLCPP_COMPONENTS_REGISTER_NODE(heightmap_sampler::SamplerNode)
