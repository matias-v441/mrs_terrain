#include "sampler_node.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp_components/register_node_macro.hpp>

namespace heightmap_sampler
{

SamplerNode::SamplerNode(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("heightmap_sampler", node_options)
{
  // Empty means "the dataset installed into this package's share directory",
  // which is the normal case; an override exists mostly for development.
  const auto dataset_dir = declare_parameter<std::string>("dataset_dir", "");
  const auto service_name = declare_parameter<std::string>("service_name", "~/sample_height");
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

}  // namespace heightmap_sampler

RCLCPP_COMPONENTS_REGISTER_NODE(heightmap_sampler::SamplerNode)
