#ifndef SAMPLER_NODE_HPP_
#define SAMPLER_NODE_HPP_

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "heightmap_sampler/height_sampler.hpp"
#include "heightmap_sampler_msgs/srv/get_height_grid.hpp"
#include "heightmap_sampler_msgs/srv/get_rgb_grid.hpp"
#include "heightmap_sampler_msgs/srv/sample_height.hpp"

namespace heightmap_sampler
{

/// Exposes a prepared heightmap dataset over services: point heights, and the
/// heights and imagery of a region of interest.
class SamplerNode : public rclcpp::Node
{
public:
  explicit SamplerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  using SampleHeight = heightmap_sampler_msgs::srv::SampleHeight;
  using GetHeightGrid = heightmap_sampler_msgs::srv::GetHeightGrid;
  using GetRgbGrid = heightmap_sampler_msgs::srv::GetRgbGrid;

  void handle_sample(
    const std::shared_ptr<SampleHeight::Request> request,
    std::shared_ptr<SampleHeight::Response> response);

  void handle_height_grid(
    const std::shared_ptr<GetHeightGrid::Request> request,
    std::shared_ptr<GetHeightGrid::Response> response);

  void handle_rgb_grid(
    const std::shared_ptr<GetRgbGrid::Request> request,
    std::shared_ptr<GetRgbGrid::Response> response);

  std::unique_ptr<HeightSampler> sampler_;
  rclcpp::Service<SampleHeight>::SharedPtr service_;
  rclcpp::Service<GetHeightGrid>::SharedPtr height_grid_service_;
  rclcpp::Service<GetRgbGrid>::SharedPtr rgb_grid_service_;
};

}  // namespace heightmap_sampler

#endif  // SAMPLER_NODE_HPP_
