#ifndef SAMPLER_NODE_HPP_
#define SAMPLER_NODE_HPP_

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "heightmap_sampler/height_sampler.hpp"
#include "heightmap_sampler_msgs/srv/sample_height.hpp"

namespace heightmap_sampler
{

/// Exposes a prepared heightmap dataset over a service.
class SamplerNode : public rclcpp::Node
{
public:
  explicit SamplerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  using SampleHeight = heightmap_sampler_msgs::srv::SampleHeight;

  void handle_sample(
    const std::shared_ptr<SampleHeight::Request> request,
    std::shared_ptr<SampleHeight::Response> response);

  std::unique_ptr<HeightSampler> sampler_;
  rclcpp::Service<SampleHeight>::SharedPtr service_;
};

}  // namespace heightmap_sampler

#endif  // SAMPLER_NODE_HPP_
