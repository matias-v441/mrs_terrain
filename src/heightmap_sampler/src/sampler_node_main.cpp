#include <cstdio>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "sampler_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    rclcpp::spin(std::make_shared<heightmap_sampler::SamplerNode>());
  } catch (const std::exception & error) {
    std::fprintf(stderr, "heightmap_sampler: %s\n", error.what());
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}
