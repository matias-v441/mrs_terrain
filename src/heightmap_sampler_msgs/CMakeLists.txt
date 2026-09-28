cmake_minimum_required(VERSION 3.16)
project(heightmap_sampler_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geographic_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SampleHeight.srv"
  DEPENDENCIES geographic_msgs
)

ament_export_dependencies(rosidl_default_runtime geographic_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
