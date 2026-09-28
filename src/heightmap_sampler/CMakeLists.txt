cmake_minimum_required(VERSION 3.16)
project(heightmap_sampler)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(geographic_msgs REQUIRED)
find_package(heightmap_sampler_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(GDAL REQUIRED)
find_package(yaml-cpp REQUIRED)

# --- the sampler library ------------------------------------------------
#
# GDAL and yaml-cpp are linked PRIVATE: the public header is a pimpl, so
# downstream packages need neither on their include path.

add_library(heightmap_sampler SHARED
  src/height_sampler.cpp
  src/manifest.cpp
  src/tile_cache.cpp
)
target_include_directories(heightmap_sampler
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(heightmap_sampler PRIVATE GDAL::GDAL yaml-cpp::yaml-cpp)
target_link_libraries(heightmap_sampler PRIVATE ament_index_cpp::ament_index_cpp)

# --- the node -----------------------------------------------------------

add_library(sampler_node_component SHARED src/sampler_node.cpp)
target_include_directories(sampler_node_component PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(sampler_node_component
  PUBLIC
    heightmap_sampler
    rclcpp::rclcpp
    ${geographic_msgs_TARGETS}
    ${heightmap_sampler_msgs_TARGETS}
  PRIVATE
    rclcpp_components::component
)
rclcpp_components_register_node(sampler_node_component
  PLUGIN "heightmap_sampler::SamplerNode"
  EXECUTABLE sampler_node_standalone)

add_executable(sampler_node src/sampler_node_main.cpp)
target_include_directories(sampler_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(sampler_node PRIVATE sampler_node_component rclcpp::rclcpp)

# A non-ROS check that this library reproduces examples/sampler.py.
add_executable(sample_height src/sample_height_cli.cpp)
target_link_libraries(sample_height PRIVATE heightmap_sampler)

# --- the dataset --------------------------------------------------------
#
# The dataset ships inside this package, so a default-constructed
# HeightSampler needs no path from the caller.  Which dataset that is, is a
# build-time decision:
#
#   colcon build --packages-up-to heightmap_sampler --cmake-args \
#     -DHEIGHTMAP_SAMPLER_DATASET_DIR=/path/to/prepared

set(HEIGHTMAP_SAMPLER_DATASET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dataset"
    CACHE PATH "Prepared dataset to install into this package's share directory")

if(EXISTS "${HEIGHTMAP_SAMPLER_DATASET_DIR}/dataset.yaml")
  message(STATUS "heightmap_sampler: bundling dataset ${HEIGHTMAP_SAMPLER_DATASET_DIR}")
  install(DIRECTORY "${HEIGHTMAP_SAMPLER_DATASET_DIR}/"
          DESTINATION share/${PROJECT_NAME}/dataset)
else()
  message(WARNING
    "heightmap_sampler: no dataset.yaml under ${HEIGHTMAP_SAMPLER_DATASET_DIR}. "
    "The package will build, but a default-constructed HeightSampler will fail "
    "at runtime. Set -DHEIGHTMAP_SAMPLER_DATASET_DIR to a prepared dataset.")
endif()

# --- install and export -------------------------------------------------

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

install(TARGETS heightmap_sampler
  EXPORT export_heightmap_sampler
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS sampler_node_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS sampler_node sample_height DESTINATION lib/${PROJECT_NAME})

ament_export_targets(export_heightmap_sampler HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(heightmap_sampler)

if(BUILD_TESTING)
  # The linters are invoked one by one rather than through ament_lint_auto:
  # this package lives inside a Python repository with its own established
  # conventions, and two of ament's defaults contradict them.  The C++ linters
  # all run, since the repository has no competing C++ style.
  find_package(ament_cmake_cppcheck REQUIRED)
  find_package(ament_cmake_cpplint REQUIRED)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_uncrustify REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)

  ament_cppcheck()
  # No file in this repository carries a copyright header.
  ament_cpplint(FILTERS -legal/copyright)
  ament_lint_cmake()
  ament_uncrustify()
  ament_xmllint()

  # ament_flake8 and ament_pep257 are deliberately not run: the Python here
  # (the launch file, the fixture generator, the parity check) follows the
  # style of the rest of the repository -- double quotes, D212 docstrings --
  # and the repository's own pytest suite is what lints it.

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_tile_grid test/test_tile_grid.cpp)
  target_include_directories(test_tile_grid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_tile_grid heightmap_sampler)

  ament_add_gtest(test_manifest test/test_manifest.cpp)
  target_include_directories(test_manifest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_manifest heightmap_sampler)

  ament_add_gtest(test_height_sampler test/test_height_sampler.cpp)
  target_include_directories(test_height_sampler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_height_sampler heightmap_sampler)
  target_compile_definitions(test_height_sampler
    PRIVATE HEIGHTMAP_SAMPLER_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
endif()

ament_package()
