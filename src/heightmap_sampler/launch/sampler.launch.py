"""Bring up the heightmap sampler service.

By default the node uses the dataset bundled into the heightmap_sampler share
directory, so no arguments are needed:

    ros2 launch heightmap_sampler sampler.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    dataset_dir = LaunchConfiguration("dataset_dir")
    tile_cache_size = LaunchConfiguration("tile_cache_size")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "dataset_dir",
                default_value="",
                description="Prepared dataset to sample; empty means the bundled one.",
            ),
            DeclareLaunchArgument(
                "tile_cache_size",
                default_value="16",
                description="How many tile rasters to keep open at once.",
            ),
            Node(
                package="heightmap_sampler",
                executable="sampler_node",
                name="heightmap_sampler",
                output="screen",
                parameters=[
                    {
                        "dataset_dir": dataset_dir,
                        "tile_cache_size": tile_cache_size,
                    }
                ],
            ),
        ]
    )
