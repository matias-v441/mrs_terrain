# heightmap_sampler

A ROS 2 Jazzy C++ library that samples terrain heights by WGS84 latitude and
longitude from a dataset prepared by [heightmap-prep](../../README.md).

It is the C++ counterpart of [`examples/sampler.py`](../../examples/sampler.py)
and implements the same contract: one horizontal transformation from
`EPSG:4326` into the dataset's stored CRS, a tile index computed from the
manifest grid, and bilinear interpolation of the stored heights. No vertical
datum transformation happens at runtime, because the stored values are already
in the dataset's target vertical datum.

## Using the library

The dataset location is resolved by the library, not by the caller:

```cpp
#include "heightmap_sampler/height_sampler.hpp"

heightmap_sampler::HeightSampler sampler;   // no path, no parameter, no config

if (auto height = sampler.sample(49.3625695, 14.2619165)) {
  use(*height);            // metres, in the dataset's vertical datum
} else {
  // No prepared height here: the tile is absent, the pixel is NoData, or the
  // point is outside the raster.
}
```

`sample` takes **latitude first**, never throws, and is safe to call from
several threads. There is a batch overload taking a `std::vector<LatLon>`.
`info()` reports what the manifest says the returned values are, so a consumer
can assert the vertical datum it expects rather than assuming one.

The constructors throw `heightmap_sampler::DatasetError` when the dataset is
missing, incomplete, of an unknown format version, or otherwise unusable —
failures worth noticing at startup rather than on the first query.

Downstream packages need only:

```cmake
find_package(heightmap_sampler REQUIRED)
target_link_libraries(your_target heightmap_sampler::heightmap_sampler)
```

The public header is a pimpl, so GDAL and yaml-cpp stay inside this package.

## The dataset

The dataset ships **inside this package**. It is committed in `dataset/`, is
installed to `share/heightmap_sampler/dataset`, and is what a default-constructed
`HeightSampler` reads. It covers the safety areas of the worlds in the
repository's `worlds/`, with heights on the WGS84 ellipsoid. After changing
those worlds, run `./regenerate.sh` from the repository root and commit the
result.

To bundle a different dataset instead, point the build at it:

```bash
colcon build --packages-up-to heightmap_sampler --cmake-args \
  -DHEIGHTMAP_SAMPLER_DATASET_DIR=/path/to/prepared
```

If no `dataset.yaml` is found there, the package still builds, with a warning,
and a default-constructed sampler fails at runtime.

To regenerate the dataset without rebuilding, write it straight into the
install tree, or pass an explicit directory to the constructor.

## The node

`sampler_node` exposes the sampler over
`heightmap_sampler_msgs/srv/SampleHeight`, which is batch-capable:

```bash
ros2 launch heightmap_sampler sampler.launch.py
# or: ros2 run heightmap_sampler sampler_node

ros2 service call /heightmap_sampler/sample_height \
  heightmap_sampler_msgs/srv/SampleHeight \
  "{points: [{latitude: 49.3625695, longitude: 14.2619165}]}"
```

```
heights=[405.6511035203645], valid=[True], success=True, message=''
```

`valid[i]` is false, and `heights[i]` NaN, where there is no prepared height;
that is a normal answer, so `success` stays true. Parameters:

| parameter | default | meaning |
| --- | --- | --- |
| `dataset_dir` | `""` | Dataset to sample; empty means the bundled one. |
| `service_name` | `~/sample_height` | |
| `tile_cache_size` | `16` | How many tile rasters to keep open at once. |

The node is also registered as an `rclcpp_components` component,
`heightmap_sampler::SamplerNode`.

## Tests

From the repository root, `./test_sampler.sh` builds this package in a private
workspace and runs everything below. Inside your own workspace:

```bash
colcon test --packages-select heightmap_sampler
colcon test-result --verbose
```

`test_bundled_dataset` checks the bundled dataset: every reference height in
its `test_points.csv`, which heightmap-prep computes at each safety area corner,
must come out of this library, both from the dataset directory and from a
default-constructed sampler reading the installed copy.

The remaining unit tests run against a committed synthetic fixture in `test/data`, whose
surface is a plane — bilinear interpolation of a plane is exact, so every
expected value is closed form. `test/make_fixture.py` regenerates it, including
the expected heights, which it derives with pyproj in the same direction the
sampler transforms; a passing test therefore also confirms that GDAL's
`EPSG:4326 -> EPSG:5514` transformation agrees with pyproj's.

The parity check against the Python reference needs the project's Python
environment, so it lives outside the colcon test set; `test_sampler.sh` runs it
on the bundled dataset. By hand:

```bash
source <workspace>/install/setup.bash
.venv/bin/python src/heightmap_sampler/test/parity_check.py src/heightmap_sampler/dataset
```

It first checks that both this library and the Python reference reproduce every
reference height in the dataset's `test_points.csv` exactly. Then it samples
random coordinates within `--radius` metres (default 100) of those points with
both, and asserts they agree, on the heights and on which points have none.
