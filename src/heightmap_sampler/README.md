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

### Regions of interest

`height_grid` and `rgb_grid` return a whole region as one stitched buffer:

```cpp
// An axis-aligned WGS84 rectangle: north-west corner, then south-east corner.
heightmap_sampler::Roi roi{{49.3630, 14.2610}, {49.3620, 14.2625}};

if (auto grid = sampler.height_grid(roi)) {
  // grid->heights: row-major float, row 0 northmost, NaN where there is no height
  // grid->geo: west/north edges, resolution_deg, width, height (EPSG:4326)
}
if (auto image = sampler.rgb_grid(roi)) {
  // image->rgb: row-major interleaved RGB; image->valid: 255 imagery / 0 none
}
```

They read the dataset's EPSG:4326 tiles (see [the dataset](#the-dataset)), on
which a latitude/longitude rectangle is axis-aligned. The grid holds exactly
the pixels that bilinear interpolation between pixel centres reads for some
point of the region. So every point of the region, the two corners included,
can be interpolated from the grid alone. On a lattice of pixel size `r` from
`(lon0, lat0)`, with `c(lon) = (lon - lon0) / r - 0.5` and
`r(lat) = (lat0 - lat) / r - 0.5`, the grid holds columns
`floor(c(west)) .. floor(c(east)) + 1` and rows `floor(r(north)) .. floor(r(south)) + 1`.
A single point gives a 2x2 grid. Imagery comes on its own, finer lattice by the
same rule.

The two calls return nothing when a tile the region needs is absent, and when
the region is inverted or off the globe. Pixels without data inside the
available tiles do not count as missing tiles: they come back as NaN or with
`valid` 0. Like `sample`, both calls never throw and are thread-safe.
`has_height_grids()` and `has_rgb_grids()` report whether the dataset holds
these tiles at all.

The EPSG:4326 heights are themselves bilinear samples of the EPSG:5514 tiles,
taken at each pixel centre. So interpolating a grid gives a close
approximation of `sample`, not the same value.
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

The dataset ships **inside this package**. It is committed in the
repository's top-level `dataset/`, which this package reaches through its
`dataset` symlink (`dataset -> ../../dataset`). It is installed to
`share/heightmap_sampler/dataset`, and is what a default-constructed
`HeightSampler` reads. It covers the safety areas of the worlds in the
repository's `worlds/`, with heights on the WGS84 ellipsoid. After changing
those worlds, run `./regenerate.sh` from the repository root and commit
`dataset/`.

Point samples read the EPSG:5514 tiles under `height/`. Region grids read the
EPSG:4326 WGS tiles of heights and orthophoto imagery under `wgs/`. The worlds
database is installed with the dataset but not read by this package.
`dataset.yaml` must be of format version 2.

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
that is a normal answer, so `success` stays true.

Regions come from `heightmap_sampler_msgs/srv/GetHeightGrid` and
`GetRgbGrid`. Each takes a `geographic_msgs/BoundingBox` whose `min_pt` is the
south-west corner and `max_pt` the north-east corner:

```bash
ros2 service call /heightmap_sampler/get_height_grid \
  heightmap_sampler_msgs/srv/GetHeightGrid \
  "{roi: {min_pt: {latitude: 49.3620, longitude: 14.2610}, max_pt: {latitude: 49.3630, longitude: 14.2625}}}"
```

The response carries a `sensor_msgs/Image` grid with row 0 northmost. Heights
are `32FC1`, with NaN where there is no height. Imagery is `rgba8`, with alpha
0 where there is none. The response also gives the grid's `west` and `north`
edges and its `resolution_deg`. `success` is false, with a reason in
`message`, when the library returns nothing.

Parameters:

| parameter | default | meaning |
| --- | --- | --- |
| `dataset_dir` | `""` | Dataset to sample; empty means the bundled one. |
| `service_name` | `~/sample_height` | |
| `height_grid_service_name` | `~/get_height_grid` | |
| `rgb_grid_service_name` | `~/get_rgb_grid` | |
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
default-constructed sampler reading the installed copy. It also checks that
there are height and imagery grids around each of those corners.

The remaining unit tests run against a committed synthetic fixture in `test/data`, whose
surface is a plane — bilinear interpolation of a plane is exact, so every
expected value is closed form. `test/make_fixture.py` regenerates it, including
the expected heights, which it derives with pyproj in the same direction the
sampler transforms; a passing test therefore also confirms that GDAL's
`EPSG:4326 -> EPSG:5514` transformation agrees with pyproj's.

The fixture also holds small EPSG:4326 tiles on power-of-two degree lattices:
a height plane with a NoData patch, and an imagery pattern with a masked patch.
One of the four tiles is absent. `test_roi` reads region grids from them, and
`test_wgs_lattice` checks the grid-extent rule on its own.

The parity check against the Python reference needs the project's Python
environment, so it lives outside the colcon test set; `test_sampler.sh` runs it
on the bundled dataset. By hand:

```bash
source <workspace>/install/setup.bash
.venv/bin/python src/heightmap_sampler/test/parity_check.py dataset
```

It first checks that both this library and the Python reference reproduce every
reference height in the dataset's `test_points.csv` exactly. Then it samples
random coordinates within `--radius` metres (default 100) of those points with
both, and asserts they agree, on the heights and on which points have none.
