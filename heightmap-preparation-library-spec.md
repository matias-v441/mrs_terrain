# Heightmap Preparation Library Specification

## 1. Purpose

This library prepares terrain heightmap datasets for fast runtime sampling by geographic coordinates.

The primary use case is:

1. acquire DMR 5G elevation data from ČÚZK,
2. preserve the source horizontal raster grid,
3. convert source elevations from Bpv to the requested output vertical datum,
4. tile and store the resulting rasters in a predictable layout,
5. write a manifest containing all metadata required by a runtime sampler.

The runtime sampler is expected to accept WGS84 longitude/latitude queries, transform them into the stored horizontal CRS, locate the correct tile, and interpolate the already-prepared elevations.

The preparation library MUST perform expensive raster acquisition and vertical datum conversion offline so the runtime sampler remains lightweight.

---

## 2. Design Principles

The implementation SHOULD follow these principles:

- Preserve the source DMR raster horizontal grid whenever practical.
- Do not reproject the entire raster to latitude/longitude merely to simplify point queries.
- Convert vertical heights once during preprocessing rather than at every runtime query.
- Keep horizontal CRS and vertical CRS metadata explicit and separate.
- Do not assume that GeoTIFF metadata contains a usable vertical CRS.
- Use deterministic tiling and naming.
- Prefer local PROJ grid files over network-dependent transformations.
- Validate all generated outputs before publishing them to the sampler.
- Avoid unnecessary resampling.
- Keep the storage representation independent from the query API representation.

---

## 3. Coordinate Reference Systems

### 3.1 Query coordinate system

The external/runtime query coordinate system is:

- WGS84 longitude/latitude
- EPSG:4326

The preparation library does not need to store rasters in EPSG:4326.

### 3.2 Preferred stored horizontal CRS

For standard ČÚZK DMR 5G data, the preferred stored horizontal CRS is:

- S-JTSK / Krovak East North
- EPSG:5514
- units: metres

The source raster affine transform MUST be preserved if the only preprocessing operation is a vertical datum conversion.

### 3.3 Source vertical datum

The standard ČÚZK DMR 5G S-JTSK product uses:

- Bpv / Baltic 1957 height
- EPSG:8357
- units: metres

### 3.4 Supported output vertical datums

The first implementation MUST support:

#### `egm96`

- EGM96 gravity-related height
- EPSG:5773
- units: metres

#### `wgs84-ellipsoid`

- WGS84 ellipsoidal height
- represented operationally through WGS84 3D CRS EPSG:4979 where needed
- units: metres

The CLI SHOULD expose readable datum names rather than requiring users to enter EPSG codes.

Recommended option:

```text
--vertical-datum [egm96 | wgs84-ellipsoid]
```

Default:

```text
egm96
```

### 3.5 Compound CRS interpretation

For transformation purposes, a DMR 5G raster in the preferred source representation is treated as:

```text
horizontal: EPSG:5514
vertical:   EPSG:8357
```

The preferred prepared representation for EGM96 output is:

```text
horizontal: EPSG:5514
vertical:   EPSG:5773
```

The horizontal affine transform remains unchanged during this conversion.

---

## 4. Vertical Transformation

### 4.1 Bpv to EGM96

The Bpv to EGM96 conversion is spatially varying and MUST be performed using PROJ/pyproj.

Conceptually:

```text
Bpv height
    ↓
ETRS89 ellipsoidal height
    ↓
WGS84 ellipsoidal height
    ↓
EGM96 height
```

The library MUST NOT approximate this conversion with a single constant offset.

### 4.2 Required PROJ grids

For the Czech transformation chain, the implementation MUST support local installation of the required PROJ grids.

Expected files include:

```text
cz_cuzk_CR-2005.tif
us_nga_egm96_15.tif
```

The exact set of required grids SHOULD be determined from the selected PROJ operation rather than hard-coded as the sole source of truth.

### 4.3 Offline operation

Production preprocessing SHOULD work without network access.

A configurable PROJ grid directory MUST be supported.

Example:

```python
from pathlib import Path
from pyproj import datadir, network

network.set_network_enabled(False)
datadir.append_data_dir(str(Path("./.proj").resolve()))
```

The grid path MUST be configured before the transformer is created.

### 4.4 Transformer selection

The implementation SHOULD use `TransformerGroup` during initialization and validation.

Requirements:

- `always_xy=True`
- `allow_ballpark=False`
- transformation must be reported as available
- the selected Czech Bpv operation SHOULD correspond to the CR-2005 transformation
- transformation failures MUST raise errors rather than silently return `inf`

Example behavior:

```python
x2, y2, z2 = transformer.transform(
    x,
    y,
    z,
    errcheck=True,
)
```

The library MUST reject outputs containing non-finite transformed elevations unless the input location is explicitly outside the supported area.

### 4.5 Pixel coordinates used for vertical transformation

Vertical corrections MUST be evaluated at raster pixel centers.

For an affine transform `T` and raster index `(row, col)`:

```python
x, y = T * (col + 0.5, row + 0.5)
```

For vectorized processing, the equivalent affine formula MAY be used directly.

The horizontal output of a vertical-only preprocessing operation MUST remain effectively equal to the input raster coordinates.

---

## 5. Input Sources

The library SHOULD support two input modes.

### 5.1 World configuration files

One or more world configuration files:

```text
world1.yaml world2.yaml ...
```

### 5.2 Worlds directory

A directory containing world configuration files:

```text
/path/to/worlds/
```

The CLI MUST distinguish between:

- multiple YAML file inputs, or
- one worlds directory input,

followed by an output directory.

Recommended usage:

```text
heightmap-prep world1.yaml world2.yaml OUTPUT_DIR
```

or:

```text
heightmap-prep WORLDS_DIR OUTPUT_DIR
```

---

## 6. World Configuration

Each world YAML SHOULD describe the geographic extent and requested prepared assets.

Recommended schema:

```yaml
name: prague

bounds:
  type: wgs84
  west: 14.20
  south: 49.95
  east: 14.70
  north: 50.25

heightmap:
  source: cuzk-dmr5g
  resolution_m: 2.0

rgb:
  enabled: false
```

Alternative polygon or more advanced region definitions MAY be added later.

The initial implementation MAY restrict world bounds to axis-aligned WGS84 bounding boxes.

---

## 7. ČÚZK DMR 5G Acquisition

### 7.1 Service

The preferred raster source is the dynamic ČÚZK DMR 5G ImageServer that exposes the data in S-JTSK / Krovak East North.

The acquisition layer SHOULD request:

```text
bboxSR    = 5514
imageSR   = 5514
format    = tiff
pixelType = F32
```

The source raster is interpreted as:

```text
horizontal: EPSG:5514
vertical:   Bpv
```

### 7.2 Input bounds

World bounds are expected to be supplied in WGS84 longitude/latitude.

Before requesting DMR data, the library MUST transform the requested area from EPSG:4326 into EPSG:5514.

For a bounding rectangle, the implementation SHOULD transform all four corners.

For larger regions, the implementation SHOULD densify the boundary before deriving the EPSG:5514 request bounds so projection curvature cannot clip the requested region.

### 7.3 Request resolution

The requested raster resolution SHOULD match the native DMR 5G raster resolution where practical.

Default:

```text
2.0 m
```

The raster dimensions SHOULD be derived from the requested projected bounds and target resolution.

### 7.4 Request tiling

Large regions MUST be split into multiple HTTP requests when they exceed server export limits.

The acquisition layer MUST:

- tile requests deterministically,
- avoid gaps,
- avoid accidental overlaps larger than necessary,
- preserve the same target resolution,
- verify that returned raster CRS and dimensions are consistent.

### 7.5 HTTP client

The recommended HTTP client is `httpx`.

The acquisition layer SHOULD support:

- configurable timeout,
- retries for transient failures,
- explicit HTTP error handling,
- optional concurrency with a bounded request count,
- optional local response cache.

---

## 8. Raster Processing

### 8.1 Raster library

Raster I/O SHOULD use Rasterio.

Primary input representation:

```python
height_bpv = dataset.read(1)
transform = dataset.transform
crs = dataset.crs
nodata = dataset.nodata
```

### 8.2 Data type

Prepared elevation rasters MUST use:

```text
float32
```

Vertical transformation calculations SHOULD be performed internally in `float64`, then converted to `float32` for storage.

### 8.3 NoData

A single explicit NoData value SHOULD be used throughout a prepared dataset.

Recommended default:

```text
-9999.0
```

NoData pixels MUST NOT be passed through vertical transformations.

### 8.4 Vertical conversion workflow

For each raster chunk or tile:

1. read source Bpv elevations,
2. generate pixel-center EPSG:5514 coordinates,
3. create a valid-data mask,
4. transform valid `(x, y, z_bpv)` values,
5. keep only the transformed Z values,
6. preserve the original affine transform,
7. write the transformed raster.

The implementation SHOULD process rasters in blocks/windows rather than allocating full-world coordinate arrays.

---

## 9. Output Tile Layout

### 9.1 Preferred tile size

Default logical output tile size:

```text
4096 × 4096 pixels
```

At 2 m resolution this represents approximately:

```text
8.192 km × 8.192 km
```

A configurable alternative such as 2048 × 2048 MAY be supported.

### 9.2 Internal GeoTIFF blocks

GeoTIFF files SHOULD use internal tiling.

Recommended defaults:

```text
blockxsize = 256
blockysize = 256
```

### 9.3 Compression

Recommended storage profile:

```text
driver     = GTiff
dtype      = float32
tiled      = true
compress   = deflate
predictor  = 3
nodata     = -9999.0
```

COG output MAY be offered as an option.

### 9.4 Tile boundaries

Output tiles SHOULD align to one deterministic global grid in EPSG:5514 rather than being independently aligned to each world.

This allows:

- direct tile index computation,
- cache reuse between overlapping worlds,
- deduplication,
- deterministic filenames.

The global grid origin MUST be stored in the manifest.

### 9.5 Border interpolation

The prepared files SHOULD NOT duplicate a one-pixel halo by default.

The downstream sampler is expected to load neighboring tiles when bilinear interpolation crosses a tile boundary.

A future optional halo mode MAY be added.

---

## 10. Output Directory Structure

Recommended structure:

```text
OUTPUT_DIR/
    dataset.yaml
    height/
        tile_<x>_<y>.tif
        tile_<x>_<y>.tif
        ...
    rgb/
        ...
```

If multiple worlds share the same prepared dataset, an optional higher-level structure MAY be used:

```text
OUTPUT_DIR/
    datasets/
        czech_dmr5_egm96/
            dataset.yaml
            height/
                ...
    worlds/
        prague.yaml
        brno.yaml
```

The first implementation MAY use the simpler single-dataset layout.

---

## 11. Dataset Manifest

The library MUST write a manifest such as `dataset.yaml`.

Recommended schema:

```yaml
format_version: 1

heightmap:
  horizontal_crs: EPSG:5514
  vertical_crs: EPSG:5773
  vertical_datum: egm96
  unit: m
  dtype: float32
  nodata: -9999.0

grid:
  resolution_x: 2.0
  resolution_y: 2.0
  tile_width_px: 4096
  tile_height_px: 4096
  origin_x: -1000000.0
  origin_y: -800000.0
  axis_order: east_north

storage:
  format: geotiff
  compression: deflate
  predictor: 3
  block_width_px: 256
  block_height_px: 256
  tile_pattern: "height/tile_{ix}_{iy}.tif"

source:
  provider: CUZK
  product: DMR5G
  source_horizontal_crs: EPSG:5514
  source_vertical_crs: EPSG:8357
  source_vertical_datum: bpv
  source_resolution_m: 2.0

transform:
  vertical:
    method: proj
    target: egm96
    network_enabled: false

worlds:
  - prague
```

The manifest MUST contain sufficient information for a sampler to locate a tile without opening every GeoTIFF.

---

## 12. Tile Naming and Indexing

Preferred deterministic naming:

```text
tile_<ix>_<iy>.tif
```

where `ix` and `iy` are integer tile-grid indices.

Given:

```text
tile_span_x = tile_width_px  * resolution_x
tile_span_y = tile_height_px * resolution_y
```

the sampler SHOULD be able to compute a tile index directly from projected coordinates.

The exact northing convention MUST be documented.

Recommended convention:

- `origin_x` is the west edge of tile `(0, 0)`
- `origin_y` is the north edge of tile `(0, 0)`
- `ix` increases eastward
- `iy` increases southward

Then:

```python
ix = floor((x - origin_x) / tile_span_x)
iy = floor((origin_y - y) / tile_span_y)
```

---

## 13. RGB Imagery

RGB imagery is optional.

CLI flag:

```text
--include-rgb
```

Default:

```text
false
```

If enabled, RGB images SHOULD:

- use the same logical tile grid as the heightmap where practical,
- be stored separately under `rgb/`,
- have their own CRS and source metadata if different,
- not affect heightmap preparation correctness.

The initial implementation MAY leave RGB acquisition as a separate adapter.

---

## 14. CLI

Recommended executable name:

```text
heightmap-prep
```

Recommended syntax:

```text
heightmap-prep INPUT... OUTPUT_DIR [OPTIONS]
```

Where `INPUT...` is either:

```text
world1.yaml world2.yaml ...
```

or:

```text
WORLDS_DIR
```

Recommended options:

```text
--vertical-datum [egm96 | wgs84-ellipsoid]
    Output height reference.
    Default: egm96

--include-rgb
    Include RGB imagery.
    Default: false

--resolution FLOAT
    Target heightmap resolution in metres.
    Default: 2.0

--tile-size INTEGER
    Logical output tile width/height in pixels.
    Default: 4096

--proj-data-dir PATH
    Directory containing local PROJ grid files.

--cache-dir PATH
    Optional cache for downloaded source rasters.

--workers INTEGER
    Number of preprocessing workers.

--overwrite
    Replace existing generated tiles.

--validate-only
    Validate an existing output dataset without downloading or converting.

--log-level [debug | info | warning | error]
```

Optional future flags:

```text
--cog
--keep-source
--source [cuzk-dmr5g | local-geotiff | laz]
```

---

## 15. Python API

Recommended top-level package name:

```text
heightmap_prep
```

Suggested modules:

```text
heightmap_prep/
    __init__.py
    cli.py
    config.py
    crs.py
    cuzk.py
    raster.py
    tiling.py
    manifest.py
    validate.py
```

### 15.1 Public API

A simple public API SHOULD be provided.

Example:

```python
from pathlib import Path
from heightmap_prep import PrepareOptions, prepare_worlds

options = PrepareOptions(
    vertical_datum="egm96",
    include_rgb=False,
    resolution_m=2.0,
    tile_size_px=4096,
    proj_data_dir=Path("./.proj"),
)

prepare_worlds(
    inputs=[Path("world1.yaml"), Path("world2.yaml")],
    output_dir=Path("./prepared"),
    options=options,
)
```

### 15.2 Core option model

Suggested model:

```python
@dataclass(frozen=True)
class PrepareOptions:
    vertical_datum: str = "egm96"
    include_rgb: bool = False
    resolution_m: float = 2.0
    tile_size_px: int = 4096
    proj_data_dir: Path | None = None
    cache_dir: Path | None = None
    workers: int = 1
    overwrite: bool = False
```

---

## 16. Source Adapter Interface

Source-specific acquisition SHOULD be isolated behind an adapter.

Suggested protocol:

```python
class HeightSource(Protocol):
    horizontal_crs: str
    vertical_crs: str
    resolution_m: float

    def fetch(
        self,
        bounds,
        output_path: Path,
    ) -> Path:
        ...
```

Initial implementation:

```text
CuzkDmr5Source
```

Possible future implementations:

```text
LocalGeoTiffSource
LazSource
```

This separation allows LAZ/PDAL input to be added without changing the output storage design.

---

## 17. Validation

A generated dataset MUST be validated before it is considered complete.

### 17.1 Manifest validation

- format version supported,
- horizontal CRS present,
- vertical CRS present,
- units present,
- resolution positive,
- tile dimensions valid,
- tile pattern valid.

### 17.2 GeoTIFF validation

For each tile:

- file opens successfully,
- CRS equals expected horizontal CRS,
- affine transform is consistent with the global tile grid,
- dimensions match expected tile dimensions, except allowed edge tiles,
- dtype is `float32`,
- NoData value matches manifest,
- all non-NoData elevations are finite.

### 17.3 Spatial consistency

Adjacent tiles MUST:

- have identical resolution,
- align exactly at shared boundaries,
- have no unintended gap,
- have no unintended overlap.

### 17.4 Vertical transformation validation

The preparation pipeline SHOULD test a small number of known points through the same PROJ transformer.

It SHOULD reject:

- `inf`,
- `nan`,
- unavailable transformations,
- ballpark-only transformations.

### 17.5 Plausibility checks

Configurable plausibility checks MAY flag:

- elevations outside an expected Czech terrain range,
- suspiciously unchanged Bpv→EGM96 outputs when a transformation was requested,
- excessive NoData fraction,
- all-zero tiles.

Plausibility warnings MUST NOT replace CRS-level validation.

---

## 18. Atomic Output and Resume Behavior

Preparation can be expensive and SHOULD be restartable.

Each output tile SHOULD be written to a temporary path first:

```text
tile_12_35.tif.tmp
```

and atomically renamed after successful validation.

Existing valid tiles SHOULD be reusable when:

- source identifier matches,
- output CRS/datum matches,
- resolution matches,
- processing version matches.

The manifest SHOULD only be marked complete after all required tiles have completed.

Recommended manifest field:

```yaml
status: complete
```

During generation:

```yaml
status: building
```

---

## 19. Reproducibility Metadata

The manifest SHOULD record:

```yaml
software:
  package_version: "..."
  python_version: "..."
  rasterio_version: "..."
  pyproj_version: "..."
  proj_version: "..."

processing:
  created_utc: "..."
  interpolation: nearest
  vertical_conversion: proj
```

If practical, record the exact selected PROJ operation description.

Example:

```yaml
proj_operation: >
  Inverse of ETRS89 to Baltic 1957 height ...
  + WGS 84 to EGM96 height ...
```

The library MAY also record checksums of local PROJ grid files.

---

## 20. Dependencies

Recommended core dependencies:

```text
numpy
rasterio
pyproj
httpx
PyYAML
```

Optional:

```text
pydantic
rich
tqdm
```

Future LAZ support may add:

```text
PDAL
```

CUDA support is not required for the first implementation.

---

## 21. Performance Requirements

The implementation SHOULD:

- process raster data window-by-window,
- avoid creating full-resolution X and Y arrays for very large regions,
- reuse one initialized PROJ transformer,
- use NumPy vectorization for each processing window,
- support bounded parallel tile processing,
- reuse HTTP connections through `httpx.Client` or `httpx.AsyncClient`.

The implementation SHOULD NOT:

- create a new PROJ transformer per pixel,
- reproject entire rasters to EPSG:4326 only for sampling convenience,
- read the entire national dataset into memory,
- depend on PROJ network access during normal production preprocessing.

---

## 22. Interaction with the Runtime Sampler

The downstream sampler is outside the core scope of this library, but the output MUST support the following efficient runtime flow:

```text
WGS84 lon/lat query
        ↓
EPSG:4326 → stored horizontal CRS
        ↓
compute tile index
        ↓
open/cache tile
        ↓
inverse affine transform
        ↓
fractional row/column
        ↓
bilinear interpolation
        ↓
prepared height
```

For the preferred ČÚZK configuration:

```text
query horizontal CRS:
    EPSG:4326

stored horizontal CRS:
    EPSG:5514

stored vertical CRS:
    EPSG:5773
```

The sampler therefore performs only a horizontal query transformation and interpolation; no vertical datum transformation is required at runtime.

---

## 23. Interpolation Contract

The preparation library stores raster samples at pixel centers.

The sampler SHOULD interpret each pixel value as located at:

```python
x, y = transform * (col + 0.5, row + 0.5)
```

For inverse lookup:

```python
col_corner, row_corner = (~transform) * (x, y)

col = col_corner - 0.5
row = row_corner - 0.5
```

The default sampler interpolation SHOULD be bilinear.

Nearest-neighbor MAY be supported for diagnostic or exact-cell access.

---

## 24. Error Handling

The library MUST fail with a clear error for:

- missing PROJ transformation grids,
- unavailable required CRS operation,
- source HTTP failure after retries,
- unsupported source CRS,
- unexpected source vertical datum,
- invalid world bounds,
- requested area outside source coverage,
- malformed GeoTIFF,
- mismatched raster resolution,
- non-finite transformation output,
- output path conflicts when overwrite is disabled.

Errors SHOULD include enough context to identify:

- world,
- source request bounds,
- tile index,
- relevant CRS,
- underlying exception.

---

## 25. Logging

At `info` level, log:

- world being processed,
- source bounds,
- number of source requests,
- output tile count,
- selected vertical transformation,
- cache hits,
- final output location.

At `debug` level, additionally log:

- request URLs excluding secrets,
- transformed bounds,
- tile transforms,
- PROJ operation descriptions,
- grid availability,
- per-tile processing timing.

---

## 26. Security and Network Behavior

The library SHOULD:

- validate that configured source URLs use HTTPS,
- enforce request timeouts,
- avoid arbitrary shell execution,
- not download PROJ grids automatically unless explicitly requested,
- allow a completely offline transformation mode once source rasters are cached.

---

## 27. Future Extensions

Potential future features:

- LAZ input with PDAL,
- EGM2008 output,
- EVRF2007 output,
- arbitrary local GeoTIFF input,
- Cloud Optimized GeoTIFF output,
- Zarr storage for very large datasets,
- GPU-assisted vertical transformation or raster processing,
- distributed preprocessing,
- RGB source adapters,
- polygonal world regions,
- checksum-based incremental rebuilds,
- multiple horizontal source CRSs,
- explicit compound CRS metadata where supported.

---

## 28. Initial Acceptance Criteria

Version 1 is complete when it can:

1. read one or more world YAML files,
2. accept WGS84 geographic bounds,
3. request ČÚZK DMR 5G data in EPSG:5514,
4. obtain Bpv float32 elevation rasters,
5. convert Bpv heights to EGM96 using local PROJ grids with network disabled,
6. preserve the EPSG:5514 raster geometry,
7. write deterministic tiled float32 GeoTIFFs,
8. write a complete `dataset.yaml`,
9. validate tile alignment and CRS consistency,
10. resume an interrupted build without regenerating valid tiles,
11. produce output directly usable by a WGS84-lon/lat runtime sampler.

---

## 29. Recommended Version-1 Defaults

```yaml
query_crs: EPSG:4326

source:
  provider: CUZK
  product: DMR5G
  horizontal_crs: EPSG:5514
  vertical_crs: EPSG:8357

output:
  horizontal_crs: EPSG:5514
  vertical_crs: EPSG:5773
  vertical_datum: egm96
  resolution_m: 2.0
  dtype: float32
  nodata: -9999.0

tiling:
  tile_size_px: 4096
  block_size_px: 256

geotiff:
  compression: deflate
  predictor: 3
  tiled: true

proj:
  network_enabled: false

sampling_contract:
  pixel_location: center
  interpolation: bilinear
```
