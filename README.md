# heightmap-prep

Prepares terrain heightmap datasets for fast runtime sampling by geographic
coordinates.

The library acquires [ČÚZK DMR 5G](https://ags.cuzk.gov.cz/dmr/) elevation data
in S-JTSK / Krovak East North (EPSG:5514), converts the source Bpv heights onto a
requested output vertical datum using local PROJ grids, and writes deterministic
tiled `float32` GeoTIFFs plus a `dataset.yaml` manifest. What it prepares is
driven by [MRS UAV system](https://github.com/ctu-mrs) world files: the tiles
cover exactly what a sampler needs anywhere inside their safety areas.

All the expensive work — HTTP acquisition and the spatially varying vertical
datum conversion — happens once, offline. A runtime sampler then only has to
transform a WGS84 lon/lat query into EPSG:5514, compute a tile index, and
interpolate. No PROJ vertical transformation is needed at query time.

```
WGS84 lon/lat query → EPSG:5514 → tile index → open/cache tile
                    → inverse affine → fractional row/col → bilinear → height
```

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

Core dependencies: `numpy`, `rasterio`, `pyproj`, `httpx`, `PyYAML`, `affine`.

### PROJ grids

The Bpv → EGM96 conversion is spatially varying and needs two local PROJ grids.
The library never downloads them by itself; fetch them once:

```bash
mkdir -p .proj
curl -o .proj/cz_cuzk_CR-2005.tif  https://cdn.proj.org/cz_cuzk_CR-2005.tif
curl -o .proj/us_nga_egm96_15.tif  https://cdn.proj.org/us_nga_egm96_15.tif
```

and point the tool at them with `--proj-data-dir ./.proj`. PROJ network access is
always disabled, so once source rasters are cached the whole conversion runs
offline.

The exact grid set is taken from the PROJ operation that is actually selected,
not hard-coded: if PROJ picks a different chain, the manifest records which
grids it used, along with their checksums.

## Usage

```bash
heightmap-prep world_a.yaml world_b.yaml OUTPUT_DIR [OPTIONS]
heightmap-prep WORLDS_DIR OUTPUT_DIR [OPTIONS]
```

The last positional argument is always the output directory; the inputs are MRS
world configs, or one directory of them.

```bash
heightmap-prep worlds/world_bechovice.yaml worlds/world_ricany.yaml ./prepared \
    --proj-data-dir ./.proj --cache-dir ./cache
heightmap-prep ./worlds ./prepared --vertical-datum wgs84-ellipsoid --workers 4
heightmap-prep ./prepared --validate-only
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--vertical-datum {egm96,wgs84-ellipsoid}` | `egm96` | Output height reference |
| `--resolution FLOAT` | `2.0` | Target resolution in metres |
| `--tile-size INTEGER` | `4096` | Logical output tile edge in pixels |
| `--proj-data-dir PATH` | – | Directory holding local PROJ grids |
| `--cache-dir PATH` | – | Cache for downloaded source rasters |
| `--workers INTEGER` | `1` | Bounded parallel tile processing |
| `--overwrite` | off | Replace existing tiles instead of reusing them |
| `--validate-only` | off | Validate an existing dataset, no network |
| `--include-rgb` | off | Accepted; RGB acquisition is not implemented in v1 |
| `--log-level` | `info` | `debug` also logs request URLs, tile transforms, PROJ operations |

Advanced knobs (`--block-size`, `--nodata`, `--grid-origin`, `--max-request-px`,
`--timeout`, `--retries`, `--cog`, `--no-plausibility-checks`) are listed by
`heightmap-prep --help`.

Exit codes: `0` success, `1` produced but failed validation, `2` the run could
not proceed.

### World configs

A world config is an MRS UAV system world file. Only its safety area is read:

```yaml
mrs_uav_managers:
  safety_area_manager:
    safety_area:
      horizontal:
        frame_name: "latlon_origin"
        points: [
          50.0905258, 14.6327381,   # lat, lon of each vertex
          50.0896023, 14.6330607,
          50.0902173, 14.6348664,
          50.0910495, 14.6346838,
        ]
```

The world is named after its file, minus any `world_` prefix. Only a safety
area in `latlon_origin` places the world on the map.

A world config that cannot be prepared is skipped with a warning, and the run
carries on with the rest. That covers a file that is missing or unreadable, a
safety area that is absent, malformed or in another frame (`world_origin`,
`local_origin`, ...), a world name already taken by an earlier config, a safety
area reaching outside the source coverage, and a safety area corner where the
source has no height. Any tiles only a skipped world needed are left out of the
dataset. The run fails only when no world is left, or on problems that affect
every world, such as the source service failing or PROJ grids missing.

The dataset holds exactly the tiles a bilinear sampler reads when queried
anywhere inside a safety area, border included, and no others:

* a query interpolates between the four pixel centres around it, so the pixels
  prepared are those around the safety area, with no further margin;
* a tile is included only if the polygon itself comes within half a pixel of
  it, so a diagonal safety area does not pull in tiles that merely share its
  bounding box;
* a safety area on or within half a pixel of a tile edge interpolates across
  it, so the adjacent tile is included too.

Within a tile only those pixels are fetched; the rest of the tile is NoData.

### Python API

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

result = prepare_worlds(
    inputs=[Path("worlds/world_bechovice.yaml"), Path("worlds/world_ricany.yaml")],
    output_dir=Path("./prepared"),
    options=options,
)
print(result.manifest.tiles, result.report.ok)
```

`prepare_worlds` also accepts `source=` to substitute the acquisition backend.

## Output

```
OUTPUT_DIR/
    dataset.yaml
    test_points.csv
    height/
        tile_<ix>_<iy>.tif
```

Tiles are single-band `float32`, `deflate`-compressed with `predictor=3`,
internally tiled at 256×256, with NoData `-9999.0`.

### Test points

`test_points.csv` is reference data for checking a sampler. It has one row per
safety area corner, with no header:

```
50.0905258,14.6327381,259.03327503733396
```

The columns are latitude, longitude and the height the dataset gives there,
interpolated from the prepared tiles as the sampling contract below prescribes,
on the manifest's vertical datum. A sampler that implements the contract
reproduces every row to within float rounding. The manifest names the file:

```yaml
test_points:
  path: test_points.csv
  count: 12
  columns: [lat, lon, height]
```

Every corner has a height: a world with a corner where the source has no data
is skipped, as described under [World configs](#world-configs).

### Tile indexing

Tiles sit on one deterministic global grid in EPSG:5514, not on a per-world
grid, so overlapping worlds share tiles and filenames are stable across runs.

* `origin_x` is the **west** edge of tile `(0, 0)`
* `origin_y` is the **north** edge of tile `(0, 0)`
* `ix` increases **eastward**, `iy` increases **southward**

```python
tile_span_x = tile_width_px  * resolution_x
tile_span_y = tile_height_px * resolution_y

ix = floor((x - origin_x) / tile_span_x)
iy = floor((origin_y - y) / tile_span_y)
```

Everything that formula needs is in `dataset.yaml`, so a sampler locates a tile
without opening any GeoTIFF.

### Grid phase

The default origin is derived from `(-1000000, -800000)` and then shifted east
and south by **less than one pixel** so that every tile edge lands on a native
DMR 5G pixel boundary — `(-999999.6, -800000.12)` at 2 m. That matters: an
out-of-phase grid would make the ImageServer resample every request, and the
source affine transform would no longer be preserved through a vertical-only
conversion. Pass `--grid-origin X Y` to override; the tool warns if the origin
you choose is out of phase.

### Interpolation contract

Pixel values are located at pixel centres:

```python
x, y = transform @ (col + 0.5, row + 0.5)
```

and for the inverse lookup:

```python
col_corner, row_corner = (~transform) @ (x, y)
col = col_corner - 0.5
row = row_corner - 0.5
```

Tiles carry no halo, so a sampler doing bilinear interpolation near a tile edge
must load the neighbouring tile. `examples/sampler.py` is a ~100-line reference
implementation of the whole runtime path.

## Sampling from C++

[`src/heightmap_sampler`](src/heightmap_sampler) is a ROS 2 Jazzy package that
implements the same contract natively, for consumers that cannot call the
Python reference:

```cpp
heightmap_sampler::HeightSampler sampler;   // no path from the caller
auto height = sampler.sample(49.3625695, 14.2619165);
```

The prepared dataset is built into the package, so a caller supplies only a
coordinate. It also ships a node exposing the sampler over a service. See
[its README](src/heightmap_sampler/README.md) for how to point it at a dataset
and how its results are checked against `examples/sampler.py`.

## Resume and atomicity

Preparation is restartable. Each tile is written to `tile_<ix>_<iy>.tif.tmp`,
validated, and only then renamed into place, so an interrupted run never leaves
a truncated file that a later run would trust. Tiles carry their provenance as
GDAL metadata (source id, vertical datum, resolution, processing version) and
are reused on a later run only when all of it matches and the file still
validates. A mismatch is an error unless `--overwrite` is given.

The manifest is `status: building` during generation and `status: complete`
afterwards. A run that fails before writing anything restores the previous
manifest, so a failed attempt cannot downgrade an already-complete dataset.

## Validation

Nothing is published before it has been checked:

* **manifest** — supported format version, CRSs and units present, positive
  resolution, valid tile dimensions and pattern;
* **each tile** — opens, CRS matches, transform is consistent with the global
  grid, dimensions, `float32`, NoData matches, all non-NoData values finite;
* **spatial consistency** — neighbouring tiles share a resolution and abut
  exactly, with no gap or overlap;
* **test points** — every row of `test_points.csv` re-samples from the tiles to
  its recorded height;
* **vertical transformation** — reference points are pushed through the real
  transformer before any raster work; `inf`, `nan`, unavailable and
  ballpark-only operations are rejected;
* **plausibility** — elevations outside the Czech range, all-zero tiles, tiles
  with no data at all and a suspiciously unchanged conversion produce
  *warnings* only, and never replace the CRS-level checks.

## Coordinate reference systems

| | CRS | |
| --- | --- | --- |
| Query | EPSG:4326 | WGS84 lon/lat |
| Stored horizontal | EPSG:5514 | S-JTSK / Krovak East North, metres |
| Source vertical | EPSG:8357 | Bpv / Baltic 1957 height |
| Stored vertical (`egm96`) | EPSG:5773 | EGM96 gravity-related height |
| Stored vertical (`wgs84-ellipsoid`) | EPSG:4979 | WGS84 ellipsoidal height |

The conversion runs `Bpv → ETRS89 ellipsoidal → WGS84 ellipsoidal → EGM96`
through PROJ with `always_xy=True` and `allow_ballpark=False`, evaluated at pixel
centres in `float64` and stored as `float32`. It is never approximated by a
constant offset: over Czechia the Bpv/EGM96 separation varies by a few decimetres,
and the ellipsoidal separation by several metres.

For `egm96` the target is the compound CRS `EPSG:5514+EPSG:5773`, so the operation
is genuinely vertical-only and the horizontal coordinates come back unchanged —
which the library asserts on every block. `wgs84-ellipsoid` has no vertical CRS of
its own, so it runs through EPSG:4979 and keeps only Z.

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

The tests never touch the network: acquisition is exercised through a synthetic
source and an `httpx` mock transport. Tests needing a real Bpv conversion are
skipped unless the PROJ grids are present in `./.proj` or at
`$HEIGHTMAP_PREP_PROJ_DIR`.

## Not implemented in version 1

RGB imagery (`--include-rgb` is accepted and warns), LAZ/PDAL input, local
GeoTIFF input, EGM2008 and EVRF2007 outputs, Zarr storage, and polygonal world
regions. The `HeightSource` adapter protocol in `sources.py` is the extension
point for new inputs; the storage design does not need to change for any of them.
