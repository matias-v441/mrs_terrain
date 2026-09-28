#!/usr/bin/env python3
"""Generate the synthetic dataset the C++ sampler tests run against.

The fixture is committed, so this only needs re-running when the expectations
in test_height_sampler.cpp change.  Run it from the repository root with the
project's own environment, so the tiles are written by exactly the stack that
writes real datasets:

    .venv/bin/python src/heightmap_sampler/test/make_fixture.py

The stored surface is a plane in tile-grid-relative coordinates,

    h = A * (x - origin_x) + B * (origin_y - y) + C

Bilinear interpolation of a plane is exact, so every expected value is closed
form -- no expectation depends on a stored sample surviving a round trip.  The
coefficients are chosen so that every pixel-centre value is exactly
representable in float32 (offsets land on multiples of 0.5, A and B are
negative powers of two), which keeps the tests to a 1e-9 tolerance.

Layout: 32x32 px tiles at 1 m, anchored in real EPSG:5514 space next to the
Temesvar field so that queries are plausible WGS84 coordinates.

    (0, 0) and (1, 0) are present and adjacent, so a query between them
           exercises interpolation across a tile seam with no halo
    (0, 1) is present and carries a NoData patch
    (1, 1) is deliberately absent

Alongside the tiles this writes ``expected.csv`` -- ``lon,lat,expected`` rows,
with ``nan`` where the sampler must report no height.  The C++ test reads that
file rather than reimplementing the WGS84 -> EPSG:5514 transformation, so a
passing test also confirms that GDAL's OGR transform agrees with pyproj's.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
import yaml
from affine import Affine
from pyproj import Transformer

HORIZONTAL_CRS = "EPSG:5514"
QUERY_CRS = "EPSG:4326"

# A round anchor a few hundred metres from the Temesvar field, so the fixture
# sits inside the Krovak projection's area of use.
ORIGIN_X = -765500.0
ORIGIN_Y = -1121200.0
RESOLUTION = 1.0
TILE_PX = 32
NODATA = -9999.0

PLANE_A = 0.25
PLANE_B = -0.125
PLANE_C = 100.0

# NoData patch in tile (0, 1), in that tile's pixel coordinates.
NODATA_ROWS = slice(8, 12)
NODATA_COLS = slice(8, 12)

PRESENT_TILES = [(0, 0), (1, 0), (0, 1)]
ABSENT_TILE = (1, 1)

SPAN = TILE_PX * RESOLUTION


def plane(x: np.ndarray | float, y: np.ndarray | float):
    """The stored surface, in tile-grid-relative coordinates."""
    return PLANE_A * (x - ORIGIN_X) + PLANE_B * (ORIGIN_Y - y) + PLANE_C


def write_tile(root: Path, ix: int, iy: int) -> None:
    west = ORIGIN_X + ix * SPAN
    north = ORIGIN_Y - iy * SPAN
    transform = Affine(RESOLUTION, 0.0, west, 0.0, -RESOLUTION, north)

    # Pixel centres, matching heightmap_prep.raster.pixel_center_coords.
    xs = west + (np.arange(TILE_PX) + 0.5) * RESOLUTION
    ys = north - (np.arange(TILE_PX) + 0.5) * RESOLUTION
    grid_x, grid_y = np.meshgrid(xs, ys)
    data = plane(grid_x, grid_y).astype("float32")

    if (ix, iy) == (0, 1):
        data[NODATA_ROWS, NODATA_COLS] = NODATA

    path = root / "height" / f"tile_{ix}_{iy}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=TILE_PX,
        height=TILE_PX,
        count=1,
        dtype="float32",
        crs=HORIZONTAL_CRS,
        transform=transform,
        nodata=NODATA,
        tiled=True,
        blockxsize=16,
        blockysize=16,
        compress="deflate",
        predictor=3,
    ) as dataset:
        dataset.write(data, 1)


def write_manifest(root: Path) -> None:
    manifest = {
        "format_version": 1,
        "status": "complete",
        "heightmap": {
            "horizontal_crs": HORIZONTAL_CRS,
            "vertical_crs": "EPSG:5773",
            "vertical_datum": "egm96",
            "unit": "m",
            "dtype": "float32",
            "nodata": NODATA,
        },
        "grid": {
            "resolution_x": RESOLUTION,
            "resolution_y": RESOLUTION,
            "tile_width_px": TILE_PX,
            "tile_height_px": TILE_PX,
            "origin_x": ORIGIN_X,
            "origin_y": ORIGIN_Y,
            "axis_order": "east_north",
            "tile_span_x": SPAN,
            "tile_span_y": SPAN,
        },
        "storage": {
            "format": "geotiff",
            "compression": "deflate",
            "predictor": 3,
            "block_width_px": 16,
            "block_height_px": 16,
            "tile_pattern": "height/tile_{ix}_{iy}.tif",
        },
        "sampling_contract": {
            "query_crs": QUERY_CRS,
            "pixel_location": "center",
            "interpolation": "bilinear",
        },
        "tiles": {"count": len(PRESENT_TILES), "index": [list(t) for t in PRESENT_TILES]},
    }
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def tile_of(x: float, y: float) -> tuple[int, int]:
    return math.floor((x - ORIGIN_X) / SPAN), math.floor((ORIGIN_Y - y) / SPAN)


def expected_height(x: float, y: float) -> float:
    """What the sampler must return at projected ``(x, y)``.

    Mirrors the contract rather than the implementation: a plane interpolates
    exactly, so the answer is the plane value unless one of the four
    surrounding pixel centres is missing, absent or NoData.
    """
    col_f = (x - ORIGIN_X) / RESOLUTION - 0.5
    row_f = (ORIGIN_Y - y) / RESOLUTION - 0.5
    col0, row0 = math.floor(col_f), math.floor(row_f)

    for drow in (0, 1):
        for dcol in (0, 1):
            cx = ORIGIN_X + (col0 + dcol + 0.5) * RESOLUTION
            cy = ORIGIN_Y - (row0 + drow + 0.5) * RESOLUTION
            ix, iy = tile_of(cx, cy)
            if (ix, iy) not in PRESENT_TILES:
                return math.nan
            col_in_tile = int(math.floor((cx - (ORIGIN_X + ix * SPAN)) / RESOLUTION))
            row_in_tile = int(math.floor(((ORIGIN_Y - iy * SPAN) - cy) / RESOLUTION))
            if not (0 <= col_in_tile < TILE_PX and 0 <= row_in_tile < TILE_PX):
                return math.nan
            if (
                (ix, iy) == (0, 1)
                and NODATA_ROWS.start <= row_in_tile < NODATA_ROWS.stop
                and NODATA_COLS.start <= col_in_tile < NODATA_COLS.stop
            ):
                return math.nan
    return plane(x, y)


def write_expectations(root: Path) -> None:
    # The cases below are expressed in projected coordinates, but what the
    # sampler is handed is a WGS84 coordinate.  PROJ's EPSG:4326 -> EPSG:5514
    # operation is not the exact inverse of its EPSG:5514 -> EPSG:4326 one --
    # they differ by about a millimetre -- so the expected height is computed
    # from the *forward* transform of the lon/lat that goes into the file, the
    # same direction examples/sampler.py and the C++ sampler take.
    to_query = Transformer.from_crs(HORIZONTAL_CRS, QUERY_CRS, always_xy=True)
    to_stored = Transformer.from_crs(QUERY_CRS, HORIZONTAL_CRS, always_xy=True)

    # Projected points chosen to cover each case of the contract.
    cases = [
        ("tile interior", ORIGIN_X + 5.3, ORIGIN_Y - 4.7),
        ("tile interior, other tile", ORIGIN_X + SPAN + 9.1, ORIGIN_Y - 12.25),
        ("across the vertical seam", ORIGIN_X + SPAN - 0.25, ORIGIN_Y - 6.5),
        ("across the vertical seam, other side", ORIGIN_X + SPAN + 0.25, ORIGIN_Y - 6.5),
        ("exactly on the vertical seam", ORIGIN_X + SPAN, ORIGIN_Y - 6.5),
        ("across the horizontal seam", ORIGIN_X + 3.5, ORIGIN_Y - SPAN + 0.25),
        ("on a pixel centre", ORIGIN_X + 10.5, ORIGIN_Y - 10.5),
        ("inside the NoData patch", ORIGIN_X + 10.2, ORIGIN_Y - SPAN - 10.2),
        ("over the absent tile", ORIGIN_X + SPAN + 16.0, ORIGIN_Y - SPAN - 16.0),
        ("west of the dataset", ORIGIN_X - 10.0, ORIGIN_Y - 16.0),
        ("north of the dataset", ORIGIN_X + 16.0, ORIGIN_Y + 10.0),
        ("far outside the dataset", ORIGIN_X + 10_000.0, ORIGIN_Y - 10_000.0),
    ]

    lines = ["# lon,lat,expected,description", "# generated by test/make_fixture.py"]
    for description, x_wanted, y_wanted in cases:
        lon, lat = to_query.transform(x_wanted, y_wanted)
        x, y = to_stored.transform(lon, lat)
        expected = expected_height(x, y)
        value = "nan" if math.isnan(expected) else repr(expected)
        lines.append(f"{lon!r},{lat!r},{value},{description}")
    (root / "expected.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent / "data" / "synthetic"
    root.mkdir(parents=True, exist_ok=True)
    for ix, iy in PRESENT_TILES:
        write_tile(root, ix, iy)
    write_manifest(root)

    absent = root / "height" / f"tile_{ABSENT_TILE[0]}_{ABSENT_TILE[1]}.tif"
    absent.unlink(missing_ok=True)

    write_expectations(root)
    print(f"wrote fixture to {root}")


if __name__ == "__main__":
    main()
