"""Reading heights back out of a prepared dataset, and the test points file.

:class:`DatasetSampler` implements the runtime sampling contract (specification
sections 22 and 23) against the tiles on disk: one EPSG:4326 -> stored-CRS
transformation, bilinear interpolation between the four surrounding pixel
centres, and no height at all when any of them is missing or NoData.  It
computes the same thing, the same way, as ``examples/sampler.py``.

The dataset ships reference heights computed this way at every safety area
corner in ``test_points.csv``, one ``lat,lon,height`` row per point, so other
sampler implementations can check themselves against it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from .crs import QUERY_CRS
from .errors import ValidationError
from .manifest import Manifest
from .tiling import TileIndex

TEST_POINTS_FILENAME = "test_points.csv"


@dataclass(frozen=True)
class ReferencePoint:
    """A WGS84 location and the height the dataset gives there."""

    lat: float
    lon: float
    height: float


class DatasetSampler:
    """Bilinear heights from a prepared dataset, NaN where there are none."""

    def __init__(self, root: Path, manifest: Manifest | None = None) -> None:
        self.root = Path(root)
        self.manifest = manifest or Manifest.read(self.root)
        self.grid = self.manifest.grid()
        self._declared = set(self.manifest.tiles)
        self._open: dict[TileIndex, rasterio.DatasetReader | None] = {}
        self._to_stored = Transformer.from_crs(
            QUERY_CRS, self.manifest.horizontal_crs, always_xy=True
        )

    def __enter__(self) -> "DatasetSampler":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        for dataset in self._open.values():
            if dataset is not None:
                dataset.close()
        self._open.clear()

    def _dataset(self, tile: TileIndex) -> rasterio.DatasetReader | None:
        if tile not in self._open:
            path = self.manifest.tile_path(self.root, tile)
            self._open[tile] = (
                rasterio.open(path) if tile in self._declared and path.is_file() else None
            )
        return self._open[tile]

    def _pixel(self, col: int, row: int) -> float:
        """The stored value of global pixel ``(col, row)``, or NaN."""
        grid = self.grid
        tile = TileIndex(col // grid.tile_width_px, row // grid.tile_height_px)
        dataset = self._dataset(tile)
        if dataset is None:
            return math.nan
        col_in_tile = col - tile.ix * grid.tile_width_px
        row_in_tile = row - tile.iy * grid.tile_height_px
        if not (0 <= col_in_tile < dataset.width and 0 <= row_in_tile < dataset.height):
            return math.nan
        value = float(dataset.read(1, window=Window(col_in_tile, row_in_tile, 1, 1))[0, 0])
        if value == self.manifest.nodata or not math.isfinite(value):
            return math.nan
        return value

    def sample(self, lon: float, lat: float) -> float:
        """Bilinearly interpolated height at a WGS84 coordinate, or NaN."""
        grid = self.grid
        x, y = self._to_stored.transform(lon, lat)
        # Values sit at pixel centres, hence the half-pixel shift (section 23).
        col_f = (x - grid.origin_x) / grid.resolution_x - 0.5
        row_f = (grid.origin_y - y) / grid.resolution_y - 0.5
        col0, row0 = math.floor(col_f), math.floor(row_f)
        fx, fy = col_f - col0, row_f - row0

        v00 = self._pixel(col0, row0)
        v10 = self._pixel(col0 + 1, row0)
        v01 = self._pixel(col0, row0 + 1)
        v11 = self._pixel(col0 + 1, row0 + 1)
        if any(math.isnan(v) for v in (v00, v10, v01, v11)):
            return math.nan
        top = v00 * (1 - fx) + v10 * fx
        bottom = v01 * (1 - fx) + v11 * fx
        return top * (1 - fy) + bottom * fy


def write_test_points(path: Path, points: Sequence[ReferencePoint]) -> None:
    """Write ``lat,lon,height`` rows atomically, at full float precision."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(f"{p.lat!r},{p.lon!r},{p.height!r}\n" for p in points), encoding="utf-8"
    )
    tmp.replace(path)


def read_test_points(path: Path) -> list[ReferencePoint]:
    """Parse a test points file."""
    path = Path(path)
    points: list[ReferencePoint] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(",")
        try:
            if len(fields) != 3:
                raise ValueError(f"expected 3 fields, got {len(fields)}")
            lat, lon, height = (float(field) for field in fields)
        except ValueError as exc:
            raise ValidationError(f"{path}:{number}: malformed test point {line!r}: {exc}") from exc
        points.append(ReferencePoint(lat=lat, lon=lon, height=height))
    return points
