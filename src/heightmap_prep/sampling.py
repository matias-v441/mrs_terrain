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

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from .crs import QUERY_CRS
from .errors import ValidationError
from .manifest import Manifest
from .tiling import TileIndex

TEST_POINTS_FILENAME = "test_points.csv"

_thread_local = threading.local()


def _query_transformer(target_crs: str) -> Transformer:
    """The EPSG:4326 -> ``target_crs`` transformer, built once per thread.

    Building one costs far more than sampling with it, and samplers are opened
    often; a transformer is not shared between threads.
    """
    cache = _thread_local.__dict__.setdefault("transformers", {})
    if target_crs not in cache:
        cache[target_crs] = Transformer.from_crs(QUERY_CRS, target_crs, always_xy=True)
    return cache[target_crs]


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
        self._to_stored = _query_transformer(self.manifest.horizontal_crs)

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

    def _pixels(self, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """The stored values of global pixels ``(cols, rows)``, NaN where there are none.

        Reads one window per tile involved, spanning the pixels asked of it.
        """
        grid = self.grid
        values = np.full(cols.shape, np.nan)
        tile_cols = cols // grid.tile_width_px
        tile_rows = rows // grid.tile_height_px
        for ix, iy in sorted(set(zip(tile_cols.tolist(), tile_rows.tolist()))):
            dataset = self._dataset(TileIndex(ix, iy))
            if dataset is None:
                continue
            here = (tile_cols == ix) & (tile_rows == iy)
            col_in = cols[here] - ix * grid.tile_width_px
            row_in = rows[here] - iy * grid.tile_height_px
            inside = (
                (col_in >= 0) & (col_in < dataset.width) & (row_in >= 0) & (row_in < dataset.height)
            )
            found = np.full(col_in.shape, np.nan)
            if inside.any():
                c, r = col_in[inside], row_in[inside]
                c0, r0 = int(c.min()), int(r.min())
                block = dataset.read(
                    1, window=Window(c0, r0, int(c.max()) - c0 + 1, int(r.max()) - r0 + 1)
                ).astype(np.float64)
                picked = block[r - r0, c - c0]
                picked[(picked == self.manifest.nodata) | ~np.isfinite(picked)] = np.nan
                found[inside] = picked
            values[here] = found
        return values

    def sample_many(self, lons: Sequence[float], lats: Sequence[float]) -> np.ndarray:
        """Bilinearly interpolated heights at WGS84 coordinates, NaN where there are none."""
        grid = self.grid
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)
        shape = np.broadcast(lons, lats).shape
        xs, ys = self._to_stored.transform(
            np.broadcast_to(lons, shape).ravel(), np.broadcast_to(lats, shape).ravel()
        )
        xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
        # Values sit at pixel centres, hence the half-pixel shift (section 23).
        col_f = (xs - grid.origin_x) / grid.resolution_x - 0.5
        row_f = (grid.origin_y - ys) / grid.resolution_y - 0.5
        col0, row0 = np.floor(col_f), np.floor(row_f)
        fx, fy = col_f - col0, row_f - row0
        col0, row0 = col0.astype(np.int64), row0.astype(np.int64)

        corners = self._pixels(
            np.concatenate([col0, col0 + 1, col0, col0 + 1]),
            np.concatenate([row0, row0, row0 + 1, row0 + 1]),
        )
        v00, v10, v01, v11 = np.split(corners, 4)
        # Any NaN corner makes the result NaN: no height without all four.
        top = v00 * (1 - fx) + v10 * fx
        bottom = v01 * (1 - fx) + v11 * fx
        return (top * (1 - fy) + bottom * fy).reshape(shape)

    def sample(self, lon: float, lat: float) -> float:
        """Bilinearly interpolated height at a WGS84 coordinate, or NaN."""
        return float(self.sample_many([lon], [lat])[0])


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
