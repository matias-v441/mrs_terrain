#!/usr/bin/env python3
"""A minimal runtime sampler for a prepared dataset.

This is not part of the library — it is the reference implementation of the
contract described in specification sections 22 and 23, and a quick way to check
that a prepared dataset really is usable from WGS84 lon/lat:

    python examples/sampler.py ./prepared 14.42 50.08

It performs one horizontal transformation, computes the tile index from the
manifest grid, and bilinearly interpolates the stored heights.  No vertical
datum transformation happens at runtime.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import rasterio
import yaml
from pyproj import Transformer


class HeightSampler:
    def __init__(self, dataset_dir: Path) -> None:
        self.root = Path(dataset_dir)
        manifest = yaml.safe_load((self.root / "dataset.yaml").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise RuntimeError(f"dataset status is {manifest.get('status')!r}, expected 'complete'")

        grid = manifest["grid"]
        self.origin_x = float(grid["origin_x"])
        self.origin_y = float(grid["origin_y"])
        self.res_x = float(grid["resolution_x"])
        self.res_y = float(grid["resolution_y"])
        self.span_x = grid["tile_width_px"] * self.res_x
        self.span_y = grid["tile_height_px"] * self.res_y
        self.pattern = manifest["storage"]["tile_pattern"]
        self.nodata = float(manifest["heightmap"]["nodata"])
        self.vertical_crs = manifest["heightmap"]["vertical_crs"]
        self.vertical_datum = manifest["heightmap"]["vertical_datum"]

        self.to_stored = Transformer.from_crs(
            "EPSG:4326", manifest["heightmap"]["horizontal_crs"], always_xy=True
        )
        self._tiles: dict[tuple[int, int], rasterio.DatasetReader] = {}

    def _tile(self, ix: int, iy: int):
        key = (ix, iy)
        if key not in self._tiles:
            path = self.root / self.pattern.format(ix=ix, iy=iy)
            self._tiles[key] = rasterio.open(path) if path.is_file() else None
        return self._tiles[key]

    def _pixel(self, x: float, y: float) -> float:
        """Nearest stored sample value at projected ``(x, y)``, or NaN."""
        ix = math.floor((x - self.origin_x) / self.span_x)
        iy = math.floor((self.origin_y - y) / self.span_y)
        print(ix,iy)
        dataset = self._tile(ix, iy)
        if dataset is None:
            print("Tile not found!")
            return math.nan
        col_corner, row_corner = (~dataset.transform) @ (x, y)
        col = int(math.floor(col_corner))
        row = int(math.floor(row_corner))
        if not (0 <= col < dataset.width and 0 <= row < dataset.height):
            print(f"Out of bounds [{row},{col}] {dataset.height}x{dataset.height}")
            return math.nan
        value = float(dataset.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
        if value == self.nodata:
            print("Missing data in the raster!")
            return math.nan
        return value

    def sample(self, lon: float, lat: float) -> float:
        """Bilinearly interpolated prepared height at a WGS84 coordinate."""
        x, y = self.to_stored.transform(lon, lat)

        # Pixel values sit at pixel centres, so shift by half a pixel before
        # taking the fractional part (specification section 23).
        col_f = (x - self.origin_x) / self.res_x - 0.5
        row_f = (self.origin_y - y) / self.res_y - 0.5
        col0, row0 = math.floor(col_f), math.floor(row_f)
        fx, fy = col_f - col0, row_f - row0

        corners = []
        for drow in (0, 1):
            for dcol in (0, 1):
                cx = self.origin_x + (col0 + dcol + 0.5) * self.res_x
                cy = self.origin_y - (row0 + drow + 0.5) * self.res_y
                corners.append(self._pixel(cx, cy))
        if any(math.isnan(v) for v in corners):
            print("One of the corners is nan")
            return math.nan
        v00, v10, v01, v11 = corners
        top = v00 * (1 - fx) + v10 * fx
        bottom = v01 * (1 - fx) + v11 * fx
        return top * (1 - fy) + bottom * fy

    def close(self) -> None:
        for dataset in self._tiles.values():
            if dataset is not None:
                dataset.close()
        self._tiles.clear()


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 1
    sampler = HeightSampler(Path(argv[1]))
    lon, lat = float(argv[2]), float(argv[3])
    height = sampler.sample(lon, lat)
    print(
        f"{lon}, {lat} -> {height:.3f} m "
        f"({sampler.vertical_datum}, {sampler.vertical_crs})"
    )
    sampler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
