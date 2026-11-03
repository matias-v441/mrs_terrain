"""Read the worlds of a prepared dataset: safety areas, origins, heights and imagery.

A consumer-side view of a dataset directory, built on ``dataset.yaml`` and
``worlds.sqlite``.  It needs neither the network nor PROJ grids::

    from heightmap_prep.world_dataset import WorldDataset

    with WorldDataset("dataset") as dataset:
        print(dataset.world_names())
        world = dataset.world("temesvar_field")
        print(world.safety_area, world.origin)
        for tile in dataset.wgs_tiles("temesvar_field"):
            print(tile.tile, tile.bounds, tile.height_path, tile.rgb_path)
        heights = dataset.read_height("temesvar_field")
        image = dataset.read_rgb("temesvar_field")

or from the shell::

    heightmap-worlds DATASET                         # list the worlds
    heightmap-worlds DATASET WORLD                   # describe one
    heightmap-worlds DATASET WORLD --export world.tif --export-height heights.tif

Heights and imagery come from the WGS tiles, cropped to the world's lat/lon
bounding box snapped outward to the WGS height lattice, which is exactly where
those tiles hold data for it.  The height and RGB images of a world therefore
cover the same area, with ``height_factor`` RGB pixels per height pixel.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import rasterio.windows
from affine import Affine

from .manifest import Manifest, WgsInfo
from .tiling import TileIndex
from .wgs import Lattice, LatticeWindow
from .worlds_db import SCHEMA_VERSION, WORLDS_DB_FILENAME


class WorldNotFoundError(KeyError):
    """The dataset was not prepared for a world of that name."""


@dataclass(frozen=True)
class Origin:
    """A world origin: as the world file gives it, and in degrees."""

    units: str
    x: float
    y: float
    #: The same point in WGS84 degrees, when it could be placed.
    lat: float | None
    lon: float | None


@dataclass(frozen=True)
class World:
    name: str
    #: Safety area vertices as ``(lat, lon)`` degrees, in world-file order.
    safety_area: tuple[tuple[float, float], ...]
    origin: Origin | None
    #: The frame ``min_z``/``max_z`` are given in (usually ``world_origin``).
    vertical_frame: str | None
    min_z: float | None
    max_z: float | None
    #: EPSG:5514 height tiles a sampler reads for this world.
    height_tiles: tuple[TileIndex, ...]
    #: WGS tiles holding this world's heights and imagery.
    wgs_tiles: tuple[TileIndex, ...]
    source_file: str | None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)`` of the safety area, in degrees."""
        lats = [lat for lat, _ in self.safety_area]
        lons = [lon for _, lon in self.safety_area]
        return min(lons), min(lats), max(lons), max(lats)


@dataclass(frozen=True)
class WgsTile:
    tile: TileIndex
    #: ``(west, south, east, north)`` in degrees.
    bounds: tuple[float, float, float, float]
    height_path: Path
    #: ``None`` when the dataset has no RGB imagery.
    rgb_path: Path | None


@dataclass
class _Image:
    transform: Affine
    crs: str

    @property
    def _shape(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        height, width = self._shape
        west, north = self.transform.c, self.transform.f
        return west, north + height * self.transform.e, west + width * self.transform.a, north


@dataclass
class RgbImage(_Image):
    """A piece of the RGB lattice, in EPSG:4326."""

    #: ``(3, height, width)`` uint8 red, green and blue.
    data: np.ndarray
    #: ``(height, width)``; True where there is imagery.
    mask: np.ndarray

    @property
    def _shape(self) -> tuple[int, int]:
        return self.mask.shape

    def write(self, path: Path) -> Path:
        """Save as an RGB GeoTIFF with an internal mask."""
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 3,
            "width": self.data.shape[2],
            "height": self.data.shape[1],
            "transform": self.transform,
            "crs": self.crs,
            "tiled": True,
            "compress": "deflate",
            "photometric": "rgb",
        }
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(path, "w", **profile) as dataset:
                dataset.write(self.data)
                dataset.write_mask(np.where(self.mask, 255, 0).astype(np.uint8))
        return Path(path)


@dataclass
class HeightImage(_Image):
    """A piece of the WGS height lattice, in EPSG:4326."""

    #: ``(height, width)`` float32 heights; NaN where there are none.
    data: np.ndarray
    vertical_crs: str

    @property
    def _shape(self) -> tuple[int, int]:
        return self.data.shape

    def write(self, path: Path) -> Path:
        """Save as a float32 GeoTIFF with NaN as NoData."""
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "width": self.data.shape[1],
            "height": self.data.shape[0],
            "transform": self.transform,
            "crs": self.crs,
            "nodata": float("nan"),
            "tiled": True,
            "compress": "deflate",
            "predictor": 3,
        }
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(self.data.astype(np.float32), 1)
        return Path(path)


class WorldDataset:
    """The worlds of a prepared dataset directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.manifest = Manifest.read(self.root)
        db_name = self.manifest.worlds_db_path or WORLDS_DB_FILENAME
        db_path = self.root / db_name
        if not db_path.is_file():
            raise FileNotFoundError(
                f"{self.root} has no worlds database ({db_name}); prepare it again with "
                "this version of heightmap-prep"
            )
        self._db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            self._db.close()
            raise ValueError(
                f"{db_path} has schema version {version}; this reader understands "
                f"{SCHEMA_VERSION}"
            )

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "WorldDataset":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- worlds ------------------------------------------------------------

    def world_names(self) -> list[str]:
        return [row[0] for row in self._db.execute("SELECT name FROM worlds ORDER BY name")]

    def worlds(self) -> list[World]:
        return [self.world(name) for name in self.world_names()]

    def world(self, name: str) -> World:
        row = self._db.execute(
            "SELECT source_file, origin_units, origin_x, origin_y, origin_lat, origin_lon, "
            "vertical_frame, min_z, max_z FROM worlds WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            raise WorldNotFoundError(
                f"no world {name!r} in {self.root}; it has {', '.join(self.world_names())}"
            )
        source_file, units, x, y, lat, lon, vertical_frame, min_z, max_z = row
        vertices = self._db.execute(
            "SELECT lat, lon FROM safety_area_vertices WHERE world = ? ORDER BY seq", (name,)
        ).fetchall()
        return World(
            name=name,
            safety_area=tuple((float(a), float(b)) for a, b in vertices),
            origin=Origin(units, x, y, lat, lon) if units is not None else None,
            vertical_frame=vertical_frame,
            min_z=min_z,
            max_z=max_z,
            height_tiles=self._tiles("world_height_tiles", name),
            wgs_tiles=self._tiles("world_wgs_tiles", name),
            source_file=source_file,
        )

    def _tiles(self, table: str, name: str) -> tuple[TileIndex, ...]:
        rows = self._db.execute(
            f"SELECT ix, iy FROM {table} WHERE world = ? ORDER BY iy, ix", (name,)
        ).fetchall()
        return tuple(TileIndex(ix, iy) for ix, iy in rows)

    def safety_area(self, name: str) -> tuple[tuple[float, float], ...]:
        """The safety area as ``(lat, lon)`` vertices."""
        return self.world(name).safety_area

    # --- WGS tiles -----------------------------------------------------------

    @property
    def wgs(self) -> WgsInfo:
        if self.manifest.wgs is None:
            raise ValueError(f"{self.root} has no WGS tiles")
        return self.manifest.wgs

    @property
    def has_rgb(self) -> bool:
        return self.manifest.wgs is not None and self.manifest.wgs.rgb is not None

    def wgs_tiles(self, name: str) -> list[WgsTile]:
        """The WGS tiles holding the heights and imagery of world ``name``."""
        rows = self._db.execute(
            "SELECT t.ix, t.iy, t.west, t.south, t.east, t.north, t.height_path, t.rgb_path "
            "FROM world_wgs_tiles w JOIN wgs_tiles t ON t.ix = w.ix AND t.iy = w.iy "
            "WHERE w.world = ? ORDER BY t.iy, t.ix",
            (self.world(name).name,),
        ).fetchall()
        return [
            WgsTile(
                TileIndex(ix, iy),
                (west, south, east, north),
                self.root / height_path,
                self.root / rgb_path if rgb_path is not None else None,
            )
            for ix, iy, west, south, east, north, height_path, rgb_path in rows
        ]

    def fill_window(self, name: str) -> LatticeWindow:
        """Where the WGS tiles hold data for world ``name``, on the height lattice."""
        return self.wgs.grid().fill_for_lonlat(self.world(name).bounds)

    def read_height(self, name: str) -> HeightImage:
        """World ``name``'s heights on the WGS height lattice."""
        lattice = self.wgs.grid().height
        target = self.fill_window(name)
        data = np.full((1, target.height, target.width), np.nan, dtype=np.float32)
        valid = np.zeros((target.height, target.width), dtype=bool)
        self._mosaic(lattice, target, [t.height_path for t in self.wgs_tiles(name)], data, valid)
        return HeightImage(
            transform=lattice.window_transform(target),
            crs=self.wgs.crs,
            data=data[0],
            vertical_crs=self.manifest.vertical_crs,
        )

    def read_rgb(self, name: str) -> RgbImage:
        """World ``name``'s imagery on the RGB lattice, covering what :meth:`read_height` does."""
        if not self.has_rgb:
            raise ValueError(f"{self.root} was prepared without RGB imagery (--include-rgb)")
        grid = self.wgs.grid()
        target = self.fill_window(name).scaled(grid.height_factor)
        paths = [t.rgb_path for t in self.wgs_tiles(name) if t.rgb_path is not None]
        data = np.zeros((3, target.height, target.width), dtype=np.uint8)
        mask = np.zeros((target.height, target.width), dtype=bool)
        self._mosaic(grid.rgb, target, paths, data, mask)
        return RgbImage(
            transform=grid.rgb.window_transform(target), crs=self.wgs.crs, data=data, mask=mask
        )

    @staticmethod
    def _mosaic(
        lattice: Lattice,
        target: LatticeWindow,
        paths: Sequence[Path],
        data: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        """Paste the part of each tile inside ``target``; tiles never overlap."""
        for path in paths:
            with rasterio.open(path) as dataset:
                window = lattice.window_of_transform(
                    dataset.transform, dataset.width, dataset.height
                )
                if window is None:
                    raise ValueError(f"{path} is not on the dataset's WGS lattice")
                overlap = window.intersection(target)
                if overlap is None:
                    continue
                col, row = overlap.offset_in(window)
                read = rasterio.windows.Window(col, row, overlap.width, overlap.height)
                pixels = dataset.read(window=read)
                good = dataset.read_masks(1, window=read) > 0
            col, row = overlap.offset_in(target)
            region = (slice(row, row + overlap.height), slice(col, col + overlap.width))
            data[(slice(None), *region)] = np.where(good, pixels, data[(slice(None), *region)])
            valid[region] = good


# --- command line ------------------------------------------------------------


def _describe(world: World, dataset: WorldDataset) -> str:
    lines = [f"world {world.name}" + (f" ({world.source_file})" if world.source_file else "")]
    if world.origin is not None:
        origin = world.origin
        where = (
            f" = {origin.lat:.7f}, {origin.lon:.7f} (lat, lon)"
            if origin.lat is not None and origin.lon is not None
            else ""
        )
        lines.append(f"  origin: {origin.units} {origin.x}, {origin.y}{where}")
    else:
        lines.append("  origin: none")
    if world.min_z is not None or world.max_z is not None:
        lines.append(f"  height: {world.min_z} .. {world.max_z} m in {world.vertical_frame}")
    lines.append(f"  safety area ({len(world.safety_area)} vertices, lat, lon):")
    lines.extend(f"    {lat:.7f}, {lon:.7f}" for lat, lon in world.safety_area)
    lines.append(
        "  EPSG:5514 tiles: " + ", ".join(f"({t.ix}, {t.iy})" for t in world.height_tiles)
    )
    for tile in dataset.wgs_tiles(world.name):
        lines.append(f"  wgs tile ({tile.tile.ix}, {tile.tile.iy}):")
        lines.append(f"    height: {tile.height_path.relative_to(dataset.root)}")
        if tile.rgb_path is not None:
            lines.append(f"    rgb: {tile.rgb_path.relative_to(dataset.root)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="heightmap-worlds",
        description="List the worlds of a prepared dataset, describe one, or export it.",
    )
    parser.add_argument("dataset", type=Path, help="prepared dataset directory")
    parser.add_argument("world", nargs="?", help="world to describe")
    parser.add_argument(
        "--export", type=Path, metavar="PATH", help="write the world's imagery as a GeoTIFF"
    )
    parser.add_argument(
        "--export-height", type=Path, metavar="PATH",
        help="write the world's WGS heights as a GeoTIFF",
    )
    args = parser.parse_args(argv)
    if (args.export or args.export_height) and not args.world:
        parser.error("--export and --export-height need a WORLD")

    try:
        with WorldDataset(args.dataset) as dataset:
            if not args.world:
                print("\n".join(dataset.world_names()))
                return 0
            world = dataset.world(args.world)
            print(_describe(world, dataset))
            if args.export:
                image = dataset.read_rgb(world.name)
                image.write(args.export)
                print(f"wrote {image.data.shape[2]}x{image.data.shape[1]} px to {args.export}")
            if args.export_height:
                heights = dataset.read_height(world.name)
                heights.write(args.export_height)
                print(
                    f"wrote {heights.data.shape[1]}x{heights.data.shape[0]} px to "
                    f"{args.export_height}"
                )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
