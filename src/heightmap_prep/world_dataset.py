"""Read the worlds of a prepared dataset: safety areas, origins and RGB imagery.

A consumer-side view of a dataset directory, built on ``dataset.yaml`` and
``worlds.sqlite``.  It needs neither the network nor PROJ grids::

    from heightmap_prep.world_dataset import WorldDataset

    with WorldDataset("dataset") as dataset:
        print(dataset.world_names())
        world = dataset.world("temesvar_field")
        print(world.safety_area, world.origin)
        for tile in dataset.rgb_tiles("temesvar_field"):
            print(tile.path, tile.bounds)
        image = dataset.read_rgb("temesvar_field", margin_m=20.0)

or from the shell::

    heightmap-worlds DATASET                         # list the worlds
    heightmap-worlds DATASET WORLD                   # describe one
    heightmap-worlds DATASET WORLD --export world.tif --margin 10
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

from .manifest import Manifest
from .rgb import RgbGrid, metres_to_degrees
from .tiling import TileIndex
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
    #: Tiles the world needs; height and RGB tiles share these indices.
    tiles: tuple[TileIndex, ...]
    source_file: str | None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)`` of the safety area, in degrees."""
        lats = [lat for lat, _ in self.safety_area]
        lons = [lon for _, lon in self.safety_area]
        return min(lons), min(lats), max(lons), max(lats)


@dataclass(frozen=True)
class RgbTile:
    tile: TileIndex
    path: Path
    #: ``(west, south, east, north)`` in degrees.
    bounds: tuple[float, float, float, float]


@dataclass
class RgbImage:
    """A piece of the RGB lattice, in EPSG:4326."""

    #: ``(3, height, width)`` uint8 red, green and blue.
    data: np.ndarray
    #: ``(height, width)``; True where there is imagery.
    mask: np.ndarray
    transform: Affine
    crs: str

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        height, width = self.mask.shape
        west, north = self.transform.c, self.transform.f
        return west, north + height * self.transform.e, west + width * self.transform.a, north

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
        tiles = self._db.execute(
            "SELECT ix, iy FROM world_tiles WHERE world = ? ORDER BY iy, ix", (name,)
        ).fetchall()
        return World(
            name=name,
            safety_area=tuple((float(a), float(b)) for a, b in vertices),
            origin=Origin(units, x, y, lat, lon) if units is not None else None,
            vertical_frame=vertical_frame,
            min_z=min_z,
            max_z=max_z,
            tiles=tuple(TileIndex(ix, iy) for ix, iy in tiles),
            source_file=source_file,
        )

    def safety_area(self, name: str) -> tuple[tuple[float, float], ...]:
        """The safety area as ``(lat, lon)`` vertices."""
        return self.world(name).safety_area

    # --- imagery -----------------------------------------------------------

    @property
    def has_rgb(self) -> bool:
        return self.manifest.rgb is not None

    def rgb_tiles(self, name: str) -> list[RgbTile]:
        """The RGB tiles holding the imagery of world ``name``."""
        if not self.has_rgb:
            raise ValueError(f"{self.root} was prepared without RGB imagery (--include-rgb)")
        rows = self._db.execute(
            "SELECT t.ix, t.iy, t.rgb_path, t.rgb_west, t.rgb_south, t.rgb_east, t.rgb_north "
            "FROM world_tiles w JOIN tiles t ON t.ix = w.ix AND t.iy = w.iy "
            "WHERE w.world = ? ORDER BY t.iy, t.ix",
            (self.world(name).name,),
        ).fetchall()
        return [
            RgbTile(TileIndex(ix, iy), self.root / path, (west, south, east, north))
            for ix, iy, path, west, south, east, north in rows
        ]

    def read_rgb(self, name: str, margin_m: float = 0.0) -> RgbImage:
        """World ``name``'s imagery, cropped to its safety area's bounding box.

        ``margin_m`` grows the box on every side; the tiles only hold imagery a
        few metres beyond the safety area, so a wide margin comes back masked.
        Where RGB tiles overlap they hold identical pixels, so the first one
        with imagery wins.
        """
        world = self.world(name)
        tiles = self.rgb_tiles(name)
        assert self.manifest.rgb is not None
        grid: RgbGrid = self.manifest.rgb.grid()

        west, south, east, north = world.bounds
        pad_lon, pad_lat = metres_to_degrees(margin_m, (south + north) / 2)
        target = grid.window_for_lonlat(
            (west - pad_lon, south - pad_lat, east + pad_lon, north + pad_lat)
        )
        data = np.zeros((3, target.height, target.width), dtype=np.uint8)
        mask = np.zeros((target.height, target.width), dtype=bool)

        for tile in tiles:
            with rasterio.open(tile.path) as dataset:
                window = grid.window_of_transform(dataset.transform, dataset.width, dataset.height)
                if window is None:
                    raise ValueError(f"{tile.path} is not on the dataset's RGB lattice")
                overlap = window.intersection(target)
                if overlap is None:
                    continue
                col, row = overlap.offset_in(window)
                read = rasterio.windows.Window(col, row, overlap.width, overlap.height)
                pixels = dataset.read(window=read)
                valid = dataset.read_masks(1, window=read) > 0
            col, row = overlap.offset_in(target)
            region = (slice(row, row + overlap.height), slice(col, col + overlap.width))
            fresh = valid & ~mask[region]
            data[(slice(None), *region)][:, fresh] = pixels[:, fresh]
            mask[region] |= fresh

        return RgbImage(
            data=data, mask=mask, transform=grid.window_transform(target), crs=self.manifest.rgb.crs
        )


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
    lines.append("  tiles: " + ", ".join(f"({t.ix}, {t.iy})" for t in world.tiles))
    if dataset.has_rgb:
        for tile in dataset.rgb_tiles(world.name):
            lines.append(f"  rgb: {tile.path.relative_to(dataset.root)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="heightmap-worlds",
        description="List the worlds of a prepared dataset, describe one, or export its imagery.",
    )
    parser.add_argument("dataset", type=Path, help="prepared dataset directory")
    parser.add_argument("world", nargs="?", help="world to describe")
    parser.add_argument(
        "--export", type=Path, metavar="PATH", help="write the world's imagery as a GeoTIFF"
    )
    parser.add_argument(
        "--margin", type=float, default=0.0, metavar="METRES",
        help="grow the exported area beyond the safety area (default: 0)",
    )
    args = parser.parse_args(argv)
    if args.export and not args.world:
        parser.error("--export needs a WORLD")

    try:
        with WorldDataset(args.dataset) as dataset:
            if not args.world:
                print("\n".join(dataset.world_names()))
                return 0
            world = dataset.world(args.world)
            print(_describe(world, dataset))
            if args.export:
                image = dataset.read_rgb(world.name, margin_m=args.margin)
                image.write(args.export)
                print(f"wrote {image.data.shape[2]}x{image.data.shape[1]} px to {args.export}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
