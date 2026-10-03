"""The worlds database: what each world is, and which tiles it needs.

``worlds.sqlite`` sits next to ``dataset.yaml``.  It records, for every world
the dataset was prepared for, the safety area, the world origin and the vertical
limits from its MRS world file, and links the world to the tiles it needs, with
the height and RGB tile files and the extent of each RGB tile.  It is rebuilt
from scratch on every run, so it always describes exactly the dataset beside it.

Schema (``PRAGMA user_version`` is :data:`SCHEMA_VERSION`)::

    meta(key PRIMARY KEY, value)
    worlds(name PRIMARY KEY, source_file, origin_units, origin_x, origin_y,
           origin_lat, origin_lon, safety_area_frame, vertical_frame, min_z, max_z,
           safety_area_wkt)
    safety_area_vertices(world, seq, lat, lon)        -- in file order
    tiles(ix, iy, height_path, rgb_path, rgb_west, rgb_south, rgb_east, rgb_north)
    world_tiles(world, ix, iy)

Coordinates are WGS84 degrees.  ``origin_x``/``origin_y`` are the world file's
values verbatim; ``origin_lat``/``origin_lon`` are the same point in degrees, or
NULL when the world file gives no usable origin.  The ``rgb_*`` columns are NULL
when the dataset has no RGB tiles.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
from pathlib import Path
from typing import Mapping, Sequence

from .config import LATLON_FRAME, WorldConfig
from .crs import QUERY_CRS, project_from
from .errors import CrsError
from .manifest import Manifest
from .tiling import TileIndex

log = logging.getLogger(__name__)

WORLDS_DB_FILENAME = "worlds.sqlite"

#: Bumped whenever the schema changes incompatibly.
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE worlds (
    name              TEXT PRIMARY KEY,
    source_file       TEXT,
    origin_units      TEXT,
    origin_x          REAL,
    origin_y          REAL,
    origin_lat        REAL,
    origin_lon        REAL,
    safety_area_frame TEXT NOT NULL,
    vertical_frame    TEXT,
    min_z             REAL,
    max_z             REAL,
    safety_area_wkt   TEXT NOT NULL
);
CREATE TABLE safety_area_vertices (
    world TEXT    NOT NULL REFERENCES worlds(name),
    seq   INTEGER NOT NULL,
    lat   REAL    NOT NULL,
    lon   REAL    NOT NULL,
    PRIMARY KEY (world, seq)
);
CREATE TABLE tiles (
    ix          INTEGER NOT NULL,
    iy          INTEGER NOT NULL,
    height_path TEXT    NOT NULL,
    rgb_path    TEXT,
    rgb_west    REAL,
    rgb_south   REAL,
    rgb_east    REAL,
    rgb_north   REAL,
    PRIMARY KEY (ix, iy)
);
CREATE TABLE world_tiles (
    world TEXT    NOT NULL REFERENCES worlds(name),
    ix    INTEGER NOT NULL,
    iy    INTEGER NOT NULL,
    PRIMARY KEY (world, ix, iy),
    FOREIGN KEY (ix, iy) REFERENCES tiles(ix, iy)
);
"""


def origin_latlon(world: WorldConfig) -> tuple[float, float] | None:
    """The world origin as ``(lat, lon)`` degrees, or ``None`` if it has none.

    A UTM origin carries no zone, so it is taken to be in the zone (and
    hemisphere) of the safety area's centroid.
    """
    origin = world.origin
    if origin is None:
        return None
    if origin.units == "LATLON":
        return origin.x, origin.y
    lon = sum(lon for lon, _ in world.points) / len(world.points)
    lat = sum(lat for _, lat in world.points) / len(world.points)
    zone = min(60, max(1, int(math.floor((lon + 180.0) / 6.0)) + 1))
    epsg = (32600 if lat >= 0 else 32700) + zone
    try:
        lons, lats = project_from([origin.x], [origin.y], source=f"EPSG:{epsg}", target=QUERY_CRS)
    except CrsError as exc:
        log.warning("world %s: cannot place its UTM origin: %s", world.name, exc)
        return None
    return float(lats[0]), float(lons[0])


def polygon_wkt(world: WorldConfig) -> str:
    """The safety area as a closed WKT polygon, in ``lon lat`` order."""
    ring = list(world.points) + [world.points[0]]
    return "POLYGON((" + ", ".join(f"{lon!r} {lat!r}" for lon, lat in ring) + "))"


def write_worlds_db(
    output_dir: Path,
    manifest: Manifest,
    worlds: Sequence[WorldConfig],
    world_tiles: Mapping[str, Sequence[TileIndex]],
    rgb_bounds: Mapping[TileIndex, tuple[float, float, float, float]] | None = None,
) -> Path:
    """Build ``worlds.sqlite`` in ``output_dir`` and record it in ``manifest``.

    ``world_tiles`` maps each world's name to the tiles it needs; ``rgb_bounds``
    gives each RGB tile's ``(west, south, east, north)`` when there are RGB tiles.
    The database is built beside its final name and renamed into place, so a
    reader never sees it half-written.
    """
    output_dir = Path(output_dir)
    path = output_dir / WORLDS_DB_FILENAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    rgb_bounds = rgb_bounds or {}

    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "query_crs": QUERY_CRS,
        "height_tile_pattern": manifest.tile_pattern,
        "height_horizontal_crs": manifest.horizontal_crs,
    }
    if manifest.rgb is not None:
        meta["rgb_tile_pattern"] = manifest.rgb.tile_pattern
        meta["rgb_crs"] = manifest.rgb.crs

    try:
        connection = sqlite3.connect(tmp)
        try:
            connection.executescript(SCHEMA)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.executemany(
                "INSERT INTO meta VALUES (?, ?)", sorted(meta.items())
            )
            for tile in manifest.tiles:
                bounds = rgb_bounds.get(tile)
                connection.execute(
                    "INSERT INTO tiles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        tile.ix,
                        tile.iy,
                        manifest.tile_relative_path(tile),
                        (
                            manifest.rgb.tile_pattern.format(ix=tile.ix, iy=tile.iy)
                            if manifest.rgb is not None
                            else None
                        ),
                        *(bounds if bounds is not None else (None, None, None, None)),
                    ),
                )
            for world in sorted(worlds, key=lambda w: w.name):
                origin = world.origin
                latlon = origin_latlon(world)
                connection.execute(
                    "INSERT INTO worlds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        world.name,
                        world.path.name if world.path is not None else None,
                        origin.units if origin else None,
                        origin.x if origin else None,
                        origin.y if origin else None,
                        latlon[0] if latlon else None,
                        latlon[1] if latlon else None,
                        LATLON_FRAME,
                        world.vertical_frame,
                        world.min_z,
                        world.max_z,
                        polygon_wkt(world),
                    ),
                )
                connection.executemany(
                    "INSERT INTO safety_area_vertices VALUES (?, ?, ?, ?)",
                    [(world.name, seq, lat, lon) for seq, (lon, lat) in enumerate(world.points)],
                )
                connection.executemany(
                    "INSERT INTO world_tiles VALUES (?, ?, ?)",
                    [(world.name, t.ix, t.iy) for t in sorted(set(world_tiles.get(world.name, ())))],
                )
            connection.commit()
        finally:
            connection.close()
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    manifest.worlds_db_path = WORLDS_DB_FILENAME
    manifest.worlds_db_schema_version = SCHEMA_VERSION
    return path
