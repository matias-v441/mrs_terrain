"""World configs and the top-level preparation options.

A *world config* is an MRS UAV system world file.  The only part of it that
matters here is the safety area: the horizontal polygon the vehicle must stay
inside.  The prepared dataset covers exactly what a sampler needs to answer a
query anywhere in that polygon.

The relevant part of such a file looks like::

    mrs_uav_managers:
      safety_area_manager:
        safety_area:
          horizontal:
            frame_name: latlon_origin
            points: [lat, lon, lat, lon, ...]

Only ``latlon_origin`` safety areas are georeferenced on their own.  A world
config that cannot be used, for that or any other reason, is skipped with a
warning rather than failing the whole run.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from .errors import ConfigError

log = logging.getLogger(__name__)

#: Extensions recognised when scanning a worlds directory.
WORLD_SUFFIXES = (".yaml", ".yml")

#: The safety area frame whose points are absolute latitude/longitude degrees.
LATLON_FRAME = "latlon_origin"

#: ``world_origin.units`` values an origin can be given in.
ORIGIN_UNITS = ("LATLON", "UTM")

#: Vertical datum identifiers accepted on the command line and in the manifest.
VERTICAL_DATUMS = ("egm96", "wgs84-ellipsoid")

#: Nominal anchor of the global tile grid in EPSG:5514, west/north of all Czech
#: data.  The effective origin is this point shifted by less than one pixel so
#: the grid lines up with the source product (see ``align_origin``).
NOMINAL_GRID_ORIGIN = (-1_000_000.0, -800_000.0)


@dataclass(frozen=True)
class WorldOrigin:
    """``mrs_uav_managers.world_origin`` as the world file gives it.

    With ``units: LATLON``, ``x`` is the latitude and ``y`` the longitude; with
    ``units: UTM`` they are the easting and northing in the UTM zone the world
    lies in.
    """

    units: str
    x: float
    y: float


@dataclass(frozen=True)
class WorldConfig:
    """One world: a named safety area polygon."""

    name: str
    #: Safety area vertices as ``(lon, lat)`` degrees, in file order.
    points: tuple[tuple[float, float], ...]
    path: Path | None = None
    #: The world origin, when the file gives a usable one.
    origin: WorldOrigin | None = None
    #: ``safety_area.vertical``: the frame ``min_z``/``max_z`` are given in.
    vertical_frame: str | None = None
    min_z: float | None = None
    max_z: float | None = None

    def describe(self) -> str:
        """A short identifier used in log lines and error messages."""
        if self.path is None:
            return self.name
        return f"{self.name} ({self.path})"


@dataclass(frozen=True)
class PrepareOptions:
    """Everything that is not derived from the world configs themselves."""

    vertical_datum: str = "egm96"
    include_rgb: bool = False
    #: North-south ground size of an RGB pixel, in metres.
    rgb_resolution_m: float = 0.25
    rgb_jpeg_quality: int = 90
    resolution_m: float = 2.0
    tile_size_px: int = 4096
    proj_data_dir: Path | None = None
    cache_dir: Path | None = None
    workers: int = 1
    overwrite: bool = False

    # Storage / grid knobs.  These have sensible version-1 defaults and are kept
    # here so the manifest can always report exactly what produced the dataset.
    block_size_px: int = 256
    nodata: float = -9999.0
    #: Write Cloud Optimized GeoTIFFs instead of plain tiled GeoTIFFs.
    cog: bool = False
    #: West/north edge of tile (0, 0).  ``None`` means "derive from
    #: :data:`NOMINAL_GRID_ORIGIN`, nudged so the grid is in phase with the
    #: source product's native pixel grid"; an explicit value is used verbatim.
    grid_origin_x: float | None = None
    grid_origin_y: float | None = None

    # Acquisition knobs.
    request_timeout_s: float = 120.0
    request_retries: int = 3
    max_request_px: int = 2048

    # Validation knobs.
    validate_tiles: bool = True
    plausibility_checks: bool = True

    def __post_init__(self) -> None:
        if self.vertical_datum not in VERTICAL_DATUMS:
            raise ConfigError(
                f"unsupported vertical datum {self.vertical_datum!r}; "
                f"expected one of {', '.join(VERTICAL_DATUMS)}"
            )
        if not self.resolution_m > 0:
            raise ConfigError(f"resolution_m must be positive, got {self.resolution_m}")
        if not self.rgb_resolution_m > 0:
            raise ConfigError(
                f"rgb_resolution_m must be positive, got {self.rgb_resolution_m}"
            )
        if not 1 <= self.rgb_jpeg_quality <= 100:
            raise ConfigError(
                f"rgb_jpeg_quality must be between 1 and 100, got {self.rgb_jpeg_quality}"
            )
        if self.tile_size_px <= 0:
            raise ConfigError(f"tile_size_px must be positive, got {self.tile_size_px}")
        if self.block_size_px <= 0 or self.block_size_px % 16:
            raise ConfigError(
                f"block_size_px must be a positive multiple of 16, got {self.block_size_px}"
            )
        if self.workers < 1:
            raise ConfigError(f"workers must be at least 1, got {self.workers}")
        if self.max_request_px <= 0:
            raise ConfigError(f"max_request_px must be positive, got {self.max_request_px}")


# --- parsing -------------------------------------------------------------


def world_name_from_path(path: Path) -> str:
    """``worlds/world_bechovice.yaml`` -> ``bechovice``."""
    stem = Path(path).stem
    for prefix in ("world_", "world-"):
        if stem.startswith(prefix) and len(stem) > len(prefix):
            return stem[len(prefix) :]
    return stem


def _latlon_pairs(points: Any, path: Path) -> tuple[tuple[float, float], ...]:
    """Split a flat ``[lat, lon, lat, lon, ...]`` list into ``(lon, lat)`` pairs."""
    if not isinstance(points, (list, tuple)):
        raise ConfigError(
            f"{path}: safety area 'points' must be a list, got {type(points).__name__}"
        )
    if len(points) % 2:
        raise ConfigError(
            f"{path}: safety area 'points' holds {len(points)} values, which is not "
            "a whole number of lat/lon pairs"
        )
    values: list[float] = []
    for index, value in enumerate(points):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: safety area points[{index}] is not a number: {value!r}")
        if not math.isfinite(float(value)):
            raise ConfigError(f"{path}: safety area points[{index}] is not finite: {value!r}")
        values.append(float(value))

    pairs = tuple((lon, lat) for lat, lon in zip(values[0::2], values[1::2]))
    if len(pairs) < 3:
        raise ConfigError(
            f"{path}: a safety area polygon needs at least 3 vertices, got {len(pairs)}"
        )
    for lon, lat in pairs:
        if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
            raise ConfigError(
                f"{path}: safety area vertex ({lat}, {lon}) is not a valid latitude/longitude"
            )
    return pairs


def _descend(document: Any, keys: Sequence[str], path: Path) -> dict[str, Any]:
    node = document
    for depth, key in enumerate(keys):
        if not isinstance(node, dict) or not isinstance(node.get(key), dict):
            raise ConfigError(
                f"{path}: has no {'.'.join(keys[: depth + 1])} section"
            )
        node = node[key]
    return node


def parse_world(document: Any, path: Path) -> WorldConfig:
    """Turn a parsed MRS world document into a :class:`WorldConfig`.

    Raises :class:`ConfigError` when the document has no usable safety area in
    ``latlon_origin``.
    """
    horizontal = _descend(
        document,
        ("mrs_uav_managers", "safety_area_manager", "safety_area", "horizontal"),
        path,
    )
    frame = str(horizontal.get("frame_name", "")).strip()
    if frame != LATLON_FRAME:
        raise ConfigError(
            f"{path}: safety area is given in frame {frame!r}, not {LATLON_FRAME!r}"
        )
    if "points" not in horizontal:
        raise ConfigError(f"{path}: safety area has no 'points'")
    vertical = document["mrs_uav_managers"]["safety_area_manager"]["safety_area"].get("vertical")
    vertical = vertical if isinstance(vertical, dict) else {}
    frame_name = vertical.get("frame_name")
    return WorldConfig(
        name=world_name_from_path(path),
        points=_latlon_pairs(horizontal["points"], path),
        path=path,
        origin=_world_origin(document["mrs_uav_managers"].get("world_origin"), path),
        vertical_frame=str(frame_name).strip() if frame_name is not None else None,
        min_z=_optional_number(vertical.get("min_z")),
        max_z=_optional_number(vertical.get("max_z")),
    )


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _world_origin(section: Any, path: Path) -> WorldOrigin | None:
    """The world origin, or ``None`` (with a debug note) when it is unusable.

    The origin only describes the world; the dataset does not depend on it, so a
    missing or malformed one never makes a world config unusable.
    """
    if not isinstance(section, dict):
        log.debug("%s: no world_origin section", path)
        return None
    units = str(section.get("units", "")).strip().upper()
    x = _optional_number(section.get("origin_x"))
    y = _optional_number(section.get("origin_y"))
    if units not in ORIGIN_UNITS or x is None or y is None:
        log.debug("%s: world_origin %r is not usable", path, section)
        return None
    if units == "LATLON" and not (-90.0 <= x <= 90.0 and -180.0 <= y <= 180.0):
        log.debug("%s: world_origin (%r, %r) is not a valid latitude/longitude", path, x, y)
        return None
    return WorldOrigin(units=units, x=x, y=y)


def load_world(path: Path) -> WorldConfig:
    """Read one MRS world file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"world config not found: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return parse_world(document, path)


# --- discovery -----------------------------------------------------------


def discover_world_paths(inputs: Sequence[Path]) -> list[Path]:
    """Resolve CLI inputs into a concrete, ordered list of world files.

    Accepts either a single worlds *directory* or one or more world *files*, as
    required by specification section 5.  Mixing the two is rejected because the
    resulting precedence would be ambiguous.
    """
    paths = [Path(item) for item in inputs]
    if not paths:
        raise ConfigError("no world config inputs were supplied")

    directories = [p for p in paths if p.is_dir()]
    if directories and len(paths) > 1:
        raise ConfigError(
            "inputs must be either a single worlds directory or a list of world files, "
            f"not both (directory: {directories[0]})"
        )

    if directories:
        worlds_dir = directories[0]
        found = sorted(
            p
            for p in worlds_dir.iterdir()
            if p.is_file() and p.suffix.lower() in WORLD_SUFFIXES
        )
        if not found:
            raise ConfigError(
                f"worlds directory {worlds_dir} contains no "
                f"{' or '.join(WORLD_SUFFIXES)} files"
            )
        return found

    return paths


def load_worlds(inputs: Sequence[Path]) -> tuple[list[WorldConfig], dict[Path, str]]:
    """Load every usable world referenced by ``inputs``.

    Returns the worlds and, for each world config that was skipped, why.  A world
    config that cannot be loaded, or whose name an earlier one already took, is
    skipped with a warning.
    """
    worlds: list[WorldConfig] = []
    skipped: dict[Path, str] = {}
    seen: dict[str, WorldConfig] = {}
    for path in discover_world_paths(inputs):
        try:
            world = load_world(path)
            previous = seen.get(world.name)
            if previous is not None:
                raise ConfigError(
                    f"{path}: world name {world.name!r} is already taken by {previous.path}"
                )
        except ConfigError as exc:
            log.warning("skipping world config: %s", exc)
            skipped[path] = str(exc)
            continue
        seen[world.name] = world
        worlds.append(world)
    return worlds, skipped


def align_origin(
    nominal: tuple[float, float],
    native: tuple[float, float] | None,
    resolution_m: float,
) -> tuple[float, float]:
    """Shift the nominal grid anchor into phase with the source pixel grid.

    ``origin_x`` moves east and ``origin_y`` moves south by less than one pixel,
    so the anchor stays west and north of all data while every tile edge falls on
    a native pixel boundary.  With no native grid to match, the nominal anchor is
    returned unchanged.
    """
    if native is None:
        return nominal
    nominal_x, nominal_y = nominal
    native_x, native_y = native
    return (
        nominal_x + (native_x - nominal_x) % resolution_m,
        nominal_y - (nominal_y - native_y) % resolution_m,
    )
