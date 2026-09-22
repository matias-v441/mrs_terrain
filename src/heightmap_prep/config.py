"""World configuration files and the top-level preparation options.

A *world* is a named geographic region that should be present in the prepared
dataset.  Worlds are described by small YAML documents (specification section 6)
and may be supplied either as individual files or as a directory of files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from .errors import ConfigError, InvalidBoundsError

#: Extensions recognised when scanning a worlds directory.
WORLD_SUFFIXES = (".yaml", ".yml")

#: The only source identifier version 1 knows how to acquire.
DEFAULT_SOURCE = "cuzk-dmr5g"

#: Vertical datum identifiers accepted on the command line and in the manifest.
VERTICAL_DATUMS = ("egm96", "wgs84-ellipsoid")

#: Nominal anchor of the global tile grid in EPSG:5514, west/north of all Czech
#: data.  The effective origin is this point shifted by less than one pixel so
#: the grid lines up with the source product (see ``align_origin``).
NOMINAL_GRID_ORIGIN = (-1_000_000.0, -800_000.0)


@dataclass(frozen=True)
class Wgs84Bounds:
    """An axis-aligned EPSG:4326 bounding box, in degrees."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        for name in ("west", "south", "east", "north"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value != value:
                raise InvalidBoundsError(f"bounds.{name} must be a finite number, got {value!r}")
        if not -180.0 <= self.west <= 180.0 or not -180.0 <= self.east <= 180.0:
            raise InvalidBoundsError(
                f"longitudes must lie in [-180, 180]: west={self.west}, east={self.east}"
            )
        if not -90.0 <= self.south <= 90.0 or not -90.0 <= self.north <= 90.0:
            raise InvalidBoundsError(
                f"latitudes must lie in [-90, 90]: south={self.south}, north={self.north}"
            )
        if self.east <= self.west:
            raise InvalidBoundsError(
                "bounds.east must be greater than bounds.west "
                f"(antimeridian-crossing worlds are not supported): "
                f"west={self.west}, east={self.east}"
            )
        if self.north <= self.south:
            raise InvalidBoundsError(
                f"bounds.north must be greater than bounds.south: "
                f"south={self.south}, north={self.north}"
            )

    @property
    def corners(self) -> tuple[tuple[float, float], ...]:
        """The four corners as ``(lon, lat)`` pairs."""
        return (
            (self.west, self.south),
            (self.west, self.north),
            (self.east, self.north),
            (self.east, self.south),
        )


@dataclass(frozen=True)
class WorldConfig:
    """One prepared region."""

    name: str
    bounds: Wgs84Bounds
    source: str = DEFAULT_SOURCE
    resolution_m: float | None = None
    rgb_enabled: bool = False
    path: Path | None = None

    def describe(self) -> str:
        """A short identifier used in log lines and error messages."""
        if self.path is None:
            return self.name
        return f"{self.name} ({self.path})"


@dataclass(frozen=True)
class PrepareOptions:
    """Everything that is not derived from the world files themselves."""

    vertical_datum: str = "egm96"
    include_rgb: bool = False
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


# --- loading -------------------------------------------------------------


def _require_mapping(value: Any, context: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: {context} must be a mapping, got {type(value).__name__}")
    return value


def _require_float(mapping: dict[str, Any], key: str, context: str, path: Path) -> float:
    if key not in mapping:
        raise ConfigError(f"{path}: {context} is missing required key {key!r}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: {context}.{key} must be a number, got {value!r}")
    return float(value)


def parse_world(document: Any, path: Path) -> WorldConfig:
    """Turn a parsed YAML document into a :class:`WorldConfig`."""
    data = _require_mapping(document, "world configuration", path)

    name = data.get("name") or path.stem
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{path}: 'name' must be a non-empty string")
    name = name.strip()

    bounds_raw = _require_mapping(data.get("bounds"), "'bounds'", path)
    bounds_type = str(bounds_raw.get("type", "wgs84")).lower()
    if bounds_type not in ("wgs84", "epsg:4326", "4326"):
        raise ConfigError(
            f"{path}: unsupported bounds type {bounds_type!r}; "
            "version 1 accepts axis-aligned WGS84 boxes only"
        )
    try:
        bounds = Wgs84Bounds(
            west=_require_float(bounds_raw, "west", "'bounds'", path),
            south=_require_float(bounds_raw, "south", "'bounds'", path),
            east=_require_float(bounds_raw, "east", "'bounds'", path),
            north=_require_float(bounds_raw, "north", "'bounds'", path),
        )
    except InvalidBoundsError as exc:
        raise InvalidBoundsError(f"{path}: {exc}") from exc

    heightmap = data.get("heightmap") or {}
    heightmap = _require_mapping(heightmap, "'heightmap'", path)
    source = str(heightmap.get("source", DEFAULT_SOURCE))
    resolution = heightmap.get("resolution_m")
    if resolution is not None:
        if isinstance(resolution, bool) or not isinstance(resolution, (int, float)):
            raise ConfigError(f"{path}: heightmap.resolution_m must be a number")
        resolution = float(resolution)
        if resolution <= 0:
            raise ConfigError(f"{path}: heightmap.resolution_m must be positive")

    rgb = data.get("rgb") or {}
    rgb = _require_mapping(rgb, "'rgb'", path)
    rgb_enabled = bool(rgb.get("enabled", False))

    return WorldConfig(
        name=name,
        bounds=bounds,
        source=source,
        resolution_m=resolution,
        rgb_enabled=rgb_enabled,
        path=path,
    )


def load_world(path: Path) -> WorldConfig:
    """Read and validate a single world YAML file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"world configuration not found: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if document is None:
        raise ConfigError(f"{path}: file is empty")
    return parse_world(document, path)


def discover_world_paths(inputs: Sequence[Path]) -> list[Path]:
    """Resolve CLI inputs into a concrete, ordered list of world files.

    Accepts either a single worlds *directory* or one or more world *files*, as
    required by specification section 5.  Mixing the two is rejected because the
    resulting precedence would be ambiguous.
    """
    paths = [Path(item) for item in inputs]
    if not paths:
        raise ConfigError("no world configuration inputs were supplied")

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

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise ConfigError(
            "world configuration not found: " + ", ".join(str(p) for p in missing)
        )
    return paths


def load_worlds(inputs: Sequence[Path]) -> list[WorldConfig]:
    """Load every world referenced by ``inputs`` and reject duplicate names."""
    worlds = [load_world(path) for path in discover_world_paths(inputs)]
    seen: dict[str, WorldConfig] = {}
    for world in worlds:
        previous = seen.get(world.name)
        if previous is not None:
            raise ConfigError(
                f"duplicate world name {world.name!r} in {previous.path} and {world.path}"
            )
        seen[world.name] = world
    return worlds


def resolve_dataset_settings(
    worlds: Iterable[WorldConfig], options: PrepareOptions
) -> tuple[str, float]:
    """Agree on the one source and one resolution the whole dataset will use.

    A dataset manifest describes a single grid, so every world contributing to it
    must want the same source product and pixel size.  A world may omit
    ``resolution_m`` and inherit the value from ``options``.
    """
    worlds = list(worlds)
    if not worlds:
        raise ConfigError("at least one world is required")

    sources = {world.source for world in worlds}
    if len(sources) > 1:
        raise ConfigError(
            "all worlds in one dataset must use the same heightmap source, got: "
            + ", ".join(sorted(sources))
        )
    source = sources.pop()

    resolutions = {
        world.resolution_m if world.resolution_m is not None else options.resolution_m
        for world in worlds
    }
    if len(resolutions) > 1:
        raise ConfigError(
            "all worlds in one dataset must use the same heightmap resolution, got: "
            + ", ".join(f"{r:g}" for r in sorted(resolutions))
        )
    return source, resolutions.pop()


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
