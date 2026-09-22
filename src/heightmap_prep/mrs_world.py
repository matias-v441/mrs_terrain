"""Reading MRS UAV system world files.

The MRS UAV system describes a flight site with a YAML file holding a world
origin and a *safety area*: a horizontal polygon the vehicle must stay inside.
Those files are the natural place to find out which terrain a world needs, so
this module turns one into the WGS84 bounding box of its safety area.

The relevant part of such a file looks like::

    mrs_uav_managers:
      world_origin:
        units: LATLON          # or UTM
        origin_x: 50.090278    # latitude, or UTM easting
        origin_y: 14.634639    # longitude, or UTM northing
      safety_area_manager:
        safety_area:
          enabled: true
          horizontal:
            frame_name: latlon_origin
            points: [lat, lon, lat, lon, ...]

``points`` is a *flat* list of coordinate pairs whose meaning depends on
``frame_name``:

``latlon_origin``
    ``x`` is latitude and ``y`` is longitude, in degrees, absolute.
``world_origin``
    ``x`` and ``y`` are metres in the world's ENU frame (x east, y north),
    relative to the world origin.
``local_origin``
    metres relative to wherever the vehicle started, which is not georeferenced;
    such a file cannot be placed on the map.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml
from pyproj import Geod, Transformer

from .config import Wgs84Bounds
from .errors import ConfigError, InvalidBoundsError

log = logging.getLogger(__name__)

#: UTM zone assumed when a world uses UTM units, which do not record one.
#: 33N covers the Czech Republic.
DEFAULT_UTM_ZONE = "33N"

#: Frames whose points are absolute latitude/longitude degrees.
LATLON_FRAMES = ("latlon_origin",)

#: Frames whose points are metres relative to the (georeferenced) world origin.
METRIC_WORLD_FRAMES = ("world_origin", "utm_origin")

#: Frames that are not georeferenced at all.
UNREFERENCED_FRAMES = ("local_origin", "fcu", "fcu_untilted")

_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class SafetyArea:
    """A safety area polygon, resolved to WGS84 degrees."""

    #: Polygon vertices as ``(lon, lat)`` pairs, in file order.
    points: tuple[tuple[float, float], ...]
    #: ``enabled`` as recorded in the file.
    enabled: bool
    #: The frame the points were written in, for diagnostics.
    frame_name: str
    min_z: float | None = None
    max_z: float | None = None

    def bounding_box(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)`` in degrees."""
        lons = [lon for lon, _ in self.points]
        lats = [lat for _, lat in self.points]
        return min(lons), min(lats), max(lons), max(lats)


@dataclass(frozen=True)
class MrsWorld:
    """One parsed MRS world file."""

    name: str
    path: Path
    origin_lat: float
    origin_lon: float
    units: str
    safety_area: SafetyArea
    utm_zone: str | None = None

    def describe(self) -> str:
        return f"{self.name} ({self.path})"


# --- helpers --------------------------------------------------------------


def _utm_epsg(zone: str) -> int:
    """EPSG code for a UTM zone such as ``33N`` or ``18S``."""
    text = str(zone).strip().upper()
    if not text or text[-1] not in "NS" or not text[:-1].isdigit():
        raise ConfigError(
            f"malformed UTM zone {zone!r}; expected a number and a hemisphere, e.g. '33N'"
        )
    number = int(text[:-1])
    if not 1 <= number <= 60:
        raise ConfigError(f"UTM zone number must be 1..60, got {number}")
    return (32600 if text[-1] == "N" else 32700) + number


def _pairs(points: Sequence[Any], context: str) -> list[tuple[float, float]]:
    """Split a flat ``[x, y, x, y, ...]`` list into pairs."""
    if not isinstance(points, (list, tuple)):
        raise ConfigError(f"{context}: 'points' must be a list, got {type(points).__name__}")
    if len(points) % 2:
        raise ConfigError(
            f"{context}: 'points' holds {len(points)} values, which is not a whole "
            "number of x/y pairs"
        )
    values: list[float] = []
    for index, value in enumerate(points):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{context}: points[{index}] is not a number: {value!r}")
        if not math.isfinite(float(value)):
            raise ConfigError(f"{context}: points[{index}] is not finite: {value!r}")
        values.append(float(value))
    pairs = list(zip(values[0::2], values[1::2]))
    if len(pairs) < 3:
        raise ConfigError(
            f"{context}: a safety area polygon needs at least 3 vertices, got {len(pairs)}"
        )
    return pairs


def _get(mapping: Any, key: str, context: str) -> Any:
    if not isinstance(mapping, dict):
        raise ConfigError(f"{context} must be a mapping, got {type(mapping).__name__}")
    if key not in mapping or mapping[key] is None:
        raise ConfigError(f"{context} is missing required key {key!r}")
    return mapping[key]


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{context} must be finite, got {value!r}")
    return number


# --- parsing --------------------------------------------------------------


def parse_mrs_world(
    document: Any,
    path: Path,
    *,
    name: str | None = None,
    utm_zone: str = DEFAULT_UTM_ZONE,
) -> MrsWorld:
    """Turn a parsed MRS world document into an :class:`MrsWorld`."""
    managers = _get(document, "mrs_uav_managers", f"{path}: the document")
    origin = _get(managers, "world_origin", f"{path}: 'mrs_uav_managers'")
    area = _get(
        _get(managers, "safety_area_manager", f"{path}: 'mrs_uav_managers'"),
        "safety_area",
        f"{path}: 'safety_area_manager'",
    )
    horizontal = _get(area, "horizontal", f"{path}: 'safety_area'")

    units = str(_get(origin, "units", f"{path}: 'world_origin'")).strip().upper()
    origin_x = _number(_get(origin, "origin_x", f"{path}: 'world_origin'"), f"{path}: origin_x")
    origin_y = _number(_get(origin, "origin_y", f"{path}: 'world_origin'"), f"{path}: origin_y")

    zone: str | None = None
    if units == "LATLON":
        origin_lat, origin_lon = origin_x, origin_y
    elif units == "UTM":
        zone = utm_zone
        transformer = Transformer.from_crs(
            f"EPSG:{_utm_epsg(zone)}", "EPSG:4326", always_xy=True
        )
        origin_lon, origin_lat = transformer.transform(origin_x, origin_y)
        log.warning(
            "%s: world origin is UTM easting/northing and the file records no zone; "
            "assuming %s, which puts the origin at %.6f, %.6f",
            path.name,
            zone,
            origin_lat,
            origin_lon,
        )
    else:
        raise ConfigError(
            f"{path}: unsupported world_origin units {units!r}; expected LATLON or UTM"
        )

    if not -90.0 <= origin_lat <= 90.0 or not -180.0 <= origin_lon <= 180.0:
        raise ConfigError(
            f"{path}: world origin ({origin_lat}, {origin_lon}) is not a valid "
            "latitude/longitude"
        )

    frame = str(_get(horizontal, "frame_name", f"{path}: 'horizontal'")).strip()
    pairs = _pairs(_get(horizontal, "points", f"{path}: 'horizontal'"), str(path))
    points = _resolve_points(
        pairs,
        frame=frame,
        path=path,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        units=units,
        zone=zone,
        origin_x=origin_x,
        origin_y=origin_y,
    )

    vertical = area.get("vertical") or {}
    enabled = bool(area.get("enabled", True))
    if not enabled:
        log.warning(
            "%s: the safety area is disabled in the file; using its polygon anyway",
            path.name,
        )

    return MrsWorld(
        name=name or world_name_from_path(path),
        path=path,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        units=units,
        utm_zone=zone,
        safety_area=SafetyArea(
            points=tuple(points),
            enabled=enabled,
            frame_name=frame,
            min_z=vertical.get("min_z"),
            max_z=vertical.get("max_z"),
        ),
    )


def _resolve_points(
    pairs: Sequence[tuple[float, float]],
    *,
    frame: str,
    path: Path,
    origin_lat: float,
    origin_lon: float,
    units: str,
    zone: str | None,
    origin_x: float,
    origin_y: float,
) -> list[tuple[float, float]]:
    """Resolve polygon vertices to absolute ``(lon, lat)`` degrees."""
    if frame in LATLON_FRAMES:
        # x is latitude and y is longitude, already absolute.
        points = [(lon, lat) for lat, lon in pairs]
        for lon, lat in points:
            if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
                raise ConfigError(
                    f"{path}: safety area vertex ({lat}, {lon}) is not a valid "
                    f"latitude/longitude; is frame_name={frame!r} correct?"
                )
        return points

    if frame in METRIC_WORLD_FRAMES:
        # Metres in the world ENU frame: x east, y north, relative to the origin.
        if units == "UTM":
            # The world frame *is* the UTM grid, so offset there and project back.
            transformer = Transformer.from_crs(
                f"EPSG:{_utm_epsg(zone or DEFAULT_UTM_ZONE)}", "EPSG:4326", always_xy=True
            )
            return [
                tuple(transformer.transform(origin_x + east, origin_y + north))
                for east, north in pairs
            ]
        # Geographic origin: step along the ellipsoid from it.
        points = []
        for east, north in pairs:
            distance = math.hypot(east, north)
            if distance == 0.0:
                points.append((origin_lon, origin_lat))
                continue
            azimuth = math.degrees(math.atan2(east, north))
            lon, lat, _ = _GEOD.fwd(origin_lon, origin_lat, azimuth, distance)
            points.append((lon, lat))
        return points

    if frame in UNREFERENCED_FRAMES:
        raise ConfigError(
            f"{path}: the safety area is expressed in frame {frame!r}, which is not "
            "georeferenced, so it cannot be placed on the map. Re-express it in "
            "'latlon_origin' or 'world_origin'."
        )

    raise ConfigError(
        f"{path}: unsupported safety area frame_name {frame!r}; expected one of "
        + ", ".join(LATLON_FRAMES + METRIC_WORLD_FRAMES)
    )


def world_name_from_path(path: Path) -> str:
    """``worlds/world_bechovice.yaml`` -> ``bechovice``."""
    stem = Path(path).stem
    for prefix in ("world_", "world-"):
        if stem.startswith(prefix) and len(stem) > len(prefix):
            return stem[len(prefix) :]
    return stem


def load_mrs_world(
    path: Path, *, name: str | None = None, utm_zone: str = DEFAULT_UTM_ZONE
) -> MrsWorld:
    """Read and validate one MRS world file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"world file not found: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if document is None:
        raise ConfigError(f"{path}: file is empty")
    return parse_mrs_world(document, path, name=name, utm_zone=utm_zone)


# --- bounds ---------------------------------------------------------------


def expand_bounds(
    west: float, south: float, east: float, north: float, margin_m: float
) -> Wgs84Bounds:
    """Grow a lon/lat box by ``margin_m`` metres on every side.

    The east/west margin is computed at whichever edge latitude is furthest from
    the equator, so the box is at least ``margin_m`` wide everywhere inside it.
    """
    if margin_m < 0:
        raise ConfigError(f"margin must not be negative, got {margin_m}")
    if margin_m == 0:
        return Wgs84Bounds(west, south, east, north)

    mid_lon = (west + east) / 2
    _, north_out, _ = _GEOD.fwd(mid_lon, north, 0.0, margin_m)
    _, south_out, _ = _GEOD.fwd(mid_lon, south, 180.0, margin_m)

    widest_lat = north if abs(north) >= abs(south) else south
    east_out, _, _ = _GEOD.fwd(east, widest_lat, 90.0, margin_m)
    west_out, _, _ = _GEOD.fwd(west, widest_lat, 270.0, margin_m)

    return Wgs84Bounds(
        west=min(west_out, west),
        south=min(south_out, south),
        east=max(east_out, east),
        north=max(north_out, north),
    )


def safety_area_bounds(world: MrsWorld, margin_m: float) -> Wgs84Bounds:
    """The margined WGS84 bounding box of ``world``'s safety area."""
    west, south, east, north = world.safety_area.bounding_box()
    try:
        return expand_bounds(west, south, east, north, margin_m)
    except InvalidBoundsError as exc:
        raise InvalidBoundsError(f"{world.describe()}: {exc}") from exc
