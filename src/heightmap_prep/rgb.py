"""The RGB pixel lattice: orthophoto tiles in the query CRS.

RGB imagery is stored in EPSG:4326, the CRS samplers are queried in, on one
global lattice of square-degree pixels anchored at ``(-180, 90)``.  Every RGB
tile is a window of that lattice, so overlapping tiles hold identical pixels
wherever they overlap and imagery never has to be resampled to be mosaicked.

RGB tile ``(ix, iy)`` belongs to height tile ``(ix, iy)``: it covers the
EPSG:4326 bounding box of the height tile's sampling region, snapped outward to
the lattice.  A consumer finds the RGB tile for a lon/lat exactly the way it
finds the height tile (project to EPSG:5514, ``floor``), and then reads the
pixel through the RGB tile's own affine transform.  Krovak East North is
rotated against the meridians, so neighbouring RGB tiles overlap slightly.

Pixels are square in degrees because the ArcGIS ``export`` operation only
renders square pixels in the requested CRS.  The resolution is given in metres
north-south; east-west pixels are ``cos(latitude)`` times that on the ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from affine import Affine

from .errors import ConfigError
from .tiling import ProjectedBounds, TileGrid, TileIndex

#: RGB tiles are stored in the query CRS.
RGB_CRS = "EPSG:4326"

RGB_TILE_PATTERN = "rgb/tile_{ix}_{iy}.tif"

#: Ground length of one degree of latitude, used to turn a resolution in metres
#: into degrees.  A nominal figure: the lattice only has to be deterministic.
METRES_PER_DEGREE = 111_320.0

#: West/north corner of lattice pixel ``(0, 0)``.
LATTICE_ORIGIN_LON = -180.0
LATTICE_ORIGIN_LAT = 90.0

#: Points per edge when projecting a rectangle into EPSG:4326: its edges are
#: curves in lon/lat, so the corners alone underestimate its bounding box.
EDGE_SAMPLES = 64

#: Tolerance, in pixels, when snapping onto the lattice, so a coordinate that
#: is a lattice line up to float rounding does not pull in an extra pixel.
SNAP_TOLERANCE_PX = 1e-6

LonLatBounds = tuple[float, float, float, float]  # west, south, east, north


@dataclass(frozen=True)
class RgbWindow:
    """A block of lattice pixels; ``col_min``/``row_min`` count from the origin."""

    col_min: int
    row_min: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ConfigError(f"degenerate RGB window {self.width}x{self.height}")

    @property
    def col_end(self) -> int:
        """One past the last column."""
        return self.col_min + self.width

    @property
    def row_end(self) -> int:
        """One past the last row."""
        return self.row_min + self.height

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.width}x{self.height} px at ({self.col_min}, {self.row_min})"

    def contains(self, other: "RgbWindow") -> bool:
        return (
            self.col_min <= other.col_min
            and self.row_min <= other.row_min
            and other.col_end <= self.col_end
            and other.row_end <= self.row_end
        )

    def intersection(self, other: "RgbWindow") -> "RgbWindow | None":
        col_min = max(self.col_min, other.col_min)
        row_min = max(self.row_min, other.row_min)
        col_end = min(self.col_end, other.col_end)
        row_end = min(self.row_end, other.row_end)
        if col_end <= col_min or row_end <= row_min:
            return None
        return RgbWindow(col_min, row_min, col_end - col_min, row_end - row_min)

    def sub_window(self, col_off: int, row_off: int, width: int, height: int) -> "RgbWindow":
        """A window given relative to this one's top-left pixel."""
        return RgbWindow(self.col_min + col_off, self.row_min + row_off, width, height)

    def offset_in(self, outer: "RgbWindow") -> tuple[int, int]:
        """``(col_off, row_off)`` of this window inside ``outer``."""
        return self.col_min - outer.col_min, self.row_min - outer.row_min

    def to_tag(self) -> str:
        return f"{self.col_min},{self.row_min},{self.width},{self.height}"

    @classmethod
    def from_tag(cls, value: str | None) -> "RgbWindow | None":
        """Parse :meth:`to_tag` output; ``None`` if it is missing or malformed."""
        if not value:
            return None
        try:
            parts = [int(part) for part in value.split(",")]
        except ValueError:
            return None
        if len(parts) != 4:
            return None
        try:
            return cls(*parts)
        except ConfigError:
            return None


@dataclass(frozen=True)
class RgbGrid:
    """The global EPSG:4326 pixel lattice RGB tiles are cut from."""

    resolution_deg: float
    origin_lon: float = LATTICE_ORIGIN_LON
    origin_lat: float = LATTICE_ORIGIN_LAT

    def __post_init__(self) -> None:
        if not self.resolution_deg > 0:
            raise ConfigError(f"RGB resolution must be positive, got {self.resolution_deg}")

    @classmethod
    def from_metres(cls, resolution_m: float) -> "RgbGrid":
        if not resolution_m > 0:
            raise ConfigError(f"RGB resolution must be positive, got {resolution_m}")
        return cls(resolution_deg=resolution_m / METRES_PER_DEGREE)

    # --- lattice <-> coordinates ----------------------------------------

    def window_bounds(self, window: RgbWindow) -> LonLatBounds:
        """``(west, south, east, north)`` of ``window`` in degrees."""
        res = self.resolution_deg
        return (
            self.origin_lon + window.col_min * res,
            self.origin_lat - window.row_end * res,
            self.origin_lon + window.col_end * res,
            self.origin_lat - window.row_min * res,
        )

    def window_transform(self, window: RgbWindow) -> Affine:
        """The north-up affine transform of a raster holding ``window``."""
        west, _, _, north = self.window_bounds(window)
        return Affine(self.resolution_deg, 0.0, west, 0.0, -self.resolution_deg, north)

    def window_for_lonlat(self, bounds: LonLatBounds) -> RgbWindow:
        """The smallest window covering ``bounds``, snapped outward to the lattice."""
        west, south, east, north = bounds
        res = self.resolution_deg
        col_min = math.floor((west - self.origin_lon) / res + SNAP_TOLERANCE_PX)
        col_end = math.ceil((east - self.origin_lon) / res - SNAP_TOLERANCE_PX)
        row_min = math.floor((self.origin_lat - north) / res + SNAP_TOLERANCE_PX)
        row_end = math.ceil((self.origin_lat - south) / res - SNAP_TOLERANCE_PX)
        return RgbWindow(
            col_min, row_min, max(1, col_end - col_min), max(1, row_end - row_min)
        )

    def window_of_transform(self, transform: Affine, width: int, height: int) -> RgbWindow | None:
        """The lattice window a raster occupies, or ``None`` if it is off the lattice."""
        res = self.resolution_deg
        if not (
            math.isclose(transform.a, res, rel_tol=1e-9)
            and math.isclose(transform.e, -res, rel_tol=1e-9)
            and transform.b == 0.0
            and transform.d == 0.0
        ):
            return None
        col = (transform.c - self.origin_lon) / res
        row = (self.origin_lat - transform.f) / res
        if abs(col - round(col)) > 1e-4 or abs(row - round(row)) > 1e-4:
            return None
        return RgbWindow(int(round(col)), int(round(row)), int(width), int(height))

    # --- from the height grid -------------------------------------------

    def window_for_projected(self, bounds: ProjectedBounds) -> RgbWindow:
        """The window covering an EPSG:5514 rectangle."""
        return self.window_for_lonlat(projected_to_lonlat_bounds(bounds))

    def tile_window(self, height_grid: TileGrid, tile: TileIndex) -> RgbWindow:
        """The window of RGB tile ``tile``: its height tile's sampling region."""
        return self.window_for_projected(height_grid.sampling_region(tile))


def projected_to_lonlat_bounds(
    bounds: ProjectedBounds, samples: int = EDGE_SAMPLES
) -> LonLatBounds:
    """The EPSG:4326 bounding box of an EPSG:5514 rectangle, edges densified."""
    from .crs import QUERY_CRS, SOURCE_HORIZONTAL_CRS, project_from

    t = np.linspace(0.0, 1.0, samples + 1)
    xs = np.concatenate(
        [
            bounds.west + t * bounds.width,
            np.full_like(t, bounds.east),
            bounds.east - t * bounds.width,
            np.full_like(t, bounds.west),
        ]
    )
    ys = np.concatenate(
        [
            np.full_like(t, bounds.north),
            bounds.north - t * bounds.height,
            np.full_like(t, bounds.south),
            bounds.south + t * bounds.height,
        ]
    )
    lons, lats = project_from(xs, ys, source=SOURCE_HORIZONTAL_CRS, target=QUERY_CRS)
    return float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())


def metres_to_degrees(margin_m: float, latitude: float) -> tuple[float, float]:
    """``(lon, lat)`` degrees spanning ``margin_m`` metres at ``latitude``."""
    lat_deg = margin_m / METRES_PER_DEGREE
    lon_deg = margin_m / (METRES_PER_DEGREE * max(math.cos(math.radians(latitude)), 1e-6))
    return lon_deg, lat_deg
