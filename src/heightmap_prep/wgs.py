"""The WGS tiling: disjoint EPSG:4326 tiles holding heights and RGB imagery.

Besides the EPSG:5514 height tiles a sampler reads, a dataset holds tiles in
EPSG:4326, the CRS samplers are queried in, for retrieving image regions.  They
are cut from one global lattice of square-degree pixels anchored at
``(-180, 90)``, independently of the EPSG:5514 grid:

* the **RGB lattice** has the orthophoto's pixel size, ``resolution_deg``;
* the **height lattice** has pixels of ``height_factor`` x ``height_factor``
  RGB pixels, with the same origin, so every height pixel covers a whole block
  of RGB pixels;
* WGS tile ``(ix, iy)`` is a block of ``tile_size_px`` x ``tile_size_px`` RGB
  pixels (``tile_size_px / height_factor`` height pixels), with
  ``ix = floor((lon + 180) / tile_span_deg)`` and
  ``iy = floor((90 - lat) / tile_span_deg)``.

Tiles therefore never overlap, and the RGB and height tiles of one index cover
exactly the same area.  Pixels are square in degrees because the ArcGIS
``export`` operation only renders square pixels in the requested CRS; the
resolution is given in metres north-south, and east-west pixels are
``cos(latitude)`` times that on the ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from affine import Affine

from .errors import ConfigError
from .tiling import TileIndex

#: WGS tiles are stored in the query CRS.
WGS_CRS = "EPSG:4326"

WGS_HEIGHT_TILE_PATTERN = "wgs/height/tile_{ix}_{iy}.tif"
WGS_RGB_TILE_PATTERN = "wgs/rgb/tile_{ix}_{iy}.tif"

#: Ground length of one degree of latitude, used to turn a resolution in metres
#: into degrees.  A nominal figure: the lattice only has to be deterministic.
METRES_PER_DEGREE = 111_320.0

#: West/north corner of lattice pixel ``(0, 0)``.
LATTICE_ORIGIN_LON = -180.0
LATTICE_ORIGIN_LAT = 90.0

#: A WGS tile's width and height in RGB pixels.
DEFAULT_TILE_SIZE_PX = 8192

#: RGB pixels per height pixel, along each axis.
DEFAULT_HEIGHT_FACTOR = 8

#: Points per edge when projecting a rectangle into another CRS: its edges are
#: curves there, so the corners alone underestimate its extent.
EDGE_SAMPLES = 64

#: Tolerance, in pixels, when snapping onto the lattice, so a coordinate that
#: is a lattice line up to float rounding does not pull in an extra pixel.
SNAP_TOLERANCE_PX = 1e-6

LonLatBounds = tuple[float, float, float, float]  # west, south, east, north


@dataclass(frozen=True)
class LatticeWindow:
    """A block of lattice pixels; ``col_min``/``row_min`` count from the origin."""

    col_min: int
    row_min: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ConfigError(f"degenerate lattice window {self.width}x{self.height}")

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

    def contains(self, other: "LatticeWindow") -> bool:
        return (
            self.col_min <= other.col_min
            and self.row_min <= other.row_min
            and other.col_end <= self.col_end
            and other.row_end <= self.row_end
        )

    def intersection(self, other: "LatticeWindow") -> "LatticeWindow | None":
        col_min = max(self.col_min, other.col_min)
        row_min = max(self.row_min, other.row_min)
        col_end = min(self.col_end, other.col_end)
        row_end = min(self.row_end, other.row_end)
        if col_end <= col_min or row_end <= row_min:
            return None
        return LatticeWindow(col_min, row_min, col_end - col_min, row_end - row_min)

    def union(self, other: "LatticeWindow") -> "LatticeWindow":
        """The smallest window holding both."""
        col_min = min(self.col_min, other.col_min)
        row_min = min(self.row_min, other.row_min)
        return LatticeWindow(
            col_min,
            row_min,
            max(self.col_end, other.col_end) - col_min,
            max(self.row_end, other.row_end) - row_min,
        )

    def scaled(self, factor: int) -> "LatticeWindow":
        """The same area on a lattice ``factor`` times finer."""
        return LatticeWindow(
            self.col_min * factor, self.row_min * factor, self.width * factor, self.height * factor
        )

    def sub_window(self, col_off: int, row_off: int, width: int, height: int) -> "LatticeWindow":
        """A window given relative to this one's top-left pixel."""
        return LatticeWindow(self.col_min + col_off, self.row_min + row_off, width, height)

    def offset_in(self, outer: "LatticeWindow") -> tuple[int, int]:
        """``(col_off, row_off)`` of this window inside ``outer``."""
        return self.col_min - outer.col_min, self.row_min - outer.row_min

    def to_tag(self) -> str:
        return f"{self.col_min},{self.row_min},{self.width},{self.height}"

    @classmethod
    def from_tag(cls, value: str | None) -> "LatticeWindow | None":
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
class Lattice:
    """A global EPSG:4326 pixel lattice of square-degree pixels."""

    resolution_deg: float
    origin_lon: float = LATTICE_ORIGIN_LON
    origin_lat: float = LATTICE_ORIGIN_LAT

    def __post_init__(self) -> None:
        if not self.resolution_deg > 0:
            raise ConfigError(f"lattice resolution must be positive, got {self.resolution_deg}")

    def window_bounds(self, window: LatticeWindow) -> LonLatBounds:
        """``(west, south, east, north)`` of ``window`` in degrees."""
        res = self.resolution_deg
        return (
            self.origin_lon + window.col_min * res,
            self.origin_lat - window.row_end * res,
            self.origin_lon + window.col_end * res,
            self.origin_lat - window.row_min * res,
        )

    def window_transform(self, window: LatticeWindow) -> Affine:
        """The north-up affine transform of a raster holding ``window``."""
        west, _, _, north = self.window_bounds(window)
        return Affine(self.resolution_deg, 0.0, west, 0.0, -self.resolution_deg, north)

    def window_for_lonlat(self, bounds: LonLatBounds) -> LatticeWindow:
        """The smallest window covering ``bounds``, snapped outward to the lattice."""
        west, south, east, north = bounds
        res = self.resolution_deg
        col_min = math.floor((west - self.origin_lon) / res + SNAP_TOLERANCE_PX)
        col_end = math.ceil((east - self.origin_lon) / res - SNAP_TOLERANCE_PX)
        row_min = math.floor((self.origin_lat - north) / res + SNAP_TOLERANCE_PX)
        row_end = math.ceil((self.origin_lat - south) / res - SNAP_TOLERANCE_PX)
        return LatticeWindow(
            col_min, row_min, max(1, col_end - col_min), max(1, row_end - row_min)
        )

    def window_of_transform(
        self, transform: Affine, width: int, height: int
    ) -> LatticeWindow | None:
        """The window a raster occupies, or ``None`` if it is off this lattice."""
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
        return LatticeWindow(int(round(col)), int(round(row)), int(width), int(height))


@dataclass(frozen=True)
class WgsGrid:
    """The WGS tiling: RGB and height lattices, cut into disjoint tiles."""

    resolution_deg: float
    tile_size_px: int = DEFAULT_TILE_SIZE_PX
    height_factor: int = DEFAULT_HEIGHT_FACTOR
    origin_lon: float = LATTICE_ORIGIN_LON
    origin_lat: float = LATTICE_ORIGIN_LAT

    def __post_init__(self) -> None:
        if not self.resolution_deg > 0:
            raise ConfigError(f"RGB resolution must be positive, got {self.resolution_deg}")
        if self.height_factor < 1 or self.tile_size_px < 1:
            raise ConfigError(
                f"WGS tile size ({self.tile_size_px}) and height factor "
                f"({self.height_factor}) must be positive"
            )
        if self.tile_size_px % self.height_factor:
            raise ConfigError(
                f"WGS tile size {self.tile_size_px} px is not a multiple of the height "
                f"factor {self.height_factor}"
            )

    @classmethod
    def from_metres(
        cls,
        resolution_m: float,
        *,
        tile_size_px: int = DEFAULT_TILE_SIZE_PX,
        height_factor: int = DEFAULT_HEIGHT_FACTOR,
    ) -> "WgsGrid":
        if not resolution_m > 0:
            raise ConfigError(f"RGB resolution must be positive, got {resolution_m}")
        return cls(
            resolution_deg=resolution_m / METRES_PER_DEGREE,
            tile_size_px=tile_size_px,
            height_factor=height_factor,
        )

    @property
    def rgb(self) -> Lattice:
        return Lattice(self.resolution_deg, self.origin_lon, self.origin_lat)

    @property
    def height(self) -> Lattice:
        return Lattice(self.resolution_deg * self.height_factor, self.origin_lon, self.origin_lat)

    @property
    def height_tile_px(self) -> int:
        return self.tile_size_px // self.height_factor

    @property
    def tile_span_deg(self) -> float:
        return self.tile_size_px * self.resolution_deg

    # --- tiles -------------------------------------------------------------

    def height_tile_window(self, tile: TileIndex) -> LatticeWindow:
        n = self.height_tile_px
        return LatticeWindow(tile.ix * n, tile.iy * n, n, n)

    def rgb_tile_window(self, tile: TileIndex) -> LatticeWindow:
        return self.height_tile_window(tile).scaled(self.height_factor)

    def tile_bounds(self, tile: TileIndex) -> LonLatBounds:
        return self.height.window_bounds(self.height_tile_window(tile))

    def tile_index(self, lon: float, lat: float) -> TileIndex:
        span = self.tile_span_deg
        return TileIndex(
            math.floor((lon - self.origin_lon) / span),
            math.floor((self.origin_lat - lat) / span),
        )

    def fill_for_lonlat(self, bounds: LonLatBounds) -> LatticeWindow:
        """``bounds`` snapped outward to the height lattice.

        Height and RGB tiles hold data exactly there: the RGB fill is this
        window scaled by ``height_factor``.
        """
        return self.height.window_for_lonlat(bounds)

    def tiles_for_window(self, window: LatticeWindow) -> list[TileIndex]:
        """The tiles a height-lattice window touches, in reading order."""
        n = self.height_tile_px
        return [
            TileIndex(ix, iy)
            for iy in range(window.row_min // n, (window.row_end - 1) // n + 1)
            for ix in range(window.col_min // n, (window.col_end - 1) // n + 1)
        ]


# --- between CRSs ----------------------------------------------------------


def densified_ring(
    west: float, south: float, east: float, north: float, samples: int = EDGE_SAMPLES
) -> tuple[np.ndarray, np.ndarray]:
    """``samples`` points per edge around a rectangle, clockwise from north-west."""
    t = np.linspace(0.0, 1.0, samples + 1)
    width, height = east - west, north - south
    xs = np.concatenate(
        [west + t * width, np.full_like(t, east), east - t * width, np.full_like(t, west)]
    )
    ys = np.concatenate(
        [np.full_like(t, north), north - t * height, np.full_like(t, south), south + t * height]
    )
    return xs, ys

