"""The deterministic global tile grid.

Output tiles are aligned to one fixed grid in the stored horizontal CRS rather
than to each world's own extent (specification section 9.4).  That makes tile
indices computable directly from projected coordinates, lets overlapping worlds
share tiles, and keeps filenames stable across runs.

Northing convention (specification section 12):

* ``origin_x`` is the **west** edge of tile ``(0, 0)``
* ``origin_y`` is the **north** edge of tile ``(0, 0)``
* ``ix`` increases **eastward**
* ``iy`` increases **southward**
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

from affine import Affine

from .errors import ConfigError


@dataclass(frozen=True, order=True)
class TileIndex:
    """Integer position of a tile on the global grid."""

    ix: int
    iy: int

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"({self.ix}, {self.iy})"


@dataclass(frozen=True)
class ProjectedBounds:
    """An axis-aligned rectangle in the stored horizontal CRS, in metres."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if self.east <= self.west or self.north <= self.south:
            raise ConfigError(
                f"degenerate projected bounds: west={self.west}, south={self.south}, "
                f"east={self.east}, north={self.north}"
            )

    @property
    def width(self) -> float:
        return self.east - self.west

    @property
    def height(self) -> float:
        return self.north - self.south

    def intersection(self, other: "ProjectedBounds") -> "ProjectedBounds | None":
        west = max(self.west, other.west)
        south = max(self.south, other.south)
        east = min(self.east, other.east)
        north = min(self.north, other.north)
        if east <= west or north <= south:
            return None
        return ProjectedBounds(west, south, east, north)

    def contains(self, other: "ProjectedBounds") -> bool:
        return (
            self.west <= other.west
            and self.south <= other.south
            and other.east <= self.east
            and other.north <= self.north
        )

    def buffered(self, margin: float) -> "ProjectedBounds":
        return ProjectedBounds(
            self.west - margin, self.south - margin, self.east + margin, self.north + margin
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)


@dataclass(frozen=True)
class PixelRange:
    """An inclusive block of pixels on a grid's global pixel lattice.

    Global pixel ``(col, row)`` is the ``col``-th pixel east of ``origin_x`` and
    the ``row``-th pixel south of ``origin_y``, so tile ``(ix, iy)`` holds columns
    ``ix * tile_width_px`` to ``(ix + 1) * tile_width_px - 1``.
    """

    col_min: int
    row_min: int
    col_max: int
    row_max: int

    def intersection(self, other: "PixelRange") -> "PixelRange | None":
        col_min = max(self.col_min, other.col_min)
        row_min = max(self.row_min, other.row_min)
        col_max = min(self.col_max, other.col_max)
        row_max = min(self.row_max, other.row_max)
        if col_max < col_min or row_max < row_min:
            return None
        return PixelRange(col_min, row_min, col_max, row_max)

    def union(self, other: "PixelRange") -> "PixelRange":
        return PixelRange(
            min(self.col_min, other.col_min),
            min(self.row_min, other.row_min),
            max(self.col_max, other.col_max),
            max(self.row_max, other.row_max),
        )


#: Slack, in pixels, when deciding which pixels a bilinear sampler may read.  A
#: query within a millimetre of a pixel-centre line can fall on either side of it
#: depending on how a sampler rounds its projection, so both sides are kept.
SAMPLING_SLACK_PX = 1e-3


@dataclass(frozen=True)
class TileGrid:
    """A regular grid of equally sized tiles anchored at a fixed origin."""

    origin_x: float
    origin_y: float
    tile_width_px: int
    tile_height_px: int
    resolution_x: float
    resolution_y: float

    def __post_init__(self) -> None:
        if self.tile_width_px <= 0 or self.tile_height_px <= 0:
            raise ConfigError(
                f"tile dimensions must be positive, got "
                f"{self.tile_width_px}x{self.tile_height_px}"
            )
        if not (self.resolution_x > 0 and self.resolution_y > 0):
            raise ConfigError(
                f"grid resolution must be positive, got "
                f"{self.resolution_x}, {self.resolution_y}"
            )

    @classmethod
    def from_options(cls, resolution_m: float, tile_size_px: int, origin_x: float, origin_y: float) -> "TileGrid":
        return cls(
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            tile_width_px=int(tile_size_px),
            tile_height_px=int(tile_size_px),
            resolution_x=float(resolution_m),
            resolution_y=float(resolution_m),
        )

    # --- spans -----------------------------------------------------------

    @property
    def tile_span_x(self) -> float:
        """Tile width in projected units."""
        return self.tile_width_px * self.resolution_x

    @property
    def tile_span_y(self) -> float:
        """Tile height in projected units."""
        return self.tile_height_px * self.resolution_y

    # --- index <-> coordinate -------------------------------------------

    def index_for_point(self, x: float, y: float) -> TileIndex:
        """The tile containing projected point ``(x, y)``."""
        return TileIndex(
            ix=math.floor((x - self.origin_x) / self.tile_span_x),
            iy=math.floor((self.origin_y - y) / self.tile_span_y),
        )

    def tile_bounds(self, tile: TileIndex) -> ProjectedBounds:
        """The projected extent covered by ``tile``."""
        west = self.origin_x + tile.ix * self.tile_span_x
        north = self.origin_y - tile.iy * self.tile_span_y
        return ProjectedBounds(
            west=west,
            south=north - self.tile_span_y,
            east=west + self.tile_span_x,
            north=north,
        )

    def tile_transform(self, tile: TileIndex) -> Affine:
        """The affine transform of ``tile``'s raster (north-up, pixel corners)."""
        bounds = self.tile_bounds(tile)
        return Affine(
            self.resolution_x, 0.0, bounds.west, 0.0, -self.resolution_y, bounds.north
        )

    # --- pixels ----------------------------------------------------------

    def tile_pixels(self, tile: TileIndex) -> PixelRange:
        """The global pixels held by ``tile``."""
        col_min = tile.ix * self.tile_width_px
        row_min = tile.iy * self.tile_height_px
        return PixelRange(
            col_min,
            row_min,
            col_min + self.tile_width_px - 1,
            row_min + self.tile_height_px - 1,
        )

    def tiles_for_pixels(self, pixels: PixelRange) -> list[TileIndex]:
        """Every tile holding a pixel of ``pixels``, in reading order."""
        return [
            TileIndex(ix, iy)
            for iy in range(
                pixels.row_min // self.tile_height_px, pixels.row_max // self.tile_height_px + 1
            )
            for ix in range(
                pixels.col_min // self.tile_width_px, pixels.col_max // self.tile_width_px + 1
            )
        ]

    def pixels_bounds(self, pixels: PixelRange) -> ProjectedBounds:
        """The projected extent covered by ``pixels``."""
        west = self.origin_x + pixels.col_min * self.resolution_x
        north = self.origin_y - pixels.row_min * self.resolution_y
        return ProjectedBounds(
            west=west,
            south=self.origin_y - (pixels.row_max + 1) * self.resolution_y,
            east=self.origin_x + (pixels.col_max + 1) * self.resolution_x,
            north=north,
        )

    def tile_window(self, tile: TileIndex, pixels: PixelRange) -> tuple[int, int, int, int]:
        """``pixels`` as a ``(col_off, row_off, width, height)`` window of ``tile``.

        ``pixels`` must lie inside the tile.
        """
        origin = self.tile_pixels(tile)
        if origin.intersection(pixels) != pixels:
            raise ValueError(f"{pixels} does not lie inside tile {tile}")
        return (
            pixels.col_min - origin.col_min,
            pixels.row_min - origin.row_min,
            pixels.col_max - pixels.col_min + 1,
            pixels.row_max - pixels.row_min + 1,
        )

    # --- what a bilinear sampler reads ----------------------------------
    #
    # A sampler answers a query at (x, y) from the four pixels whose centres
    # surround it: columns floor(u - 0.5) and floor(u - 0.5) + 1, where
    # u = (x - origin_x) / resolution_x, and likewise for rows.

    def sampling_pixels(self, xs: Sequence[float], ys: Sequence[float]) -> PixelRange:
        """Every pixel a sampler may read for a query inside the box around ``(xs, ys)``."""
        us = [(x - self.origin_x) / self.resolution_x - 0.5 for x in xs]
        vs = [(self.origin_y - y) / self.resolution_y - 0.5 for y in ys]
        return PixelRange(
            col_min=math.floor(min(us) - SAMPLING_SLACK_PX),
            row_min=math.floor(min(vs) - SAMPLING_SLACK_PX),
            col_max=math.floor(max(us) + SAMPLING_SLACK_PX) + 1,
            row_max=math.floor(max(vs) + SAMPLING_SLACK_PX) + 1,
        )

    def sampling_region(self, tile: TileIndex) -> ProjectedBounds:
        """Where a query must fall for a sampler to read any pixel of ``tile``.

        That is the tile grown by half a pixel on every side: a query up to half
        a pixel outside the tile still interpolates towards its edge pixels.
        """
        pixels = self.tile_pixels(tile)
        reach = 0.5 + SAMPLING_SLACK_PX
        return ProjectedBounds(
            west=self.origin_x + (pixels.col_min - reach) * self.resolution_x,
            east=self.origin_x + (pixels.col_max + 1 + reach) * self.resolution_x,
            north=self.origin_y - (pixels.row_min - reach) * self.resolution_y,
            south=self.origin_y - (pixels.row_max + 1 + reach) * self.resolution_y,
        )


# --- polygons ------------------------------------------------------------


def _segment_meets_bounds(
    x0: float, y0: float, x1: float, y1: float, bounds: ProjectedBounds
) -> bool:
    """Whether a segment touches a closed rectangle (Liang-Barsky clipping)."""
    t_enter, t_exit = 0.0, 1.0
    dx, dy = x1 - x0, y1 - y0
    for p, q in (
        (-dx, x0 - bounds.west),
        (dx, bounds.east - x0),
        (-dy, y0 - bounds.south),
        (dy, bounds.north - y0),
    ):
        if p == 0.0:
            if q < 0.0:
                return False
            continue
        t = q / p
        if p < 0.0:
            t_enter = max(t_enter, t)
        else:
            t_exit = min(t_exit, t)
        if t_enter > t_exit:
            return False
    return True


def _point_in_polygon(px: float, py: float, xs: Sequence[float], ys: Sequence[float]) -> bool:
    inside = False
    j = len(xs) - 1
    for i in range(len(xs)):
        if (ys[i] > py) != (ys[j] > py):
            crossing = xs[i] + (py - ys[i]) * (xs[j] - xs[i]) / (ys[j] - ys[i])
            if px < crossing:
                inside = not inside
        j = i
    return inside


def polygon_intersects_bounds(
    xs: Sequence[float], ys: Sequence[float], bounds: ProjectedBounds
) -> bool:
    """Whether the closed polygon with vertices ``(xs, ys)`` meets ``bounds``."""
    count = len(xs)
    for i in range(count):
        j = (i + 1) % count
        if _segment_meets_bounds(xs[i], ys[i], xs[j], ys[j], bounds):
            return True
    # No edge reaches the rectangle, so it is either entirely inside the
    # polygon or entirely outside it.
    return _point_in_polygon(
        (bounds.west + bounds.east) / 2, (bounds.south + bounds.north) / 2, xs, ys
    )


def tile_filename(tile: TileIndex, pattern: str = "height/tile_{ix}_{iy}.tif") -> str:
    """Render the deterministic relative path of a tile."""
    return pattern.format(ix=tile.ix, iy=tile.iy)


def parse_tile_filename(name: str) -> TileIndex | None:
    """Recover a tile index from ``tile_<ix>_<iy>.tif``; ``None`` if it is not one."""
    stem = name.rsplit("/", 1)[-1]
    if not stem.startswith("tile_") or not stem.endswith(".tif"):
        return None
    parts = stem[len("tile_") : -len(".tif")].split("_")
    if len(parts) != 2:
        return None
    try:
        return TileIndex(int(parts[0]), int(parts[1]))
    except ValueError:
        return None


def iter_blocks(
    width: int, height: int, block_width: int, block_height: int
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(col_off, row_off, width, height)`` windows covering an array."""
    for row_off in range(0, height, block_height):
        rows = min(block_height, height - row_off)
        for col_off in range(0, width, block_width):
            cols = min(block_width, width - col_off)
            yield col_off, row_off, cols, rows
