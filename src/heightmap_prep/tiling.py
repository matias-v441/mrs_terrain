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
from typing import Iterator

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

    def union(self, other: "ProjectedBounds") -> "ProjectedBounds":
        return ProjectedBounds(
            min(self.west, other.west),
            min(self.south, other.south),
            max(self.east, other.east),
            max(self.north, other.north),
        )

    def buffered(self, margin: float) -> "ProjectedBounds":
        return ProjectedBounds(
            self.west - margin, self.south - margin, self.east + margin, self.north + margin
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)


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

    def tiles_for_bounds(self, bounds: ProjectedBounds) -> list[TileIndex]:
        """Every tile that overlaps ``bounds``, in deterministic order.

        A coordinate sitting exactly on a tile edge belongs to the tile it opens,
        so an extent ending exactly on a boundary does not pull in the next tile.
        """
        ix_min = math.floor((bounds.west - self.origin_x) / self.tile_span_x)
        ix_max = math.ceil((bounds.east - self.origin_x) / self.tile_span_x) - 1
        iy_min = math.floor((self.origin_y - bounds.north) / self.tile_span_y)
        iy_max = math.ceil((self.origin_y - bounds.south) / self.tile_span_y) - 1
        return [
            TileIndex(ix, iy)
            for iy in range(iy_min, max(iy_max, iy_min) + 1)
            for ix in range(ix_min, max(ix_max, ix_min) + 1)
        ]

    # --- pixel windows ---------------------------------------------------

    def pixel_window_for_bounds(
        self, tile: TileIndex, bounds: ProjectedBounds
    ) -> tuple[int, int, int, int] | None:
        """Pixel window ``(col_off, row_off, width, height)`` of ``bounds``.

        The window is snapped **outward** to whole pixels and clipped to the
        tile, so no requested ground area is ever lost to rounding.  Returns
        ``None`` when ``bounds`` does not overlap the tile.
        """
        tile_bounds = self.tile_bounds(tile)
        overlap = tile_bounds.intersection(bounds)
        if overlap is None:
            return None
        col_off = math.floor((overlap.west - tile_bounds.west) / self.resolution_x)
        col_end = math.ceil((overlap.east - tile_bounds.west) / self.resolution_x)
        row_off = math.floor((tile_bounds.north - overlap.north) / self.resolution_y)
        row_end = math.ceil((tile_bounds.north - overlap.south) / self.resolution_y)
        col_off = max(0, min(col_off, self.tile_width_px))
        col_end = max(0, min(col_end, self.tile_width_px))
        row_off = max(0, min(row_off, self.tile_height_px))
        row_end = max(0, min(row_end, self.tile_height_px))
        if col_end <= col_off or row_end <= row_off:
            return None
        return col_off, row_off, col_end - col_off, row_end - row_off

    def window_bounds(
        self, tile: TileIndex, window: tuple[int, int, int, int]
    ) -> ProjectedBounds:
        """The projected extent of a pixel window inside ``tile``."""
        col_off, row_off, width, height = window
        tile_bounds = self.tile_bounds(tile)
        west = tile_bounds.west + col_off * self.resolution_x
        north = tile_bounds.north - row_off * self.resolution_y
        return ProjectedBounds(
            west=west,
            south=north - height * self.resolution_y,
            east=west + width * self.resolution_x,
            north=north,
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
