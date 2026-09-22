"""The height source adapter interface.

Acquisition is isolated behind a small protocol so that new inputs (local
GeoTIFFs, LAZ via PDAL, …) can be added without touching the storage design
(specification section 16).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

from .errors import OutOfCoverageError, ResolutionMismatchError
from .tiling import ProjectedBounds

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceBlock:
    """One decoded piece of source raster, already on the requested pixel grid."""

    bounds: ProjectedBounds
    data: np.ndarray  # shape (height, width), float32/float64
    nodata: float


@runtime_checkable
class HeightSource(Protocol):
    """Minimal adapter contract (specification section 16)."""

    horizontal_crs: str
    vertical_crs: str
    resolution_m: float

    def fetch(self, bounds: ProjectedBounds, output_path: Path) -> Path:
        """Acquire ``bounds`` and write it as a GeoTIFF at ``output_path``."""
        ...


class BaseHeightSource:
    """Shared behaviour for sources that can serve arbitrary pixel windows.

    Subclasses implement :meth:`_request_block` for a single, already
    size-limited request; this base class handles splitting a large area into
    several requests, stitching the results, and the file-producing
    :meth:`fetch` entry point of :class:`HeightSource`.
    """

    horizontal_crs: str = "EPSG:5514"
    vertical_crs: str = "EPSG:8357"
    resolution_m: float = 2.0
    nodata: float = -9999.0
    #: Largest request the backend accepts, in pixels.
    max_request_width_px: int = 2048
    max_request_height_px: int = 2048

    # --- to implement ----------------------------------------------------

    @property
    def source_id(self) -> str:
        """Stable identifier recorded on generated tiles for resume checks."""
        raise NotImplementedError

    def coverage(self) -> ProjectedBounds | None:
        """Known extent of the product, or ``None`` when unbounded."""
        return None

    def native_grid_origin(self) -> tuple[float, float] | None:
        """A point on a corner of a native source pixel, or ``None`` if unknown.

        The pipeline phase-aligns the output tile grid to this point so that
        requested windows fall on native pixel boundaries and the source is
        never resampled (specification sections 2 and 3.2).
        """
        return None

    def _request_block(self, bounds: ProjectedBounds, width: int, height: int) -> np.ndarray:
        """Fetch exactly ``width`` x ``height`` pixels covering ``bounds``."""
        raise NotImplementedError

    # --- provided --------------------------------------------------------

    def check_coverage(self, bounds: ProjectedBounds, context: str = "") -> None:
        """Raise when ``bounds`` lies entirely outside the product's extent."""
        coverage = self.coverage()
        if coverage is None:
            return
        if coverage.intersection(bounds) is None:
            where = f" ({context})" if context else ""
            raise OutOfCoverageError(
                f"requested area{where} {bounds.as_tuple()} lies outside the "
                f"{self.source_id} coverage {coverage.as_tuple()} "
                f"in {self.horizontal_crs}"
            )

    def split_requests(
        self, bounds: ProjectedBounds, width: int, height: int
    ) -> Iterator[tuple[ProjectedBounds, int, int, int, int]]:
        """Split one window into server-sized requests without gaps or overlap.

        Yields ``(sub_bounds, col_off, row_off, sub_width, sub_height)`` where the
        offsets are relative to the top-left of the requested window.
        """
        res_x = bounds.width / width
        res_y = bounds.height / height
        max_w = max(1, int(self.max_request_width_px))
        max_h = max(1, int(self.max_request_height_px))
        for row_off in range(0, height, max_h):
            rows = min(max_h, height - row_off)
            for col_off in range(0, width, max_w):
                cols = min(max_w, width - col_off)
                west = bounds.west + col_off * res_x
                north = bounds.north - row_off * res_y
                sub = ProjectedBounds(
                    west=west,
                    south=north - rows * res_y,
                    east=west + cols * res_x,
                    north=north,
                )
                yield sub, col_off, row_off, cols, rows

    def read_block(self, bounds: ProjectedBounds, width: int, height: int) -> np.ndarray:
        """Return ``height`` x ``width`` source samples covering ``bounds``.

        The area is split into as many backend requests as the server limits
        require; the pieces tile the window exactly, so the mosaic has neither
        gaps nor overlaps.
        """
        if width <= 0 or height <= 0:
            raise ResolutionMismatchError(
                f"invalid block size {width}x{height} requested from {self.source_id}"
            )
        self.check_coverage(bounds)

        requests = list(self.split_requests(bounds, width, height))
        if len(requests) == 1:
            sub, _, _, cols, rows = requests[0]
            return self._request_block(sub, cols, rows)

        log.debug(
            "%s: %d requests for %dx%d px window %s",
            self.source_id,
            len(requests),
            width,
            height,
            bounds.as_tuple(),
        )
        mosaic = np.full((height, width), self.nodata, dtype=np.float32)
        for sub, col_off, row_off, cols, rows in requests:
            block = self._request_block(sub, cols, rows)
            mosaic[row_off : row_off + rows, col_off : col_off + cols] = block
        return mosaic

    def read_bounds(self, bounds: ProjectedBounds) -> SourceBlock:
        """Read ``bounds`` at the source's own resolution, snapped outward."""
        width = max(1, int(math.ceil(bounds.width / self.resolution_m - 1e-9)))
        height = max(1, int(math.ceil(bounds.height / self.resolution_m - 1e-9)))
        snapped = ProjectedBounds(
            west=bounds.west,
            south=bounds.north - height * self.resolution_m,
            east=bounds.west + width * self.resolution_m,
            north=bounds.north,
        )
        return SourceBlock(
            bounds=snapped,
            data=self.read_block(snapped, width, height),
            nodata=self.nodata,
        )

    def fetch(self, bounds: ProjectedBounds, output_path: Path) -> Path:
        """:class:`HeightSource` entry point: acquire ``bounds`` to a GeoTIFF."""
        from .raster import write_raster  # imported here to avoid a cycle

        block = self.read_bounds(bounds)
        transform = _north_up_transform(block.bounds, block.data.shape[1], block.data.shape[0])
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_raster(
            output_path,
            block.data.astype(np.float32, copy=False),
            transform=transform,
            crs=self.horizontal_crs,
            nodata=self.nodata,
        )
        return output_path


def _north_up_transform(bounds: ProjectedBounds, width: int, height: int):
    from affine import Affine

    return Affine(bounds.width / width, 0.0, bounds.west, 0.0, -bounds.height / height, bounds.north)
