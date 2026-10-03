"""Raster I/O and the vertical conversion workflow (specification section 8).

Heights are converted once, here, so the runtime sampler never has to touch
PROJ.  The horizontal affine transform of the source raster is carried through
untouched; only band values change.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import rasterio
import rasterio.windows
from affine import Affine
from rasterio.crs import CRS as RioCRS

from .crs import VerticalConverter
from .errors import MalformedRasterError, OutputConflictError
from .tiling import iter_blocks

log = logging.getLogger(__name__)

#: Storage dtype for every prepared elevation raster.
OUTPUT_DTYPE = "float32"

#: Default NoData value used throughout a prepared dataset.
DEFAULT_NODATA = -9999.0

#: Window size used when generating pixel-centre coordinates, in pixels.  Small
#: enough that no full-resolution coordinate array is ever allocated.
DEFAULT_CONVERSION_BLOCK_PX = 512

#: GDAL metadata keys stamped onto generated tiles, used for resume checks.
TAG_PACKAGE_VERSION = "HEIGHTMAP_PREP_VERSION"
TAG_PROCESSING_VERSION = "HEIGHTMAP_PREP_PROCESSING_VERSION"
TAG_SOURCE_ID = "HEIGHTMAP_PREP_SOURCE"
TAG_VERTICAL_DATUM = "HEIGHTMAP_PREP_VERTICAL_DATUM"
TAG_VERTICAL_CRS = "HEIGHTMAP_PREP_VERTICAL_CRS"
TAG_RESOLUTION = "HEIGHTMAP_PREP_RESOLUTION_M"

#: Bumped whenever generated pixel values could change for identical inputs.
PROCESSING_VERSION = "1"

#: RGB tile provenance, next to the shared package/processing/source tags.
TAG_RGB_RESOLUTION = "HEIGHTMAP_PREP_RGB_RESOLUTION_M"
TAG_RGB_JPEG_QUALITY = "HEIGHTMAP_PREP_RGB_JPEG_QUALITY"
#: The lattice window that holds imagery, as ``col,row,width,height``.
TAG_RGB_FILLED = "HEIGHTMAP_PREP_RGB_FILLED"

#: Bumped whenever generated RGB pixels could change for identical inputs.
RGB_PROCESSING_VERSION = "1"

#: Internal block edge of RGB tiles.  RGB tiles are tens of thousands of pixels
#: across, so larger blocks keep the (mostly empty) block index small.
RGB_BLOCK_PX = 512


def storage_profile(
    *,
    width: int,
    height: int,
    transform: Affine,
    crs: str | RioCRS,
    nodata: float = DEFAULT_NODATA,
    block_size_px: int = 256,
    compression: str = "deflate",
    predictor: int = 3,
    cog: bool = False,
) -> dict[str, object]:
    """The recommended storage profile from specification section 9.3."""
    profile: dict[str, object] = {
        "driver": "COG" if cog else "GTiff",
        "dtype": OUTPUT_DTYPE,
        "count": 1,
        "width": int(width),
        "height": int(height),
        "transform": transform,
        "crs": RioCRS.from_user_input(crs) if isinstance(crs, str) else crs,
        "nodata": float(nodata),
    }
    if cog:
        profile.update(
            {
                "COMPRESS": compression.upper(),
                "PREDICTOR": "YES" if predictor == 3 else str(predictor),
                "BLOCKSIZE": int(block_size_px),
            }
        )
    else:
        profile.update(
            {
                "tiled": True,
                "blockxsize": int(block_size_px),
                "blockysize": int(block_size_px),
                "compress": compression,
                "predictor": int(predictor),
                "BIGTIFF": "IF_SAFER",
            }
        )
    return profile


def write_raster(
    path: Path,
    data: np.ndarray,
    *,
    transform: Affine,
    crs: str | RioCRS,
    nodata: float = DEFAULT_NODATA,
    block_size_px: int = 256,
    compression: str = "deflate",
    predictor: int = 3,
    cog: bool = False,
    tags: Mapping[str, str] | None = None,
) -> Path:
    """Write a single-band float32 raster using the dataset storage profile."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise MalformedRasterError(f"expected a 2-D array, got shape {data.shape}")
    profile = storage_profile(
        width=data.shape[1],
        height=data.shape[0],
        transform=transform,
        crs=crs,
        nodata=nodata,
        block_size_px=block_size_px,
        compression=compression,
        predictor=predictor,
        cog=cog,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(data.astype(np.float32, copy=False), 1)
        if tags:
            dataset.update_tags(**{k: str(v) for k, v in tags.items()})
    return path


def write_raster_atomic(
    path: Path,
    data: np.ndarray,
    *,
    overwrite: bool = False,
    validate: Callable[[Path], None] | None = None,
    **kwargs: object,
) -> Path:
    """Write to ``<path>.tmp``, validate it, and only then rename it into place.

    Specification section 18: an interrupted run must never leave a truncated or
    unvalidated tile behind that a later run would mistake for valid output.
    ``validate`` is called with the temporary path and should raise on failure.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise OutputConflictError(
            f"{path} already exists; pass --overwrite to replace it"
        )
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        write_raster(tmp, data, **kwargs)  # type: ignore[arg-type]
        if validate is not None:
            validate(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def rgb_storage_profile(
    *,
    width: int,
    height: int,
    transform: Affine,
    crs: str | RioCRS,
    jpeg_quality: int = 90,
    block_size_px: int = RGB_BLOCK_PX,
) -> dict[str, object]:
    """Three-band uint8 YCbCr JPEG, tiled and sparse; validity is an internal mask."""
    return {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 3,
        "width": int(width),
        "height": int(height),
        "transform": transform,
        "crs": RioCRS.from_user_input(crs) if isinstance(crs, str) else crs,
        "tiled": True,
        "blockxsize": int(block_size_px),
        "blockysize": int(block_size_px),
        "interleave": "pixel",
        "photometric": "ycbcr",
        "compress": "jpeg",
        "jpeg_quality": int(jpeg_quality),
        # Blocks nobody writes are left out of the file entirely.
        "sparse_ok": True,
        "BIGTIFF": "IF_SAFER",
    }


def write_rgb_window_atomic(
    path: Path,
    rgba: np.ndarray,
    *,
    profile: Mapping[str, object],
    col_off: int,
    row_off: int,
    overwrite: bool = False,
    validate: Callable[[Path], None] | None = None,
    tags: Mapping[str, str] | None = None,
) -> Path:
    """Write ``rgba`` into a window of an otherwise empty RGB tile, atomically.

    The tile is created at its full size, but only the window is written: the
    rest stays sparse and masked out, so a tile far larger than memory costs
    only what its imagery does.  Alpha becomes the internal mask.  Otherwise
    like :func:`write_raster_atomic`.
    """
    rgba = np.asarray(rgba)
    if rgba.ndim != 3 or rgba.shape[0] != 4 or rgba.dtype != np.uint8:
        raise MalformedRasterError(
            f"expected a (4, height, width) uint8 RGBA array, got {rgba.dtype} {rgba.shape}"
        )
    path = Path(path)
    if path.exists() and not overwrite:
        raise OutputConflictError(
            f"{path} already exists; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    window = rasterio.windows.Window(col_off, row_off, rgba.shape[2], rgba.shape[1])
    valid = rgba[3] > 0
    try:
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(tmp, "w", **profile) as dataset:
                dataset.write(np.where(valid, rgba[:3], 0).astype(np.uint8), window=window)
                dataset.write_mask(np.where(valid, 255, 0).astype(np.uint8), window=window)
                if tags:
                    dataset.update_tags(**{k: str(v) for k, v in tags.items()})
        if validate is not None:
            validate(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_tags(path: Path) -> dict[str, str]:
    """Read GDAL metadata tags from a raster, or ``{}`` if it cannot be opened."""
    try:
        with rasterio.open(path) as dataset:
            return dict(dataset.tags())
    except Exception:
        return {}


def provenance_tags(
    *,
    package_version: str,
    source_id: str,
    vertical_datum: str,
    vertical_crs: str,
    resolution_m: float,
) -> dict[str, str]:
    """Tags stamped on each tile so a later run can tell whether to reuse it."""
    return {
        TAG_PACKAGE_VERSION: package_version,
        TAG_PROCESSING_VERSION: PROCESSING_VERSION,
        TAG_SOURCE_ID: source_id,
        TAG_VERTICAL_DATUM: vertical_datum,
        TAG_VERTICAL_CRS: vertical_crs,
        TAG_RESOLUTION: f"{resolution_m:.10g}",
    }


# --- pixel centres -------------------------------------------------------


def pixel_center_coords(
    transform: Affine,
    col_off: int,
    row_off: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Projected coordinates of the centres of a pixel window.

    Implements ``x, y = T * (col + 0.5, row + 0.5)`` from specification section
    4.5 in vectorised form.  Arrays are returned broadcast to ``(height, width)``
    without materialising both full grids for the common north-up case.
    """
    cols = np.arange(col_off, col_off + width, dtype=np.float64) + 0.5
    rows = np.arange(row_off, row_off + height, dtype=np.float64) + 0.5
    a, b, c, d, e, f = (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )
    if b == 0.0 and d == 0.0:
        # North-up: x varies only along columns, y only along rows.
        x = np.broadcast_to((a * cols + c)[np.newaxis, :], (height, width))
        y = np.broadcast_to((e * rows + f)[:, np.newaxis], (height, width))
        return x, y
    x = a * cols[np.newaxis, :] + b * rows[:, np.newaxis] + c
    y = d * cols[np.newaxis, :] + e * rows[:, np.newaxis] + f
    return x, y


# --- vertical conversion -------------------------------------------------


@dataclass
class ConversionStats:
    """Counters gathered while converting one raster."""

    pixels_total: int = 0
    pixels_valid: int = 0
    pixels_converted: int = 0
    min_delta: float = float("inf")
    max_delta: float = float("-inf")

    @property
    def nodata_fraction(self) -> float:
        if not self.pixels_total:
            return 0.0
        return 1.0 - self.pixels_valid / self.pixels_total

    @property
    def max_abs_delta(self) -> float:
        if self.pixels_converted == 0:
            return 0.0
        return max(abs(self.min_delta), abs(self.max_delta))

    def merge(self, other: "ConversionStats") -> None:
        self.pixels_total += other.pixels_total
        self.pixels_valid += other.pixels_valid
        self.pixels_converted += other.pixels_converted
        self.min_delta = min(self.min_delta, other.min_delta)
        self.max_delta = max(self.max_delta, other.max_delta)


def convert_vertical(
    data: np.ndarray,
    transform: Affine,
    converter: VerticalConverter,
    *,
    nodata: float = DEFAULT_NODATA,
    source_nodata: float | None = None,
    block_size_px: int = DEFAULT_CONVERSION_BLOCK_PX,
    col_off: int = 0,
    row_off: int = 0,
    context: str = "",
    stats: ConversionStats | None = None,
) -> np.ndarray:
    """Convert source heights onto the output datum, window by window.

    ``transform`` is the affine transform of the raster ``data`` belongs to, and
    ``col_off``/``row_off`` locate ``data`` inside it.  NoData pixels are copied
    through untouched and never enter a transformation (specification section
    8.3); the returned array keeps the input's shape and the original transform
    stays valid for it.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise MalformedRasterError(f"expected a 2-D array, got shape {data.shape}")
    if source_nodata is None:
        source_nodata = nodata

    height, width = data.shape
    out = np.full(data.shape, np.float32(nodata), dtype=np.float32)
    local_stats = ConversionStats(pixels_total=height * width)

    for bx, by, bw, bh in iter_blocks(width, height, block_size_px, block_size_px):
        block = data[by : by + bh, bx : bx + bw]
        source = block.astype(np.float64, copy=False)

        valid = np.isfinite(source)
        if source_nodata is not None and np.isfinite(source_nodata):
            valid &= source != float(source_nodata)
        if nodata is not None and np.isfinite(nodata) and nodata != source_nodata:
            valid &= source != float(nodata)

        count = int(np.count_nonzero(valid))
        local_stats.pixels_valid += count
        if count == 0:
            continue

        x, y = pixel_center_coords(transform, col_off + bx, row_off + by, bw, bh)
        z_in = source[valid]
        z_out = converter.transform_heights(
            np.asarray(x)[valid], np.asarray(y)[valid], z_in, context=context
        )

        block_out = out[by : by + bh, bx : bx + bw]
        block_out[valid] = z_out.astype(np.float32, copy=False)

        delta = z_out - z_in
        local_stats.pixels_converted += count
        local_stats.min_delta = min(local_stats.min_delta, float(delta.min()))
        local_stats.max_delta = max(local_stats.max_delta, float(delta.max()))

    if stats is not None:
        stats.merge(local_stats)
    return out


def convert_raster_file(
    src_path: Path,
    dst_path: Path,
    converter: VerticalConverter,
    *,
    nodata: float = DEFAULT_NODATA,
    block_size_px: int = 256,
    overwrite: bool = False,
    tags: Mapping[str, str] | None = None,
) -> ConversionStats:
    """Convert a whole GeoTIFF on disk, preserving its horizontal geometry."""
    stats = ConversionStats()
    with rasterio.open(src_path) as dataset:
        transform = dataset.transform
        crs = dataset.crs
        source_nodata = dataset.nodata
        data = dataset.read(1)
    converted = convert_vertical(
        data,
        transform,
        converter,
        nodata=nodata,
        source_nodata=source_nodata,
        context=str(src_path),
        stats=stats,
    )
    write_raster_atomic(
        dst_path,
        converted,
        overwrite=overwrite,
        transform=transform,
        crs=crs,
        nodata=nodata,
        block_size_px=block_size_px,
        tags=tags,
    )
    return stats
