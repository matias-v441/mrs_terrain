"""The preparation pipeline: worlds in, validated tiled dataset out.

This is the orchestration layer that ties acquisition (:mod:`.cuzk`), vertical
conversion (:mod:`.crs`, :mod:`.raster`), the global tile grid (:mod:`.tiling`),
the manifest (:mod:`.manifest`) and validation (:mod:`.validate`) together.
It also cuts every world's lat/lon bounding box into disjoint EPSG:4326 WGS
tiles of heights and, optionally, orthophoto imagery (:mod:`.wgs`), and it
records the worlds in ``worlds.sqlite`` (:mod:`.worlds_db`).

The EPSG:5514 tiles hold exactly what a bilinear sampler reads when queried
anywhere inside the worlds' lat/lon bounding boxes, as the WGS tiles cover
them, borders included.  A world config that
cannot be prepared is skipped with a warning; only problems that affect every
world, such as a failing source or missing PROJ grids, stop the run.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Mapping, Sequence

import numpy as np
import rasterio

from . import __version__
from .config import (
    NOMINAL_GRID_ORIGIN,
    PrepareOptions,
    WorldConfig,
    align_origin,
    load_worlds,
)
from .crs import (
    SOURCE_HORIZONTAL_CRS,
    SOURCE_VERTICAL_CRS,
    SOURCE_VERTICAL_DATUM,
    VERTICAL_TARGETS,
    VerticalConverter,
    VerticalOperation,
    build_converter,
    project_points,
)
from .cuzk import CuzkDmr5Source, CuzkOrthophotoSource
from .errors import (
    ConfigError,
    HeightmapPrepError,
    OutOfCoverageError,
    OutputConflictError,
    SourceError,
    UnexpectedVerticalDatumError,
    ValidationError,
)
from .manifest import (
    MANIFEST_FILENAME,
    Manifest,
    WgsInfo,
    WgsRgbInfo,
    grid_file_checksums,
    software_versions,
)
from .raster import (
    ConversionStats,
    PROCESSING_VERSION,
    RGB_BLOCK_PX,
    RGB_PROCESSING_VERSION,
    TAG_FILLED,
    TAG_PACKAGE_VERSION,
    TAG_PROCESSING_VERSION,
    TAG_RESOLUTION,
    TAG_RGB_FILLED,
    TAG_RGB_JPEG_QUALITY,
    TAG_RGB_RESOLUTION,
    TAG_SOURCE_ID,
    TAG_VERTICAL_CRS,
    TAG_VERTICAL_DATUM,
    TAG_WGS_FILLED,
    WGS_HEIGHT_PROCESSING_VERSION,
    convert_vertical,
    provenance_tags,
    read_tags,
    rgb_storage_profile,
    storage_profile,
    write_height_window_atomic,
    write_raster_atomic,
    write_rgb_window_atomic,
)
from .sampling import TEST_POINTS_FILENAME, DatasetSampler, ReferencePoint, write_test_points
from .sources import BaseHeightSource, BaseRgbSource
from .tiling import (
    PixelRange,
    ProjectedBounds,
    TileGrid,
    TileIndex,
    polygon_intersects_bounds,
)
from .validate import (
    ValidationReport,
    validate_dataset,
    validate_rgb_tile,
    validate_tile,
    validate_wgs_height_tile,
    validate_vertical_transformation,
)
from .wgs import LatticeWindow, WgsGrid, densified_ring
from .worlds_db import write_worlds_db

log = logging.getLogger(__name__)


@dataclass
class WorldPlan:
    """A world's footprint on the global tile grid."""

    world: WorldConfig
    #: Every EPSG:5514 pixel a sampler may read for a query inside ``wgs_fill``.
    pixels: PixelRange
    tiles: list[TileIndex]
    #: The safety area's lat/lon bounding box, snapped outward to the WGS
    #: height lattice: where the WGS tiles hold data for this world.
    wgs_fill: LatticeWindow
    wgs_tiles: list[TileIndex]


@dataclass
class TileOutcome:
    """What happened to one tile during a run."""

    tile: TileIndex
    action: str  # "written" | "reused"
    path: Path | None = None
    stats: ConversionStats | None = None
    duration_s: float = 0.0


@dataclass
class PrepareResult:
    """Everything a caller might want to know about a completed run."""

    output_dir: Path
    manifest_path: Path
    manifest: Manifest
    worlds: list[WorldConfig]
    plans: list[WorldPlan] = field(default_factory=list)
    outcomes: list[TileOutcome] = field(default_factory=list)
    #: One per WGS tile.
    wgs_outcomes: list[TileOutcome] = field(default_factory=list)
    #: One per WGS tile; empty unless RGB imagery was requested.
    rgb_outcomes: list[TileOutcome] = field(default_factory=list)
    report: ValidationReport = field(default_factory=ValidationReport)
    test_points: list[ReferencePoint] = field(default_factory=list)
    #: Each world config that was skipped, and why.
    skipped: dict[Path, str] = field(default_factory=dict)

    @property
    def tiles_written(self) -> list[TileIndex]:
        return [o.tile for o in self.outcomes if o.action == "written"]

    @property
    def tiles_reused(self) -> list[TileIndex]:
        return [o.tile for o in self.outcomes if o.action == "reused"]


# --- planning ------------------------------------------------------------


def plan_world(
    world: WorldConfig, grid: TileGrid, wgs_grid: WgsGrid, coverage: ProjectedBounds | None
) -> WorldPlan:
    """Find what ``world`` needs: its WGS tiles and the EPSG:5514 pixels behind them.

    The WGS tiles hold data over the safety area's lat/lon bounding box, snapped
    outward to the WGS height lattice.  Every EPSG:5514 pixel a sampler reads
    anywhere in that box is fetched, which covers the safety area itself too.
    Its edges are curves in EPSG:5514, so they are densified before projecting.
    """
    wgs_fill = wgs_grid.fill_for_lonlat(world.bounds)
    lons, lats = densified_ring(*wgs_grid.height.window_bounds(wgs_fill))
    xs, ys = project_points(lons, lats, target_epsg=5514)
    pixels = grid.sampling_pixels(xs, ys)
    needed = grid.pixels_bounds(pixels)
    if coverage is not None and not coverage.contains(needed):
        raise OutOfCoverageError(
            f"world {world.describe()}: the safety area needs EPSG:5514 extent "
            f"{tuple(round(v, 1) for v in needed.as_tuple())}, which is not inside the "
            f"source coverage {coverage.as_tuple()}"
        )
    # The box is a curved quadrilateral in EPSG:5514, so a tile its EPSG:5514
    # bounding box merely touches may still be outside it.
    tiles = [
        tile
        for tile in grid.tiles_for_pixels(pixels)
        if polygon_intersects_bounds(xs, ys, grid.sampling_region(tile))
    ]
    log.debug(
        "world %s: EPSG:5514 extent %s, pixels %s, WGS fill %s",
        world.name,
        tuple(round(v, 3) for v in needed.as_tuple()),
        pixels,
        wgs_fill,
    )
    return WorldPlan(
        world=world,
        pixels=pixels,
        tiles=tiles,
        wgs_fill=wgs_fill,
        wgs_tiles=wgs_grid.tiles_for_window(wgs_fill),
    )


def merge_tile_requirements(
    plans: Sequence[WorldPlan], grid: TileGrid
) -> dict[TileIndex, PixelRange]:
    """For each tile, the pixels that actually have to be fetched.

    Worlds rarely fill a whole tile, so only the union of the requesting worlds'
    pixels inside that tile is acquired; the rest of the tile stays NoData.
    Fetching the union rather than each world separately means overlapping worlds
    share one request.
    """
    required: dict[TileIndex, PixelRange] = {}
    for plan in plans:
        for tile in plan.tiles:
            inside = plan.pixels.intersection(grid.tile_pixels(tile))
            previous = required.get(tile)
            required[tile] = inside if previous is None else previous.union(inside)
    # Reading order, so a run's tile order never depends on the worker count.
    return dict(sorted(required.items(), key=lambda item: (item[0].iy, item[0].ix)))


def merge_wgs_requirements(
    plans: Sequence[WorldPlan], wgs_grid: WgsGrid
) -> dict[TileIndex, LatticeWindow]:
    """For each WGS tile, the height-lattice window its worlds need filled.

    As for :func:`merge_tile_requirements`, worlds sharing a tile share one
    window: the smallest one holding all their fills.
    """
    required: dict[TileIndex, LatticeWindow] = {}
    for plan in plans:
        for tile in plan.wgs_tiles:
            inside = plan.wgs_fill.intersection(wgs_grid.height_tile_window(tile))
            if inside is None:  # pragma: no cover - wgs_tiles come from the fill
                continue
            previous = required.get(tile)
            required[tile] = inside if previous is None else previous.union(inside)
    return dict(sorted(required.items(), key=lambda item: (item[0].iy, item[0].ix)))


# --- tile production -----------------------------------------------------


class _TileBuilder:
    """Builds one tile at a time; safe to call from several worker threads."""

    def __init__(
        self,
        *,
        grid: TileGrid,
        manifest: Manifest,
        output_dir: Path,
        source: BaseHeightSource,
        converter: VerticalConverter,
        options: PrepareOptions,
    ) -> None:
        self.grid = grid
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.source = source
        self.converter = converter
        self.options = options
        self.modified = False
        self._modified_lock = Lock()
        self.expected_tags = provenance_tags(
            package_version=__version__,
            source_id=source.source_id,
            vertical_datum=options.vertical_datum,
            vertical_crs=manifest.vertical_crs,
            resolution_m=manifest.resolution_x,
        )

    # --- resume ----------------------------------------------------------

    def _is_reusable(self, path: Path, tile: TileIndex, pixels: PixelRange) -> tuple[bool, str]:
        """Whether an existing tile was produced by an equivalent run and holds ``pixels``."""
        tags = read_tags(path)
        if not tags:
            return False, "tile carries no provenance metadata"
        for key in (
            TAG_SOURCE_ID,
            TAG_VERTICAL_DATUM,
            TAG_VERTICAL_CRS,
            TAG_RESOLUTION,
            TAG_PROCESSING_VERSION,
        ):
            expected = self.expected_tags[key]
            actual = tags.get(key)
            if actual != expected:
                return False, f"{key} is {actual!r}, this run produces {expected!r}"
        filled = PixelRange.from_tag(tags.get(TAG_FILLED))
        if filled is None or not filled.contains(pixels):
            return False, "it holds heights for less than the worlds now need"

        report, _ = validate_tile(
            path,
            self.manifest,
            tile,
            grid=self.grid,
            plausibility=False,
            report=ValidationReport(),
        )
        if report.errors:
            return False, f"existing tile failed validation: {report.errors[0]}"
        return True, ""

    # --- build -----------------------------------------------------------

    def build(self, tile: TileIndex, pixels: PixelRange) -> TileOutcome:
        started = time.monotonic()
        path = self.manifest.tile_path(self.output_dir, tile)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            reusable, reason = self._is_reusable(path, tile, pixels)
            if reusable and not self.options.overwrite:
                log.debug("tile %s: reusing %s", tile, path.name)
                return TileOutcome(tile, "reused", path, duration_s=time.monotonic() - started)
            if not self.options.overwrite:
                raise OutputConflictError(
                    f"tile {tile}: {path} already exists but cannot be reused "
                    f"({reason}); pass --overwrite to replace it"
                )
            log.debug("tile %s: regenerating (%s)", tile, reason or "overwrite requested")

        col_off, row_off, width, height = self.grid.tile_window(tile, pixels)
        fetch_bounds = self.grid.pixels_bounds(pixels)
        transform = self.grid.tile_transform(tile)

        try:
            raw = self.source.read_block(fetch_bounds, width, height)
        except SourceError as exc:
            raise SourceError(
                f"tile {tile} (bounds {fetch_bounds.as_tuple()} in "
                f"{self.source.horizontal_crs}): {exc}"
            ) from exc

        stats = ConversionStats()
        converted = convert_vertical(
            raw,
            transform,
            self.converter,
            nodata=self.options.nodata,
            source_nodata=self.source.nodata,
            col_off=col_off,
            row_off=row_off,
            context=f"tile {tile}",
            stats=stats,
        )

        data = np.full(
            (self.grid.tile_height_px, self.grid.tile_width_px),
            np.float32(self.options.nodata),
            dtype=np.float32,
        )
        data[row_off : row_off + height, col_off : col_off + width] = converted

        if self.options.plausibility_checks and stats.pixels_converted:
            if stats.max_abs_delta < 1e-6:
                log.warning(
                    "tile %s: the %s conversion changed no height by more than "
                    "1 micrometre; check that the source really is %s",
                    tile,
                    self.options.vertical_datum,
                    SOURCE_VERTICAL_DATUM,
                )

        with self._modified_lock:
            self.modified = True
        write_raster_atomic(
            path,
            data,
            overwrite=True,
            validate=lambda tmp: self._validate_pending(tmp, tile),
            transform=transform,
            crs=self.manifest.horizontal_crs,
            nodata=self.options.nodata,
            block_size_px=self.options.block_size_px,
            compression=self.manifest.compression,
            predictor=self.manifest.predictor,
            cog=self.options.cog,
            tags={**self.expected_tags, TAG_FILLED: pixels.to_tag()},
        )

        duration = time.monotonic() - started
        log.debug(
            "tile %s: %d/%d valid px, delta %.3f..%.3f m, %.2fs",
            tile,
            stats.pixels_valid,
            stats.pixels_total,
            stats.min_delta if stats.pixels_converted else 0.0,
            stats.max_delta if stats.pixels_converted else 0.0,
            duration,
        )
        return TileOutcome(tile, "written", path, stats=stats, duration_s=duration)

    def _validate_pending(self, tmp_path: Path, tile: TileIndex) -> None:
        """Validate the temporary file before it is renamed into place."""
        report, _ = validate_tile(
            tmp_path,
            self.manifest,
            tile,
            grid=self.grid,
            plausibility=self.options.plausibility_checks,
            report=ValidationReport(),
        )
        for warning in report.warnings:
            log.warning("%s", warning)
        report.raise_for_errors(f"tile {tile}")


class _WgsHeightTileBuilder:
    """Builds one WGS height tile at a time from the finished EPSG:5514 tiles.

    Each pixel is the dataset's own bilinear sample at the pixel centre, so a
    WGS height pixel is exactly what a sampler returns there.  The tiles are
    derived from data already on disk, so they are rebuilt on every run.
    """

    def __init__(self, *, manifest: Manifest, output_dir: Path, source_id: str) -> None:
        assert manifest.wgs is not None
        self.manifest = manifest
        self.wgs_grid = manifest.wgs.grid()
        self.output_dir = Path(output_dir)
        self.tags = {
            TAG_PACKAGE_VERSION: __version__,
            TAG_PROCESSING_VERSION: WGS_HEIGHT_PROCESSING_VERSION,
            TAG_SOURCE_ID: source_id,
            TAG_VERTICAL_DATUM: manifest.vertical_datum,
            TAG_VERTICAL_CRS: manifest.vertical_crs,
        }

    def build(self, tile: TileIndex, fill: LatticeWindow) -> TileOutcome:
        started = time.monotonic()
        path = self.manifest.wgs_height_path(self.output_dir, tile)
        window = self.wgs_grid.height_tile_window(tile)
        lattice = self.wgs_grid.height
        res = lattice.resolution_deg
        lons = lattice.origin_lon + (np.arange(fill.col_min, fill.col_end) + 0.5) * res
        lats = lattice.origin_lat - (np.arange(fill.row_min, fill.row_end) + 0.5) * res
        with DatasetSampler(self.output_dir, self.manifest) as sampler:
            heights = sampler.sample_many(lons[np.newaxis, :], lats[:, np.newaxis])

        col_off, row_off = fill.offset_in(window)
        write_height_window_atomic(
            path,
            heights,
            profile=storage_profile(
                width=window.width,
                height=window.height,
                transform=lattice.window_transform(window),
                crs=self.manifest.wgs.crs,
                nodata=self.manifest.nodata,
                block_size_px=self.manifest.wgs.height_block_px,
            ),
            col_off=col_off,
            row_off=row_off,
            overwrite=True,
            validate=lambda tmp: self._validate_pending(tmp, tile),
            tags={**self.tags, TAG_WGS_FILLED: fill.to_tag()},
        )
        duration = time.monotonic() - started
        log.debug(
            "wgs height tile %s: %d/%d px of %s with heights, %.2fs",
            tile,
            int(np.isfinite(heights).sum()),
            heights.size,
            fill,
            duration,
        )
        return TileOutcome(tile, "written", path, duration_s=duration)

    def _validate_pending(self, tmp_path: Path, tile: TileIndex) -> None:
        report = validate_wgs_height_tile(
            tmp_path, self.manifest, tile, self.output_dir, report=ValidationReport()
        )
        for warning in report.warnings:
            log.warning("%s", warning)
        report.raise_for_errors(f"wgs height tile {tile}")


class _RgbTileBuilder:
    """Builds one WGS RGB tile at a time; thread safe like :class:`_TileBuilder`.

    The file spans the whole WGS tile, but only imagery for the height-lattice
    window the worlds need is fetched, so it covers exactly the area the WGS
    height tile of the same index holds heights for.
    """

    def __init__(
        self,
        *,
        manifest: Manifest,
        output_dir: Path,
        source: BaseRgbSource,
        options: PrepareOptions,
    ) -> None:
        assert manifest.wgs is not None and manifest.wgs.rgb is not None
        self.wgs_grid = manifest.wgs.grid()
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.source = source
        self.options = options
        self.expected_tags = {
            TAG_PACKAGE_VERSION: __version__,
            TAG_PROCESSING_VERSION: RGB_PROCESSING_VERSION,
            TAG_SOURCE_ID: source.source_id,
            TAG_RGB_RESOLUTION: f"{options.rgb_resolution_m:.10g}",
            TAG_RGB_JPEG_QUALITY: str(options.rgb_jpeg_quality),
        }

    def _is_reusable(
        self, path: Path, tile: TileIndex, window: LatticeWindow, fill: LatticeWindow
    ) -> tuple[bool, str]:
        """Whether an existing RGB tile has this run's extent, provenance and imagery."""
        tags = read_tags(path)
        if not tags:
            return False, "tile carries no provenance metadata"
        for key in (TAG_SOURCE_ID, TAG_RGB_RESOLUTION, TAG_RGB_JPEG_QUALITY, TAG_PROCESSING_VERSION):
            expected = self.expected_tags[key]
            actual = tags.get(key)
            if actual != expected:
                return False, f"{key} is {actual!r}, this run produces {expected!r}"
        filled = LatticeWindow.from_tag(tags.get(TAG_RGB_FILLED))
        if filled is None or not filled.contains(fill):
            return False, "it holds imagery for less than the worlds now need"
        try:
            with rasterio.open(path) as dataset:
                existing = self.wgs_grid.rgb.window_of_transform(
                    dataset.transform, dataset.width, dataset.height
                )
        except Exception as exc:
            return False, f"cannot open it: {exc}"
        if existing != window:
            return False, f"it spans {existing}, this run expects {window}"
        report = validate_rgb_tile(path, self.manifest, tile, report=ValidationReport())
        if report.errors:
            return False, f"existing tile failed validation: {report.errors[0]}"
        return True, ""

    def build(self, tile: TileIndex, height_fill: LatticeWindow) -> TileOutcome:
        started = time.monotonic()
        path = self.manifest.wgs_rgb_path(self.output_dir, tile)
        path.parent.mkdir(parents=True, exist_ok=True)

        window = self.wgs_grid.rgb_tile_window(tile)
        fill = height_fill.scaled(self.wgs_grid.height_factor)

        if path.exists():
            reusable, reason = self._is_reusable(path, tile, window, fill)
            if reusable and not self.options.overwrite:
                log.debug("rgb tile %s: reusing %s", tile, path.name)
                return TileOutcome(tile, "reused", path, duration_s=time.monotonic() - started)
            if not self.options.overwrite:
                raise OutputConflictError(
                    f"rgb tile {tile}: {path} already exists but cannot be reused "
                    f"({reason}); pass --overwrite to replace it"
                )
            log.debug("rgb tile %s: regenerating (%s)", tile, reason or "overwrite requested")

        lattice = self.wgs_grid.rgb
        try:
            rgba = self.source.read_rgba(lattice, fill)
        except SourceError as exc:
            raise SourceError(
                f"rgb tile {tile} (bounds {lattice.window_bounds(fill)} in "
                f"{self.source.crs}): {exc}"
            ) from exc

        rgb = self.manifest.wgs.rgb
        col_off, row_off = fill.offset_in(window)
        write_rgb_window_atomic(
            path,
            rgba,
            profile=rgb_storage_profile(
                width=window.width,
                height=window.height,
                transform=lattice.window_transform(window),
                crs=self.manifest.wgs.crs,
                jpeg_quality=self.options.rgb_jpeg_quality,
                block_size_px=rgb.block_width_px,
            ),
            col_off=col_off,
            row_off=row_off,
            overwrite=True,
            validate=lambda tmp: self._validate_pending(tmp, tile),
            tags={**self.expected_tags, TAG_RGB_FILLED: fill.to_tag()},
        )
        duration = time.monotonic() - started
        log.debug(
            "rgb tile %s: %s of a %s tile, %.2fs", tile, fill, f"{window.width}x{window.height}", duration
        )
        return TileOutcome(tile, "written", path, duration_s=duration)

    def _validate_pending(self, tmp_path: Path, tile: TileIndex) -> None:
        report = validate_rgb_tile(tmp_path, self.manifest, tile, report=ValidationReport())
        for warning in report.warnings:
            log.warning("%s", warning)
        report.raise_for_errors(f"rgb tile {tile}")


# --- entry points --------------------------------------------------------


def _build_rgb_source(options: PrepareOptions) -> BaseRgbSource:
    return CuzkOrthophotoSource(
        timeout_s=options.request_timeout_s,
        retries=options.request_retries,
        cache_dir=options.cache_dir,
    )


def _wgs_info(options: PrepareOptions, rgb_source: BaseRgbSource | None) -> WgsInfo:
    grid = WgsGrid.from_metres(
        options.rgb_resolution_m,
        tile_size_px=options.wgs_tile_size_px,
        height_factor=options.wgs_height_factor,
    )
    rgb = (
        WgsRgbInfo(
            source_id=rgb_source.source_id,
            source_service_url=getattr(rgb_source, "service_url", None),
            source_provider=getattr(rgb_source, "provider", "CUZK"),
            source_product=getattr(rgb_source, "product", "ORTOFOTO"),
            jpeg_quality=options.rgb_jpeg_quality,
            block_width_px=RGB_BLOCK_PX,
            block_height_px=RGB_BLOCK_PX,
            processing_version=RGB_PROCESSING_VERSION,
        )
        if rgb_source is not None
        else None
    )
    return WgsInfo(
        resolution_m=options.rgb_resolution_m,
        resolution_deg=grid.resolution_deg,
        tile_size_px=grid.tile_size_px,
        height_factor=grid.height_factor,
        height_processing_version=WGS_HEIGHT_PROCESSING_VERSION,
        rgb=rgb,
    )


def _build_source(options: PrepareOptions) -> BaseHeightSource:
    return CuzkDmr5Source(
        resolution_m=options.resolution_m,
        nodata=options.nodata,
        timeout_s=options.request_timeout_s,
        retries=options.request_retries,
        cache_dir=options.cache_dir,
        max_request_px=options.max_request_px,
    )


def _grid_origin(
    source: BaseHeightSource, options: PrepareOptions, resolution_m: float
) -> tuple[float, float]:
    """Decide where tile (0, 0) starts.

    An explicit origin is honoured verbatim; otherwise the nominal anchor is
    nudged by less than one pixel so that tile edges coincide with the source
    product's own pixel boundaries and nothing has to be resampled.
    """
    native = source.native_grid_origin()
    aligned = align_origin(NOMINAL_GRID_ORIGIN, native, resolution_m)

    origin_x = options.grid_origin_x if options.grid_origin_x is not None else aligned[0]
    origin_y = options.grid_origin_y if options.grid_origin_y is not None else aligned[1]

    if native is not None:
        for label, value, native_value in (
            ("origin_x", origin_x, native[0]),
            ("origin_y", origin_y, native[1]),
        ):
            offset = (value - native_value) % resolution_m
            offset = min(offset, resolution_m - offset)
            if offset > 1e-6:
                log.warning(
                    "grid %s=%r is %.3f m out of phase with the %s pixel grid; "
                    "the source will be resampled",
                    label,
                    value,
                    offset,
                    source.source_id,
                )
    return origin_x, origin_y


def _build_manifest(
    *,
    worlds: Sequence[WorldConfig],
    grid: TileGrid,
    source: BaseHeightSource,
    operation: VerticalOperation,
    options: PrepareOptions,
) -> Manifest:
    """Describe the dataset this run is about to produce.

    Everything a sampler needs to find a tile, plus the provenance from
    specification section 19: the exact PROJ operation, the grids it used and
    their checksums, and the versions of everything that shaped the pixels.
    """
    target = VERTICAL_TARGETS[options.vertical_datum]
    checksums = (
        grid_file_checksums(
            Path(options.proj_data_dir) / name for name in operation.grid_names
        )
        if options.proj_data_dir
        else {}
    )
    return Manifest(
        horizontal_crs=SOURCE_HORIZONTAL_CRS,
        vertical_crs=target.vertical_crs,
        vertical_datum=options.vertical_datum,
        nodata=options.nodata,
        resolution_x=grid.resolution_x,
        resolution_y=grid.resolution_y,
        tile_width_px=grid.tile_width_px,
        tile_height_px=grid.tile_height_px,
        origin_x=grid.origin_x,
        origin_y=grid.origin_y,
        block_width_px=options.block_size_px,
        block_height_px=options.block_size_px,
        storage_format="cog" if options.cog else "geotiff",
        source_id=source.source_id,
        source_horizontal_crs=source.horizontal_crs,
        source_vertical_crs=source.vertical_crs,
        source_vertical_datum=SOURCE_VERTICAL_DATUM,
        source_resolution_m=source.resolution_m,
        source_service_url=getattr(source, "service_url", None),
        network_enabled=False,
        proj_operation=operation.description,
        proj_operation_accuracy_m=operation.accuracy,
        proj_grids=list(operation.grid_names),
        proj_grid_checksums=checksums,
        worlds=[world.name for world in worlds],
        software=software_versions(__version__),
        interpolation=getattr(source, "interpolation", "nearest"),
        processing_version=PROCESSING_VERSION,
    )


def _read_manifest_bytes(output_dir: Path) -> bytes | None:
    path = Path(output_dir) / MANIFEST_FILENAME
    try:
        return path.read_bytes()
    except OSError:
        return None


def _restore_manifest(output_dir: Path, previous: bytes | None) -> None:
    """Put back the manifest a failed run replaced, or remove the new one."""
    path = Path(output_dir) / MANIFEST_FILENAME
    try:
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(previous)
            tmp.replace(path)
    except OSError as exc:  # pragma: no cover - best effort during teardown
        log.warning("could not restore the previous manifest in %s: %s", output_dir, exc)


def _tile_directories(output_dir: Path, manifest: Manifest) -> list[Path]:
    """The directory of every kind of tile the dataset holds."""
    patterns = [manifest.tile_pattern]
    if manifest.wgs is not None:
        patterns.append(manifest.wgs.height_tile_pattern)
        if manifest.wgs.rgb is not None:
            patterns.append(manifest.wgs.rgb.tile_pattern)
    return [output_dir / Path(pattern.format(ix=0, iy=0)).parent for pattern in patterns]


def _clean_stale_temporaries(output_dir: Path, manifest: Manifest) -> None:
    for directory in _tile_directories(output_dir, manifest):
        if not directory.is_dir():
            continue
        for leftover in directory.glob("*.tmp"):
            log.debug("removing leftover temporary file %s", leftover)
            leftover.unlink(missing_ok=True)


def _skip(skipped: dict[Path, str], world: WorldConfig, reason: str) -> None:
    log.warning("skipping world config: %s", reason)
    skipped[world.path or Path(world.name)] = reason


def _corner_heights(
    output_dir: Path, manifest: Manifest, worlds: Sequence[WorldConfig]
) -> dict[tuple[float, float], float]:
    """The height at every safety area corner, keyed by ``(lat, lon)``; NaN where none."""
    heights: dict[tuple[float, float], float] = {}
    with DatasetSampler(output_dir, manifest) as sampler:
        for world in worlds:
            for lon, lat in world.points:
                if (lat, lon) not in heights:
                    heights[(lat, lon)] = sampler.sample(lon, lat)
    return heights


def _discard_unneeded_tiles(
    output_dir: Path, manifest: Manifest, plans: Sequence[WorldPlan]
) -> None:
    """Remove the tiles that only skipped worlds needed."""
    needed = {tile for plan in plans for tile in plan.tiles}
    for tile in manifest.tiles:
        if tile not in needed:
            manifest.tile_path(output_dir, tile).unlink(missing_ok=True)
            log.info("removed tile %s, which only skipped worlds needed", tile)
    manifest.set_tiles(list(needed))


def _write_test_points(
    output_dir: Path,
    manifest: Manifest,
    worlds: Sequence[WorldConfig],
    heights: dict[tuple[float, float], float],
) -> list[ReferencePoint]:
    """Write the height at every safety area corner to the test points file."""
    points: list[ReferencePoint] = []
    seen: set[tuple[float, float]] = set()
    for world in worlds:
        for lon, lat in world.points:
            if (lat, lon) not in seen:
                seen.add((lat, lon))
                points.append(ReferencePoint(lat=lat, lon=lon, height=heights[(lat, lon)]))

    write_test_points(Path(output_dir) / TEST_POINTS_FILENAME, points)
    manifest.test_points_path = TEST_POINTS_FILENAME
    manifest.test_points_count = len(points)
    return points


def prepare_worlds(
    inputs: Sequence[Path],
    output_dir: Path,
    options: PrepareOptions | None = None,
    *,
    source: BaseHeightSource | None = None,
    rgb_source: BaseRgbSource | None = None,
) -> PrepareResult:
    """Prepare every world config in ``inputs`` into a dataset under ``output_dir``.

    ``source`` may be supplied to substitute the acquisition backend (used by
    the tests and by future adapters); by default the ČÚZK DMR 5G ImageServer is
    used.  ``rgb_source`` likewise substitutes the ČÚZK orthophoto service when
    ``options.include_rgb`` is set.
    """
    options = options or PrepareOptions()
    output_dir = Path(output_dir)

    worlds, skipped = load_worlds([Path(p) for p in inputs])
    if not worlds:
        raise ConfigError(
            f"none of the {len(skipped)} world config(s) could be used; nothing to prepare"
        )
    resolution_m = options.resolution_m
    log.info(
        "preparing %d world(s) at %g m onto the %s vertical datum: %s",
        len(worlds),
        resolution_m,
        options.vertical_datum,
        ", ".join(world.name for world in worlds),
    )

    # PROJ must be configured before the first transformer is built.
    converter = build_converter(
        options.vertical_datum,
        proj_data_dir=options.proj_data_dir,
        network_enabled=False,
    )
    operation = converter.operation
    log.info("selected vertical transformation: %s", operation.description)
    log.debug("PROJ grids in use: %s", ", ".join(operation.grid_names) or "none")

    vertical_report = validate_vertical_transformation(converter)
    vertical_report.raise_for_errors("vertical transformation validation")
    for warning in vertical_report.warnings:
        log.warning("%s", warning)

    owned_source = source is None
    height_source = source or _build_source(options)
    owned_rgb_source = rgb_source is None
    if not options.include_rgb:
        rgb_source = None
    elif rgb_source is None:
        rgb_source = _build_rgb_source(options)

    try:
        if height_source.vertical_crs != SOURCE_VERTICAL_CRS:
            # The operation above was built from SOURCE_VERTICAL_CRS, so a source
            # on any other datum would be silently mis-converted.
            raise UnexpectedVerticalDatumError(
                f"source {height_source.source_id!r} declares vertical CRS "
                f"{height_source.vertical_crs}, but the {options.vertical_datum} "
                f"conversion is defined from {SOURCE_VERTICAL_CRS} "
                f"({SOURCE_VERTICAL_DATUM})"
            )

        if abs(height_source.resolution_m - resolution_m) > 1e-9:
            log.warning(
                "source native resolution is %g m but %g m was requested; "
                "the service will resample",
                height_source.resolution_m,
                resolution_m,
            )

        grid = TileGrid.from_options(
            resolution_m, options.tile_size_px, *_grid_origin(height_source, options, resolution_m)
        )
        log.debug(
            "global tile grid: origin (%.3f, %.3f), %d x %d px, span %g x %g m",
            grid.origin_x,
            grid.origin_y,
            grid.tile_width_px,
            grid.tile_height_px,
            grid.tile_span_x,
            grid.tile_span_y,
        )

        wgs_info = _wgs_info(options, rgb_source)
        wgs_grid = wgs_info.grid()
        coverage = height_source.coverage()
        plans: list[WorldPlan] = []
        rgb_coverage = rgb_source.coverage() if rgb_source is not None else None
        for world in worlds:
            try:
                plan = plan_world(world, grid, wgs_grid, coverage)
                needed = grid.pixels_bounds(plan.pixels)
                if rgb_coverage is not None and not rgb_coverage.contains(needed):
                    raise OutOfCoverageError(
                        f"world {world.describe()}: the safety area needs EPSG:5514 extent "
                        f"{tuple(round(v, 1) for v in needed.as_tuple())}, which is not "
                        f"inside the {rgb_source.source_id} coverage {rgb_coverage.as_tuple()}"
                    )
                plans.append(plan)
            except HeightmapPrepError as exc:
                _skip(skipped, world, str(exc))
        if not plans:
            raise ConfigError(
                "no world config's safety area can be prepared from the source; "
                "nothing to prepare"
            )
        worlds = [plan.world for plan in plans]
        required = merge_tile_requirements(plans, grid)

        for plan in plans:
            log.info(
                "world %s: %d-vertex safety area -> tile(s) %s",
                plan.world.name,
                len(plan.world.points),
                ", ".join(str(tile) for tile in plan.tiles),
            )

        manifest = _build_manifest(
            worlds=worlds,
            grid=grid,
            source=height_source,
            operation=operation,
            options=options,
        )
        manifest.wgs = wgs_info
        manifest.set_tiles(list(required))

        output_dir.mkdir(parents=True, exist_ok=True)
        _clean_stale_temporaries(output_dir, manifest)
        previous_manifest = _read_manifest_bytes(output_dir)
        manifest_path = manifest.write(output_dir)
        log.info(
            "dataset manifest written to %s (status: %s, %d tile(s) planned)",
            manifest_path,
            manifest.status,
            len(required),
        )

        builder = _TileBuilder(
            grid=grid,
            manifest=manifest,
            output_dir=output_dir,
            source=height_source,
            converter=converter,
            options=options,
        )

        try:
            outcomes = _run_builder(builder, required, options.workers)
        except BaseException:
            # A run that never got as far as writing a tile must not leave an
            # existing complete dataset marked as "building".
            if not builder.modified:
                _restore_manifest(output_dir, previous_manifest)
            raise

        written = sum(1 for o in outcomes if o.action == "written")
        log.info("tiles: %d written, %d reused", written, len(outcomes) - written)
        cache_hits = getattr(height_source, "cache_hits", None)
        if cache_hits is not None:
            log.info(
                "source requests: %d performed, %d served from cache",
                getattr(height_source, "requests_made", 0),
                cache_hits,
            )

        # Every point of a safety area must be samplable, so a world with a corner
        # the source has no height for cannot be part of the dataset.
        heights = _corner_heights(output_dir, manifest, worlds)
        kept: list[WorldPlan] = []
        for plan in plans:
            missing = [
                f"{lat}, {lon}"
                for lon, lat in plan.world.points
                if not math.isfinite(heights[(lat, lon)])
            ]
            if missing:
                _skip(
                    skipped,
                    plan.world,
                    f"world {plan.world.describe()}: the source has no height around "
                    f"safety area corner(s) {'; '.join(missing)}",
                )
            else:
                kept.append(plan)
        if not kept:
            raise ValidationError(
                "the source has no height around the safety area corners of any world; "
                "nothing to publish"
            )
        if len(kept) < len(plans):
            plans = kept
            worlds = [plan.world for plan in plans]
            manifest.worlds = [world.name for world in worlds]
            _discard_unneeded_tiles(output_dir, manifest, plans)
            remaining = set(manifest.tiles)
            outcomes = [o for o in outcomes if o.tile in remaining]

        # The WGS tiles are cut for the worlds still in the dataset only.
        wgs_required = merge_wgs_requirements(plans, wgs_grid)
        wgs_info.set_tiles(list(wgs_required))
        wgs_builder = _WgsHeightTileBuilder(
            manifest=manifest, output_dir=output_dir, source_id=height_source.source_id
        )
        # Sequential: each build opens its own sampler over the finished tiles.
        wgs_outcomes = _run_builder(wgs_builder, wgs_required, workers=1)
        log.info("wgs height tiles: %d written", len(wgs_outcomes))

        rgb_outcomes: list[TileOutcome] = []
        if rgb_source is not None:
            rgb_builder = _RgbTileBuilder(
                manifest=manifest,
                output_dir=output_dir,
                source=rgb_source,
                options=options,
            )
            rgb_outcomes = _run_builder(rgb_builder, wgs_required, options.workers)
            written = sum(1 for o in rgb_outcomes if o.action == "written")
            log.info("rgb tiles: %d written, %d reused", written, len(rgb_outcomes) - written)

        test_points = _write_test_points(output_dir, manifest, worlds, heights)
        log.info(
            "wrote %d test point(s) to %s", len(test_points), output_dir / TEST_POINTS_FILENAME
        )
        worlds_db = write_worlds_db(
            output_dir,
            manifest,
            worlds,
            {plan.world.name: plan.tiles for plan in plans},
            {plan.world.name: plan.wgs_tiles for plan in plans},
        )
        manifest.write(output_dir)
        log.info("recorded %d world(s) in %s", len(worlds), worlds_db)

        if options.validate_tiles:
            report = validate_dataset(output_dir, plausibility=options.plausibility_checks)
        else:
            report = ValidationReport()
        report.log()
        report.raise_for_errors(f"dataset {output_dir}")

        manifest.mark_complete()
        manifest_path = manifest.write(output_dir)
        log.info("dataset complete: %s", output_dir.resolve())

        return PrepareResult(
            output_dir=output_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            worlds=worlds,
            plans=plans,
            outcomes=outcomes,
            wgs_outcomes=wgs_outcomes,
            rgb_outcomes=rgb_outcomes,
            report=report,
            test_points=test_points,
            skipped=skipped,
        )
    finally:
        for owned, opened in ((owned_source, height_source), (owned_rgb_source, rgb_source)):
            close = getattr(opened, "close", None)
            if owned and callable(close):
                close()


def _run_builder(
    builder: _TileBuilder | _WgsHeightTileBuilder | _RgbTileBuilder,
    required: Mapping[TileIndex, PixelRange | LatticeWindow],
    workers: int,
) -> list[TileOutcome]:
    """Build every required tile, optionally across a bounded thread pool.

    ``builder`` builds one tile per call from what that tile needs: an
    EPSG:5514 pixel range or a WGS height-lattice window.
    """
    items = list(required.items())
    if workers <= 1 or len(items) <= 1:
        return [builder.build(tile, pixels) for tile, pixels in items]

    outcomes: list[TileOutcome] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tile") as pool:
        futures = {pool.submit(builder.build, tile, pixels): tile for tile, pixels in items}
        try:
            for future in as_completed(futures):
                outcomes.append(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    outcomes.sort(key=lambda outcome: (outcome.tile.iy, outcome.tile.ix))
    return outcomes


def validate_only(output_dir: Path) -> ValidationReport:
    """Validate an existing dataset without downloading or converting anything."""
    output_dir = Path(output_dir)
    log.info("validating existing dataset %s", output_dir.resolve())
    report = validate_dataset(output_dir, plausibility=True, require_complete=True)
    report.log()
    return report
