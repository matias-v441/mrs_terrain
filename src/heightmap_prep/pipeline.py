"""The preparation pipeline: worlds in, validated tiled dataset out.

This is the orchestration layer that ties acquisition (:mod:`.cuzk`), vertical
conversion (:mod:`.crs`, :mod:`.raster`), the global tile grid (:mod:`.tiling`),
the manifest (:mod:`.manifest`) and validation (:mod:`.validate`) together.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Sequence

import numpy as np

from . import __version__
from .config import (
    NOMINAL_GRID_ORIGIN,
    PrepareOptions,
    WorldConfig,
    align_origin,
    load_worlds,
    resolve_dataset_settings,
)
from .crs import (
    SOURCE_HORIZONTAL_CRS,
    SOURCE_VERTICAL_CRS,
    SOURCE_VERTICAL_DATUM,
    VERTICAL_TARGETS,
    VerticalConverter,
    VerticalOperation,
    build_converter,
    wgs84_bounds_to_projected,
)
from .cuzk import CuzkDmr5Source
from .errors import (
    ConfigError,
    OutOfCoverageError,
    OutputConflictError,
    SourceError,
    UnexpectedVerticalDatumError,
)
from .manifest import (
    MANIFEST_FILENAME,
    Manifest,
    grid_file_checksums,
    software_versions,
)
from .raster import (
    ConversionStats,
    PROCESSING_VERSION,
    TAG_PROCESSING_VERSION,
    TAG_RESOLUTION,
    TAG_SOURCE_ID,
    TAG_VERTICAL_CRS,
    TAG_VERTICAL_DATUM,
    convert_vertical,
    provenance_tags,
    read_tags,
    write_raster_atomic,
)
from .sources import BaseHeightSource
from .tiling import ProjectedBounds, TileGrid, TileIndex
from .validate import ValidationReport, validate_dataset, validate_tile, validate_vertical_transformation

log = logging.getLogger(__name__)


@dataclass
class WorldPlan:
    """A world's footprint on the global tile grid."""

    world: WorldConfig
    projected_bounds: ProjectedBounds
    tiles: list[TileIndex]


@dataclass
class TileOutcome:
    """What happened to one tile during a run."""

    tile: TileIndex
    action: str  # "written" | "reused" | "skipped"
    path: Path | None = None
    stats: ConversionStats | None = None
    duration_s: float = 0.0
    reason: str = ""


@dataclass
class PrepareResult:
    """Everything a caller might want to know about a completed run."""

    output_dir: Path
    manifest_path: Path
    manifest: Manifest
    worlds: list[WorldConfig]
    plans: list[WorldPlan] = field(default_factory=list)
    outcomes: list[TileOutcome] = field(default_factory=list)
    report: ValidationReport = field(default_factory=ValidationReport)

    @property
    def tiles_written(self) -> list[TileIndex]:
        return [o.tile for o in self.outcomes if o.action == "written"]

    @property
    def tiles_reused(self) -> list[TileIndex]:
        return [o.tile for o in self.outcomes if o.action == "reused"]

    @property
    def tiles_skipped(self) -> list[TileIndex]:
        return [o.tile for o in self.outcomes if o.action == "skipped"]


# --- planning ------------------------------------------------------------


def plan_world(
    world: WorldConfig, grid: TileGrid, coverage: ProjectedBounds | None
) -> WorldPlan:
    """Project a world's WGS84 box and list the tiles it touches."""
    bounds_tuple = wgs84_bounds_to_projected(
        world.bounds.west,
        world.bounds.south,
        world.bounds.east,
        world.bounds.north,
        target_epsg=5514,
    )
    projected = ProjectedBounds(*bounds_tuple)
    log.debug(
        "world %s: WGS84 %s -> EPSG:5514 %s",
        world.name,
        (world.bounds.west, world.bounds.south, world.bounds.east, world.bounds.north),
        projected.as_tuple(),
    )

    if coverage is not None:
        clipped = coverage.intersection(projected)
        if clipped is None:
            raise OutOfCoverageError(
                f"world {world.describe()}: requested area "
                f"{projected.as_tuple()} (EPSG:5514) does not intersect the source "
                f"coverage {coverage.as_tuple()}"
            )
        if clipped.as_tuple() != projected.as_tuple():
            log.warning(
                "world %s: requested area is partly outside source coverage; "
                "clipping to %s",
                world.name,
                clipped.as_tuple(),
            )
        projected = clipped

    return WorldPlan(world=world, projected_bounds=projected, tiles=grid.tiles_for_bounds(projected))


def merge_tile_requirements(plans: Sequence[WorldPlan], grid: TileGrid) -> dict[TileIndex, ProjectedBounds]:
    """For each tile, the area that actually has to be fetched.

    Worlds rarely fill a whole tile, so only the union of the requesting worlds'
    footprints inside that tile is acquired; the rest of the tile stays NoData.
    Fetching the union rather than each world separately means overlapping worlds
    share one request.
    """
    required: dict[TileIndex, ProjectedBounds] = {}
    for plan in plans:
        for tile in plan.tiles:
            overlap = grid.tile_bounds(tile).intersection(plan.projected_bounds)
            if overlap is None:
                continue
            previous = required.get(tile)
            required[tile] = overlap if previous is None else previous.union(overlap)
    # Reading order, so a run's tile order never depends on the worker count.
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

    def _is_reusable(self, path: Path, tile: TileIndex) -> tuple[bool, str]:
        """Whether an existing tile was produced by an equivalent run."""
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

    def build(self, tile: TileIndex, required: ProjectedBounds) -> TileOutcome:
        started = time.monotonic()
        path = self.manifest.tile_path(self.output_dir, tile)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            reusable, reason = self._is_reusable(path, tile)
            if reusable and not self.options.overwrite:
                log.debug("tile %s: reusing %s", tile, path.name)
                return TileOutcome(tile, "reused", path, duration_s=time.monotonic() - started)
            if not self.options.overwrite:
                raise OutputConflictError(
                    f"tile {tile}: {path} already exists but cannot be reused "
                    f"({reason}); pass --overwrite to replace it"
                )
            log.debug("tile %s: regenerating (%s)", tile, reason or "overwrite requested")

        window = self.grid.pixel_window_for_bounds(tile, required)
        if window is None:  # pragma: no cover - merge_tile_requirements excludes these
            return TileOutcome(tile, "skipped", reason="no overlap with requested area")

        coverage = self.source.coverage()
        fetch_bounds = self.grid.window_bounds(tile, window)
        if coverage is not None and coverage.intersection(fetch_bounds) is None:
            log.info("tile %s: entirely outside source coverage, skipping", tile)
            return TileOutcome(tile, "skipped", reason="outside source coverage")

        col_off, row_off, width, height = window
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
            tags=self.expected_tags,
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


# --- entry points --------------------------------------------------------


def _build_source(
    source_id: str, resolution_m: float, options: PrepareOptions
) -> BaseHeightSource:
    if source_id != "cuzk-dmr5g":
        raise ConfigError(
            f"unsupported heightmap source {source_id!r}; "
            "version 1 implements 'cuzk-dmr5g'"
        )
    return CuzkDmr5Source(
        resolution_m=resolution_m,
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
    dataset_bounds: ProjectedBounds | None,
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
        world_bounds_wgs84={
            world.name: [
                world.bounds.west,
                world.bounds.south,
                world.bounds.east,
                world.bounds.north,
            ]
            for world in worlds
        },
        dataset_bounds=list(dataset_bounds.as_tuple()) if dataset_bounds else None,
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


def _clean_stale_temporaries(output_dir: Path, manifest: Manifest) -> None:
    height_dir = output_dir / Path(manifest.tile_relative_path(TileIndex(0, 0))).parent
    if not height_dir.is_dir():
        return
    for leftover in height_dir.glob("*.tmp"):
        log.debug("removing leftover temporary file %s", leftover)
        leftover.unlink(missing_ok=True)


def prepare_worlds(
    inputs: Sequence[Path],
    output_dir: Path,
    options: PrepareOptions | None = None,
    *,
    source: BaseHeightSource | None = None,
) -> PrepareResult:
    """Prepare every world in ``inputs`` into a dataset under ``output_dir``.

    ``source`` may be supplied to substitute the acquisition backend (used by
    the tests and by future adapters); by default the ČÚZK DMR 5G ImageServer is
    used.
    """
    options = options or PrepareOptions()
    output_dir = Path(output_dir)

    worlds = load_worlds([Path(p) for p in inputs])
    source_id, resolution_m = resolve_dataset_settings(worlds, options)
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
    height_source = source or _build_source(source_id, resolution_m, options)

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

        coverage = height_source.coverage()
        plans = [plan_world(world, grid, coverage) for world in worlds]
        required = merge_tile_requirements(plans, grid)

        dataset_bounds: ProjectedBounds | None = None
        for plan in plans:
            log.info(
                "world %s: EPSG:5514 bounds %s -> %d tile(s)",
                plan.world.name,
                tuple(round(v, 1) for v in plan.projected_bounds.as_tuple()),
                len(plan.tiles),
            )
            dataset_bounds = (
                plan.projected_bounds
                if dataset_bounds is None
                else dataset_bounds.union(plan.projected_bounds)
            )

        manifest = _build_manifest(
            worlds=worlds,
            grid=grid,
            source=height_source,
            operation=operation,
            options=options,
            dataset_bounds=dataset_bounds,
        )
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
        reused = sum(1 for o in outcomes if o.action == "reused")
        skipped = [o for o in outcomes if o.action == "skipped"]
        log.info(
            "tiles: %d written, %d reused, %d skipped", written, reused, len(skipped)
        )
        cache_hits = getattr(height_source, "cache_hits", None)
        if cache_hits is not None:
            log.info(
                "source requests: %d performed, %d served from cache",
                getattr(height_source, "requests_made", 0),
                cache_hits,
            )

        # Tiles that turned out to be entirely outside coverage are not part of
        # the dataset, so the manifest must not promise them.
        if skipped:
            manifest.set_tiles(
                [o.tile for o in outcomes if o.action in ("written", "reused")]
            )

        if options.validate_tiles:
            report = validate_dataset(
                output_dir,
                plausibility=options.plausibility_checks,
                expected_tiles=manifest.tiles,
            )
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
            report=report,
        )
    finally:
        if owned_source:
            close = getattr(height_source, "close", None)
            if callable(close):
                close()


def _run_builder(
    builder: _TileBuilder,
    required: dict[TileIndex, ProjectedBounds],
    workers: int,
) -> list[TileOutcome]:
    """Build every required tile, optionally across a bounded thread pool."""
    items = list(required.items())
    if workers <= 1 or len(items) <= 1:
        return [builder.build(tile, bounds) for tile, bounds in items]

    outcomes: list[TileOutcome] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tile") as pool:
        futures = {pool.submit(builder.build, tile, bounds): tile for tile, bounds in items}
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
