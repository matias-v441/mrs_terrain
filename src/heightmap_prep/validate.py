"""Dataset validation (specification section 17).

Nothing is published to the sampler before it has been checked: the manifest
must be self-consistent, every tile must sit exactly on the declared global
grid, and the vertical transformation must have been a real, non-ballpark
operation.  Plausibility checks are reported separately as warnings and never
stand in for the CRS-level checks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
import rasterio.enums
import rasterio.windows

from .crs import (
    VerticalConverter,
    check_known_points,
    crs_matches,
    epsg_code_of,
)
from .errors import ValidationError
from .manifest import FORMAT_VERSION, STATUS_COMPLETE, Manifest
from .sampling import DatasetSampler, read_test_points
from .tiling import TileGrid, TileIndex, parse_tile_filename

log = logging.getLogger(__name__)

#: Plausible prepared-elevation range for Czech terrain, in metres.  Wide enough
#: to cover both EGM96 heights and WGS84 ellipsoidal heights (~+45 m).
PLAUSIBLE_MIN_M = -100.0
PLAUSIBLE_MAX_M = 2100.0

#: Affine components must match the grid to this many metres.
TRANSFORM_TOLERANCE_M = 1e-6

#: A test point must re-sample to its recorded height within this many metres.
TEST_POINT_TOLERANCE_M = 1e-6


@dataclass
class ValidationReport:
    """Accumulated findings from one validation pass."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tiles_checked: int = 0
    tiles_missing: list[TileIndex] = field(default_factory=list)
    unexpected_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def extend(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.tiles_checked += other.tiles_checked
        self.tiles_missing.extend(other.tiles_missing)
        self.unexpected_files.extend(other.unexpected_files)

    def raise_for_errors(self, context: str = "dataset validation") -> None:
        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise ValidationError(f"{context} failed:\n  - {joined}")

    def log(self, level: int = logging.INFO) -> None:
        for warning in self.warnings:
            log.warning("%s", warning)
        if self.ok:
            log.log(level, "validation passed (%d tile(s) checked)", self.tiles_checked)


# --- 17.1 manifest -------------------------------------------------------


def validate_manifest(manifest: Manifest, report: ValidationReport | None = None) -> ValidationReport:
    """Check that the manifest is internally consistent and usable."""
    report = report or ValidationReport()

    if manifest.format_version != FORMAT_VERSION:
        report.error(
            f"unsupported manifest format_version {manifest.format_version}; "
            f"this build understands {FORMAT_VERSION}"
        )
    if not manifest.horizontal_crs:
        report.error("manifest heightmap.horizontal_crs is missing")
    elif epsg_code_of(manifest.horizontal_crs) is None:
        report.error(
            f"manifest heightmap.horizontal_crs {manifest.horizontal_crs!r} "
            "is not a resolvable CRS"
        )
    if not manifest.vertical_crs:
        report.error("manifest heightmap.vertical_crs is missing")
    if not manifest.unit:
        report.error("manifest heightmap.unit is missing")
    elif manifest.unit != "m":
        report.error(f"manifest heightmap.unit must be 'm', got {manifest.unit!r}")
    if manifest.dtype != "float32":
        report.error(f"manifest heightmap.dtype must be 'float32', got {manifest.dtype!r}")
    if not math.isfinite(manifest.nodata):
        report.error(f"manifest heightmap.nodata must be finite, got {manifest.nodata}")

    if not (manifest.resolution_x > 0 and manifest.resolution_y > 0):
        report.error(
            f"manifest grid resolution must be positive, got "
            f"{manifest.resolution_x} x {manifest.resolution_y}"
        )
    if manifest.tile_width_px <= 0 or manifest.tile_height_px <= 0:
        report.error(
            f"manifest grid tile dimensions must be positive, got "
            f"{manifest.tile_width_px} x {manifest.tile_height_px}"
        )
    if not (math.isfinite(manifest.origin_x) and math.isfinite(manifest.origin_y)):
        report.error("manifest grid origin must be finite")
    if manifest.axis_order != "east_north":
        report.error(
            f"manifest grid.axis_order must be 'east_north', got {manifest.axis_order!r}"
        )

    if not manifest.tile_pattern:
        report.error("manifest storage.tile_pattern is missing")
    else:
        _check_pattern(manifest.tile_pattern, "storage.tile_pattern", report)
    if manifest.block_width_px <= 0 or manifest.block_height_px <= 0:
        report.error("manifest storage block dimensions must be positive")

    if manifest.rgb is not None:
        rgb = manifest.rgb
        if epsg_code_of(rgb.crs) != 4326:
            report.error(f"manifest rgb.crs must be EPSG:4326, got {rgb.crs!r}")
        if not (rgb.resolution_deg > 0 and rgb.resolution_m > 0):
            report.error(
                f"manifest rgb resolution must be positive, got {rgb.resolution_m} m "
                f"/ {rgb.resolution_deg} deg"
            )
        _check_pattern(rgb.tile_pattern, "rgb.tile_pattern", report)
        if rgb.tile_pattern == manifest.tile_pattern:
            report.error("manifest rgb.tile_pattern must differ from storage.tile_pattern")

    return report


def _check_pattern(pattern: str, key: str, report: ValidationReport) -> None:
    if "{ix}" not in pattern or "{iy}" not in pattern:
        report.error(f"manifest {key} {pattern!r} must contain both {{ix}} and {{iy}} placeholders")
        return
    try:
        rendered = pattern.format(ix=0, iy=0)
    except (KeyError, IndexError, ValueError) as exc:
        report.error(f"manifest {key} {pattern!r} is malformed: {exc}")
        return
    if Path(rendered).is_absolute() or ".." in Path(rendered).parts:
        report.error(
            f"manifest {key} {pattern!r} must be a relative path that stays inside the "
            "dataset directory"
        )


# --- 17.2 per-tile GeoTIFF ----------------------------------------------


@dataclass
class TileFacts:
    """What validation learned about one tile, reused for spatial checks."""

    tile: TileIndex
    path: Path
    width: int
    height: int
    resolution_x: float
    resolution_y: float
    west: float
    north: float
    valid_min: float | None = None
    valid_max: float | None = None
    nodata_fraction: float = 0.0


def validate_tile(
    path: Path,
    manifest: Manifest,
    tile: TileIndex,
    *,
    grid: TileGrid | None = None,
    plausibility: bool = True,
    report: ValidationReport | None = None,
) -> tuple[ValidationReport, TileFacts | None]:
    """Validate one prepared GeoTIFF against the manifest's global grid."""
    report = report or ValidationReport()
    grid = grid or manifest.grid()
    expected_epsg = epsg_code_of(manifest.horizontal_crs)

    try:
        dataset = rasterio.open(path)
    except Exception as exc:
        report.error(f"tile {tile}: cannot open {path}: {exc}")
        return report, None

    with dataset:
        report.tiles_checked += 1

        if expected_epsg is not None and not crs_matches(dataset.crs, expected_epsg):
            report.error(
                f"tile {tile} ({path.name}): CRS is EPSG:{epsg_code_of(dataset.crs)}, "
                f"expected EPSG:{expected_epsg}"
            )

        if dataset.dtypes[0] != "float32":
            report.error(
                f"tile {tile} ({path.name}): dtype is {dataset.dtypes[0]}, expected float32"
            )
        if dataset.count != 1:
            report.error(
                f"tile {tile} ({path.name}): has {dataset.count} bands, expected 1"
            )

        if dataset.nodata is None:
            report.error(f"tile {tile} ({path.name}): no NoData value is set")
        elif not math.isclose(float(dataset.nodata), manifest.nodata, rel_tol=0, abs_tol=1e-9):
            report.error(
                f"tile {tile} ({path.name}): NoData is {dataset.nodata}, "
                f"manifest declares {manifest.nodata}"
            )

        transform = dataset.transform
        if abs(transform.b) > TRANSFORM_TOLERANCE_M or abs(transform.d) > TRANSFORM_TOLERANCE_M:
            report.error(
                f"tile {tile} ({path.name}): transform is rotated/skewed "
                f"(b={transform.b}, d={transform.d}); prepared tiles must be north-up"
            )
        if not math.isclose(abs(transform.a), manifest.resolution_x, rel_tol=0, abs_tol=TRANSFORM_TOLERANCE_M):
            report.error(
                f"tile {tile} ({path.name}): x resolution is {abs(transform.a)}, "
                f"manifest declares {manifest.resolution_x}"
            )
        if not math.isclose(abs(transform.e), manifest.resolution_y, rel_tol=0, abs_tol=TRANSFORM_TOLERANCE_M):
            report.error(
                f"tile {tile} ({path.name}): y resolution is {abs(transform.e)}, "
                f"manifest declares {manifest.resolution_y}"
            )
        if transform.e > 0:
            report.error(
                f"tile {tile} ({path.name}): raster is south-up (e={transform.e}); "
                "prepared tiles must be north-up"
            )

        expected = grid.tile_bounds(tile)
        if not math.isclose(transform.c, expected.west, rel_tol=0, abs_tol=TRANSFORM_TOLERANCE_M):
            report.error(
                f"tile {tile} ({path.name}): west edge is {transform.c}, "
                f"global grid expects {expected.west}"
            )
        if not math.isclose(transform.f, expected.north, rel_tol=0, abs_tol=TRANSFORM_TOLERANCE_M):
            report.error(
                f"tile {tile} ({path.name}): north edge is {transform.f}, "
                f"global grid expects {expected.north}"
            )

        # Full-size tiles are the norm; a clipped edge tile is allowed as long as
        # it is not larger than the grid cell and still starts at the tile origin.
        if dataset.width > manifest.tile_width_px or dataset.height > manifest.tile_height_px:
            report.error(
                f"tile {tile} ({path.name}): {dataset.width}x{dataset.height} px "
                f"exceeds the declared tile size "
                f"{manifest.tile_width_px}x{manifest.tile_height_px}"
            )
        elif (
            dataset.width != manifest.tile_width_px
            or dataset.height != manifest.tile_height_px
        ):
            report.warn(
                f"tile {tile} ({path.name}): {dataset.width}x{dataset.height} px is a "
                f"clipped edge tile (declared tile size is "
                f"{manifest.tile_width_px}x{manifest.tile_height_px})"
            )

        facts = TileFacts(
            tile=tile,
            path=path,
            width=dataset.width,
            height=dataset.height,
            resolution_x=abs(transform.a),
            resolution_y=abs(transform.e),
            west=transform.c,
            north=transform.f,
        )

        # Value checks, block by block so a 4096² tile is never fully resident.
        total = 0
        valid_count = 0
        vmin = math.inf
        vmax = -math.inf
        nonfinite = 0
        nodata_value = dataset.nodata
        for _, window in dataset.block_windows(1):
            block = dataset.read(1, window=window)
            total += block.size
            if nodata_value is None:
                valid = np.ones(block.shape, dtype=bool)
            else:
                valid = block != np.float32(nodata_value)
            values = block[valid]
            nonfinite += int(np.count_nonzero(~np.isfinite(values)))
            finite = values[np.isfinite(values)]
            valid_count += int(values.size)
            if finite.size:
                vmin = min(vmin, float(finite.min()))
                vmax = max(vmax, float(finite.max()))

        if nonfinite:
            report.error(
                f"tile {tile} ({path.name}): {nonfinite} non-NoData pixel(s) are "
                "not finite"
            )

        facts.nodata_fraction = 1.0 - (valid_count / total if total else 0.0)
        if valid_count:
            facts.valid_min = vmin
            facts.valid_max = vmax

        if plausibility:
            _plausibility_checks(facts, manifest, report)

    return report, facts


def _plausibility_checks(
    facts: TileFacts, manifest: Manifest, report: ValidationReport
) -> None:
    """Section 17.5 checks.  These only ever produce warnings."""
    name = facts.path.name
    # Tiles only hold the pixels around the safety areas, so being mostly NoData
    # is normal; holding no data at all is not.
    if facts.valid_min is None:
        report.warn(
            f"tile {facts.tile} ({name}): no valid pixel at all; the source may "
            "have no data there"
        )
        return
    if facts.valid_min == 0.0 and facts.valid_max == 0.0:
        report.warn(f"tile {facts.tile} ({name}): every valid pixel is exactly 0.0")
    if facts.valid_min < PLAUSIBLE_MIN_M or facts.valid_max > PLAUSIBLE_MAX_M:
        report.warn(
            f"tile {facts.tile} ({name}): elevations span "
            f"{facts.valid_min:.1f}..{facts.valid_max:.1f} m, outside the plausible "
            f"range {PLAUSIBLE_MIN_M:.0f}..{PLAUSIBLE_MAX_M:.0f} m"
        )


# --- 17.3 spatial consistency -------------------------------------------


def validate_spatial_consistency(
    facts: Sequence[TileFacts],
    manifest: Manifest,
    report: ValidationReport | None = None,
) -> ValidationReport:
    """Check that neighbouring tiles abut exactly, with no gap or overlap."""
    report = report or ValidationReport()
    if not facts:
        return report

    grid = manifest.grid()
    by_index = {f.tile: f for f in facts}

    resolutions = {(round(f.resolution_x, 9), round(f.resolution_y, 9)) for f in facts}
    if len(resolutions) > 1:
        report.error(
            "adjacent tiles have differing resolutions: "
            + ", ".join(f"{rx}x{ry}" for rx, ry in sorted(resolutions))
        )

    for tile, fact in sorted(by_index.items()):
        east_neighbour = by_index.get(TileIndex(tile.ix + 1, tile.iy))
        if east_neighbour is not None:
            seam = fact.west + fact.width * fact.resolution_x
            if not math.isclose(seam, east_neighbour.west, rel_tol=0, abs_tol=TRANSFORM_TOLERANCE_M):
                gap = east_neighbour.west - seam
                report.error(
                    f"tiles {tile} and {east_neighbour.tile} do not meet: "
                    f"{'gap' if gap > 0 else 'overlap'} of {abs(gap):.6f} m at "
                    f"x={seam:.3f}"
                )
        south_neighbour = by_index.get(TileIndex(tile.ix, tile.iy + 1))
        if south_neighbour is not None:
            seam = fact.north - fact.height * fact.resolution_y
            if not math.isclose(seam, south_neighbour.north, rel_tol=0, abs_tol=TRANSFORM_TOLERANCE_M):
                gap = seam - south_neighbour.north
                report.error(
                    f"tiles {tile} and {south_neighbour.tile} do not meet: "
                    f"{'gap' if gap > 0 else 'overlap'} of {abs(gap):.6f} m at "
                    f"y={seam:.3f}"
                )

        expected = grid.tile_bounds(tile)
        if not math.isclose(fact.west, expected.west, rel_tol=0, abs_tol=TRANSFORM_TOLERANCE_M) or not math.isclose(
            fact.north, expected.north, rel_tol=0, abs_tol=TRANSFORM_TOLERANCE_M
        ):
            report.error(
                f"tile {tile} is not aligned to the global grid: origin "
                f"({fact.west}, {fact.north}) vs expected "
                f"({expected.west}, {expected.north})"
            )

    return report


# --- 17.4 vertical transformation ---------------------------------------


def validate_vertical_transformation(
    converter: VerticalConverter, report: ValidationReport | None = None
) -> ValidationReport:
    """Run reference points through the real transformer before any raster work."""
    report = report or ValidationReport()
    operation = converter.operation

    missing = [grid.name for grid in operation.grids if not grid.available]
    if missing:
        report.error(
            f"selected vertical operation needs unavailable PROJ grid(s): "
            f"{', '.join(missing)}"
        )
        return report

    try:
        results = check_known_points(converter)
    except Exception as exc:
        report.error(f"vertical transformation smoke test failed: {exc}")
        return report

    for result in results:
        if not math.isfinite(result.z_output):
            report.error(
                f"vertical transformation returned a non-finite height at "
                f"{result.name}"
            )
        elif abs(result.delta) > 200.0:
            report.warn(
                f"vertical transformation moves {result.name} by {result.delta:+.3f} m, "
                "which is larger than expected for a Czech datum change"
            )
    log.debug(
        "vertical smoke test: %s",
        ", ".join(f"{r.name}{r.delta:+.3f}m" for r in results),
    )
    return report


# --- test points ---------------------------------------------------------


def validate_test_points(
    output_dir: Path, manifest: Manifest, report: ValidationReport | None = None
) -> ValidationReport:
    """Check that every test point re-samples from the tiles to its recorded height."""
    report = report or ValidationReport()
    relative = manifest.test_points_path
    if relative is None:
        report.warn("manifest declares no test points")
        return report
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        report.error(
            f"manifest test_points.path {relative!r} must be a relative path that "
            "stays inside the dataset directory"
        )
        return report

    path = Path(output_dir) / relative
    if not path.is_file():
        report.error(f"test points file {path} is missing")
        return report
    try:
        points = read_test_points(path)
    except ValidationError as exc:
        report.error(str(exc))
        return report
    if len(points) != manifest.test_points_count:
        report.error(
            f"{path.name} holds {len(points)} point(s), manifest declares "
            f"{manifest.test_points_count}"
        )

    with DatasetSampler(output_dir, manifest) as sampler:
        for point in points:
            height = sampler.sample(point.lon, point.lat)
            if not math.isfinite(point.height) or not math.isfinite(height):
                report.error(
                    f"test point {point.lat}, {point.lon}: recorded height "
                    f"{point.height!r}, the tiles give {height!r}"
                )
            elif abs(height - point.height) > TEST_POINT_TOLERANCE_M:
                report.error(
                    f"test point {point.lat}, {point.lon}: recorded height "
                    f"{point.height!r} m, the tiles give {height!r} m"
                )
    return report


# --- RGB tiles -------------------------------------------------------------

#: How far, in metres, an RGB tile may fall short of its height tile's sampling
#: region before that is an error.  Generous enough for the difference between
#: PROJ's grid-based and Helmert S-JTSK -> WGS84 operations, which is below 1 m.
RGB_EXTENT_TOLERANCE_M = 2.0

#: Fewer valid pixels than this, as a fraction of a tile's filled window, is
#: worth a warning: the source should cover every safety area completely.
RGB_MIN_COVERAGE = 0.99


def validate_rgb_tile(
    path: Path,
    manifest: Manifest,
    tile: TileIndex,
    *,
    report: ValidationReport | None = None,
) -> ValidationReport:
    """Validate one RGB tile: lattice, extent, layout, mask and imagery."""
    from .raster import TAG_RGB_FILLED
    from .rgb import RgbWindow, projected_to_lonlat_bounds

    report = report or ValidationReport()
    if manifest.rgb is None:
        report.error(f"rgb tile {tile}: the manifest declares no RGB tiles")
        return report
    rgb_grid = manifest.rgb.grid()
    name = path.name

    try:
        dataset = rasterio.open(path)
    except Exception as exc:
        report.error(f"rgb tile {tile}: cannot open {path}: {exc}")
        return report

    with dataset:
        report.tiles_checked += 1
        if not crs_matches(dataset.crs, 4326):
            report.error(
                f"rgb tile {tile} ({name}): CRS is EPSG:{epsg_code_of(dataset.crs)}, "
                "expected EPSG:4326"
            )
        if dataset.count != 3 or any(dtype != "uint8" for dtype in dataset.dtypes):
            report.error(
                f"rgb tile {tile} ({name}): has {dataset.count} {dataset.dtypes[0]} "
                "band(s), expected 3 uint8"
            )
        if any(rasterio.enums.MaskFlags.per_dataset not in flags for flags in dataset.mask_flag_enums):
            report.error(f"rgb tile {tile} ({name}): has no internal per-dataset mask")

        window = rgb_grid.window_of_transform(dataset.transform, dataset.width, dataset.height)
        if window is None:
            report.error(
                f"rgb tile {tile} ({name}): geotransform {tuple(dataset.transform)[:6]} "
                f"is not on the {manifest.rgb.resolution_deg} deg RGB lattice"
            )
            return report

        # The tile must cover every query its height tile answers.
        west, south, east, north = rgb_grid.window_bounds(window)
        need = projected_to_lonlat_bounds(manifest.grid().sampling_region(tile))
        lat = (need[1] + need[3]) / 2
        slack_lat = RGB_EXTENT_TOLERANCE_M / 111_320.0
        slack_lon = slack_lat / max(math.cos(math.radians(lat)), 1e-6)
        if (
            west > need[0] + slack_lon
            or east < need[2] - slack_lon
            or south > need[1] + slack_lat
            or north < need[3] - slack_lat
        ):
            report.error(
                f"rgb tile {tile} ({name}): covers {(west, south, east, north)}, which "
                f"does not contain its height tile's extent {need}"
            )

        filled = RgbWindow.from_tag(dataset.tags().get(TAG_RGB_FILLED))
        if filled is None or not window.contains(filled):
            report.error(
                f"rgb tile {tile} ({name}): {TAG_RGB_FILLED} is "
                f"{dataset.tags().get(TAG_RGB_FILLED)!r}, not a window inside the tile"
            )
            return report
        col_off, row_off = filled.offset_in(window)
        mask = dataset.read_masks(
            1, window=rasterio.windows.Window(col_off, row_off, filled.width, filled.height)
        )
        coverage = float(np.count_nonzero(mask)) / mask.size
        if coverage == 0.0:
            report.error(f"rgb tile {tile} ({name}): holds no imagery")
        elif coverage < RGB_MIN_COVERAGE:
            report.warn(
                f"rgb tile {tile} ({name}): only {coverage:.1%} of the area around the "
                "safety areas has imagery"
            )
    return report


# --- worlds database ---------------------------------------------------------


def validate_worlds_db(
    output_dir: Path, manifest: Manifest, report: ValidationReport | None = None
) -> ValidationReport:
    """Check that ``worlds.sqlite`` describes exactly the dataset beside it."""
    import sqlite3

    from .worlds_db import SCHEMA_VERSION

    report = report or ValidationReport()
    if manifest.worlds_db_path is None:
        return report
    path = Path(output_dir) / manifest.worlds_db_path
    if not path.is_file():
        report.error(f"worlds database {manifest.worlds_db_path} is declared but missing")
        return report
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        report.error(f"worlds database {path}: cannot open: {exc}")
        return report
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION or manifest.worlds_db_schema_version != SCHEMA_VERSION:
            report.error(
                f"worlds database {path.name}: schema version {version} (manifest says "
                f"{manifest.worlds_db_schema_version}), this build understands {SCHEMA_VERSION}"
            )
            return report
        names = {row[0] for row in connection.execute("SELECT name FROM worlds")}
        if names != set(manifest.worlds):
            report.error(
                f"worlds database {path.name}: holds worlds {sorted(names)}, the manifest "
                f"declares {sorted(manifest.worlds)}"
            )
        for name, count in connection.execute(
            "SELECT w.name, COUNT(v.seq) FROM worlds w "
            "LEFT JOIN safety_area_vertices v ON v.world = w.name GROUP BY w.name"
        ):
            if count < 3:
                report.error(
                    f"worlds database {path.name}: world {name} has a {count}-vertex safety area"
                )
        declared = set(manifest.tiles)
        for world, ix, iy in connection.execute("SELECT world, ix, iy FROM world_tiles"):
            if TileIndex(ix, iy) not in declared:
                report.error(
                    f"worlds database {path.name}: world {world} needs tile ({ix}, {iy}), "
                    "which the manifest does not declare"
                )
        tiles = {TileIndex(ix, iy) for ix, iy in connection.execute("SELECT ix, iy FROM tiles")}
        if tiles != declared:
            report.error(
                f"worlds database {path.name}: lists {len(tiles)} tile(s), the manifest "
                f"declares {len(declared)}"
            )
    except sqlite3.Error as exc:
        report.error(f"worlds database {path}: {exc}")
    finally:
        connection.close()
    return report


# --- whole dataset -------------------------------------------------------


def validate_dataset(
    output_dir: Path,
    *,
    plausibility: bool = True,
    require_complete: bool = False,
    expected_tiles: Iterable[TileIndex] | None = None,
) -> ValidationReport:
    """Validate a prepared dataset directory end to end.

    This is what ``--validate-only`` runs: it opens nothing but the manifest and
    the tiles it declares, and needs neither the network nor PROJ grids.
    """
    output_dir = Path(output_dir)
    report = ValidationReport()

    try:
        manifest = Manifest.read(output_dir)
    except ValidationError as exc:
        report.error(str(exc))
        return report

    validate_manifest(manifest, report)
    if report.errors:
        return report

    if require_complete and manifest.status != STATUS_COMPLETE:
        report.error(
            f"dataset status is {manifest.status!r}, expected {STATUS_COMPLETE!r}"
        )

    tiles = list(expected_tiles) if expected_tiles is not None else list(manifest.tiles)
    if not tiles:
        report.warn("manifest declares no tiles")

    grid = manifest.grid()
    facts: list[TileFacts] = []
    for tile in tiles:
        path = manifest.tile_path(output_dir, tile)
        if not path.is_file():
            report.tiles_missing.append(tile)
            report.error(f"tile {tile}: declared in the manifest but {path} is missing")
            continue
        _, tile_facts = validate_tile(
            path, manifest, tile, grid=grid, plausibility=plausibility, report=report
        )
        if tile_facts is not None:
            facts.append(tile_facts)

    validate_spatial_consistency(facts, manifest, report)
    validate_test_points(output_dir, manifest, report)

    if manifest.rgb is not None:
        for tile in tiles:
            path = manifest.rgb_tile_path(output_dir, tile)
            if not path.is_file():
                report.tiles_missing.append(tile)
                report.error(f"rgb tile {tile}: declared in the manifest but {path} is missing")
                continue
            validate_rgb_tile(path, manifest, tile, report=report)

    validate_worlds_db(output_dir, manifest, report)

    # Stray files usually mean an interrupted run or a stale manifest.
    _check_stray_files(output_dir, manifest.tile_pattern, tiles, report)
    if manifest.rgb is not None:
        _check_stray_files(output_dir, manifest.rgb.tile_pattern, tiles, report)

    return report


def _check_stray_files(
    output_dir: Path, pattern: str, tiles: Sequence[TileIndex], report: ValidationReport
) -> None:
    declared = {pattern.format(ix=tile.ix, iy=tile.iy) for tile in tiles}
    directory = output_dir / Path(pattern.format(ix=0, iy=0)).parent
    if not directory.is_dir():
        return
    for candidate in sorted(directory.glob("*.tif")):
        relative = candidate.relative_to(output_dir).as_posix()
        if relative not in declared and parse_tile_filename(candidate.name):
            report.unexpected_files.append(relative)
            report.warn(f"{relative} is present but not declared in the manifest")
    for leftover in sorted(directory.glob("*.tmp")):
        report.warn(
            f"{leftover.relative_to(output_dir).as_posix()} is a leftover "
            "temporary file from an interrupted run"
        )
