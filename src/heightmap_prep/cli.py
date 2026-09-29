"""Command line interface (specification section 14).

::

    heightmap-prep world_a.yaml world_b.yaml OUTPUT_DIR [OPTIONS]
    heightmap-prep WORLDS_DIR OUTPUT_DIR [OPTIONS]

World configs are MRS UAV system world files; the dataset covers their safety
areas.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import PrepareOptions, VERTICAL_DATUMS
from .errors import HeightmapPrepError
from .pipeline import prepare_worlds, validate_only

log = logging.getLogger("heightmap_prep")

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

EPILOG = """\
examples:
  heightmap-prep worlds/world_bechovice.yaml worlds/world_ricany.yaml ./prepared \
      --proj-data-dir ./.proj
  heightmap-prep ./worlds ./prepared --vertical-datum wgs84-ellipsoid
  heightmap-prep ./prepared --validate-only

The last positional argument is always the output directory.  Preceding
arguments are either one worlds directory or a list of MRS world config files.
The prepared tiles cover every point of each world's safety area, which must be
given in latlon_origin.  A world config that cannot be prepared -- unreadable,
no latlon_origin safety area, outside the source coverage, no source data at a
safety area corner -- is skipped with a warning.  test_points.csv in the output holds the height at every safety area
corner, as lat,lon,height rows, for checking samplers against.

exit codes:
  0  success
  1  the dataset was produced or read but failed validation
  2  the run could not proceed (bad configuration, missing PROJ grids,
     source failure, output conflict)
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heightmap-prep",
        description=(
            "Prepare ČÚZK DMR 5G heightmaps for a WGS84 lon/lat runtime sampler: "
            "acquire in EPSG:5514, convert Bpv heights onto the requested vertical "
            "datum, and write deterministic tiled float32 GeoTIFFs covering the "
            "safety areas of MRS world configs, with a manifest."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="INPUT... OUTPUT_DIR",
        help="one worlds directory or several MRS world config files, then the "
        "output directory",
    )
    parser.add_argument(
        "--vertical-datum",
        choices=VERTICAL_DATUMS,
        default="egm96",
        help="output height reference (default: egm96)",
    )
    parser.add_argument(
        "--include-rgb",
        action="store_true",
        help="include RGB imagery (not implemented in version 1)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=2.0,
        metavar="FLOAT",
        help="target heightmap resolution in metres (default: 2.0)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=4096,
        metavar="INTEGER",
        help="logical output tile width/height in pixels (default: 4096)",
    )
    parser.add_argument(
        "--proj-data-dir",
        type=Path,
        metavar="PATH",
        help="directory containing local PROJ grid files "
        "(cz_cuzk_CR-2005.tif, us_nga_egm96_15.tif)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        metavar="PATH",
        help="optional cache for downloaded source rasters",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="INTEGER",
        help="number of preprocessing workers (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing generated tiles instead of reusing them",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an existing output dataset without downloading or converting",
    )
    parser.add_argument(
        "--log-level",
        choices=sorted(LOG_LEVELS),
        default="info",
        help="logging verbosity (default: info)",
    )

    advanced = parser.add_argument_group("advanced")
    advanced.add_argument(
        "--block-size",
        type=int,
        default=256,
        metavar="INTEGER",
        help="internal GeoTIFF block size in pixels (default: 256)",
    )
    advanced.add_argument(
        "--nodata",
        type=float,
        default=-9999.0,
        metavar="FLOAT",
        help="NoData value for the prepared dataset (default: -9999.0)",
    )
    advanced.add_argument(
        "--grid-origin",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="west/north edge of tile (0, 0) in EPSG:5514 (default: derived from "
        "-1000000 -800000, shifted by under one pixel to stay in phase with the "
        "source product's native grid)",
    )
    advanced.add_argument(
        "--max-request-px",
        type=int,
        default=2048,
        metavar="INTEGER",
        help="largest source request edge in pixels (default: 2048)",
    )
    advanced.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help="per-request HTTP timeout (default: 120)",
    )
    advanced.add_argument(
        "--retries",
        type=int,
        default=3,
        metavar="INTEGER",
        help="attempts per source request (default: 3)",
    )
    advanced.add_argument(
        "--cog",
        action="store_true",
        help="write Cloud Optimized GeoTIFFs (with overviews) instead of plain "
        "tiled GeoTIFFs",
    )
    advanced.add_argument(
        "--no-plausibility-checks",
        action="store_true",
        help="skip the optional plausibility warnings",
    )
    parser.add_argument("--version", action="version", version=f"heightmap-prep {__version__}")
    return parser


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=LOG_LEVELS[level_name],
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Third-party loggers are extremely chatty and would drown out our own
    # output; request URLs belong at debug level, which this module logs itself.
    level = LOG_LEVELS[level_name]
    for noisy in ("rasterio", "rasterio._env", "rasterio._io", "httpx", "httpcore", "fiona"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def options_from_args(args: argparse.Namespace) -> PrepareOptions:
    return PrepareOptions(
        vertical_datum=args.vertical_datum,
        include_rgb=args.include_rgb,
        resolution_m=args.resolution,
        tile_size_px=args.tile_size,
        proj_data_dir=args.proj_data_dir,
        cache_dir=args.cache_dir,
        workers=args.workers,
        overwrite=args.overwrite,
        block_size_px=args.block_size,
        nodata=args.nodata,
        grid_origin_x=args.grid_origin[0] if args.grid_origin else None,
        grid_origin_y=args.grid_origin[1] if args.grid_origin else None,
        request_timeout_s=args.timeout,
        request_retries=args.retries,
        max_request_px=args.max_request_px,
        cog=args.cog,
        plausibility_checks=not args.no_plausibility_checks,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    paths = [Path(p) for p in args.paths]
    output_dir = paths[-1]
    inputs = paths[:-1]

    if args.validate_only:
        # With --validate-only the inputs are optional: only the dataset matters.
        target = output_dir
        try:
            report = validate_only(target)
        except HeightmapPrepError as exc:
            log.error("%s", exc)
            return 2
        if not report.ok:
            for message in report.errors:
                log.error("%s", message)
            return 1
        return 0

    if not inputs:
        parser.error(
            "expected at least one world input before OUTPUT_DIR; "
            f"got only {output_dir}"
        )

    if args.include_rgb:
        log.warning(
            "--include-rgb was requested but RGB acquisition is not implemented "
            "in version 1; only heightmaps will be prepared"
        )

    try:
        options = options_from_args(args)
        result = prepare_worlds(inputs, output_dir, options)
    except HeightmapPrepError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        log.error("interrupted; partial output left in %s for a later resume", output_dir)
        return 130

    log.info(
        "prepared %d tile(s) for %d world(s) in %s",
        len(result.manifest.tiles),
        len(result.worlds),
        result.output_dir.resolve(),
    )
    if result.skipped:
        log.warning(
            "skipped %d world config(s), see the warnings above: %s",
            len(result.skipped),
            ", ".join(path.name for path in result.skipped),
        )
    return 0 if result.report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
