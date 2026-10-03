"""The ``dataset.yaml`` manifest (specification sections 11, 18 and 19).

The manifest is the contract between this library and the runtime sampler: it
carries enough grid metadata to locate any tile from projected coordinates
without opening a single GeoTIFF, plus the provenance needed to reproduce the
dataset.
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from .errors import ValidationError
from .rgb import LATTICE_ORIGIN_LAT, LATTICE_ORIGIN_LON, RGB_CRS, RGB_TILE_PATTERN, RgbGrid
from .tiling import TileGrid, TileIndex

#: Only this manifest layout is understood.
FORMAT_VERSION = 1

MANIFEST_FILENAME = "dataset.yaml"

STATUS_BUILDING = "building"
STATUS_COMPLETE = "complete"

DEFAULT_TILE_PATTERN = "height/tile_{ix}_{iy}.tif"

#: Column order of the test points file, which has no header row.
TEST_POINT_COLUMNS = ("lat", "lon", "height")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def software_versions(package_version: str) -> dict[str, str]:
    """Versions of everything that can influence the generated pixels."""
    import pyproj
    import rasterio

    return {
        "package_version": package_version,
        "python_version": platform.python_version(),
        "rasterio_version": rasterio.__version__,
        "gdal_version": str(rasterio.__gdal_version__),
        "pyproj_version": pyproj.__version__,
        "proj_version": str(pyproj.proj_version_str),
    }


def grid_file_checksums(paths: Iterable[Path]) -> dict[str, str]:
    """MD5 of each local PROJ grid file that could be located."""
    checksums: dict[str, str] = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        checksums[path.name] = digest.hexdigest()
    return checksums


#: How RGB tile ``(ix, iy)`` relates to height tile ``(ix, iy)``.
RGB_TILE_EXTENT = (
    "rgb tile (ix, iy) covers the EPSG:4326 bounding box of height tile (ix, iy) "
    "grown by half a height pixel, snapped outward to the lattice; find it as for "
    "the height tile, then locate pixels with its own geotransform. Only pixels "
    "around the safety areas hold imagery; the internal mask marks them."
)


@dataclass
class RgbInfo:
    """The ``rgb`` section: orthophoto tiles in the query CRS."""

    resolution_m: float
    resolution_deg: float
    source_id: str
    source_service_url: str | None = None
    source_provider: str = "CUZK"
    source_product: str = "ORTOFOTO"
    jpeg_quality: int = 90
    block_width_px: int = 512
    block_height_px: int = 512
    crs: str = RGB_CRS
    tile_pattern: str = RGB_TILE_PATTERN
    origin_lon: float = LATTICE_ORIGIN_LON
    origin_lat: float = LATTICE_ORIGIN_LAT
    processing_version: str = "1"

    def grid(self) -> RgbGrid:
        return RgbGrid(self.resolution_deg, self.origin_lon, self.origin_lat)

    def to_dict(self) -> dict[str, Any]:
        source: dict[str, Any] = {
            "provider": self.source_provider,
            "product": self.source_product,
            "source_id": self.source_id,
            "reprojection": "server",
        }
        if self.source_service_url:
            source["service_url"] = self.source_service_url
        return {
            "crs": self.crs,
            "bands": ["red", "green", "blue"],
            "dtype": "uint8",
            "mask": "internal",
            "resolution_m": float(self.resolution_m),
            "resolution_deg": float(self.resolution_deg),
            "lattice_origin_lon": float(self.origin_lon),
            "lattice_origin_lat": float(self.origin_lat),
            "tile_pattern": self.tile_pattern,
            "tile_extent": RGB_TILE_EXTENT,
            "storage": {
                "format": "geotiff",
                "compression": "jpeg",
                "photometric": "ycbcr",
                "jpeg_quality": int(self.jpeg_quality),
                "block_width_px": int(self.block_width_px),
                "block_height_px": int(self.block_height_px),
            },
            "source": source,
            "processing_version": self.processing_version,
        }

    @classmethod
    def from_dict(cls, section: Any, *, origin: str) -> "RgbInfo":
        if not isinstance(section, dict):
            raise ValidationError(f"{origin}: section 'rgb' must be a mapping")
        storage = section.get("storage") or {}
        source = section.get("source") or {}
        for key in ("resolution_m", "resolution_deg", "tile_pattern"):
            if section.get(key) is None:
                raise ValidationError(f"{origin}: missing required key rgb.{key}")
        return cls(
            resolution_m=float(section["resolution_m"]),
            resolution_deg=float(section["resolution_deg"]),
            source_id=str(source.get("source_id", "")),
            source_service_url=source.get("service_url"),
            source_provider=str(source.get("provider", "")),
            source_product=str(source.get("product", "")),
            jpeg_quality=int(storage.get("jpeg_quality", 90)),
            block_width_px=int(storage.get("block_width_px", 512)),
            block_height_px=int(storage.get("block_height_px", 512)),
            crs=str(section.get("crs", RGB_CRS)),
            tile_pattern=str(section["tile_pattern"]),
            origin_lon=float(section.get("lattice_origin_lon", LATTICE_ORIGIN_LON)),
            origin_lat=float(section.get("lattice_origin_lat", LATTICE_ORIGIN_LAT)),
            processing_version=str(section.get("processing_version", "1")),
        )


@dataclass
class Manifest:
    """In-memory view of ``dataset.yaml``."""

    # heightmap
    horizontal_crs: str
    vertical_crs: str
    vertical_datum: str
    nodata: float
    dtype: str = "float32"
    unit: str = "m"

    # grid
    resolution_x: float = 2.0
    resolution_y: float = 2.0
    tile_width_px: int = 4096
    tile_height_px: int = 4096
    origin_x: float = -1_000_000.0
    origin_y: float = -800_000.0
    axis_order: str = "east_north"

    # storage
    storage_format: str = "geotiff"
    compression: str = "deflate"
    predictor: int = 3
    block_width_px: int = 256
    block_height_px: int = 256
    tile_pattern: str = DEFAULT_TILE_PATTERN

    # source
    source_provider: str = "CUZK"
    source_product: str = "DMR5G"
    source_horizontal_crs: str = "EPSG:5514"
    source_vertical_crs: str = "EPSG:8357"
    source_vertical_datum: str = "bpv"
    source_resolution_m: float = 2.0
    source_id: str = "cuzk-dmr5g"
    source_service_url: str | None = None

    # transform
    vertical_method: str = "proj"
    network_enabled: bool = False
    proj_operation: str = ""
    proj_operation_accuracy_m: float | None = None
    proj_grids: list[str] = field(default_factory=list)
    proj_grid_checksums: dict[str, str] = field(default_factory=dict)

    # bookkeeping
    status: str = STATUS_BUILDING
    format_version: int = FORMAT_VERSION
    worlds: list[str] = field(default_factory=list)
    tiles: list[TileIndex] = field(default_factory=list)
    #: ``lat,lon,height`` reference samples at every safety area corner,
    #: relative to the dataset directory.
    test_points_path: str | None = None
    test_points_count: int = 0
    #: Orthophoto tiles, one per height tile; ``None`` when not prepared.
    rgb: RgbInfo | None = None
    #: SQLite database of the worlds, relative to the dataset directory.
    worlds_db_path: str | None = None
    worlds_db_schema_version: int = 0

    # reproducibility
    software: dict[str, str] = field(default_factory=dict)
    created_utc: str = field(default_factory=_utc_now)
    updated_utc: str = field(default_factory=_utc_now)
    interpolation: str = "nearest"
    vertical_conversion: str = "proj"
    processing_version: str = "1"

    # --- derived ---------------------------------------------------------

    @property
    def tile_span_x(self) -> float:
        return self.tile_width_px * self.resolution_x

    @property
    def tile_span_y(self) -> float:
        return self.tile_height_px * self.resolution_y

    def grid(self) -> TileGrid:
        """The tile grid this manifest describes."""
        return TileGrid(
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            tile_width_px=self.tile_width_px,
            tile_height_px=self.tile_height_px,
            resolution_x=self.resolution_x,
            resolution_y=self.resolution_y,
        )

    def tile_relative_path(self, tile: TileIndex) -> str:
        return self.tile_pattern.format(ix=tile.ix, iy=tile.iy)

    def tile_path(self, root: Path, tile: TileIndex) -> Path:
        return Path(root) / self.tile_relative_path(tile)

    def rgb_tile_path(self, root: Path, tile: TileIndex) -> Path:
        if self.rgb is None:
            raise ValidationError("the dataset has no RGB tiles")
        return Path(root) / self.rgb.tile_pattern.format(ix=tile.ix, iy=tile.iy)

    # --- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "format_version": self.format_version,
            "status": self.status,
            "heightmap": {
                "horizontal_crs": self.horizontal_crs,
                "vertical_crs": self.vertical_crs,
                "vertical_datum": self.vertical_datum,
                "unit": self.unit,
                "dtype": self.dtype,
                "nodata": float(self.nodata),
            },
            "grid": {
                "resolution_x": float(self.resolution_x),
                "resolution_y": float(self.resolution_y),
                "tile_width_px": int(self.tile_width_px),
                "tile_height_px": int(self.tile_height_px),
                "origin_x": float(self.origin_x),
                "origin_y": float(self.origin_y),
                "axis_order": self.axis_order,
                # Redundant but convenient: the sampler's tile-index arithmetic.
                "tile_span_x": float(self.tile_span_x),
                "tile_span_y": float(self.tile_span_y),
                "index_convention": (
                    "origin is the west/north edge of tile (0, 0); "
                    "ix increases eastward, iy increases southward; "
                    "ix = floor((x - origin_x) / tile_span_x), "
                    "iy = floor((origin_y - y) / tile_span_y)"
                ),
            },
            "storage": {
                "format": self.storage_format,
                "compression": self.compression,
                "predictor": int(self.predictor),
                "block_width_px": int(self.block_width_px),
                "block_height_px": int(self.block_height_px),
                "tile_pattern": self.tile_pattern,
            },
            "source": {
                "provider": self.source_provider,
                "product": self.source_product,
                "source_id": self.source_id,
                "source_horizontal_crs": self.source_horizontal_crs,
                "source_vertical_crs": self.source_vertical_crs,
                "source_vertical_datum": self.source_vertical_datum,
                "source_resolution_m": float(self.source_resolution_m),
            },
            "transform": {
                "vertical": {
                    "method": self.vertical_method,
                    "target": self.vertical_datum,
                    "network_enabled": bool(self.network_enabled),
                    "proj_operation": self.proj_operation,
                    "proj_operation_accuracy_m": self.proj_operation_accuracy_m,
                    "proj_grids": list(self.proj_grids),
                    "proj_grid_checksums": dict(self.proj_grid_checksums),
                }
            },
            "sampling_contract": {
                "query_crs": "EPSG:4326",
                "pixel_location": "center",
                "interpolation": "bilinear",
            },
            "worlds": list(self.worlds),
            "tiles": {
                "count": len(self.tiles),
                "index": [[tile.ix, tile.iy] for tile in self.tiles],
            },
            "software": dict(self.software),
            "processing": {
                "created_utc": self.created_utc,
                "updated_utc": self.updated_utc,
                "interpolation": self.interpolation,
                "vertical_conversion": self.vertical_conversion,
                "processing_version": self.processing_version,
            },
        }
        if self.source_service_url:
            document["source"]["service_url"] = self.source_service_url
        if self.test_points_path is not None:
            document["test_points"] = {
                "path": self.test_points_path,
                "count": int(self.test_points_count),
                "columns": list(TEST_POINT_COLUMNS),
            }
        if self.rgb is not None:
            document["rgb"] = self.rgb.to_dict()
        if self.worlds_db_path is not None:
            document["worlds_db"] = {
                "path": self.worlds_db_path,
                "schema_version": int(self.worlds_db_schema_version),
            }
        return document

    @classmethod
    def from_dict(cls, document: Any, *, origin: str = "manifest") -> "Manifest":
        if not isinstance(document, dict):
            raise ValidationError(f"{origin}: manifest must be a mapping")

        def section(key: str) -> dict[str, Any]:
            value = document.get(key)
            if value is None:
                raise ValidationError(f"{origin}: missing required section {key!r}")
            if not isinstance(value, dict):
                raise ValidationError(f"{origin}: section {key!r} must be a mapping")
            return value

        heightmap = section("heightmap")
        grid = section("grid")
        storage = section("storage")
        source = document.get("source") or {}
        transform = document.get("transform") or {}
        vertical = (transform.get("vertical") or {}) if isinstance(transform, dict) else {}
        processing = document.get("processing") or {}
        tiles_section = document.get("tiles") or {}
        test_points = document.get("test_points") or {}
        if not isinstance(test_points, dict):
            raise ValidationError(f"{origin}: section 'test_points' must be a mapping")
        worlds_db = document.get("worlds_db") or {}
        if not isinstance(worlds_db, dict):
            raise ValidationError(f"{origin}: section 'worlds_db' must be a mapping")
        rgb = (
            RgbInfo.from_dict(document["rgb"], origin=origin)
            if document.get("rgb") is not None
            else None
        )

        def need(mapping: dict[str, Any], key: str, where: str) -> Any:
            if key not in mapping or mapping[key] is None:
                raise ValidationError(f"{origin}: missing required key {where}.{key}")
            return mapping[key]

        tiles: list[TileIndex] = []
        for entry in tiles_section.get("index", []) or []:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValidationError(f"{origin}: malformed tile index entry {entry!r}")
            tiles.append(TileIndex(int(entry[0]), int(entry[1])))

        return cls(
            format_version=int(document.get("format_version", FORMAT_VERSION)),
            status=str(document.get("status", STATUS_BUILDING)),
            horizontal_crs=str(need(heightmap, "horizontal_crs", "heightmap")),
            vertical_crs=str(need(heightmap, "vertical_crs", "heightmap")),
            vertical_datum=str(heightmap.get("vertical_datum", "")),
            unit=str(need(heightmap, "unit", "heightmap")),
            dtype=str(heightmap.get("dtype", "float32")),
            nodata=float(need(heightmap, "nodata", "heightmap")),
            resolution_x=float(need(grid, "resolution_x", "grid")),
            resolution_y=float(need(grid, "resolution_y", "grid")),
            tile_width_px=int(need(grid, "tile_width_px", "grid")),
            tile_height_px=int(need(grid, "tile_height_px", "grid")),
            origin_x=float(need(grid, "origin_x", "grid")),
            origin_y=float(need(grid, "origin_y", "grid")),
            axis_order=str(grid.get("axis_order", "east_north")),
            storage_format=str(storage.get("format", "geotiff")),
            compression=str(storage.get("compression", "deflate")),
            predictor=int(storage.get("predictor", 3)),
            block_width_px=int(storage.get("block_width_px", 256)),
            block_height_px=int(storage.get("block_height_px", 256)),
            tile_pattern=str(need(storage, "tile_pattern", "storage")),
            source_provider=str(source.get("provider", "")),
            source_product=str(source.get("product", "")),
            source_id=str(source.get("source_id", "")),
            source_horizontal_crs=str(source.get("source_horizontal_crs", "")),
            source_vertical_crs=str(source.get("source_vertical_crs", "")),
            source_vertical_datum=str(source.get("source_vertical_datum", "")),
            source_resolution_m=float(source.get("source_resolution_m", 0.0) or 0.0),
            source_service_url=source.get("service_url"),
            vertical_method=str(vertical.get("method", "proj")),
            network_enabled=bool(vertical.get("network_enabled", False)),
            proj_operation=str(vertical.get("proj_operation", "")),
            proj_operation_accuracy_m=(
                float(vertical["proj_operation_accuracy_m"])
                if vertical.get("proj_operation_accuracy_m") is not None
                else None
            ),
            proj_grids=list(vertical.get("proj_grids", []) or []),
            proj_grid_checksums=dict(vertical.get("proj_grid_checksums", {}) or {}),
            worlds=[str(w) for w in (document.get("worlds") or [])],
            tiles=tiles,
            test_points_path=(
                str(test_points["path"]) if test_points.get("path") is not None else None
            ),
            test_points_count=int(test_points.get("count", 0) or 0),
            rgb=rgb,
            worlds_db_path=(
                str(worlds_db["path"]) if worlds_db.get("path") is not None else None
            ),
            worlds_db_schema_version=int(worlds_db.get("schema_version", 0) or 0),
            software=dict(document.get("software") or {}),
            created_utc=str(processing.get("created_utc", _utc_now())),
            updated_utc=str(processing.get("updated_utc", _utc_now())),
            interpolation=str(processing.get("interpolation", "nearest")),
            vertical_conversion=str(processing.get("vertical_conversion", "proj")),
            processing_version=str(processing.get("processing_version", "1")),
        )

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.to_dict(), sort_keys=False, default_flow_style=False, allow_unicode=True
        )

    def write(self, output_dir: Path, filename: str = MANIFEST_FILENAME) -> Path:
        """Write the manifest atomically so it is never observed half-written."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        tmp = path.with_name(path.name + ".tmp")
        self.updated_utc = _utc_now()
        tmp.write_text(self.to_yaml(), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def read(cls, output_dir: Path, filename: str = MANIFEST_FILENAME) -> "Manifest":
        path = Path(output_dir)
        if path.is_dir():
            path = path / filename
        if not path.is_file():
            raise ValidationError(f"manifest not found: {path}")
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValidationError(f"{path}: invalid YAML: {exc}") from exc
        return cls.from_dict(document, origin=str(path))

    # --- mutation --------------------------------------------------------

    def set_tiles(self, tiles: Sequence[TileIndex]) -> None:
        self.tiles = sorted(set(tiles))

    def mark_complete(self) -> None:
        self.status = STATUS_COMPLETE
