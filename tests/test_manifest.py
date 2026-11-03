"""The dataset manifest (specification sections 11, 18 and 19)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from heightmap_prep.errors import ValidationError
from heightmap_prep.manifest import (
    FORMAT_VERSION,
    MANIFEST_FILENAME,
    STATUS_BUILDING,
    STATUS_COMPLETE,
    Manifest,
    WgsInfo,
    WgsRgbInfo,
    grid_file_checksums,
    software_versions,
)
from heightmap_prep.tiling import TileIndex


def sample() -> Manifest:
    manifest = Manifest(
        horizontal_crs="EPSG:5514",
        vertical_crs="EPSG:5773",
        vertical_datum="egm96",
        nodata=-9999.0,
        worlds=["prague"],
        proj_operation="Inverse of ETRS89 to Baltic 1957 height ... + WGS 84 to EGM96 height ...",
        proj_grids=["cz_cuzk_CR-2005.tif", "us_nga_egm96_15.tif"],
        software=software_versions("1.0.0"),
    )
    manifest.set_tiles([TileIndex(31, 29), TileIndex(30, 29), TileIndex(31, 29)])
    return manifest


def test_the_document_has_every_section_the_specification_lists() -> None:
    document = sample().to_dict()
    assert document["format_version"] == FORMAT_VERSION
    assert set(document) >= {
        "format_version",
        "status",
        "heightmap",
        "grid",
        "storage",
        "source",
        "transform",
        "worlds",
        "software",
        "processing",
    }
    assert document["heightmap"] == {
        "horizontal_crs": "EPSG:5514",
        "vertical_crs": "EPSG:5773",
        "vertical_datum": "egm96",
        "unit": "m",
        "dtype": "float32",
        "nodata": -9999.0,
    }
    assert document["storage"]["tile_pattern"] == "height/tile_{ix}_{iy}.tif"
    assert document["source"]["source_vertical_crs"] == "EPSG:8357"
    assert document["transform"]["vertical"]["network_enabled"] is False


def test_a_sampler_can_locate_a_tile_from_the_manifest_alone() -> None:
    document = sample().to_dict()["grid"]
    # Everything the runtime formula in section 12 needs.
    for key in ("origin_x", "origin_y", "tile_span_x", "tile_span_y", "resolution_x"):
        assert key in document
    assert document["tile_span_x"] == document["tile_width_px"] * document["resolution_x"]
    assert document["axis_order"] == "east_north"


def test_tiles_are_deduplicated_and_sorted() -> None:
    assert sample().tiles == [TileIndex(30, 29), TileIndex(31, 29)]


def test_status_starts_at_building_and_is_marked_complete() -> None:
    manifest = sample()
    assert manifest.status == STATUS_BUILDING
    manifest.mark_complete()
    assert manifest.to_dict()["status"] == STATUS_COMPLETE


def test_round_trip_through_yaml(tmp_path: Path) -> None:
    original = sample()
    original.write(tmp_path)
    restored = Manifest.read(tmp_path)
    assert restored.to_dict() == original.to_dict()


def test_test_points_round_trip(tmp_path: Path) -> None:
    original = sample()
    assert "test_points" not in original.to_dict()
    original.test_points_path = "test_points.csv"
    original.test_points_count = 12
    assert original.to_dict()["test_points"] == {
        "path": "test_points.csv",
        "count": 12,
        "columns": ["lat", "lon", "height"],
    }
    original.write(tmp_path)
    restored = Manifest.read(tmp_path)
    assert (restored.test_points_path, restored.test_points_count) == ("test_points.csv", 12)


def test_the_wgs_section_round_trips(tmp_path: Path) -> None:
    original = sample()
    original.wgs = WgsInfo(
        resolution_m=0.25,
        resolution_deg=0.25 / 111_320.0,
        tiles=[TileIndex(10568, 2170), TileIndex(10567, 2170)],
        rgb=WgsRgbInfo(source_id="cuzk-ortofoto", jpeg_quality=80),
    )
    original.write(tmp_path)
    document = yaml.safe_load((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    section = document["wgs"]
    assert section["tile_size_px"] == 8192 and section["height"]["tile_width_px"] == 1024
    assert section["height"]["tile_pattern"] == "wgs/height/tile_{ix}_{iy}.tif"
    assert section["rgb"]["tile_pattern"] == "wgs/rgb/tile_{ix}_{iy}.tif"
    assert section["tiles"]["index"] == [[10567, 2170], [10568, 2170]]

    restored = Manifest.read(tmp_path)
    assert restored.wgs is not None and restored.wgs.rgb is not None
    assert restored.wgs.grid() == original.wgs.grid()
    assert restored.wgs.tiles == [TileIndex(10567, 2170), TileIndex(10568, 2170)]
    assert restored.wgs.rgb.jpeg_quality == 80
    assert restored.wgs_height_path(tmp_path, TileIndex(1, 2)) == (
        tmp_path / "wgs" / "height" / "tile_1_2.tif"
    )


def test_the_manifest_has_no_bounds() -> None:
    document = sample().to_dict()
    assert "world_bounds_wgs84" not in document
    assert "dataset_bounds" not in document["grid"]


def test_write_is_atomic_and_leaves_no_temporary(tmp_path: Path) -> None:
    path = sample().write(tmp_path)
    assert path.name == MANIFEST_FILENAME
    assert not list(tmp_path.glob("*.tmp"))


def test_yaml_is_human_readable_and_ordered(tmp_path: Path) -> None:
    text = (sample().write(tmp_path)).read_text(encoding="utf-8")
    assert text.startswith("format_version: 2\nstatus: building\n")
    assert "!!python" not in text
    yaml.safe_load(text)


def test_tile_paths_follow_the_pattern(tmp_path: Path) -> None:
    manifest = sample()
    assert manifest.tile_relative_path(TileIndex(31, 29)) == "height/tile_31_29.tif"
    assert manifest.tile_path(tmp_path, TileIndex(31, 29)) == tmp_path / "height" / "tile_31_29.tif"


def test_grid_is_reconstructed_from_the_manifest() -> None:
    grid = sample().grid()
    assert grid.tile_span_x == 4096 * 2.0
    assert grid.index_for_point(*(x + 1.0 for x in (grid.origin_x, grid.origin_y - 1.0))) == TileIndex(0, 0)


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="manifest not found"):
        Manifest.read(tmp_path)


def test_malformed_manifest_yaml_is_reported(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_FILENAME).write_text("heightmap: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid YAML"):
        Manifest.read(tmp_path)


def test_a_manifest_missing_a_section_is_reported(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_FILENAME).write_text("format_version: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="missing required section 'heightmap'"):
        Manifest.read(tmp_path)


def test_a_manifest_missing_a_key_is_reported(tmp_path: Path) -> None:
    document = sample().to_dict()
    del document["heightmap"]["nodata"]
    (tmp_path / MANIFEST_FILENAME).write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValidationError, match="heightmap.nodata"):
        Manifest.read(tmp_path)


def test_malformed_tile_entries_are_reported(tmp_path: Path) -> None:
    document = sample().to_dict()
    document["tiles"]["index"] = [[1, 2, 3]]
    (tmp_path / MANIFEST_FILENAME).write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValidationError, match="malformed tile index"):
        Manifest.read(tmp_path)


# --- reproducibility (section 19) ----------------------------------------


def test_software_versions_record_everything_that_affects_the_pixels() -> None:
    versions = software_versions("1.2.3")
    assert versions["package_version"] == "1.2.3"
    for key in ("python_version", "rasterio_version", "pyproj_version", "proj_version"):
        assert versions[key]


def test_grid_checksums_cover_the_files_that_exist(tmp_path: Path) -> None:
    (tmp_path / "grid.tif").write_bytes(b"abc")
    checksums = grid_file_checksums([tmp_path / "grid.tif", tmp_path / "absent.tif"])
    assert checksums == {"grid.tif": "900150983cd24fb0d6963f7d28e17f72"}
