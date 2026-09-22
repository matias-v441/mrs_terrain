"""Raster storage and the vertical conversion workflow (specification section 8)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from heightmap_prep.crs import build_converter, epsg_code_of
from heightmap_prep.errors import OutputConflictError
from heightmap_prep.raster import (
    ConversionStats,
    convert_vertical,
    pixel_center_coords,
    provenance_tags,
    read_tags,
    storage_profile,
    write_raster,
    write_raster_atomic,
)

TRANSFORM = Affine(2.0, 0.0, -743_000.0, 0.0, -2.0, -1_043_000.0)


# --- pixel centres (section 4.5) -----------------------------------------


def test_pixel_centres_match_the_specification_formula() -> None:
    x, y = pixel_center_coords(TRANSFORM, 0, 0, 3, 2)
    for row in range(2):
        for col in range(3):
            expected = TRANSFORM @ (col + 0.5, row + 0.5)
            assert (x[row, col], y[row, col]) == pytest.approx(expected)


def test_pixel_centres_respect_the_window_offset() -> None:
    x, y = pixel_center_coords(TRANSFORM, 10, 20, 2, 2)
    assert (x[0, 0], y[0, 0]) == pytest.approx(TRANSFORM @ (10.5, 20.5))


def test_pixel_centres_handle_a_rotated_transform() -> None:
    rotated = Affine(2.0, 0.5, -743_000.0, 0.25, -2.0, -1_043_000.0)
    x, y = pixel_center_coords(rotated, 0, 0, 2, 2)
    assert (x[1, 1], y[1, 1]) == pytest.approx(rotated @ (1.5, 1.5))


def test_north_up_centres_are_not_materialised_twice() -> None:
    # The north-up path broadcasts 1-D coordinate vectors rather than building
    # two full grids (specification section 21).
    x, y = pixel_center_coords(TRANSFORM, 0, 0, 1024, 1024)
    assert x.base is not None and y.base is not None
    assert x.shape == y.shape == (1024, 1024)


# --- storage profile (section 9.3) ---------------------------------------


def test_storage_profile_matches_the_recommended_defaults() -> None:
    profile = storage_profile(
        width=256, height=256, transform=TRANSFORM, crs="EPSG:5514"
    )
    assert profile["driver"] == "GTiff"
    assert profile["dtype"] == "float32"
    assert profile["tiled"] is True
    assert profile["compress"] == "deflate"
    assert profile["predictor"] == 3
    assert profile["nodata"] == -9999.0
    assert profile["blockxsize"] == profile["blockysize"] == 256


def test_written_rasters_use_the_storage_profile(tmp_path: Path) -> None:
    data = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
    path = write_raster(tmp_path / "t.tif", data, transform=TRANSFORM, crs="EPSG:5514")
    with rasterio.open(path) as dataset:
        assert dataset.dtypes == ("float32",)
        assert dataset.nodata == -9999.0
        assert dataset.profile["compress"] == "deflate"
        assert dataset.profile["tiled"] is True
        assert dataset.block_shapes == [(256, 256)]
        # GDAL reports the predictor through IMAGE_STRUCTURE, not the profile.
        assert dataset.tags(ns="IMAGE_STRUCTURE")["PREDICTOR"] == "3"
        assert epsg_code_of(dataset.crs) == 5514
        assert dataset.transform == TRANSFORM
        np.testing.assert_array_equal(dataset.read(1), data)


def test_tags_round_trip(tmp_path: Path) -> None:
    tags = provenance_tags(
        package_version="1.0.0",
        source_id="cuzk-dmr5g",
        vertical_datum="egm96",
        vertical_crs="EPSG:5773",
        resolution_m=2.0,
    )
    path = write_raster(
        tmp_path / "t.tif",
        np.zeros((16, 16), dtype=np.float32),
        transform=TRANSFORM,
        crs="EPSG:5514",
        tags=tags,
    )
    stored = read_tags(path)
    assert {k: stored[k] for k in tags} == tags


def test_read_tags_of_a_non_raster_is_empty(tmp_path: Path) -> None:
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"not a tiff")
    assert read_tags(broken) == {}


# --- atomic writes (section 18) ------------------------------------------


def test_atomic_write_leaves_no_temporary_behind(tmp_path: Path) -> None:
    path = write_raster_atomic(
        tmp_path / "t.tif",
        np.zeros((16, 16), dtype=np.float32),
        transform=TRANSFORM,
        crs="EPSG:5514",
    )
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_refuses_to_clobber_without_overwrite(tmp_path: Path) -> None:
    kwargs = dict(transform=TRANSFORM, crs="EPSG:5514")
    data = np.zeros((16, 16), dtype=np.float32)
    write_raster_atomic(tmp_path / "t.tif", data, **kwargs)
    with pytest.raises(OutputConflictError, match="already exists"):
        write_raster_atomic(tmp_path / "t.tif", data, **kwargs)
    write_raster_atomic(tmp_path / "t.tif", data, overwrite=True, **kwargs)


def test_a_failed_validation_discards_the_temporary_and_keeps_the_original(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t.tif"
    original = np.full((16, 16), 1.0, dtype=np.float32)
    write_raster_atomic(path, original, transform=TRANSFORM, crs="EPSG:5514")

    def reject(_: Path) -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        write_raster_atomic(
            path,
            np.full((16, 16), 2.0, dtype=np.float32),
            overwrite=True,
            validate=reject,
            transform=TRANSFORM,
            crs="EPSG:5514",
        )
    assert not list(tmp_path.glob("*.tmp"))
    with rasterio.open(path) as dataset:
        np.testing.assert_array_equal(dataset.read(1), original)


# --- vertical conversion (section 8.4) -----------------------------------


def test_conversion_leaves_nodata_untouched(proj_dir: Path) -> None:
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    data = np.full((8, 8), 300.0, dtype=np.float32)
    data[2, 3] = -9999.0
    data[5, 5] = -9999.0
    stats = ConversionStats()
    out = convert_vertical(data, TRANSFORM, converter, stats=stats)

    assert out.dtype == np.float32
    assert out[2, 3] == -9999.0 and out[5, 5] == -9999.0
    assert stats.pixels_total == 64
    assert stats.pixels_valid == 62
    assert stats.pixels_converted == 62
    assert stats.nodata_fraction == pytest.approx(2 / 64)
    valid = out[out != -9999.0]
    assert valid.size == 62
    assert np.all(np.isfinite(valid))
    # Bpv -> EGM96 over Prague is a shift of a few centimetres.
    assert np.all(np.abs(valid - 300.0) < 1.0)
    assert np.all(np.abs(valid - 300.0) > 1e-4)


def test_conversion_is_independent_of_the_block_size(proj_dir: Path) -> None:
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    rng = np.random.default_rng(7)
    data = (300.0 + rng.normal(0.0, 20.0, size=(37, 53))).astype(np.float32)
    whole = convert_vertical(data, TRANSFORM, converter, block_size_px=4096)
    blocked = convert_vertical(data, TRANSFORM, converter, block_size_px=8)
    np.testing.assert_allclose(whole, blocked, rtol=0, atol=0)


def test_conversion_honours_the_window_offset(proj_dir: Path) -> None:
    """A window converted in place must match the same pixels of the whole raster."""
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    data = np.full((40, 40), 300.0, dtype=np.float32)
    whole = convert_vertical(data, TRANSFORM, converter)
    window = convert_vertical(
        data[8:16, 12:20], TRANSFORM, converter, col_off=12, row_off=8
    )
    np.testing.assert_allclose(window, whole[8:16, 12:20], rtol=0, atol=0)


def test_source_nodata_is_translated_to_the_dataset_nodata(proj_dir: Path) -> None:
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    data = np.full((4, 4), 300.0, dtype=np.float32)
    data[1, 1] = -32768.0
    out = convert_vertical(data, TRANSFORM, converter, nodata=-9999.0, source_nodata=-32768.0)
    assert out[1, 1] == -9999.0
    assert out[0, 0] != -9999.0


def test_non_finite_source_pixels_become_nodata(proj_dir: Path) -> None:
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    data = np.full((4, 4), 300.0, dtype=np.float32)
    data[0, 0] = np.nan
    data[0, 1] = np.inf
    out = convert_vertical(data, TRANSFORM, converter)
    assert out[0, 0] == -9999.0 and out[0, 1] == -9999.0
    assert np.all(np.isfinite(out))


def test_an_all_nodata_block_needs_no_transformation(proj_dir: Path) -> None:
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    data = np.full((8, 8), -9999.0, dtype=np.float32)
    stats = ConversionStats()
    out = convert_vertical(data, TRANSFORM, converter, stats=stats)
    assert np.all(out == -9999.0)
    assert stats.pixels_converted == 0
    assert stats.max_abs_delta == 0.0
    assert stats.nodata_fraction == 1.0


def test_convert_raster_file_preserves_geometry_and_converts_heights(
    tmp_path: Path, proj_dir: Path
) -> None:
    """The file-level helper a LocalGeoTiffSource would build on."""
    from heightmap_prep.raster import convert_raster_file

    source = tmp_path / "bpv.tif"
    data = np.full((32, 32), 300.0, dtype=np.float32)
    data[0, 0] = -32768.0
    write_raster(source, data, transform=TRANSFORM, crs="EPSG:5514", nodata=-32768.0)

    destination = tmp_path / "egm96.tif"
    stats = convert_raster_file(
        source, destination, build_converter("egm96", proj_data_dir=proj_dir)
    )

    assert stats.pixels_valid == 32 * 32 - 1
    with rasterio.open(source) as before, rasterio.open(destination) as after:
        assert after.transform == before.transform
        assert epsg_code_of(after.crs) == epsg_code_of(before.crs)
        assert after.dtypes == ("float32",)
        assert after.nodata == -9999.0
        converted = after.read(1)
    assert converted[0, 0] == -9999.0
    valid = converted[converted != -9999.0]
    assert np.all(np.abs(valid - 300.0) < 1.0) and np.all(valid != 300.0)
