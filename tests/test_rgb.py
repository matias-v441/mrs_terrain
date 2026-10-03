"""The RGB lattice, the ČÚZK orthophoto source and RGB tile storage."""

from __future__ import annotations

import warnings
from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile

from heightmap_prep.cuzk import (
    DEFAULT_ORTHO_SERVICE_URL,
    ORTHO_COVERAGE,
    ORTHO_MAX_IMAGE_PX,
    CuzkOrthophotoSource,
)
from heightmap_prep.errors import (
    ConfigError,
    MalformedRasterError,
    OutputConflictError,
    ResolutionMismatchError,
    SourceError,
    SourceHttpError,
)
from heightmap_prep.raster import rgb_storage_profile, write_rgb_window_atomic
from heightmap_prep.rgb import (
    METRES_PER_DEGREE,
    RgbGrid,
    RgbWindow,
    projected_to_lonlat_bounds,
)
from heightmap_prep.tiling import TileGrid, TileIndex

from conftest import SyntheticRgbSource

GRID = TileGrid.from_options(2.0, 128, -999_999.6, -800_000.12)
RGB = RgbGrid.from_metres(1.0)

#: Around the conftest test area, near Prague.
LONLAT = (14.4166, 50.0762, 14.4239, 50.0808)


# --- the lattice ---------------------------------------------------------


def test_the_resolution_is_square_degrees_from_metres_north_south() -> None:
    assert RgbGrid.from_metres(0.25).resolution_deg == pytest.approx(0.25 / METRES_PER_DEGREE)


def test_a_non_positive_resolution_is_refused() -> None:
    with pytest.raises(ConfigError):
        RgbGrid.from_metres(0.0)


def test_windows_snap_outward_onto_the_lattice() -> None:
    window = RGB.window_for_lonlat(LONLAT)
    west, south, east, north = RGB.window_bounds(window)
    res = RGB.resolution_deg
    assert west <= LONLAT[0] < west + res
    assert east - res < LONLAT[2] <= east
    assert south <= LONLAT[1] < south + res
    assert north - res < LONLAT[3] <= north


def test_a_lattice_edge_does_not_pull_in_an_extra_pixel() -> None:
    window = RgbWindow(1000, 2000, 30, 40)
    assert RGB.window_for_lonlat(RGB.window_bounds(window)) == window


def test_a_transform_maps_back_to_its_window() -> None:
    window = RgbWindow(86_666_048, 17_770_634, 981, 669)
    transform = RGB.window_transform(window)
    assert RGB.window_of_transform(transform, 981, 669) == window


def test_an_off_lattice_transform_is_recognised() -> None:
    window = RgbWindow(100, 100, 10, 10)
    transform = RGB.window_transform(window)
    shifted = transform @ transform.translation(0.5, 0.0)
    assert RGB.window_of_transform(shifted, 10, 10) is None
    assert RGB.window_of_transform(transform @ transform.scale(2.0), 10, 10) is None


def test_an_rgb_tile_covers_its_height_tiles_sampling_region() -> None:
    tile = GRID.index_for_point(-743_000.0, -1_044_000.0)
    window = RGB.tile_window(GRID, tile)
    need = projected_to_lonlat_bounds(GRID.sampling_region(tile))
    west, south, east, north = RGB.window_bounds(window)
    assert west <= need[0] and south <= need[1] and east >= need[2] and north >= need[3]


def test_densified_bounds_exceed_the_corners_alone() -> None:
    """Krovak edges are curves in lon/lat; corners alone would cut them off."""
    bounds = GRID.tile_bounds(TileIndex(1000, 950))
    dense = projected_to_lonlat_bounds(bounds)
    corners = projected_to_lonlat_bounds(bounds, samples=1)
    assert dense[0] <= corners[0] and dense[1] <= corners[1]
    assert dense[2] >= corners[2] and dense[3] >= corners[3]


def test_neighbouring_rgb_tiles_overlap_on_the_same_lattice() -> None:
    tile = GRID.index_for_point(-743_000.0, -1_044_000.0)
    left = RGB.tile_window(GRID, tile)
    right = RGB.tile_window(GRID, TileIndex(tile.ix + 1, tile.iy))
    assert left.intersection(right) is not None


def test_window_tags_round_trip() -> None:
    window = RgbWindow(12, 34, 56, 78)
    assert RgbWindow.from_tag(window.to_tag()) == window
    for bad in (None, "", "1,2,3", "a,b,c,d", "1,2,0,4"):
        assert RgbWindow.from_tag(bad) is None


# --- splitting and stitching ---------------------------------------------


def test_large_windows_are_split_into_server_sized_requests() -> None:
    source = SyntheticRgbSource(max_request_px=100)
    window = RgbWindow(1_000_000, 500_000, 250, 130)
    mosaic = source.read_rgba(RGB, window)
    assert len(source.requests) == 3 * 2
    whole = SyntheticRgbSource().read_rgba(RGB, window)
    np.testing.assert_array_equal(mosaic, whole)


# --- the ČÚZK orthophoto service ------------------------------------------


def make_png(width: int, height: int, *, bands: int = 4, value: int = 100) -> bytes:
    data = np.full((bands, height, width), value, dtype=np.uint8)
    if bands == 4:
        data[3] = 255
        data[3, 0, 0] = 0
    with warnings.catch_warnings(), MemoryFile() as memfile:
        warnings.simplefilter("ignore", NotGeoreferencedWarning)  # PNGs never are
        with memfile.open(
            driver="PNG", width=width, height=height, count=bands, dtype="uint8"
        ) as dataset:
            dataset.write(data)
        return memfile.read()


def ortho_server(*, extent_shift: float = 0.0, bands: int = 4, error: dict | None = None):
    """A mock export endpoint: renders, then serves the image at ``href``."""
    calls = {"export": 0, "image": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/export"):
            calls["export"] += 1
            if error is not None:
                return httpx.Response(200, json={"error": error})
            params = request.url.params
            west, south, east, north = (float(v) for v in params["bbox"].split(","))
            width, height = (int(v) for v in params["size"].split(","))
            href = f"https://ags.cuzk.gov.cz/out/{width}x{height}.png"
            return httpx.Response(
                200,
                json={
                    "href": href,
                    "width": width,
                    "height": height,
                    "extent": {
                        "xmin": west + extent_shift,
                        "ymin": south,
                        "xmax": east + extent_shift,
                        "ymax": north,
                        "spatialReference": {"wkid": 4326, "latestWkid": 4326},
                    },
                },
            )
        calls["image"] += 1
        width, height = (int(v) for v in request.url.path.rsplit("/", 1)[-1][:-4].split("x"))
        return httpx.Response(
            200, content=make_png(width, height, bands=bands), headers={"content-type": "image/png"}
        )

    return handler, calls


def ortho(handler, **kwargs) -> CuzkOrthophotoSource:
    return CuzkOrthophotoSource(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_backoff_s=0.0,
        **kwargs,
    )


WINDOW = RgbWindow(1_000_000, 500_000, 40, 30)


def test_the_orthophoto_service_defaults() -> None:
    source = CuzkOrthophotoSource()
    assert source.service_url == DEFAULT_ORTHO_SERVICE_URL
    assert source.source_id == "cuzk-ortofoto"
    assert source.coverage() == ORTHO_COVERAGE
    assert source.max_request_width_px == ORTHO_MAX_IMAGE_PX


def test_plain_http_is_refused_for_imagery() -> None:
    with pytest.raises(SourceError, match="HTTPS"):
        CuzkOrthophotoSource(service_url="http://ags.cuzk.gov.cz/x/MapServer")


def test_imagery_is_rendered_in_epsg_4326_then_downloaded() -> None:
    seen: list[httpx.Request] = []
    handler, calls = ortho_server()

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    with ortho(recording) as source:
        rgba = source.read_rgba(RGB, WINDOW)
    assert rgba.shape == (4, 30, 40) and rgba.dtype == np.uint8
    assert rgba[3, 0, 0] == 0 and rgba[3, 1, 1] == 255
    assert calls == {"export": 1, "image": 1}
    params = seen[0].url.params
    assert params["bboxSR"] == params["imageSR"] == "4326"
    assert params["size"] == "40,30"
    assert params["format"] == "png32" and params["f"] == "json"
    west, south, east, north = (float(v) for v in params["bbox"].split(","))
    # Square pixels in degrees, as ArcGIS renders them.
    assert (east - west) / 40 == pytest.approx((north - south) / 30, rel=1e-9)


def test_rgb_without_alpha_is_fully_valid() -> None:
    handler, _ = ortho_server(bands=3)
    with ortho(handler) as source:
        assert source.read_rgba(RGB, WINDOW)[3].min() == 255


def test_a_widened_extent_is_rejected() -> None:
    handler, calls = ortho_server(extent_shift=RGB.resolution_deg)
    with ortho(handler) as source, pytest.raises(MalformedRasterError, match="covers"):
        source.read_rgba(RGB, WINDOW)
    assert calls["image"] == 0


def test_a_json_error_is_reported() -> None:
    handler, _ = ortho_server(error={"code": 400, "message": "Invalid bbox"})
    with ortho(handler) as source, pytest.raises(SourceHttpError, match="Invalid bbox"):
        source.read_rgba(RGB, WINDOW)


def test_an_image_of_the_wrong_size_is_rejected() -> None:
    handler, _ = ortho_server()

    def wrong_size(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".png"):
            return httpx.Response(200, content=make_png(5, 5), headers={"content-type": "image/png"})
        return handler(request)

    with ortho(wrong_size) as source, pytest.raises(ResolutionMismatchError):
        source.read_rgba(RGB, WINDOW)


def test_transient_render_failures_are_retried() -> None:
    handler, calls = ortho_server()
    attempts = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/export"):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(503, content=b"busy")
        return handler(request)

    with ortho(flaky) as source:
        assert source.read_rgba(RGB, WINDOW).shape == (4, 30, 40)
    assert attempts["n"] == 2


def test_the_cache_avoids_a_second_render(tmp_path: Path) -> None:
    handler, calls = ortho_server()
    with ortho(handler, cache_dir=tmp_path) as first:
        expected = first.read_rgba(RGB, WINDOW)
    with ortho(handler, cache_dir=tmp_path) as second:
        np.testing.assert_array_equal(second.read_rgba(RGB, WINDOW), expected)
        assert second.cache_hits == 1 and second.requests_made == 0
    assert calls == {"export": 1, "image": 1}
    assert len(list(tmp_path.glob("*.png"))) == 1


# --- storage -------------------------------------------------------------


def write_tile(path: Path, *, window: RgbWindow, fill: RgbWindow, rgba=None, **kwargs) -> Path:
    if rgba is None:
        rgba = SyntheticRgbSource().read_rgba(RGB, fill)
    col_off, row_off = fill.offset_in(window)
    return write_rgb_window_atomic(
        path,
        rgba,
        profile=rgb_storage_profile(
            width=window.width,
            height=window.height,
            transform=RGB.window_transform(window),
            crs="EPSG:4326",
        ),
        col_off=col_off,
        row_off=row_off,
        **kwargs,
    )


def test_only_the_filled_window_holds_imagery(tmp_path: Path) -> None:
    window = RgbWindow(1_000_000, 500_000, 3000, 2000)
    fill = RgbWindow(1_001_000, 500_700, 300, 200)
    path = write_tile(tmp_path / "tile.tif", window=window, fill=fill)
    with rasterio.open(path) as dataset:
        assert (dataset.width, dataset.height, dataset.count) == (3000, 2000, 3)
        assert dataset.compression.value == "JPEG"
        mask = dataset.read_masks(1)
        assert mask[700:900, 1000:1300].min() == 255
        assert np.count_nonzero(mask) == 300 * 200
        rgb = dataset.read(window=rasterio.windows.Window(1000, 700, 300, 200)).astype(int)
    expected = SyntheticRgbSource().read_rgba(RGB, fill)[:3].astype(int)
    assert np.abs(rgb - expected).mean() < 4  # JPEG is lossy, but not by much
    # Empty blocks are left out of the file.
    assert path.stat().st_size < 3000 * 2000 * 3 // 20


def test_alpha_becomes_the_mask(tmp_path: Path) -> None:
    window = RgbWindow(0, 0, 64, 64)
    rgba = np.full((4, 64, 64), 200, dtype=np.uint8)
    rgba[3, :, :10] = 0
    path = write_tile(tmp_path / "tile.tif", window=window, fill=window, rgba=rgba)
    with rasterio.open(path) as dataset:
        mask = dataset.read_masks(1)
    assert mask[:, :10].max() == 0 and mask[:, 10:].min() == 255


def test_an_existing_tile_is_not_overwritten_by_default(tmp_path: Path) -> None:
    window = RgbWindow(0, 0, 16, 16)
    path = write_tile(tmp_path / "tile.tif", window=window, fill=window)
    with pytest.raises(OutputConflictError):
        write_tile(path, window=window, fill=window)


def test_a_failed_validation_leaves_nothing_behind(tmp_path: Path) -> None:
    window = RgbWindow(0, 0, 16, 16)

    def reject(tmp: Path) -> None:
        raise MalformedRasterError("no")

    with pytest.raises(MalformedRasterError):
        write_tile(tmp_path / "tile.tif", window=window, fill=window, validate=reject)
    assert list(tmp_path.iterdir()) == []


def test_only_rgba_uint8_is_accepted(tmp_path: Path) -> None:
    window = RgbWindow(0, 0, 4, 4)
    with pytest.raises(MalformedRasterError):
        write_tile(tmp_path / "t.tif", window=window, fill=window, rgba=np.zeros((3, 4, 4), np.uint8))
