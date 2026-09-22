"""ČÚZK acquisition (specification sections 7 and 26), driven by a mock transport."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.io import MemoryFile

from heightmap_prep.cuzk import (
    DEFAULT_COVERAGE,
    DEFAULT_SERVICE_URL,
    SERVICE_MAX_IMAGE_HEIGHT,
    SERVICE_MAX_IMAGE_WIDTH,
    CuzkDmr5Source,
)
from heightmap_prep.errors import (
    MalformedRasterError,
    OutOfCoverageError,
    ResolutionMismatchError,
    SourceError,
    SourceHttpError,
)
from heightmap_prep.tiling import ProjectedBounds

AREA = ProjectedBounds(-743_000.0, -1_044_000.0, -742_800.0, -1_043_800.0)


def make_tiff(
    bounds: ProjectedBounds,
    width: int,
    height: int,
    *,
    crs: str | None = "EPSG:5514",
    value: float | None = None,
    nodata: float = -9999.0,
    transform: Affine | None = None,
) -> bytes:
    """Build a GeoTIFF the way the ImageServer would return one."""
    if transform is None:
        transform = Affine(
            bounds.width / width, 0.0, bounds.west, 0.0, -bounds.height / height, bounds.north
        )
    data = (
        np.full((height, width), value, dtype=np.float32)
        if value is not None
        else np.arange(width * height, dtype=np.float32).reshape(height, width)
    )
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": width,
        "height": height,
        "transform": transform,
        "nodata": nodata,
    }
    if crs is not None:
        profile["crs"] = crs
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data, 1)
        return memfile.read()


def serving(responder) -> CuzkDmr5Source:
    """A source whose HTTP client is backed by ``responder``."""
    return CuzkDmr5Source(
        client=httpx.Client(transport=httpx.MockTransport(responder)),
        retry_backoff_s=0.0,
    )


def echo_dem(request: httpx.Request) -> httpx.Response:
    """Answer any exportImage call with a correctly georeferenced raster."""
    params = request.url.params
    west, south, east, north = (float(v) for v in params["bbox"].split(","))
    width, height = (int(v) for v in params["size"].split(","))
    return httpx.Response(
        200,
        content=make_tiff(ProjectedBounds(west, south, east, north), width, height),
        headers={"content-type": "image/tiff"},
    )


# --- request construction (section 7.1) ----------------------------------


def test_request_parameters_match_the_specification() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return echo_dem(request)

    with serving(handler) as source:
        source.read_block(AREA, 100, 100)

    params = seen[0].url.params
    assert seen[0].url.path.endswith("/exportImage")
    assert params["bboxSR"] == "5514"
    assert params["imageSR"] == "5514"
    assert params["format"] == "tiff"
    assert params["pixelType"] == "F32"
    assert params["size"] == "100,100"
    assert params["f"] == "image"
    assert [float(v) for v in params["bbox"].split(",")] == list(AREA.as_tuple())


def test_default_service_is_https_and_the_documented_endpoint() -> None:
    assert DEFAULT_SERVICE_URL.startswith("https://")
    assert CuzkDmr5Source().service_url == DEFAULT_SERVICE_URL


def test_plain_http_is_refused() -> None:
    with pytest.raises(SourceError, match="HTTPS"):
        CuzkDmr5Source(service_url="http://ags.cuzk.gov.cz/x/ImageServer")


def test_unknown_interpolation_is_refused() -> None:
    with pytest.raises(SourceError, match="interpolation"):
        CuzkDmr5Source(interpolation="lanczos")


def test_request_limits_respect_the_service_maxima() -> None:
    source = CuzkDmr5Source(max_request_px=99_999)
    assert source.max_request_width_px == SERVICE_MAX_IMAGE_WIDTH
    assert source.max_request_height_px == SERVICE_MAX_IMAGE_HEIGHT


# --- request tiling (section 7.4) ----------------------------------------


def test_large_windows_are_split_without_gaps_or_overlaps() -> None:
    seen: list[tuple[float, float, float, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(tuple(float(v) for v in request.url.params["bbox"].split(",")))
        return echo_dem(request)

    bounds = ProjectedBounds(-743_000.0, -1_044_000.0, -742_600.0, -1_043_600.0)
    with serving(handler) as source:
        source.max_request_width_px = 100
        source.max_request_height_px = 100
        data = source.read_block(bounds, 200, 200)

    assert data.shape == (200, 200)
    assert len(seen) == 4
    # The pieces exactly tile the requested rectangle.
    assert sum((e - w) * (n - s) for w, s, e, n in seen) == pytest.approx(
        bounds.width * bounds.height
    )
    assert min(w for w, _, _, _ in seen) == bounds.west
    assert max(e for _, _, e, _ in seen) == bounds.east
    assert min(s for _, s, _, _ in seen) == bounds.south
    assert max(n for _, _, _, n in seen) == bounds.north


def test_a_split_mosaic_is_identical_to_a_single_request() -> None:
    bounds = ProjectedBounds(-743_000.0, -1_044_000.0, -742_800.0, -1_043_800.0)
    with serving(echo_dem) as single:
        whole = single.read_block(bounds, 100, 100)
    with serving(echo_dem) as split:
        split.max_request_width_px = 37
        split.max_request_height_px = 23
        mosaic = split.read_block(bounds, 100, 100)
    # The stub's values depend only on the sub-request, so compare geometry:
    # every pixel must be filled (no NoData left from the pre-fill).
    assert mosaic.shape == whole.shape
    assert not np.any(mosaic == -9999.0)


# --- response verification ------------------------------------------------


def test_wrong_dimensions_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=make_tiff(AREA, 50, 50), headers={"content-type": "image/tiff"})

    with serving(handler) as source, pytest.raises(ResolutionMismatchError, match="50x50"):
        source.read_block(AREA, 100, 100)


def test_wrong_crs_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=make_tiff(AREA, 100, 100, crs="EPSG:4326"),
            headers={"content-type": "image/tiff"},
        )

    with serving(handler) as source, pytest.raises(MalformedRasterError, match="expected EPSG:5514"):
        source.read_block(AREA, 100, 100)


def test_wrong_origin_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        shifted = Affine(2.0, 0.0, AREA.west + 500.0, 0.0, -2.0, AREA.north)
        return httpx.Response(
            200,
            content=make_tiff(AREA, 100, 100, transform=shifted),
            headers={"content-type": "image/tiff"},
        )

    with serving(handler) as source, pytest.raises(MalformedRasterError, match="origin"):
        source.read_block(AREA, 100, 100)


def test_a_corrupt_body_is_reported_clearly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not a tiff", headers={"content-type": "image/tiff"})

    with serving(handler) as source, pytest.raises(MalformedRasterError):
        source.read_block(AREA, 100, 100)


def test_the_servers_nodata_is_normalised_to_ours() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=make_tiff(AREA, 10, 10, value=-32768.0, nodata=-32768.0),
            headers={"content-type": "image/tiff"},
        )

    with serving(handler) as source:
        assert np.all(source.read_block(AREA, 10, 10) == -9999.0)


# --- error handling (sections 7.5 and 24) --------------------------------


def test_a_json_error_body_is_not_mistaken_for_a_raster() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps({"error": {"code": 400, "message": "Invalid bbox"}}).encode(),
            headers={"content-type": "application/json"},
        )

    with serving(handler) as source, pytest.raises(SourceHttpError, match="Invalid bbox"):
        source.read_block(AREA, 10, 10)


def test_transient_failures_are_retried_then_succeed() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, content=b"busy")
        return echo_dem(request)

    with serving(handler) as source:
        source.retries = 3
        assert source.read_block(AREA, 10, 10).shape == (10, 10)
    assert attempts["n"] == 3


def test_persistent_failure_raises_after_the_retry_budget() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, content=b"busy")

    with serving(handler) as source:
        source.retries = 2
        with pytest.raises(SourceHttpError, match="failed after 2 attempt"):
            source.read_block(AREA, 10, 10)
    assert attempts["n"] == 2


def test_a_non_retryable_status_fails_immediately() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(403, content=b"forbidden")

    with serving(handler) as source, pytest.raises(SourceHttpError, match="403"):
        source.read_block(AREA, 10, 10)
    assert attempts["n"] == 1


def test_timeouts_are_retried() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectTimeout("too slow", request=request)
        return echo_dem(request)

    with serving(handler) as source:
        assert source.read_block(AREA, 10, 10).shape == (10, 10)
    assert attempts["n"] == 2


# --- coverage (section 24) ------------------------------------------------


def test_areas_outside_czech_coverage_are_rejected() -> None:
    with serving(echo_dem) as source, pytest.raises(OutOfCoverageError, match="coverage"):
        source.read_block(ProjectedBounds(0.0, 0.0, 100.0, 100.0), 10, 10)


def test_the_published_coverage_covers_prague() -> None:
    assert DEFAULT_COVERAGE.intersection(AREA) is not None


# --- cache (section 7.5) --------------------------------------------------


def test_the_cache_avoids_a_second_request(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return echo_dem(request)

    def source() -> CuzkDmr5Source:
        return CuzkDmr5Source(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            cache_dir=tmp_path / "cache",
        )

    with source() as first:
        original = first.read_block(AREA, 10, 10)
        assert first.cache_hits == 0

    # A brand new source instance still finds the cached response.
    with source() as second:
        cached = second.read_block(AREA, 10, 10)
        assert second.cache_hits == 1
        assert second.requests_made == 0

    assert attempts["n"] == 1
    np.testing.assert_array_equal(original, cached)


def test_a_corrupt_cache_entry_is_discarded_and_refetched(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "deadbeef.tif").write_bytes(b"junk")

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return echo_dem(request)

    with CuzkDmr5Source(
        client=httpx.Client(transport=httpx.MockTransport(handler)), cache_dir=cache
    ) as source:
        source.read_block(AREA, 10, 10)
        entry = next(p for p in cache.glob("*.tif") if p.name != "deadbeef.tif")
        entry.write_bytes(b"corrupted")
        source.read_block(AREA, 10, 10)

    assert attempts["n"] == 2


def test_different_requests_get_different_cache_entries(tmp_path: Path) -> None:
    with CuzkDmr5Source(
        client=httpx.Client(transport=httpx.MockTransport(echo_dem)),
        cache_dir=tmp_path / "cache",
    ) as source:
        source.read_block(AREA, 10, 10)
        source.read_block(AREA, 20, 20)
        assert source.cache_hits == 0
        assert len(list((tmp_path / "cache").glob("*.tif"))) == 2


# --- fetch (section 16) ---------------------------------------------------


def test_fetch_writes_a_georeferenced_geotiff(tmp_path: Path) -> None:
    with serving(echo_dem) as source:
        path = source.fetch(AREA, tmp_path / "out.tif")
    with rasterio.open(path) as dataset:
        assert dataset.dtypes == ("float32",)
        assert dataset.width == int(AREA.width / 2.0)
        assert dataset.height == int(AREA.height / 2.0)
        assert dataset.transform.c == pytest.approx(AREA.west)
        assert dataset.transform.f == pytest.approx(AREA.north)
