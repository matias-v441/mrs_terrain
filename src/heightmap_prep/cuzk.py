"""ČÚZK acquisition: DMR 5G heights (specification section 7) and orthophoto.

The dynamic ArcGIS ImageServer publishes DMR 5G directly in S-JTSK / Krovak East
North with Bpv heights, which is exactly the representation this library wants to
store, so rasters are requested in the source grid and never reprojected.

The orthophoto MapServer is cached in EPSG:5514 but renders any extent in
EPSG:4326 on request, datum shift included, so RGB imagery is fetched directly in
the query CRS.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile

from .crs import SOURCE_HORIZONTAL_CRS, SOURCE_VERTICAL_CRS, crs_matches, epsg_code_of
from .errors import (
    MalformedRasterError,
    ResolutionMismatchError,
    SourceError,
    SourceHttpError,
)
from .sources import BaseHeightSource, BaseRgbSource
from .tiling import ProjectedBounds

log = logging.getLogger(__name__)

#: Dynamic DMR 5G ImageServer, serving EPSG:5514 / Bpv.
DEFAULT_SERVICE_URL = (
    "https://ags.cuzk.gov.cz/arcgis2/rest/services/dmr5g/ImageServer"
)

#: Published extent of the service, in EPSG:5514 metres.  Used to reject requests
#: outside Czech coverage without needing a network round-trip.
DEFAULT_COVERAGE = ProjectedBounds(
    west=-904_703.6, south=-1_227_414.12, east=-431_605.6, north=-935_118.12
)

#: Native product resolution.
NATIVE_RESOLUTION_M = 2.0

#: A point known to lie on a corner of a native DMR 5G pixel, in EPSG:5514.
#: The service extent is itself pixel-aligned, which was confirmed by
#: oversampling the mosaic with nearest-neighbour resampling: value changes fall
#: on x = -904703.6 + 2k and y = -935118.12 + 2k.  Aligning the output tile grid
#: to this phase means requested bboxes land on native pixel boundaries, so the
#: server returns native samples instead of resampling them.
NATIVE_GRID_ORIGIN_X = DEFAULT_COVERAGE.west
NATIVE_GRID_ORIGIN_Y = DEFAULT_COVERAGE.north

#: Server export limits reported by the service description.
SERVICE_MAX_IMAGE_WIDTH = 15_000
SERVICE_MAX_IMAGE_HEIGHT = 4_100

#: HTTP statuses worth retrying.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class _ArcGisHttp:
    """Pooled HTTP access to an ArcGIS REST service, with retries and a disk cache.

    Mixed into the ČÚZK sources, which declare the fields it reads:
    ``service_url``, ``timeout_s``, ``retries``, ``retry_backoff_s``,
    ``cache_dir``, ``client`` and ``user_agent``.
    """

    service_url: str
    timeout_s: float
    retries: int
    retry_backoff_s: float
    cache_dir: Path | None
    client: httpx.Client | None
    user_agent: str
    _owned_client: httpx.Client | None
    _cache_hits: int
    _requests_made: int

    def _check_service_url(self) -> None:
        parsed = urlparse(self.service_url)
        if parsed.scheme != "https":
            raise SourceError(
                f"source service URL must use HTTPS, got {self.service_url!r}"
            )

    def _prepare_cache_dir(self) -> None:
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def requests_made(self) -> int:
        return self._requests_made

    # --- lifecycle -------------------------------------------------------

    def _http(self) -> httpx.Client:
        """One pooled client, so connections are reused across requests."""
        if self.client is not None:
            return self.client
        if self._owned_client is None:
            self._owned_client = httpx.Client(
                timeout=httpx.Timeout(self.timeout_s),
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
        return self._owned_client

    def close(self) -> None:
        if self._owned_client is not None:
            self._owned_client.close()
            self._owned_client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- cache -----------------------------------------------------------

    def _cache_path(self, params: dict[str, str], suffix: str = ".tif") -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps(
            {"url": self.service_url, **params}, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}{suffix}"

    @staticmethod
    def _cache_store(cache_path: Path | None, payload: bytes) -> None:
        if cache_path is None:
            return
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            tmp.write_bytes(payload)
            tmp.replace(cache_path)
        except OSError as exc:  # pragma: no cover - cache is best effort
            log.warning("could not write cache entry %s: %s", cache_path, exc)
            tmp.unlink(missing_ok=True)

    # --- requests --------------------------------------------------------

    def _get(
        self,
        url: str,
        params: dict[str, str] | None,
        *,
        describe: str,
        expect_json: bool = False,
    ) -> bytes:
        """GET ``url``, retrying transient failures; ``describe`` names the request.

        ArcGIS reports failures as JSON even when an image was asked for, so
        unless ``expect_json`` a JSON body is a refusal, not a payload.
        """
        client = self._http()
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                response = client.get(url, params=params)
                log.debug("GET %s -> %s", response.url, response.status_code)
                if response.status_code in RETRYABLE_STATUS:
                    last_error = SourceHttpError(
                        f"{self.service_url} returned HTTP {response.status_code}"
                    )
                elif response.status_code != 200:
                    raise SourceHttpError(
                        f"{self.service_url} returned HTTP {response.status_code} "
                        f"for {describe}: {response.text[:300]}"
                    )
                else:
                    content_type = response.headers.get("content-type", "")
                    if not expect_json and "json" in content_type.lower():
                        raise SourceHttpError(
                            f"{self.service_url} refused {describe}: {response.text[:300]}"
                        )
                    if not response.content:
                        last_error = SourceHttpError(
                            f"{self.service_url} returned an empty body for {describe}"
                        )
                    else:
                        self._requests_made += 1
                        return response.content
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc

            if attempt < self.retries:
                delay = self.retry_backoff_s * (2 ** (attempt - 1))
                log.warning(
                    "request for %s failed (%s); retry %d/%d in %.1fs",
                    describe,
                    last_error,
                    attempt,
                    self.retries - 1,
                    delay,
                )
                time.sleep(delay)

        raise SourceHttpError(
            f"{self.service_url} failed after {self.retries} attempt(s) for {describe}"
        ) from last_error


@dataclass
class CuzkDmr5Source(_ArcGisHttp, BaseHeightSource):
    """Fetches DMR 5G elevation rasters from the ČÚZK ImageServer."""

    service_url: str = DEFAULT_SERVICE_URL
    resolution_m: float = NATIVE_RESOLUTION_M
    nodata: float = -9999.0
    timeout_s: float = 120.0
    retries: int = 3
    retry_backoff_s: float = 1.5
    cache_dir: Path | None = None
    #: Resampling the server applies when our grid is offset from the source grid.
    interpolation: str = "nearest"
    max_request_px: int = 2048
    client: httpx.Client | None = None
    user_agent: str = "heightmap-prep/1.0 (+https://pypi.org/project/heightmap-prep)"

    horizontal_crs: str = SOURCE_HORIZONTAL_CRS
    vertical_crs: str = SOURCE_VERTICAL_CRS

    _owned_client: httpx.Client | None = field(default=None, init=False, repr=False)
    _cache_hits: int = field(default=0, init=False, repr=False)
    _requests_made: int = field(default=0, init=False, repr=False)

    _INTERPOLATIONS = {
        "nearest": "RSP_NearestNeighbor",
        "bilinear": "RSP_BilinearInterpolation",
        "cubic": "RSP_CubicConvolution",
    }

    def __post_init__(self) -> None:
        self._check_service_url()
        if self.interpolation not in self._INTERPOLATIONS:
            raise SourceError(
                f"unsupported source interpolation {self.interpolation!r}; "
                f"expected one of {', '.join(sorted(self._INTERPOLATIONS))}"
            )
        if self.resolution_m <= 0:
            raise SourceError(f"resolution_m must be positive, got {self.resolution_m}")
        self._prepare_cache_dir()
        self.max_request_width_px = min(int(self.max_request_px), SERVICE_MAX_IMAGE_WIDTH)
        self.max_request_height_px = min(int(self.max_request_px), SERVICE_MAX_IMAGE_HEIGHT)

    # --- identity --------------------------------------------------------

    @property
    def source_id(self) -> str:
        return "cuzk-dmr5g"

    def coverage(self) -> ProjectedBounds:
        return DEFAULT_COVERAGE

    def native_grid_origin(self) -> tuple[float, float]:
        return (NATIVE_GRID_ORIGIN_X, NATIVE_GRID_ORIGIN_Y)

    # --- requests --------------------------------------------------------

    def _params(self, bounds: ProjectedBounds, width: int, height: int) -> dict[str, str]:
        return {
            "bbox": "{:.6f},{:.6f},{:.6f},{:.6f}".format(*bounds.as_tuple()),
            "bboxSR": "5514",
            "imageSR": "5514",
            "size": f"{width},{height}",
            "format": "tiff",
            "pixelType": "F32",
            "noData": f"{self.nodata:g}",
            "noDataInterpretation": "esriNoDataMatchAny",
            "interpolation": self._INTERPOLATIONS[self.interpolation],
            "f": "image",
        }

    def _download(self, params: dict[str, str]) -> bytes:
        return self._get(
            f"{self.service_url}/exportImage",
            params,
            describe=f"bbox {params['bbox']} size {params['size']}",
        )

    def _decode(
        self, payload: bytes, bounds: ProjectedBounds, width: int, height: int
    ) -> np.ndarray:
        """Decode and verify one returned GeoTIFF."""
        # The service embeds EPSG:5514 GeoTIFF keys that differ slightly from the
        # EPSG registry definition; trust the registry and drop GDAL's warning.
        try:
            with rasterio.Env(GTIFF_SRS_SOURCE="EPSG"), MemoryFile(payload) as memfile, memfile.open() as dataset:
                if dataset.count < 1:
                    raise MalformedRasterError(
                        f"DMR 5G response for bbox {bounds.as_tuple()} has no bands"
                    )
                if dataset.width != width or dataset.height != height:
                    raise ResolutionMismatchError(
                        f"DMR 5G returned {dataset.width}x{dataset.height} px for "
                        f"bbox {bounds.as_tuple()}, expected {width}x{height}"
                    )
                if not crs_matches(dataset.crs, 5514):
                    raise MalformedRasterError(
                        f"DMR 5G response for bbox {bounds.as_tuple()} is in "
                        f"EPSG:{epsg_code_of(dataset.crs)}, expected EPSG:5514"
                    )
                expected_res_x = bounds.width / width
                expected_res_y = bounds.height / height
                actual_res_x = abs(dataset.transform.a)
                actual_res_y = abs(dataset.transform.e)
                tolerance = 1e-6 * max(expected_res_x, expected_res_y, 1.0)
                if (
                    abs(actual_res_x - expected_res_x) > tolerance
                    or abs(actual_res_y - expected_res_y) > tolerance
                ):
                    raise ResolutionMismatchError(
                        f"DMR 5G returned pixel size {actual_res_x}x{actual_res_y} m "
                        f"for bbox {bounds.as_tuple()}, expected "
                        f"{expected_res_x}x{expected_res_y} m"
                    )
                origin_tolerance = 0.5 * expected_res_x
                if (
                    abs(dataset.transform.c - bounds.west) > origin_tolerance
                    or abs(dataset.transform.f - bounds.north) > origin_tolerance
                ):
                    raise MalformedRasterError(
                        f"DMR 5G returned origin ({dataset.transform.c}, "
                        f"{dataset.transform.f}) for bbox {bounds.as_tuple()}, "
                        f"expected ({bounds.west}, {bounds.north})"
                    )

                data = dataset.read(1).astype(np.float32, copy=False)
                src_nodata = dataset.nodata
        except (MalformedRasterError, ResolutionMismatchError):
            raise
        except rasterio.errors.RasterioError as exc:
            raise MalformedRasterError(
                f"could not decode the DMR 5G response for bbox "
                f"{bounds.as_tuple()}: {exc}"
            ) from exc

        # Normalise whatever the server used onto this dataset's single NoData.
        if src_nodata is not None and not np.isnan(src_nodata):
            if float(src_nodata) != float(self.nodata):
                data = np.where(data == np.float32(src_nodata), np.float32(self.nodata), data)
        data = np.where(np.isfinite(data), data, np.float32(self.nodata)).astype(
            np.float32, copy=False
        )
        return data

    def _request_block(
        self, bounds: ProjectedBounds, width: int, height: int
    ) -> np.ndarray:
        params = self._params(bounds, width, height)
        cache_path = self._cache_path(params)

        if cache_path is not None and cache_path.is_file():
            try:
                payload = cache_path.read_bytes()
                data = self._decode(payload, bounds, width, height)
                self._cache_hits += 1
                log.debug("cache hit for bbox %s", params["bbox"])
                return data
            except (MalformedRasterError, ResolutionMismatchError, OSError) as exc:
                log.warning("discarding unusable cache entry %s: %s", cache_path, exc)
                cache_path.unlink(missing_ok=True)

        payload = self._download(params)
        data = self._decode(payload, bounds, width, height)
        self._cache_store(cache_path, payload)
        return data


# --- orthophoto ------------------------------------------------------------

#: Cached ORTOFOTO map service; its cache is in EPSG:5514.
DEFAULT_ORTHO_SERVICE_URL = (
    "https://ags.cuzk.gov.cz/arcgis1/rest/services/ORTOFOTO/MapServer"
)

#: Full extent the service reports, in EPSG:5514 metres.
ORTHO_COVERAGE = ProjectedBounds(
    west=-907_841.06, south=-1_230_916.87, east=-416_691.67, north=-932_111.73
)

#: ``maxImageWidth`` / ``maxImageHeight`` of the service.
ORTHO_MAX_IMAGE_PX = 4096

#: How far, in pixels, the extent the server rendered may differ from the one
#: requested.  ArcGIS widens a bbox whose aspect ratio differs from the image
#: size; requests are built to match exactly, so anything visible is an error.
ORTHO_EXTENT_TOLERANCE_PX = 0.25


@dataclass
class CuzkOrthophotoSource(_ArcGisHttp, BaseRgbSource):
    """Fetches ČÚZK orthophoto imagery, rendered by the server in EPSG:4326.

    Each request is an ``export`` with ``f=json``, which renders the image and
    reports the extent it actually covers; that extent is checked against the
    request before the PNG it points to is downloaded.  ``format=tiff`` would
    save a round trip but comes back without georeferencing, so nothing could
    be checked.
    """

    service_url: str = DEFAULT_ORTHO_SERVICE_URL
    timeout_s: float = 120.0
    retries: int = 3
    retry_backoff_s: float = 1.5
    cache_dir: Path | None = None
    max_request_px: int = ORTHO_MAX_IMAGE_PX
    client: httpx.Client | None = None
    user_agent: str = "heightmap-prep/1.0 (+https://pypi.org/project/heightmap-prep)"

    _owned_client: httpx.Client | None = field(default=None, init=False, repr=False)
    _cache_hits: int = field(default=0, init=False, repr=False)
    _requests_made: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._check_service_url()
        if self.max_request_px <= 0:
            raise SourceError(f"max_request_px must be positive, got {self.max_request_px}")
        self._prepare_cache_dir()
        self.max_request_width_px = min(int(self.max_request_px), ORTHO_MAX_IMAGE_PX)
        self.max_request_height_px = min(int(self.max_request_px), ORTHO_MAX_IMAGE_PX)

    @property
    def source_id(self) -> str:
        return "cuzk-ortofoto"

    def coverage(self) -> ProjectedBounds:
        return ORTHO_COVERAGE

    # --- requests --------------------------------------------------------

    def _params(
        self, bounds: tuple[float, float, float, float], width: int, height: int
    ) -> dict[str, str]:
        return {
            "bbox": "{:.12f},{:.12f},{:.12f},{:.12f}".format(*bounds),
            "bboxSR": "4326",
            "imageSR": "4326",
            "size": f"{width},{height}",
            "format": "png32",
            "transparent": "true",
            "f": "json",
        }

    def _render(self, params: dict[str, str], bounds, width: int, height: int) -> str:
        """Ask the server to render the image; returns the URL of the result."""
        describe = f"bbox {params['bbox']} size {params['size']}"
        body = self._get(
            f"{self.service_url}/export", params, describe=describe, expect_json=True
        )
        try:
            reply = json.loads(body)
        except ValueError as exc:
            raise SourceHttpError(
                f"{self.service_url} answered {describe} with something that is not "
                f"JSON: {body[:200]!r}"
            ) from exc
        if not isinstance(reply, dict):
            raise SourceHttpError(f"{self.service_url} answered {describe} with {reply!r}")
        if "error" in reply:
            raise SourceHttpError(f"{self.service_url} refused {describe}: {reply['error']}")
        href = reply.get("href")
        extent = reply.get("extent") or {}
        if not href or not isinstance(extent, dict):
            raise SourceHttpError(
                f"{self.service_url} answered {describe} without an image: {str(reply)[:300]}"
            )

        if reply.get("width") not in (None, width) or reply.get("height") not in (None, height):
            raise ResolutionMismatchError(
                f"orthophoto rendered {reply.get('width')}x{reply.get('height')} px for "
                f"{describe}, expected {width}x{height}"
            )
        wkid = (extent.get("spatialReference") or {}).get("wkid")
        if wkid not in (None, 4326):
            raise MalformedRasterError(
                f"orthophoto for {describe} was rendered in wkid {wkid}, expected 4326"
            )
        west, south, east, north = bounds
        tolerance = ORTHO_EXTENT_TOLERANCE_PX * (east - west) / width
        try:
            got = tuple(float(extent[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedRasterError(
                f"orthophoto for {describe} came with an unreadable extent {extent!r}"
            ) from exc
        if any(abs(a - b) > tolerance for a, b in zip(got, bounds)):
            raise MalformedRasterError(
                f"orthophoto for {describe} covers {got}, not the requested {bounds}"
            )
        return str(href)

    def _decode(self, payload: bytes, describe: str, width: int, height: int) -> np.ndarray:
        """Decode one rendered PNG into ``(4, height, width)`` RGBA."""
        try:
            # A PNG has no georeferencing; the extent was checked when it was rendered.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                memfile = MemoryFile(payload)
                dataset = memfile.open()
            with memfile, dataset:
                if dataset.width != width or dataset.height != height:
                    raise ResolutionMismatchError(
                        f"orthophoto for {describe} is {dataset.width}x{dataset.height} "
                        f"px, expected {width}x{height}"
                    )
                if dataset.count not in (3, 4) or dataset.dtypes[0] != "uint8":
                    raise MalformedRasterError(
                        f"orthophoto for {describe} has {dataset.count} "
                        f"{dataset.dtypes[0]} band(s), expected RGB or RGBA uint8"
                    )
                data = dataset.read()
        except (MalformedRasterError, ResolutionMismatchError):
            raise
        except rasterio.errors.RasterioError as exc:
            raise MalformedRasterError(
                f"could not decode the orthophoto for {describe}: {exc}"
            ) from exc
        if data.shape[0] == 3:
            alpha = np.full((1, height, width), 255, dtype=np.uint8)
            data = np.concatenate([data, alpha])
        return data.astype(np.uint8, copy=False)

    def _request_rgba(
        self, bounds: tuple[float, float, float, float], width: int, height: int
    ) -> np.ndarray:
        params = self._params(bounds, width, height)
        describe = f"bbox {params['bbox']} size {params['size']}"
        cache_path = self._cache_path(params, suffix=".png")

        if cache_path is not None and cache_path.is_file():
            try:
                data = self._decode(cache_path.read_bytes(), describe, width, height)
                self._cache_hits += 1
                log.debug("cache hit for orthophoto %s", describe)
                return data
            except (MalformedRasterError, ResolutionMismatchError, OSError) as exc:
                log.warning("discarding unusable cache entry %s: %s", cache_path, exc)
                cache_path.unlink(missing_ok=True)

        href = self._render(params, bounds, width, height)
        payload = self._get(href, None, describe=f"the rendered image of {describe}")
        data = self._decode(payload, describe, width, height)
        self._cache_store(cache_path, payload)
        return data
