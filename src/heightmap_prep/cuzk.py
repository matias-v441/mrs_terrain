"""ČÚZK DMR 5G acquisition (specification section 7).

The dynamic ArcGIS ImageServer publishes DMR 5G directly in S-JTSK / Krovak East
North with Bpv heights, which is exactly the representation this library wants to
store, so rasters are requested in the source grid and never reprojected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import rasterio
from rasterio.io import MemoryFile

from .crs import SOURCE_HORIZONTAL_CRS, SOURCE_VERTICAL_CRS, crs_matches, epsg_code_of
from .errors import (
    MalformedRasterError,
    ResolutionMismatchError,
    SourceError,
    SourceHttpError,
)
from .sources import BaseHeightSource
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


@dataclass
class CuzkDmr5Source(BaseHeightSource):
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
        parsed = urlparse(self.service_url)
        if parsed.scheme != "https":
            raise SourceError(
                f"source service URL must use HTTPS, got {self.service_url!r}"
            )
        if self.interpolation not in self._INTERPOLATIONS:
            raise SourceError(
                f"unsupported source interpolation {self.interpolation!r}; "
                f"expected one of {', '.join(sorted(self._INTERPOLATIONS))}"
            )
        if self.resolution_m <= 0:
            raise SourceError(f"resolution_m must be positive, got {self.resolution_m}")
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
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

    def __enter__(self) -> "CuzkDmr5Source":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

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

    def _cache_path(self, params: dict[str, str]) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps(
            {"url": self.service_url, **params}, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.tif"

    def _download(self, params: dict[str, str]) -> bytes:
        url = f"{self.service_url}/exportImage"
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
                        f"for bbox {params['bbox']} size {params['size']}: "
                        f"{response.text[:300]}"
                    )
                else:
                    content_type = response.headers.get("content-type", "")
                    if "json" in content_type.lower():
                        # ArcGIS reports failures as JSON even when f=image.
                        raise SourceHttpError(
                            f"{self.service_url} refused bbox {params['bbox']} "
                            f"size {params['size']}: {response.text[:300]}"
                        )
                    if not response.content:
                        last_error = SourceHttpError(
                            f"{self.service_url} returned an empty body for "
                            f"bbox {params['bbox']}"
                        )
                    else:
                        self._requests_made += 1
                        return response.content
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc

            if attempt < self.retries:
                delay = self.retry_backoff_s * (2 ** (attempt - 1))
                log.warning(
                    "request for bbox %s failed (%s); retry %d/%d in %.1fs",
                    params["bbox"],
                    last_error,
                    attempt,
                    self.retries - 1,
                    delay,
                )
                time.sleep(delay)

        raise SourceHttpError(
            f"{self.service_url} failed after {self.retries} attempt(s) for "
            f"bbox {params['bbox']} size {params['size']}"
        ) from last_error

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

        if cache_path is not None:
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            try:
                tmp.write_bytes(payload)
                tmp.replace(cache_path)
            except OSError as exc:  # pragma: no cover - cache is best effort
                log.warning("could not write cache entry %s: %s", cache_path, exc)
                tmp.unlink(missing_ok=True)
        return data
