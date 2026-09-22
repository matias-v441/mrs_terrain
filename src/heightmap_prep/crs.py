"""Horizontal and vertical coordinate handling.

This module owns every interaction with PROJ.  It deliberately keeps the
horizontal CRS and the vertical CRS separate (specification section 2): the
horizontal grid of a DMR 5G raster is preserved untouched while only the heights
are moved from Bpv onto the requested output datum.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from pyproj import CRS, Transformer, datadir, network
from pyproj.transformer import TransformerGroup

from .errors import (
    CrsError,
    MissingGridError,
    NonFiniteTransformError,
    TransformationUnavailableError,
    UnsupportedSourceCrsError,
)

log = logging.getLogger(__name__)

#: The query CRS used by the downstream runtime sampler.
QUERY_CRS = "EPSG:4326"

#: Stored horizontal CRS for ČÚZK DMR 5G: S-JTSK / Krovak East North, metres.
SOURCE_HORIZONTAL_CRS = "EPSG:5514"

#: Source vertical CRS for ČÚZK DMR 5G: Bpv / Baltic 1957 height, metres.
SOURCE_VERTICAL_CRS = "EPSG:8357"
SOURCE_VERTICAL_DATUM = "bpv"

#: Largest residual horizontal movement tolerated by a vertical-only conversion.
HORIZONTAL_PRESERVATION_TOLERANCE_M = 1e-3

_AUTHORITY_RE = re.compile(r'AUTHORITY\s*\[\s*"EPSG"\s*,\s*"(\d+)"\s*\]', re.IGNORECASE)
_ID_RE = re.compile(r'ID\s*\[\s*"EPSG"\s*,\s*(\d+)\s*\]', re.IGNORECASE)


@dataclass(frozen=True)
class VerticalTarget:
    """One supported output height reference."""

    #: Readable name exposed on the CLI and recorded in the manifest.
    name: str
    #: CRS used as the PROJ transformation target.
    target_crs: str
    #: CRS recorded in the manifest as the vertical CRS of stored heights.
    vertical_crs: str
    #: ``True`` when the transformation keeps x/y in the stored horizontal CRS.
    preserves_horizontal: bool
    description: str


#: Output datums required by specification section 3.4.
VERTICAL_TARGETS: dict[str, VerticalTarget] = {
    "egm96": VerticalTarget(
        name="egm96",
        # A compound target keeps the horizontal part in EPSG:5514, so the
        # operation is genuinely vertical-only and x/y come back unchanged.
        target_crs=f"{SOURCE_HORIZONTAL_CRS}+EPSG:5773",
        vertical_crs="EPSG:5773",
        preserves_horizontal=True,
        description="EGM96 gravity-related height",
    ),
    "wgs84-ellipsoid": VerticalTarget(
        name="wgs84-ellipsoid",
        # There is no vertical CRS for "WGS 84 ellipsoidal height", so the
        # operation runs through the WGS 84 3D CRS and only Z is kept.
        target_crs="EPSG:4979",
        vertical_crs="EPSG:4979",
        preserves_horizontal=False,
        description="WGS84 ellipsoidal height (via EPSG:4979)",
    ),
}


# --- PROJ environment ----------------------------------------------------

_proj_lock = threading.Lock()
_configured_dirs: set[str] = set()


def configure_proj(
    proj_data_dir: Path | None = None, *, network_enabled: bool = False
) -> None:
    """Point PROJ at local grid files and decide whether it may use the network.

    Must be called *before* any transformer is created (specification section
    4.3); PROJ caches its search path when it first builds an operation.
    """
    with _proj_lock:
        network.set_network_enabled(network_enabled)
        if proj_data_dir is not None:
            resolved = str(Path(proj_data_dir).resolve())
            if not Path(resolved).is_dir():
                raise CrsError(f"PROJ data directory does not exist: {resolved}")
            if resolved not in _configured_dirs:
                datadir.append_data_dir(resolved)
                _configured_dirs.add(resolved)
            log.debug("PROJ data directories now include %s", resolved)
        log.debug(
            "PROJ network access %s; base data dir %s",
            "enabled" if network_enabled else "disabled",
            datadir.get_data_dir(),
        )


def proj_version() -> str:
    """The PROJ library version string, for reproducibility metadata."""
    import pyproj

    return str(pyproj.proj_version_str)


# --- CRS identity --------------------------------------------------------


def epsg_code_of(crs_like: object) -> int | None:
    """Best-effort EPSG code for a CRS that may carry a degraded WKT.

    The ČÚZK ImageServer returns GeoTIFFs whose CRS is a ``LOCAL_CS`` carrying
    ``AUTHORITY["EPSG","5514"]``.  ``to_epsg()`` refuses such a definition, so
    the authority node is read directly as a fallback.
    """
    if crs_like is None:
        return None
    try:
        code = CRS.from_user_input(crs_like).to_epsg()
    except Exception:  # pragma: no cover - defensive; handled by WKT fallback
        code = None
    if code is not None:
        return int(code)

    try:
        wkt = crs_like.to_wkt()  # type: ignore[union-attr]
    except Exception:
        wkt = str(crs_like)
    matches = _AUTHORITY_RE.findall(wkt) or _ID_RE.findall(wkt)
    if matches:
        return int(matches[-1])
    return None


def crs_matches(crs_like: object, expected_epsg: int) -> bool:
    """Whether ``crs_like`` denotes ``expected_epsg``, tolerating odd WKT."""
    return epsg_code_of(crs_like) == expected_epsg


def require_horizontal_crs(crs_like: object, expected_epsg: int, context: str) -> None:
    """Raise unless ``crs_like`` is the expected stored horizontal CRS."""
    if not crs_matches(crs_like, expected_epsg):
        raise UnsupportedSourceCrsError(
            f"{context}: expected horizontal CRS EPSG:{expected_epsg}, "
            f"got {crs_like!r} (resolved EPSG code: {epsg_code_of(crs_like)})"
        )


# --- vertical operation selection ---------------------------------------


@dataclass(frozen=True)
class GridInfo:
    """One PROJ grid file referenced by the selected operation."""

    name: str
    available: bool
    url: str = ""


@dataclass(frozen=True)
class VerticalOperation:
    """The concrete PROJ operation chosen for the Bpv → output conversion."""

    target: VerticalTarget
    source_crs: str
    description: str
    pipeline: str
    accuracy: float | None
    grids: tuple[GridInfo, ...]
    network_enabled: bool

    @property
    def grid_names(self) -> tuple[str, ...]:
        return tuple(grid.name for grid in self.grids)


def _grids_of(transformer: Transformer) -> tuple[GridInfo, ...]:
    grids: list[GridInfo] = []
    for operation in transformer.operations or ():
        for grid in operation.grids:
            grids.append(
                GridInfo(
                    name=grid.short_name,
                    available=bool(grid.available),
                    url=getattr(grid, "url", "") or "",
                )
            )
    # Preserve order while removing duplicates.
    seen: set[str] = set()
    unique: list[GridInfo] = []
    for grid in grids:
        if grid.name not in seen:
            seen.add(grid.name)
            unique.append(grid)
    return tuple(unique)


def _prefers_cr2005(transformer: Transformer) -> bool:
    return any("CR-2005".lower() in g.name.lower() for g in _grids_of(transformer))


def select_vertical_operation(
    datum: str,
    *,
    source_horizontal_crs: str = SOURCE_HORIZONTAL_CRS,
    source_vertical_crs: str = SOURCE_VERTICAL_CRS,
    network_enabled: bool = False,
) -> VerticalOperation:
    """Choose the PROJ operation that converts source heights onto ``datum``.

    Uses :class:`~pyproj.transformer.TransformerGroup` with ``always_xy=True``
    and ``allow_ballpark=False``, prefers the Czech CR-2005 based chain, and
    raises a descriptive error when required grids are absent.
    """
    try:
        target = VERTICAL_TARGETS[datum]
    except KeyError:
        raise CrsError(
            f"unsupported vertical datum {datum!r}; "
            f"expected one of {', '.join(sorted(VERTICAL_TARGETS))}"
        ) from None

    source_crs = f"{source_horizontal_crs}+{source_vertical_crs}"
    try:
        src = CRS.from_user_input(source_crs)
        dst = CRS.from_user_input(target.target_crs)
    except Exception as exc:  # pragma: no cover - only on a broken PROJ install
        raise CrsError(f"could not build CRS for {source_crs} -> {target.target_crs}: {exc}") from exc

    group = TransformerGroup(src, dst, always_xy=True, allow_ballpark=False)

    if not group.transformers:
        missing: list[str] = []
        for operation in group.unavailable_operations:
            for grid in operation.grids:
                if not grid.available and grid.short_name not in missing:
                    missing.append(grid.short_name)
        if missing:
            raise MissingGridError(
                f"no usable PROJ operation for {source_crs} -> {target.target_crs}: "
                f"missing grid file(s) {', '.join(missing)}. Install them into the "
                "directory passed as --proj-data-dir (they can be downloaded from "
                "https://cdn.proj.org/).",
                tuple(missing),
            )
        raise TransformationUnavailableError(
            f"PROJ reports no non-ballpark operation for {source_crs} -> "
            f"{target.target_crs}"
        )

    chosen = next(
        (t for t in group.transformers if _prefers_cr2005(t)), group.transformers[0]
    )
    if not _prefers_cr2005(chosen):
        log.warning(
            "selected vertical operation does not use the Czech CR-2005 grid: %s",
            chosen.description,
        )

    grids = _grids_of(chosen)
    unavailable = [g.name for g in grids if not g.available]
    if unavailable:  # pragma: no cover - TransformerGroup already filters these
        raise MissingGridError(
            f"selected operation {chosen.description!r} needs unavailable grid(s) "
            f"{', '.join(unavailable)}",
            tuple(unavailable),
        )

    operation = VerticalOperation(
        target=target,
        source_crs=source_crs,
        description=chosen.description,
        pipeline=chosen.to_proj4(),
        accuracy=chosen.accuracy if chosen.accuracy is not None and chosen.accuracy >= 0 else None,
        grids=grids,
        network_enabled=network_enabled,
    )
    log.debug("selected vertical operation: %s", operation.description)
    log.debug("vertical pipeline: %s", operation.pipeline)
    return operation


# --- vertical conversion -------------------------------------------------


class VerticalConverter:
    """Applies one selected PROJ operation to arrays of pixel-centre heights.

    A single :class:`~pyproj.Transformer` is *not* thread-safe, so the converter
    keeps one transformer per thread, each rebuilt from the same PROJ pipeline
    string.  Every thread therefore runs the identical operation while still
    honouring "reuse one initialized PROJ transformer" per worker.
    """

    def __init__(self, operation: VerticalOperation) -> None:
        self.operation = operation
        self._local = threading.local()

    @property
    def transformer(self) -> Transformer:
        transformer = getattr(self._local, "transformer", None)
        if transformer is None:
            transformer = Transformer.from_pipeline(self.operation.pipeline)
            self._local.transformer = transformer
        return transformer

    def transform_heights(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        context: str = "",
    ) -> np.ndarray:
        """Convert source heights at ``(x, y)`` onto the output datum.

        ``x``/``y`` are EPSG:5514 pixel-centre coordinates and ``z`` holds source
        Bpv heights.  Only the transformed Z values are returned; the horizontal
        components are discarded after being checked for drift.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        if x.shape != y.shape or x.shape != z.shape:
            raise CrsError(
                f"x, y and z must have identical shapes, got {x.shape}, {y.shape}, {z.shape}"
            )
        if x.size == 0:
            return np.empty_like(z)

        try:
            out_x, out_y, out_z = self.transformer.transform(
                x.ravel(), y.ravel(), z.ravel(), errcheck=True
            )
        except Exception as exc:
            where = f" ({context})" if context else ""
            raise NonFiniteTransformError(
                f"vertical transformation failed{where} using "
                f"{self.operation.description!r}: {exc}"
            ) from exc

        out_x = np.asarray(out_x)
        out_y = np.asarray(out_y)
        out_z = np.asarray(out_z)

        if not np.all(np.isfinite(out_z)):
            bad = int(np.count_nonzero(~np.isfinite(out_z)))
            where = f" ({context})" if context else ""
            raise NonFiniteTransformError(
                f"vertical transformation produced {bad} non-finite height(s) of "
                f"{out_z.size}{where}; the requested area is probably outside the "
                f"support of {self.operation.description!r}"
            )

        if self.operation.target.preserves_horizontal:
            drift = max(
                float(np.max(np.abs(out_x - x.ravel()))),
                float(np.max(np.abs(out_y - y.ravel()))),
            )
            if drift > HORIZONTAL_PRESERVATION_TOLERANCE_M:
                where = f" ({context})" if context else ""
                raise CrsError(
                    f"vertical-only conversion moved horizontal coordinates by "
                    f"{drift:.6f} m{where}, exceeding the "
                    f"{HORIZONTAL_PRESERVATION_TOLERANCE_M} m tolerance"
                )

        return out_z.reshape(z.shape)


def build_converter(
    datum: str,
    *,
    proj_data_dir: Path | None = None,
    network_enabled: bool = False,
    configure: bool = True,
) -> VerticalConverter:
    """Configure PROJ (once) and return a converter for ``datum``."""
    if configure:
        configure_proj(proj_data_dir, network_enabled=network_enabled)
    return VerticalConverter(
        select_vertical_operation(datum, network_enabled=network_enabled)
    )


# --- horizontal helpers --------------------------------------------------


def _transformer_4326_to(target_epsg: int) -> Transformer:
    return Transformer.from_crs(QUERY_CRS, f"EPSG:{target_epsg}", always_xy=True)


def densified_boundary(
    west: float, south: float, east: float, north: float, samples_per_edge: int
) -> tuple[np.ndarray, np.ndarray]:
    """Points along the edges of a lon/lat box, including all four corners.

    Projection curvature means the projected bounding box of a large region is
    not the projection of its corners, so the boundary is sampled before the
    projected extent is derived (specification section 7.2).
    """
    n = max(int(samples_per_edge), 2)
    lon = np.linspace(west, east, n)
    lat = np.linspace(south, north, n)
    xs = np.concatenate([lon, lon, np.full(n, west), np.full(n, east)])
    ys = np.concatenate([np.full(n, south), np.full(n, north), lat, lat])
    return xs, ys


def wgs84_bounds_to_projected(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    target_epsg: int = 5514,
    samples_per_edge: int = 64,
) -> tuple[float, float, float, float]:
    """Project a WGS84 box into ``target_epsg`` without clipping its interior."""
    lons, lats = densified_boundary(west, south, east, north, samples_per_edge)
    xs, ys = _transformer_4326_to(target_epsg).transform(lons, lats, errcheck=True)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        raise CrsError(
            f"could not project bounds ({west}, {south}, {east}, {north}) "
            f"into EPSG:{target_epsg}: transformation returned non-finite values"
        )
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


# --- validation helpers --------------------------------------------------

#: A handful of well-spread Czech locations used as a transformer smoke test.
KNOWN_POINTS_WGS84: tuple[tuple[str, float, float, float], ...] = (
    ("prague", 14.42, 50.08, 200.0),
    ("brno", 16.61, 49.20, 220.0),
    ("ostrava", 18.29, 49.84, 210.0),
    ("plzen", 13.38, 49.75, 310.0),
    ("snezka", 15.74, 50.74, 1600.0),
)


@dataclass(frozen=True)
class KnownPointResult:
    name: str
    x: float
    y: float
    z_source: float
    z_output: float

    @property
    def delta(self) -> float:
        return self.z_output - self.z_source


def check_known_points(
    converter: VerticalConverter,
    points: Sequence[tuple[str, float, float, float]] = KNOWN_POINTS_WGS84,
) -> list[KnownPointResult]:
    """Push a few reference points through the real transformer.

    Rejects ``inf``/``nan`` results, which is what specification section 17.4
    asks the pipeline to do before any raster work begins.
    """
    to_projected = _transformer_4326_to(5514)
    results: list[KnownPointResult] = []
    for name, lon, lat, z in points:
        x, y = to_projected.transform(lon, lat, errcheck=True)
        z_out = float(
            converter.transform_heights(
                np.array([x]), np.array([y]), np.array([float(z)]), context=f"known point {name}"
            )[0]
        )
        results.append(KnownPointResult(name=name, x=float(x), y=float(y), z_source=float(z), z_output=z_out))
    return results
