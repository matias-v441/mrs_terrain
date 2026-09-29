"""Shared fixtures.

The tests never touch the network: acquisition is exercised through a synthetic
:class:`SyntheticSource` or through an ``httpx`` mock transport.  The tests that
need a real Bpv conversion are skipped unless the PROJ grids named in
specification section 4.2 are available locally.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from heightmap_prep.cuzk import NATIVE_GRID_ORIGIN_X, NATIVE_GRID_ORIGIN_Y
from heightmap_prep.sources import BaseHeightSource
from heightmap_prep.tiling import ProjectedBounds

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_GRIDS = ("cz_cuzk_CR-2005.tif", "us_nga_egm96_15.tif")

#: A small area inside Czech coverage, near Prague, used by most tests.
TEST_AREA = ProjectedBounds(
    west=-743_500.0, south=-1_044_500.0, east=-742_500.0, north=-1_043_500.0
)


def find_proj_dir() -> Path | None:
    """Locate a directory holding the Czech transformation grids, if any."""
    candidates = []
    env = os.environ.get("HEIGHTMAP_PREP_PROJ_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(REPO_ROOT / ".proj")
    for candidate in candidates:
        if all((candidate / name).is_file() for name in REQUIRED_GRIDS):
            return candidate.resolve()
    return None


@pytest.fixture(scope="session")
def proj_dir() -> Path:
    directory = find_proj_dir()
    if directory is None:
        pytest.skip(
            "PROJ grids not installed; put "
            f"{' and '.join(REQUIRED_GRIDS)} in ./.proj or set "
            "HEIGHTMAP_PREP_PROJ_DIR"
        )
    return directory


class SyntheticSource(BaseHeightSource):
    """A deterministic stand-in for the ČÚZK service.

    Produces a smooth, analytically known Bpv surface so tests can assert on
    exact values, and counts requests so request splitting can be observed.
    """

    horizontal_crs = "EPSG:5514"
    vertical_crs = "EPSG:8357"

    def __init__(
        self,
        *,
        resolution_m: float = 2.0,
        nodata: float = -9999.0,
        coverage: ProjectedBounds | None = TEST_AREA,
        max_request_px: int = 512,
        nodata_region: ProjectedBounds | None = None,
        native_origin: tuple[float, float] | None = (
            NATIVE_GRID_ORIGIN_X,
            NATIVE_GRID_ORIGIN_Y,
        ),
    ) -> None:
        self.resolution_m = resolution_m
        self.nodata = nodata
        self._coverage = coverage
        self._native_origin = native_origin
        self.nodata_region = nodata_region
        self.max_request_width_px = max_request_px
        self.max_request_height_px = max_request_px
        self.requests: list[tuple[ProjectedBounds, int, int]] = []

    @property
    def source_id(self) -> str:
        return "synthetic"

    def coverage(self) -> ProjectedBounds | None:
        return self._coverage

    def native_grid_origin(self) -> tuple[float, float] | None:
        return self._native_origin

    @staticmethod
    def height_at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """A gentle synthetic Bpv surface around 300 m."""
        return (
            300.0
            + 0.01 * (x + 743_000.0)
            + 0.005 * (y + 1_044_000.0)
            + 5.0 * np.sin((x + y) / 500.0)
        )

    def _request_block(
        self, bounds: ProjectedBounds, width: int, height: int
    ) -> np.ndarray:
        self.requests.append((bounds, width, height))
        res_x = bounds.width / width
        res_y = bounds.height / height
        xs = bounds.west + (np.arange(width) + 0.5) * res_x
        ys = bounds.north - (np.arange(height) + 0.5) * res_y
        data = self.height_at(xs[np.newaxis, :], ys[:, np.newaxis]).astype(np.float32)
        if self.nodata_region is not None:
            hole = self.nodata_region
            mask = (
                (xs[np.newaxis, :] >= hole.west)
                & (xs[np.newaxis, :] <= hole.east)
                & (ys[:, np.newaxis] >= hole.south)
                & (ys[:, np.newaxis] <= hole.north)
            )
            data = np.where(mask, np.float32(self.nodata), data)
        return data


@pytest.fixture
def synthetic_source() -> SyntheticSource:
    return SyntheticSource()


def mrs_world(points: str, frame: str = "latlon_origin") -> str:
    """An MRS world file whose safety area has the given flat ``points`` list."""
    return (
        "mrs_uav_managers:\n"
        "  world_origin:\n"
        '    units: "LATLON"\n'
        "    origin_x: 50.0785\n"
        "    origin_y: 14.4205\n"
        "  safety_area_manager:\n"
        "    safety_area:\n"
        "      enabled: true\n"
        "      horizontal:\n"
        f'        frame_name: "{frame}"\n'
        f"        points: [{points}]\n"
        "      vertical:\n"
        '        frame_name: "world_origin"\n'
        "        max_z: 30.0\n"
        "        min_z: 1.0\n"
    )


#: A skewed quadrilateral, about 450 m across, inside ``TEST_AREA``.
TEST_WORLD_POINTS = (
    "50.0800925, 14.4166195, 50.0762346, 14.4181500, "
    "50.0771058, 14.4238870, 50.0807978, 14.4225335"
)

#: A second, overlapping safety area east of the first.
SECOND_WORLD_POINTS = (
    "50.0812342, 14.4213118, 50.0790319, 14.4220648, "
    "50.0795531, 14.4259025, 50.0817554, 14.4251496"
)


@pytest.fixture
def world_file(tmp_path: Path) -> Path:
    """A world whose safety area lies inside the synthetic source's coverage."""
    path = tmp_path / "world_testworld.yaml"
    path.write_text(mrs_world(TEST_WORLD_POINTS), encoding="utf-8")
    return path
