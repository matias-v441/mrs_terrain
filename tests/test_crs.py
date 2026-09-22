"""Vertical transformation handling (specification sections 3, 4 and 17.4)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer

from heightmap_prep.crs import (
    HORIZONTAL_PRESERVATION_TOLERANCE_M,
    SOURCE_HORIZONTAL_CRS,
    SOURCE_VERTICAL_CRS,
    VERTICAL_TARGETS,
    VerticalConverter,
    build_converter,
    check_known_points,
    configure_proj,
    crs_matches,
    densified_boundary,
    epsg_code_of,
    select_vertical_operation,
    wgs84_bounds_to_projected,
)
from heightmap_prep.errors import CrsError, NonFiniteTransformError

# The degraded CRS the ČÚZK ImageServer actually returns.
LOCAL_CS_WKT = (
    'LOCAL_CS["S-JTSK / Krovak East North",UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","5514"]]'
)


# --- CRS identity ---------------------------------------------------------


def test_epsg_code_of_a_normal_crs() -> None:
    assert epsg_code_of("EPSG:5514") == 5514


def test_epsg_code_is_recovered_from_a_degraded_local_cs() -> None:
    # to_epsg() gives up on this WKT; the authority node still identifies it.
    assert epsg_code_of(LOCAL_CS_WKT) == 5514
    assert crs_matches(LOCAL_CS_WKT, 5514)
    assert not crs_matches(LOCAL_CS_WKT, 4326)


def test_epsg_code_of_nothing() -> None:
    assert epsg_code_of(None) is None


# --- targets --------------------------------------------------------------


def test_both_required_output_datums_are_defined() -> None:
    assert set(VERTICAL_TARGETS) == {"egm96", "wgs84-ellipsoid"}
    assert VERTICAL_TARGETS["egm96"].vertical_crs == "EPSG:5773"
    assert VERTICAL_TARGETS["egm96"].preserves_horizontal is True
    assert VERTICAL_TARGETS["wgs84-ellipsoid"].vertical_crs == "EPSG:4979"


def test_unknown_datum_is_rejected() -> None:
    with pytest.raises(CrsError, match="unsupported vertical datum"):
        select_vertical_operation("egm2008")


# --- operation selection --------------------------------------------------


def test_selected_operation_uses_the_cr2005_grid(proj_dir: Path) -> None:
    configure_proj(proj_dir, network_enabled=False)
    operation = select_vertical_operation("egm96")
    # PROJ names the step "ETRS89 to Baltic 1957 height (2)"; the CR-2005
    # transformation is identified by the grid file it pulls in.
    assert "cz_cuzk_CR-2005.tif" in operation.grid_names
    assert "Baltic 1957 height" in operation.description
    assert "us_nga_egm96_15.tif" in operation.grid_names
    assert all(grid.available for grid in operation.grids)
    assert operation.network_enabled is False
    assert operation.source_crs == f"{SOURCE_HORIZONTAL_CRS}+{SOURCE_VERTICAL_CRS}"


def test_pipeline_string_round_trips(proj_dir: Path) -> None:
    configure_proj(proj_dir, network_enabled=False)
    operation = select_vertical_operation("egm96")
    rebuilt = Transformer.from_pipeline(operation.pipeline)
    x, y, z = -743_011.72, -1_043_823.18, 200.0
    assert rebuilt.transform(x, y, z, errcheck=True)[2] == pytest.approx(
        VerticalConverter(operation).transform_heights(
            np.array([x]), np.array([y]), np.array([z])
        )[0]
    )


# --- conversion -----------------------------------------------------------


def test_bpv_to_egm96_is_a_small_spatially_varying_shift(proj_dir: Path) -> None:
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    results = check_known_points(converter)
    deltas = [r.delta for r in results]
    assert all(math.isfinite(d) for d in deltas)
    # Over Czechia the Bpv/EGM96 separation is well under a metre ...
    assert all(abs(d) < 1.0 for d in deltas)
    # ... but it is not a constant offset (specification section 4.1).
    assert max(deltas) - min(deltas) > 0.05


def test_bpv_to_wgs84_ellipsoid_matches_the_czech_geoid_undulation(proj_dir: Path) -> None:
    converter = build_converter("wgs84-ellipsoid", proj_data_dir=proj_dir)
    deltas = [r.delta for r in check_known_points(converter)]
    assert all(40.0 < d < 50.0 for d in deltas)


def test_horizontal_coordinates_survive_a_vertical_only_conversion(proj_dir: Path) -> None:
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    x = np.array([-743_011.72, -742_000.0, -744_000.0])
    y = np.array([-1_043_823.18, -1_044_000.0, -1_042_000.0])
    z = np.array([200.0, 210.0, 220.0])
    out = converter.transform_heights(x, y, z)
    # transform_heights only returns Z, so re-run the operation to inspect x/y.
    ox, oy, _ = converter.transformer.transform(x, y, z, errcheck=True)
    assert np.max(np.abs(np.asarray(ox) - x)) < HORIZONTAL_PRESERVATION_TOLERANCE_M
    assert np.max(np.abs(np.asarray(oy) - y)) < HORIZONTAL_PRESERVATION_TOLERANCE_M
    assert out.shape == z.shape


def test_conversion_preserves_array_shape(proj_dir: Path) -> None:
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    x = np.full((3, 4), -743_011.72)
    y = np.full((3, 4), -1_043_823.18)
    z = np.full((3, 4), 200.0)
    assert converter.transform_heights(x, y, z).shape == (3, 4)


def test_empty_input_is_handled(proj_dir: Path) -> None:
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    empty = np.array([])
    assert converter.transform_heights(empty, empty, empty).size == 0


def test_mismatched_shapes_are_rejected(proj_dir: Path) -> None:
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    with pytest.raises(CrsError, match="identical shapes"):
        converter.transform_heights(np.zeros(3), np.zeros(4), np.zeros(3))


def test_points_far_outside_the_grid_raise_rather_than_return_inf(proj_dir: Path) -> None:
    converter = build_converter("egm96", proj_data_dir=proj_dir)
    # Somewhere in the Pacific, far outside the Czech CR-2005 grid.
    with pytest.raises(NonFiniteTransformError):
        converter.transform_heights(
            np.array([5.0e6]), np.array([5.0e6]), np.array([100.0])
        )


def test_transformers_are_per_thread(proj_dir: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    converter = build_converter("egm96", proj_data_dir=proj_dir)
    seen: list[int] = []

    def run() -> float:
        seen.append(id(converter.transformer))
        return float(
            converter.transform_heights(
                np.array([-743_011.72]), np.array([-1_043_823.18]), np.array([200.0])
            )[0]
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: run(), range(8)))

    assert len(set(seen)) > 1  # several threads, several transformers
    assert len(set(round(r, 9) for r in results)) == 1  # identical answers


# --- horizontal helpers ---------------------------------------------------


def test_densified_boundary_includes_the_corners() -> None:
    xs, ys = densified_boundary(14.2, 49.95, 14.7, 50.25, samples_per_edge=8)
    points = set(zip(xs.tolist(), ys.tolist()))
    for corner in ((14.2, 49.95), (14.2, 50.25), (14.7, 50.25), (14.7, 49.95)):
        assert corner in points


@pytest.mark.parametrize(
    "box",
    [
        (12.0, 48.5, 18.9, 51.1),  # Czechia: corners already bound the area
        (5.0, 45.0, 25.0, 55.0),
        (14.0, 40.0, 15.0, 60.0),  # tall: Krovak curvature bites hard
    ],
)
def test_densification_never_shrinks_the_projected_extent(box) -> None:
    dense = wgs84_bounds_to_projected(*box, samples_per_edge=512)
    sparse = wgs84_bounds_to_projected(*box, samples_per_edge=2)  # corners only
    assert dense[0] <= sparse[0] and dense[1] <= sparse[1]
    assert dense[2] >= sparse[2] and dense[3] >= sparse[3]


def test_densification_recovers_area_that_corner_projection_would_clip() -> None:
    # Krovak is an oblique conic: along a meridian the projected northing bulges
    # past both corners, so corner-only projection would clip the region
    # (specification section 7.2).
    box = (14.0, 40.0, 15.0, 60.0)
    dense = wgs84_bounds_to_projected(*box, samples_per_edge=512)
    sparse = wgs84_bounds_to_projected(*box, samples_per_edge=2)
    assert dense[3] - sparse[3] > 1000.0


def test_projected_bounds_land_inside_czechia() -> None:
    west, south, east, north = wgs84_bounds_to_projected(14.2, 49.95, 14.7, 50.25)
    assert -905_000 < west < -431_000
    assert -1_228_000 < south < -935_000
    assert east > west and north > south


def test_configure_proj_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(CrsError, match="does not exist"):
        configure_proj(tmp_path / "absent")
