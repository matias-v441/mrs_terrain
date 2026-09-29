#!/usr/bin/env python3
"""Check that the C++ sampler reproduces examples/sampler.py exactly.

This is the test that matters: the C++ library exists only to give ROS nodes
the answers the Python reference gives.  It is deliberately outside the colcon
test set, because it needs a real prepared dataset (which is gitignored) and
the project's Python environment.

    source <workspace>/install/setup.bash
    .venv/bin/python src/heightmap_sampler/test/parity_check.py ./prepared

First, every point in the dataset's test points file (the safety area corners,
with the heights heightmap-prep sampled there) must come out of both samplers
with exactly the recorded height.  Then random points are drawn within
--radius metres of randomly chosen test points, so they land both on and off
the prepared data; where there is none, both implementations must agree on that.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import random
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "examples"))

from sampler import HeightSampler  # noqa: E402


def cpp_sample(executable: list[str], dataset: Path, lon: float, lat: float) -> float:
    """One height from the C++ CLI, or NaN when it reports none."""
    result = subprocess.run(
        [*executable, repr(lon), repr(lat), str(dataset)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sample_height failed: {result.stderr.strip()}")
    return float(result.stdout.splitlines()[0])


def reference_sample(reference: HeightSampler, lon: float, lat: float) -> float:
    """One height from the Python reference, without whatever it prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return reference.sample(lon, lat)


def test_points(dataset: Path) -> list[tuple[float, float, float]]:
    """The dataset's ``(lat, lon, height)`` reference points."""
    manifest = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8"))
    entry = manifest.get("test_points") or {}
    if not entry.get("path"):
        raise SystemExit(f"{dataset}/dataset.yaml declares no test points")
    rows = (dataset / entry["path"]).read_text(encoding="utf-8").split()
    return [tuple(float(v) for v in row.split(",")) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="a prepared dataset directory")
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--radius",
        type=float,
        default=100.0,
        help="how far from a test point random points may fall, in metres (default: 100)",
    )
    parser.add_argument(
        "--executable",
        default="ros2 run heightmap_sampler sample_height",
        help="how to invoke the C++ CLI",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    executable = args.executable.split()

    points = test_points(dataset)
    reference = HeightSampler(dataset)

    mismatches = 0
    for lat, lon, height in points:
        expected = reference_sample(reference, lon, lat)
        actual = cpp_sample(executable, dataset, lon, lat)
        if expected != height or not abs(actual - height) <= args.tolerance:
            print(
                f"MISMATCH test point {lat!r}, {lon!r}: file={height!r} "
                f"python={expected!r} cpp={actual!r}"
            )
            mismatches += 1

    rng = random.Random(args.seed)
    metres_per_degree = 111_320.0

    both_missing = 0
    worst = 0.0
    for _ in range(args.points):
        centre_lat, centre_lon, _ = rng.choice(points)
        dlat = args.radius / metres_per_degree
        dlon = dlat / math.cos(math.radians(centre_lat))
        lat = centre_lat + rng.uniform(-dlat, dlat)
        lon = centre_lon + rng.uniform(-dlon, dlon)

        expected = reference_sample(reference, lon, lat)
        actual = cpp_sample(executable, dataset, lon, lat)

        if math.isnan(expected) and math.isnan(actual):
            both_missing += 1
            continue
        if math.isnan(expected) != math.isnan(actual):
            print(f"MISMATCH availability at {lon!r}, {lat!r}: python={expected} cpp={actual}")
            mismatches += 1
            continue

        difference = abs(expected - actual)
        worst = max(worst, difference)
        if difference > args.tolerance:
            print(f"MISMATCH value at {lon!r}, {lat!r}: python={expected!r} cpp={actual!r}")
            mismatches += 1

    reference.close()

    sampled = args.points - both_missing
    print(
        f"{len(points)} test points and {args.points} random points: {sampled} random "
        f"with data, {both_missing} with none in both, worst difference {worst:.3e} m, "
        f"{mismatches} mismatches"
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
