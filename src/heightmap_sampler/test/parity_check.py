#!/usr/bin/env python3
"""Check that the C++ sampler reproduces examples/sampler.py exactly.

This is the test that matters: the C++ library exists only to give ROS nodes
the answers the Python reference gives.  It is deliberately outside the colcon
test set, because it needs a real prepared dataset (which is gitignored) and
the project's Python environment.

    source <workspace>/install/setup.bash
    .venv/bin/python src/heightmap_sampler/test/parity_check.py ./prepared

Points are drawn from the dataset's own world bounds, so most of them land on
real data; add --outside to also probe beyond the prepared area, where both
implementations must agree that there is no height.
"""

from __future__ import annotations

import argparse
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


def query_bounds(dataset: Path) -> tuple[float, float, float, float]:
    manifest = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8"))
    worlds = manifest.get("world_bounds_wgs84") or {}
    if not worlds:
        raise SystemExit(f"{dataset}/dataset.yaml has no world_bounds_wgs84 to sample within")
    west = min(b[0] for b in worlds.values())
    south = min(b[1] for b in worlds.values())
    east = max(b[2] for b in worlds.values())
    north = max(b[3] for b in worlds.values())
    return west, south, east, north


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="a prepared dataset directory")
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--outside",
        action="store_true",
        help="grow the sampled area so that points fall outside the prepared data too",
    )
    parser.add_argument(
        "--executable",
        default="ros2 run heightmap_sampler sample_height",
        help="how to invoke the C++ CLI",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    executable = args.executable.split()

    west, south, east, north = query_bounds(dataset)
    if args.outside:
        margin_x = (east - west) * 0.5
        margin_y = (north - south) * 0.5
        west, east = west - margin_x, east + margin_x
        south, north = south - margin_y, north + margin_y

    reference = HeightSampler(dataset)
    rng = random.Random(args.seed)

    mismatches = 0
    both_missing = 0
    worst = 0.0
    for _ in range(args.points):
        lon = rng.uniform(west, east)
        lat = rng.uniform(south, north)

        expected = reference.sample(lon, lat)
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
        f"{args.points} points: {sampled} with data, {both_missing} with none in both, "
        f"worst difference {worst:.3e} m, {mismatches} mismatches"
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
