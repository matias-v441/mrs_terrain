#!/usr/bin/env bash
# Build and test the heightmap_sampler ROS 2 packages.
#
#   ./test_sampler.sh
#
# Builds heightmap_sampler_msgs and heightmap_sampler in a private colcon
# workspace under _colcon/ (so it never interferes with the workspace this
# repository sits in), runs their tests -- unit tests, the bundled-dataset
# tests and the linters -- and finally checks the C++ sampler against the
# Python reference sampler on the bundled dataset.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

WS=_colcon

if [[ -z "${ROS_DISTRO:-}" ]]; then
  # ROS setup scripts read unset variables.
  set +u
  source /opt/ros/jazzy/setup.bash
  set -u
fi

mkdir -p "$WS"
touch "$WS/COLCON_IGNORE"
colcon_args=(--base-paths src --build-base "$WS/build" --install-base "$WS/install")

echo "==> Building"
colcon --log-base "$WS/log" build "${colcon_args[@]}" --packages-up-to heightmap_sampler

echo "==> Running the colcon tests"
colcon --log-base "$WS/log" test "${colcon_args[@]}" --packages-select heightmap_sampler
colcon test-result --test-result-base "$WS/build/heightmap_sampler" --verbose

echo "==> Checking the C++ sampler against the Python reference"
source scripts/python_env.sh
set +u
source "$WS/install/setup.bash"
set -u
.venv/bin/python src/heightmap_sampler/test/parity_check.py src/heightmap_sampler/dataset \
  --executable "$WS/install/heightmap_sampler/lib/heightmap_sampler/sample_height" \
  --points 300
