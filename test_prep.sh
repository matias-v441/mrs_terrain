#!/usr/bin/env bash
# Run the heightmap-prep (Python) test suite.
#
#   ./test_prep.sh                  # extra arguments go to pytest,
#   ./test_prep.sh -k pipeline -x   # e.g. -k, -x or -v
#
# The first run creates .venv and downloads the PROJ grids, without which the
# tests that need a real vertical transformation would be skipped.  The tests
# never touch the network themselves.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

source scripts/python_env.sh

.venv/bin/python -m pytest "$@"
