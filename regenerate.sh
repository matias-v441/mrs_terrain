#!/usr/bin/env bash
# Regenerate dataset/ from the world configs in worlds/, then commit it.
# heightmap_sampler bundles it through its src/heightmap_sampler/dataset symlink.
#
#   ./regenerate.sh                 # extra arguments go to heightmap-prep,
#   ./regenerate.sh --workers 4     # e.g. --workers or --log-level debug
#
# The dataset includes the ČÚZK orthophoto (rgb/) and the worlds database
# (worlds.sqlite).  The first run creates .venv and downloads the PROJ grids.
# Downloaded source rasters and imagery are cached in cache/, so regenerating
# after a small change is quick.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATASET=dataset
VERTICAL_DATUM=wgs84-ellipsoid

source scripts/python_env.sh

# Build into a fresh directory and swap it in only once it is complete, so a
# failed run leaves the committed dataset alone and a successful one leaves no
# tiles behind for worlds that have since been removed.
staging=$(mktemp -d "_regenerate.XXXXXX")
trap 'rm -rf "$staging"' EXIT

.venv/bin/heightmap-prep worlds "$staging/dataset" \
  --vertical-datum "$VERTICAL_DATUM" \
  --proj-data-dir .proj \
  --cache-dir cache \
  --include-rgb \
  "$@"

rm -rf "$DATASET"
mv "$staging/dataset" "$DATASET"

echo
echo "==> $DATASET regenerated; commit it:"
git status --short -- "$DATASET"
