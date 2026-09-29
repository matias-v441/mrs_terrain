# Sourced by the scripts in the repository root, from the repository root.
#
# Makes sure the Python environment in .venv and the PROJ grids in .proj
# exist, creating them on first use.

PROJ_GRIDS=(cz_cuzk_CR-2005.tif us_nga_egm96_15.tif)

if [[ ! -x .venv/bin/heightmap-prep ]]; then
  echo "==> Creating the Python environment in .venv"
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -e ".[dev]"
fi

mkdir -p .proj
for grid in "${PROJ_GRIDS[@]}"; do
  if [[ ! -f ".proj/$grid" ]]; then
    echo "==> Downloading PROJ grid $grid"
    curl --fail --silent --show-error --location -o ".proj/$grid.part" "https://cdn.proj.org/$grid"
    mv ".proj/$grid.part" ".proj/$grid"
  fi
done
