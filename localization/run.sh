#!/usr/bin/env bash
# One command from the raw data to the submission CSV (D-016).
#
#   ./run.sh <run.mcap> [out.csv]
#
# On first use it builds the artifacts from the training bag named in the config (caches, masks and
# the relocalization map from all training laps), then localizes <run.mcap> causally.
#   CONFIG=configs/final.yaml (default)   PYTHON=python (default)
set -euo pipefail
cd "$(dirname "$0")"
RUN_BAG=$(cd "$(dirname "${1:?usage: ./run.sh <run.mcap> [out.csv]}")" && pwd)/$(basename "$1")
OUT=${2:-results/submission.csv}
CONFIG=${CONFIG:-configs/final.yaml}
PY=${PYTHON:-python}
cfg() { $PY -c "import sys; sys.path.insert(0, '.'); from camloc import config; c = config.load('$CONFIG'); print($1)"; }
ART=$(cfg "c['artifacts']")
MAP=$(cfg "c['reloc']['map_name']")

[ -f "$ART/state.npz" ]         || $PY scripts/prepare.py --config "$CONFIG"
[ -f "$ART/masks/fl.png" ]      || $PY scripts/build_masks.py --config "$CONFIG"
[ -f "$ART/$MAP/meta.json" ]    || $PY scripts/build_map.py --config "$CONFIG" --name "$MAP"
mkdir -p "$(dirname "$OUT")"
$PY scripts/run.py --bag "$RUN_BAG" --out "$OUT" --config "$CONFIG"
