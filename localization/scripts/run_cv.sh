#!/usr/bin/env bash
# Cross-validation (D-021): for each fold, build its map (if missing), run the whole bag continuously
# from the first pose, score it; then pool the folds. Folds run one after another so the runtime
# numbers are measured on an otherwise idle machine.
#
#   scripts/run_cv.sh [configs/cv_f1.yaml configs/cv_f2.yaml configs/cv_f3.yaml]
#   CV_OUT=results/cv/cv_score_swapped scripts/run_cv.sh configs/cv_f1_swapped.yaml ...   # pooled score name
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
if [ $# -gt 0 ]; then CONFIGS=("$@"); else CONFIGS=(configs/cv_f1.yaml configs/cv_f2.yaml configs/cv_f3.yaml); fi
cfg() { $PY -c "import sys; sys.path.insert(0, '.'); from camloc import config; c = config.load('$1'); print($2)"; }

for C in "${CONFIGS[@]}"; do
  ART=$(cfg "$C" "c['artifacts']"); MAP=$(cfg "$C" "c['reloc']['map_name']")
  BAG=$(cfg "$C" "c['bag']"); RES=$(cfg "$C" "c['results']")
  mkdir -p "$RES/fullrun"
  echo "=== $C: map $MAP ($(date +%T))"
  [ -f "$ART/$MAP/meta.json" ] || $PY scripts/build_map.py --config "$C" --name "$MAP"
  echo "=== $C: continuous run ($(date +%T))"
  $PY scripts/run.py --config "$C" --bag "$BAG" --out "$RES/fullrun/submission.csv"
  $PY scripts/score_run.py "$RES/fullrun/submission.csv" --config "$C" | tee "$RES/fullrun/score.txt"
  $PY scripts/plot_lap.py --config "$C" --run "$RES/fullrun/submission.csv" --laps $(cfg "$C" "' '.join(c['split']['val'])")
done
CV_OUT=${CV_OUT:-results/cv/cv_score}
$PY scripts/cv_score.py "${CONFIGS[@]}" --out "$CV_OUT.json" | tee "$CV_OUT.txt"
echo "=== done ($(date +%T))"
