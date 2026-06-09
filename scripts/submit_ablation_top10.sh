#!/usr/bin/env bash
# Submit top 10 unscored TabPFN ablation CSVs to Kaggle (by shift vs V3).
set -euo pipefail
cd "$(dirname "$0")/.."
COMP="${KAGGLE_COMPETITION:-nycu-data-mining-assignment-3}"
OUT=output
FILES=(
  submission_tabpfn_xgb_blend_20260530_151303_02.csv
  submission_tabpfn_gbdt_blend_20260529_132544_02.csv
  submission_tabpfn_tuned_bal_20260531_133446_01.csv
  submission_tabpfn_tuned_roc_20260531_133600_01.csv
  submission_tabpfn_noise_0.05_20260602_044412_01.csv
  submission_tabpfn_noise_0.02_20260602_044211_01.csv
  submission_tabpfn_noise_0.01_20260602_044011_01.csv
  submission_tabpfn_ft_sub_20e_20260601_113603_01.csv
  submission_tabpfn_ft_sub_30e_20260601_113629_01.csv
  submission_tabpfn_oof_stacking_20260529_132544_01.csv
)
for f in "${FILES[@]}"; do
  path="$OUT/$f"
  if [[ ! -f "$path" ]]; then
    echo "MISSING: $path" >&2
    exit 1
  fi
  msg="ablation: ${f%.csv}"
  echo "Submitting $f ..."
  kaggle competitions submit -c "$COMP" -f "$path" -m "$msg"
done
echo "Done: ${#FILES[@]} submissions."
