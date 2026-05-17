# DM2026-Assignment-3 — HAR Classification

## Quick Start

```bash
jupyter nbconvert --to notebook --execute --inplace HAR_Analysis.ipynb
# or open Jupyter and Run All Cells
```

## Architecture

```
model/
├── train.py      # tune_xgboost, tune_lightgbm, cv_evaluate (Optuna + GroupKFold-5)
├── utils.py      # data loading (42 base features), generate_submission (auto-tracks to SUBMISSIONS.md)
└── (activations, gradients, metrics, linear_model from base — unused, for DM2026-Assignment-1/2 compat)
```

**`model/train.py`**: Uses Optuna (TPESampler + MedianPruner) to tune XGBClassifier + LGBMClassifier. GroupKFold-5 by user. SMOTE applied inside each fold (prevents data leakage). 150 trials each.

**`model/utils.py`**: `_aggregate_file()` extracts 42 features (6 cols × 7 stats: mean, std, min, max, q25, q50, q75). `generate_submission()` auto-versions files (v1, v2, v3...) and appends to `output/SUBMISSIONS.md`.

**Data**: 60 train users + 40 test users. Each CSV = 300 timesteps of accelerometer readings (mean_x, mean_y, mean_z, std_x, std_y, std_z). 6 activity classes (0-5). Test set has no labels.

## Kaggle

- Competition: `nycu-data-mining-assignment-3` (invite-only)
- Metric: Accuracy (higher is better)
- 5 submissions/day limit
- Output: `output/submission_*.csv`

## Submission History

| File | Score | Features | Tuning | Notes |
|------|-------|----------|--------|-------|
| submission_lgb.csv | **0.7645** | 42 | 50 trials, no SMOTE | best v1 |
| submission_xgb.csv | 0.7358 | 42 | 50 trials, no SMOTE | |
| submission_ensemble.csv | 0.7587 | 42 | soft vote | |
| submission_baseline_rf.csv | 0.7597 | 42 | sklearn RF | |
| submission_lgb_v2.csv | 0.7453 | 82 | 50 trials, no SMOTE | WORSE — extra features hurt |
| submission_xgb_v2.csv | 0.7535 | 82 | 50 trials, no SMOTE | |
| submission_ensemble_v2.csv | ? | 82 | soft vote | NOT YET SUBMITTED |
| submission_baseline_rf_v2.csv | ? | 82 | sklearn RF | NOT YET SUBMITTED |

## Current State (v3 — NOT YET RUN)

- **Features**: 42 base (reverted from 82 — skew/kurt/jerk/SMA/corr hurt LightGBM)
- **Trials**: 150 per model (up from 50)
- **SMOTE**: Enabled inside each fold
- Expected: LightGBM > 0.7645

## Running v3

```bash
git pull
rm -f data/train_*.npy data/test_*.npy  # clear cached features
# Run All Cells in HAR_Analysis.ipynb
```

## Dependencies

- numpy, pandas, matplotlib, scikit-learn, xgboost, lightgbm, optuna, imbalanced-learn
- Install: `pip install -e .` (runs setup.py which also downloads Kaggle data)
- Kaggle auth: `~/.kaggle/kaggle.json` or `~/.kaggle/access_token` (auto-converted)
