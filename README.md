# NYCU Data Mining (Spring 2026) Assignment 3: Human Activity Recognition

Kaggle competition: predict activity label (0–5) from wrist accelerometer data.

## File Structure

```text
.
├── HAR_Analysis.ipynb           # Report notebook: EDA, baseline, final summary
├── README.md
├── .gitignore
├── data/
│   ├── train/train/User_001-060/  # 11,020 labeled CSV files
│   ├── test/test/User_061-100/    # 6,849 unlabeled CSV files
│   └── sample_submission.csv      # Kaggle submission template
├── model/
│   ├── __init__.py
│   ├── cnn.py                   # 1D CNN raw-sequence experiment
│   ├── sequence.py              # Raw sequence loaders
│   ├── train.py                 # Tree CV/tuning helpers
│   └── utils.py                 # Data loading, aggregation, submission
├── scripts/
│   └── run_balanced_candidates.py # Script-based training experiments
└── output/                      # Generated submission CSVs
```

## Current Best

Best public Kaggle score so far: `0.7653` from `output/submission_lgb_20260518_185105_02.csv`.

May 19 follow-up submissions did not improve it:

| File | Public Score | Notes |
|------|--------------|-------|
| `submission_lgb_acc_no_smote_20260519_053954_01.csv` | 0.7515 | accuracy-tuned LightGBM, no SMOTE |
| `submission_lgb_acc_smote_20260519_055054_01.csv` | 0.7572 | accuracy-tuned LightGBM, SMOTE |
| `submission_cnn_raw_sequence_20260519_060812_01.csv` | 0.6709 | small raw-sequence 1D CNN |

## Running

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter nbconvert --to notebook --execute HAR_Analysis.ipynb
```

For training experiments, prefer the Python script:

```bash
python scripts/run_balanced_candidates.py --tree-trials 60 --cnn-epochs 60 --include-xgb
```
