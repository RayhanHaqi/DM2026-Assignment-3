# NYCU Data Mining (Spring 2026) Assignment 3: Human Activity Recognition

Kaggle competition: predict activity label (0–5) from wrist accelerometer data.

## File Structure

```text
.
├── HAR_Analysis.ipynb           # Main notebook: EDA, baseline, submission
├── README.md
├── .gitignore
├── data/
│   ├── train/train/User_001-060/  # 11,020 labeled CSV files
│   ├── test/test/User_061-100/    # 6,849 unlabeled CSV files
│   └── sample_submission.csv      # Kaggle submission template
├── model/
│   ├── __init__.py
│   └── utils.py                 # Data loading, aggregation, submission
└── output/                      # Generated submission CSVs
```

## Running

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter nbconvert --to notebook --execute HAR_Analysis.ipynb
```
