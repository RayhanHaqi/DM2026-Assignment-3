# NYCU Data Mining (Spring 2026) Assignment 3: Human Activity Recognition

Kaggle competition: predict activity label (0–5) from wrist accelerometer data.

**Student:** Muhammad Rayhan Athaillah — ID **313540001**  
**GitHub:** https://github.com/RayhanHaqi/DM2026-Assignment-3  
**Competition:** `nycu-data-mining-assignment-3` (Kaggle display name must be **313540001**)

## Final Best

| Property | Value |
|----------|-------|
| **Public score** | **0.7897** |
| **Submission** | `output/submission_tabpfn_v3_cal_ft_c08_c1092_b18_t058_20260609_094157_01.csv` |
| **Model** | TabPFN V3 on 91 targeted-temporal features + boost180 post-hoc calibration |
| **Recipe** | `cut0=0.80`, `cut1=0.92`, `boost=1.80`, `temp=0.58` |

Score progression: LightGBM 0.7653 → XGB targeted temporal 0.7691 → TabPFN 0.7823 → TabPFN V3 0.7830 → boost180 cal 0.7883 → **finetune cut1=0.92: 0.7897**.

Full experiment log: `output/SUBMISSIONS.md`. Analysis notebook: `HAR_Analysis.ipynb`. PDF report: `report/report.pdf` (submit to E3 as `DM_asg3_313540001.pdf`).

## File Structure

```text
.
├── HAR_Analysis.ipynb              # EDA, baselines, final Kaggle story (§6–§7)
├── README.md
├── AGENTS.md                       # Agent/dev quick reference
├── report/
│   └── report.tex                  # Assignment PDF source
├── data/
│   ├── train/train/User_001-060/   # 11,020 labeled CSV files
│   ├── test/test/User_061-100/     # 6,849 unlabeled CSV files
│   └── sample_submission.csv
├── model/
│   ├── train.py                    # XGB/LGB Optuna + GroupKFold CV
│   ├── utils.py                    # 42-feature aggregation, submissions
│   ├── tabpfn_model.py             # TabPFN fit/OOF/predict
│   ├── v3_probability.py           # Post-hoc class priors + temperature
│   ├── prob_blend.py               # Probability mixing helpers
│   └── ...                         # sequence, CNN, Hjorth, validation, etc.
├── scripts/
│   ├── run_balanced_candidates.py  # Tree / temporal / plateau experiments
│   ├── cache_tabpfn_v3_probs.py    # One-time TabPFN V3 prob cache (GPU)
│   ├── run_tabpfn_v3_calibration.py
│   ├── rank_finetune_cuts.py       # Rank calibration near 0.7897 anchor
│   └── run_tabpfn_finetune.py
└── output/
    ├── submission_*.csv            # Kaggle submissions
    ├── SUBMISSIONS.md              # Score tracker
    └── prob_cache/                 # Cached TabPFN probabilities
```

## Running

Install dependencies (downloads Kaggle data via `setup.py`):

```bash
pip install -e .
```

Kaggle auth required for data download: place credentials at `~/.kaggle/kaggle.json` or `~/.kaggle/access_token` before install.

For TabPFN experiments, also install: `pip install tabpfn`

Execute the analysis notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace HAR_Analysis.ipynb
```

### Reproduce final submission (cache-only calibration)

One-time GPU step to cache TabPFN V3 probabilities:

```bash
PYTHONPATH=. python scripts/cache_tabpfn_v3_probs.py --output-path output/prob_cache/tabpfn_v3_91f1.npz --device cuda
```

Rank calibration candidates without uploading to Kaggle:

```bash
PYTHONPATH=. python scripts/rank_finetune_cuts.py --phase1 --top 25 --write-top 0
```

### Tree experiments (historical path)

```bash
python scripts/run_balanced_candidates.py --targeted-20260522 --tree-trials 150 --n-jobs 4
```

## Assignment Deliverables

1. **Kaggle CSV** — best submission above; Id/Label format, 6849 rows.
2. **Public GitHub** — this repository with runnable code and instructions.
3. **Report PDF** — `report/report.pdf` → upload to E3 as `DM_asg3_313540001.pdf` (GitHub link inside report).
4. **Kaggle team name** — display name **313540001**.

Report sections (40%): EDA, preprocessing with quantified gains, temporal/label alignment, ablation study — see `report/report.tex` and `HAR_Analysis.ipynb`.

## What Did Not Work (post-0.7897)

Micro-calibration near the anchor (0.7883), low-confidence XGB overrides (0.7880), TabPFN+XGB probability blends (0.7794), and coarse re-calibration on the same cache (0.7857–0.7858) all regressed publicly despite small grouped-OOF gains.
