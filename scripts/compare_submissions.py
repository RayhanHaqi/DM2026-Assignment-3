#!/usr/bin/env python
"""Compare submission CSVs vs a baseline (agreement, MD5, label distribution)."""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.validation import distribution_max_delta_pp, prediction_distribution, prediction_shift

VALID_LABELS = (0, 1, 2, 3, 4, 5)


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _load_labels(path):
    frame = pd.read_csv(path)
    if "Label" not in frame.columns:
        raise ValueError(f"{path} missing Label column")
    return frame["Label"].astype(int).to_numpy()


def compare_one(baseline_path, candidate_path):
    baseline_preds = _load_labels(baseline_path)
    candidate_preds = _load_labels(candidate_path)
    shift = prediction_shift(candidate_preds, baseline_preds)
    base_dist = prediction_distribution(baseline_preds)
    cand_dist = prediction_distribution(candidate_preds)
    prop_delta = distribution_max_delta_pp(cand_dist, base_dist, len(baseline_preds))
    same_md5 = _md5(baseline_path) == _md5(candidate_path)
    return {
        "candidate": str(candidate_path),
        "md5": _md5(candidate_path)[:8],
        "same_md5_as_baseline": same_md5,
        "changed": shift.changed,
        "total": shift.total,
        "shift_pct": shift.percent,
        "max_class_prop_delta_pp": prop_delta,
        "baseline_dist": base_dist,
        "candidate_dist": cand_dist,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Kaggle submission CSV files.")
    parser.add_argument("--baseline", required=True, help="Baseline submission CSV (e.g. TabPFN V3)")
    parser.add_argument("candidates", nargs="+", help="Candidate CSV paths to compare")
    parser.add_argument(
        "--min-shift-pct",
        type=float,
        default=1.0,
        help="Flag candidates below this %% label change vs baseline",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise SystemExit(f"Baseline not found: {baseline_path}")

    print(f"Baseline: {baseline_path}  md5={_md5(baseline_path)[:8]}")
    base_dist = prediction_distribution(_load_labels(baseline_path))
    print(f"  Label distribution: {base_dist}\n")

    rows = []
    for candidate in args.candidates:
        row = compare_one(baseline_path, Path(candidate))
        rows.append(row)
        flag = []
        if row["same_md5_as_baseline"]:
            flag.append("DUPLICATE_MD5")
        if row["shift_pct"] < args.min_shift_pct:
            flag.append("LOW_SHIFT")
        flag_txt = f"  [{', '.join(flag)}]" if flag else ""
        print(
            f"{Path(row['candidate']).name}\n"
            f"  md5={row['md5']}  shift={row['changed']}/{row['total']} "
            f"({row['shift_pct']:.2f}%)  max class Δ={row['max_class_prop_delta_pp']:.2f} pp"
            f"{flag_txt}\n"
            f"  dist={row['candidate_dist']}\n"
        )

    print("--- Summary (sort by shift_pct desc) ---")
    for row in sorted(rows, key=lambda r: r["shift_pct"], reverse=True):
        print(
            f"{row['shift_pct']:6.2f}%  md5={row['md5']}  "
            f"classΔ={row['max_class_prop_delta_pp']:.2f}pp  "
            f"{Path(row['candidate']).name}"
        )


if __name__ == "__main__":
    main()
