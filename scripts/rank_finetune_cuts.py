#!/usr/bin/env python
"""Rank --finetune-cuts candidates by label delta vs boost180 anchor."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from model.prob_cache import load_v3_prob_cache
from model.submission_audit import gate_submission, load_labels, prediction_distribution
from model.utils import generate_submission
from model.v3_diagnostics import diagnose_vs_v3
from model.v3_probability import apply_class_priors, apply_temperature, decode_labels
from scripts.run_tabpfn_v3_calibration import (
    FINETUNE_BOOST180,
    FINETUNE_CUT0,
    FINETUNE_CUT1,
    FINETUNE_TEMPS,
    _fmt_temp,
)

DEFAULT_CACHE = ROOT / "output" / "prob_cache" / "tabpfn_v3_91f1.npz"
DEFAULT_BASELINE = ROOT / "output" / "submission_tabpfn_v3_20260601_180045_01.csv"
DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_TRACKER = ROOT / "output" / "SUBMISSIONS.md"
ANCHOR_RECIPE = dict(cut0=0.80, cut1=0.90, boost=1.80, temp=0.60)
SUBMITTED_PRED_MD5 = {"9a02efc8", "93972e8b", "a2746db2"}


def _decode(cache: dict, cut0: float, cut1: float, boost: float, temp: float) -> np.ndarray:
    mult = np.array([cut0, cut1, boost, 1.0, boost, boost], dtype=float)
    adjusted = apply_temperature(apply_class_priors(cache["test_proba"], mult), temp)
    return decode_labels(adjusted, cache["classes"])


def _class_counts(preds: np.ndarray) -> dict[int, int]:
    dist = prediction_distribution(preds)
    return {int(k): int(v) for k, v in dist.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank finetune-cut candidates vs boost180.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--write-top", type=int, default=0, help="Write top N unsubmitted boost=1.80 picks")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--relax-class2", action="store_true")
    parser.add_argument(
        "--min-diff-anchor",
        type=float,
        default=0.0,
        help="Skip picks within this %% label change of boost180 anchor (avoid near-dup writes)",
    )
    args = parser.parse_args()

    cache = load_v3_prob_cache(args.cache)
    baseline = load_labels(args.baseline)
    anchor = _decode(cache, **ANCHOR_RECIPE)
    anchor_md5 = hashlib.md5(anchor.astype(int).tobytes()).hexdigest()[:8]
    anchor_c2 = int((anchor == 2).sum())

    rows: list[dict] = []
    seen_md5: set[str] = set()
    for cut0, cut1, boost, temp in itertools.product(
        FINETUNE_CUT0, FINETUNE_CUT1, FINETUNE_BOOST180, FINETUNE_TEMPS
    ):
        preds = _decode(cache, cut0, cut1, boost, temp)
        pred_md5 = hashlib.md5(preds.astype(int).tobytes()).hexdigest()
        if pred_md5 in seen_md5:
            continue
        seen_md5.add(pred_md5)
        pred_prefix = pred_md5[:8]

        diff_anchor = float(np.mean(preds != anchor) * 100.0)
        diff_v3 = float(np.mean(preds != baseline) * 100.0)
        c2 = int((preds == 2).sum())
        diag = diagnose_vs_v3(baseline, cache["test_proba"], preds)
        counts = _class_counts(preds)

        rows.append(
            dict(
                cut0=cut0,
                cut1=cut1,
                boost=boost,
                temp=temp,
                md5=pred_prefix,
                shift_v3=diag.shift_pct,
                diff_anchor=diff_anchor,
                diff_v3=diff_v3,
                c2=c2,
                c2_delta=abs(c2 - anchor_c2),
                submitted=pred_prefix in SUBMITTED_PRED_MD5,
                counts=counts,
            )
        )

    def rank_key(r: dict) -> tuple:
        c2_penalty = 0 if 107 <= r["c2"] <= 109 else abs(r["c2"] - anchor_c2)
        boost_penalty = 0 if r["boost"] == 1.80 else 1
        return (
            r["submitted"],
            boost_penalty,
            c2_penalty,
            r["diff_anchor"],
            abs(r["shift_v3"] - 1.97),
        )

    ranked = sorted(rows, key=rank_key)
    unsubmitted = [r for r in ranked if not r["submitted"]]

    anchor_shift = diagnose_vs_v3(baseline, cache["test_proba"], anchor).shift_pct
    print(f"Anchor boost180: md5={anchor_md5} c2={anchor_c2} shift_vs_v3={anchor_shift:.2f}%")
    print(f"Unique candidates: {len(rows)} | unsubmitted: {len(unsubmitted)}")
    print()
    print(
        f"{'rank':>4}  {'cut0':>4} {'cut1':>4} {'boost':>5} {'t':>4}  "
        f"{'md5':>8}  {'diff180':>7} {'shift':>6} {'c2':>3}  submitted"
    )
    for i, r in enumerate(ranked[: args.top], start=1):
        flag = "YES" if r["submitted"] else ""
        print(
            f"{i:4d}  {r['cut0']:4.2f} {r['cut1']:4.2f} {r['boost']:5.3f} {r['temp']:4.2f}  "
            f"{r['md5']:>8}  {r['diff_anchor']:6.2f}% {r['shift_v3']:5.2f}% {r['c2']:3d}  {flag}"
        )

    print("\n--- Top picks (unsubmitted, boost=1.80, c2 107-109) ---")
    picks = [
        r
        for r in ranked
        if not r["submitted"]
        and r["boost"] == 1.80
        and 107 <= r["c2"] <= 109
        and r["diff_anchor"] >= args.min_diff_anchor
    ]
    picks = sorted(
        picks,
        key=lambda r: (abs(r["shift_v3"] - 1.97), r["diff_anchor"]),
    )
    for r in picks[:5]:
        print(
            f"  cut0={r['cut0']} cut1={r['cut1']} boost={r['boost']} t={r['temp']}  "
            f"md5={r['md5']} diff180={r['diff_anchor']:.2f}% shift={r['shift_v3']:.2f}% c2={r['c2']}"
        )

    if args.write_top <= 0:
        return

    test_ids = pd.Series(cache["test_ids"])
    written: list[Path] = []
    for r in picks[: args.write_top]:
        preds = _decode(cache, r["cut0"], r["cut1"], r["boost"], r["temp"])
        boost_tag = str(r["boost"]).replace(".", "")
        name = (
            f"ft_c{str(r['cut0']).replace('.', '')}_c1{str(r['cut1']).replace('.', '')}"
            f"_b{boost_tag}_t{_fmt_temp(r['temp'])}"
        )
        notes = (
            f"V3 cal {name}; shift={r['shift_v3']:.2f}%; diff_vs_boost180={r['diff_anchor']:.2f}%; "
            f"c2={r['c2']}; pred_md5={r['md5']}"
        )
        out_path = generate_submission(
            test_ids,
            preds,
            args.output_dir / f"submission_tabpfn_v3_cal_{name}.csv",
            model="TabPFN V3 calibration",
            features="91 targeted temporal + post-hoc proba",
            notes=notes,
        )
        gate_kwargs: dict = dict(
            min_shift_pct=1.0,
            max_shift_pct=10.0,
            baseline_proba=cache["test_proba"],
            best_public_score=0.7883,
        )
        if args.relax_class2:
            gate_kwargs["class2_range"] = None
        row = gate_submission(
            out_path,
            args.baseline,
            args.tracker,
            args.output_dir,
            **gate_kwargs,
        )
        if row.block_submit:
            print(f"BLOCK {out_path.name}:")
            for reason in row.block_reasons:
                print(f"  - {reason}")
            out_path.unlink(missing_ok=True)
            continue
        written.append(out_path)
        print(f"PASS {out_path.name}  md5_pred={r['md5']}  shift={row.shift_pct:.2f}%")

    print(f"\nWrote {len(written)} submission(s)")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
