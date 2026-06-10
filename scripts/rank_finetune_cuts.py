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
from model.utils import generate_submission, load_train_data
from model.v3_diagnostics import diagnose_vs_v3
from model.v3_probability import apply_class_priors, apply_temperature, decode_labels
from sklearn.metrics import accuracy_score, f1_score
from scripts.run_tabpfn_v3_calibration import (
    FINETUNE_BOOST180,
    FINETUNE_CUT0,
    FINETUNE_CUT1,
    FINETUNE_TEMPS,
    _fmt_temp,
)

DEFAULT_CACHE = ROOT / "output" / "prob_cache" / "tabpfn_v3_91f1.npz"
DEFAULT_BASELINE = ROOT / "output" / "submission_tabpfn_v3_20260601_180045_01.csv"
DEFAULT_PUBLIC_BEST = (
    ROOT / "output" / "submission_tabpfn_v3_cal_ft_c08_c1092_b18_t058_20260609_094157_01.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_TRACKER = ROOT / "output" / "SUBMISSIONS.md"
BOOST180_RECIPE = dict(cut0=0.80, cut1=0.90, boost=1.80, temp=0.60)
PUBLIC_BEST_RECIPE = dict(cut0=0.80, cut1=0.92, boost=1.80, temp=0.58)
PHASE1_CUT0 = [0.78, 0.80, 0.82]
PHASE1_CUT1 = [0.91, 0.92, 0.93]
PHASE1_TEMPS = [0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62]
PHASE1_BOOST = [1.80]
SUBMITTED_PRED_MD5 = {
    "9a02efc8",
    "93972e8b",
    "a2746db2",
    "5a474150",
    "61be51f5",
}


def _calibrated_labels(proba: np.ndarray, classes: np.ndarray, cut0: float, cut1: float, boost: float, temp: float) -> np.ndarray:
    mult = np.array([cut0, cut1, boost, 1.0, boost, boost], dtype=float)
    adjusted = apply_temperature(apply_class_priors(proba, mult), temp)
    return decode_labels(adjusted, classes)


def _decode(cache: dict, cut0: float, cut1: float, boost: float, temp: float) -> np.ndarray:
    return _calibrated_labels(cache["test_proba"], cache["classes"], cut0, cut1, boost, temp)


def _decode_oof(cache: dict, cut0: float, cut1: float, boost: float, temp: float) -> np.ndarray:
    return _calibrated_labels(cache["oof_proba"], cache["classes"], cut0, cut1, boost, temp)


def _class_counts(preds: np.ndarray) -> dict[int, int]:
    dist = prediction_distribution(preds)
    return {int(k): int(v) for k, v in dist.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank finetune-cut candidates vs boost180.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--gate-baseline",
        type=Path,
        default=None,
        help="Gate/write baseline CSV (default: --public-best in --phase1 else --baseline)",
    )
    parser.add_argument("--public-best", type=Path, default=DEFAULT_PUBLIC_BEST)
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Phase-1 grid around 0.7897 public best; rank by OOF accuracy vs public anchor",
    )
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--write-top", type=int, default=0, help="Write top N unsubmitted boost=1.80 picks")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--relax-class2", action="store_true")
    parser.add_argument(
        "--min-diff-anchor",
        type=float,
        default=0.0,
        help="Skip picks within this %% label change of ranking anchor (avoid near-dup writes)",
    )
    parser.add_argument("--min-shift-pct", type=float, default=None)
    parser.add_argument("--max-shift-pct", type=float, default=None)
    parser.add_argument("--best-public-score", type=float, default=None)
    args = parser.parse_args()

    gate_baseline = args.gate_baseline or (args.public_best if args.phase1 else args.baseline)
    min_shift = 0.03 if args.min_shift_pct is None and args.phase1 else (args.min_shift_pct or 1.0)
    max_shift = 0.35 if args.max_shift_pct is None and args.phase1 else (args.max_shift_pct or 10.0)
    best_public = 0.7897 if args.best_public_score is None and args.phase1 else (args.best_public_score or 0.7883)

    cache = load_v3_prob_cache(args.cache)
    baseline = load_labels(args.baseline)
    gate_labels = load_labels(gate_baseline)
    public_best = _decode(cache, **PUBLIC_BEST_RECIPE)
    if args.public_best.exists():
        public_best = load_labels(args.public_best)
    boost180 = _decode(cache, **BOOST180_RECIPE)
    ranking_anchor = public_best if args.phase1 else boost180
    anchor_md5 = hashlib.md5(ranking_anchor.astype(int).tobytes()).hexdigest()[:8]
    anchor_c2 = int((ranking_anchor == 2).sum())

    y_train = None
    anchor_oof_acc = None
    if args.phase1:
        if "oof_proba" not in cache:
            raise SystemExit("Cache missing oof_proba; run scripts/cache_tabpfn_v3_probs.py")
        train_path = Path("data/train/train")
        if not train_path.exists():
            raise SystemExit("Need data/train/train for Phase-1 OOF ranking")
        _, y_train, _, _ = load_train_data(str(train_path))
        y_train = np.asarray(y_train, dtype=int)
        anchor_oof_preds = _decode_oof(cache, **PUBLIC_BEST_RECIPE)
        anchor_oof_acc = float(accuracy_score(y_train, anchor_oof_preds))

    cut0_values = PHASE1_CUT0 if args.phase1 else FINETUNE_CUT0
    cut1_values = PHASE1_CUT1 if args.phase1 else FINETUNE_CUT1
    temp_values = PHASE1_TEMPS if args.phase1 else FINETUNE_TEMPS
    boost_values = PHASE1_BOOST if args.phase1 else FINETUNE_BOOST180

    rows: list[dict] = []
    seen_md5: set[str] = set()
    for cut0, cut1, boost, temp in itertools.product(
        cut0_values, cut1_values, boost_values, temp_values
    ):
        preds = _decode(cache, cut0, cut1, boost, temp)
        pred_md5 = hashlib.md5(preds.astype(int).tobytes()).hexdigest()
        if pred_md5 in seen_md5:
            continue
        seen_md5.add(pred_md5)
        pred_prefix = pred_md5[:8]

        diff_anchor = float(np.mean(preds != ranking_anchor) * 100.0)
        diff_v3 = float(np.mean(preds != baseline) * 100.0)
        c2 = int((preds == 2).sum())
        diag = diagnose_vs_v3(baseline, cache["test_proba"], preds)
        counts = _class_counts(preds)
        oof_acc = None
        oof_f1 = None
        if args.phase1 and y_train is not None:
            oof_preds = _decode_oof(cache, cut0, cut1, boost, temp)
            oof_acc = float(accuracy_score(y_train, oof_preds))
            oof_f1 = float(f1_score(y_train, oof_preds, average="macro", zero_division=0))

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
                oof_acc=oof_acc,
                oof_f1=oof_f1,
            )
        )

    def rank_key(r: dict) -> tuple:
        c2_penalty = 0 if 107 <= r["c2"] <= 109 else abs(r["c2"] - anchor_c2)
        boost_penalty = 0 if r["boost"] == 1.80 else 1
        if args.phase1 and r["oof_acc"] is not None:
            return (
                r["submitted"],
                -r["oof_acc"],
                c2_penalty,
                -r["oof_f1"],
                r["diff_anchor"],
            )
        return (
            r["submitted"],
            boost_penalty,
            c2_penalty,
            r["diff_anchor"],
            abs(r["shift_v3"] - 1.97),
        )

    ranked = sorted(rows, key=rank_key)
    unsubmitted = [r for r in ranked if not r["submitted"]]

    anchor_shift = float(np.mean(_decode(cache, **PUBLIC_BEST_RECIPE) != ranking_anchor) * 100.0)
    anchor_label = "public_best" if args.phase1 else "boost180"
    print(f"Anchor {anchor_label}: md5={anchor_md5} c2={anchor_c2} shift_vs_self={anchor_shift:.2f}%")
    if anchor_oof_acc is not None:
        print(f"Public-best anchor OOF accuracy: {anchor_oof_acc:.4f}")
    print(f"Unique candidates: {len(rows)} | unsubmitted: {len(unsubmitted)}")
    print()
    if args.phase1:
        print(
            f"{'rank':>4}  {'cut0':>4} {'cut1':>4} {'boost':>5} {'t':>4}  "
            f"{'md5':>8}  {'oof':>7} {'f1':>7} {'diff':>6} {'c2':>3}  sub"
        )
        for i, r in enumerate(ranked[: args.top], start=1):
            flag = "YES" if r["submitted"] else ""
            print(
                f"{i:4d}  {r['cut0']:4.2f} {r['cut1']:4.2f} {r['boost']:5.3f} {r['temp']:4.2f}  "
                f"{r['md5']:>8}  {r['oof_acc']:7.4f} {r['oof_f1']:7.4f} "
                f"{r['diff_anchor']:5.2f}% {r['c2']:3d}  {flag}"
            )
    else:
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

    c2_lo, c2_hi = (107, 109) if args.phase1 else (107, 109)
    print(f"\n--- Top picks (unsubmitted, boost=1.80, c2 {c2_lo}-{c2_hi}) ---")
    picks = [
        r
        for r in ranked
        if not r["submitted"]
        and r["boost"] == 1.80
        and c2_lo <= r["c2"] <= c2_hi
        and r["diff_anchor"] >= args.min_diff_anchor
        and (not args.phase1 or r["diff_anchor"] <= max_shift)
        and (not args.phase1 or anchor_oof_acc is None or r["oof_acc"] >= anchor_oof_acc - 0.001)
    ]
    if args.phase1:
        picks = sorted(
            picks,
            key=lambda r: (-r["oof_acc"], r["c2_delta"], r["diff_anchor"]),
        )
    else:
        picks = sorted(
            picks,
            key=lambda r: (abs(r["shift_v3"] - 1.97), r["diff_anchor"]),
        )
    for r in picks[:5]:
        if args.phase1:
            print(
                f"  cut0={r['cut0']} cut1={r['cut1']} boost={r['boost']} t={r['temp']}  "
                f"md5={r['md5']} oof={r['oof_acc']:.4f} diff={r['diff_anchor']:.2f}% c2={r['c2']}"
            )
        else:
            print(
                f"  cut0={r['cut0']} cut1={r['cut1']} boost={r['boost']} t={r['temp']}  "
                f"md5={r['md5']} diff180={r['diff_anchor']:.2f}% shift={r['shift_v3']:.2f}% c2={r['c2']}"
            )

    if args.write_top <= 0:
        return

    test_ids = pd.Series(cache["test_ids"])
    written: list[Path] = []
    if args.phase1:
        picks = sorted(
            picks,
            key=lambda r: (-r["diff_anchor"], -r["oof_acc"], r["c2_delta"]),
        )
    for r in picks:
        if len(written) >= args.write_top:
            break
        preds = _decode(cache, r["cut0"], r["cut1"], r["boost"], r["temp"])
        boost_tag = str(r["boost"]).replace(".", "")
        name = (
            f"ft_c{str(r['cut0']).replace('.', '')}_c1{str(r['cut1']).replace('.', '')}"
            f"_b{boost_tag}_t{_fmt_temp(r['temp'])}"
        )
        oof_note = f"; oof={r['oof_acc']:.4f}" if r.get("oof_acc") is not None else ""
        notes = (
            f"V3 cal {name}; diff_vs_anchor={r['diff_anchor']:.2f}%; "
            f"c2={r['c2']}; pred_md5={r['md5']}{oof_note}"
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
            min_shift_pct=min_shift,
            max_shift_pct=max_shift,
            baseline_proba=cache["test_proba"],
            best_public_score=best_public,
            class2_range=(100, 120) if args.phase1 else "auto",
            max_near_dup_shift_pct=0.08 if args.phase1 else 0.3,
        )
        if args.relax_class2:
            gate_kwargs["class2_range"] = None
        row = gate_submission(
            out_path,
            gate_baseline,
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
