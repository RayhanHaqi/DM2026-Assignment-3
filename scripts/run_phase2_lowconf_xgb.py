#!/usr/bin/env python
"""Phase 2: narrow low-confidence XGB hard-label override on 0.7897 anchor."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.prob_cache import load_v3_prob_cache
from model.submission_audit import gate_submission, load_labels
from model.utils import generate_submission, load_train_data
from model.v3_diagnostics import diagnose_vs_v3, passes_confidence_shift_gate
from model.v3_probability import (
    apply_class_priors,
    apply_temperature,
    decode_labels,
    low_confidence_mask,
)

DEFAULT_CACHE = ROOT / "output" / "prob_cache" / "tabpfn_v3_91f1.npz"
DEFAULT_ANCHOR = (
    ROOT / "output" / "submission_tabpfn_v3_cal_ft_c08_c1092_b18_t058_20260609_094157_01.csv"
)
DEFAULT_XGB = ROOT / "output" / "submission_xgb_targeted_temporal_20260521_143424_01.csv"
DEFAULT_GBDT_CACHE = ROOT / "output" / "cache" / "gbdt_oof.npz"
DEFAULT_TRACKER = ROOT / "output" / "SUBMISSIONS.md"
DEFAULT_OUTPUT_DIR = ROOT / "output"

PUBLIC_BEST_RECIPE = dict(cut0=0.80, cut1=0.92, boost=1.80, temp=0.58)
PHASE2_MAX_P = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55]
PHASE2_MARGINS = [0.08, 0.10, 0.12, 0.14, 0.15]
PHASE2_ONLY_FROM = (1,)
PHASE2_PROTECT = (2,)


def _calibrated_labels(proba: np.ndarray, classes: np.ndarray, recipe: dict) -> np.ndarray:
    mult = np.array(
        [recipe["cut0"], recipe["cut1"], recipe["boost"], 1.0, recipe["boost"], recipe["boost"]],
        dtype=float,
    )
    adjusted = apply_temperature(apply_class_priors(proba, mult), recipe["temp"])
    return decode_labels(adjusted, classes)


def _restricted_override(
    base_preds: np.ndarray,
    base_proba: np.ndarray,
    alt_preds: np.ndarray,
    *,
    max_confidence: float,
    max_margin: float,
    protect_classes: tuple[int, ...],
    only_from: tuple[int, ...] | None,
) -> np.ndarray:
    low = low_confidence_mask(
        base_proba, max_confidence=max_confidence, max_margin=max_margin
    )
    mask = low & (base_preds != alt_preds)
    if only_from is not None:
        mask &= np.isin(base_preds, list(only_from))
    if protect_classes:
        mask &= ~np.isin(base_preds, list(protect_classes))
    result = np.asarray(base_preds, dtype=int).copy()
    result[mask] = alt_preds[mask]
    return result


def _fmt_num(value: float) -> str:
    return str(value).replace(".", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 low-conf XGB override on public-best anchor.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--xgb-csv", type=Path, default=DEFAULT_XGB)
    parser.add_argument("--gbdt-cache", type=Path, default=DEFAULT_GBDT_CACHE)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--write-top", type=int, default=0)
    parser.add_argument("--min-shift-pct", type=float, default=0.25)
    parser.add_argument("--max-shift-pct", type=float, default=1.0)
    parser.add_argument("--best-public-score", type=float, default=0.7897)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cache = load_v3_prob_cache(args.cache)
    anchor = load_labels(args.anchor)
    xgb = load_labels(args.xgb_csv)
    if len(anchor) != len(xgb):
        raise SystemExit("anchor and xgb label length mismatch")

    gbdt = np.load(args.gbdt_cache)
    classes = cache["classes"]
    xgb_oof = decode_labels(gbdt["xgb_oof_proba"], classes)
    anchor_oof = _calibrated_labels(cache["oof_proba"], classes, PUBLIC_BEST_RECIPE)

    train_path = ROOT / "data" / "train" / "train"
    if not train_path.exists():
        raise SystemExit("Need data/train/train for OOF ranking")
    _, y_train, _, _ = load_train_data(str(train_path))
    y_train = np.asarray(y_train, dtype=int)
    anchor_oof_acc = float(accuracy_score(y_train, anchor_oof))
    anchor_c2 = int((anchor == 2).sum())

    rows: list[dict] = []
    seen_md5: set[str] = set()
    for max_p, margin in itertools.product(PHASE2_MAX_P, PHASE2_MARGINS):
        preds = _restricted_override(
            anchor,
            cache["test_proba"],
            xgb,
            max_confidence=max_p,
            max_margin=margin,
            protect_classes=PHASE2_PROTECT,
            only_from=PHASE2_ONLY_FROM,
        )
        pred_md5 = hashlib.md5(preds.astype(int).tobytes()).hexdigest()
        if pred_md5 in seen_md5:
            continue
        seen_md5.add(pred_md5)

        oof_preds = _restricted_override(
            anchor_oof,
            cache["oof_proba"],
            xgb_oof,
            max_confidence=max_p,
            max_margin=margin,
            protect_classes=PHASE2_PROTECT,
            only_from=PHASE2_ONLY_FROM,
        )
        oof_acc = float(accuracy_score(y_train, oof_preds))
        oof_f1 = float(f1_score(y_train, oof_preds, average="macro", zero_division=0))
        diag = diagnose_vs_v3(anchor, cache["test_proba"], preds)
        conf_ok, conf_reasons = passes_confidence_shift_gate(diag)
        c2 = int((preds == 2).sum())
        rows.append(
            dict(
                max_p=max_p,
                margin=margin,
                md5=pred_md5[:8],
                shift=diag.shift_pct,
                diff=float(np.mean(preds != anchor) * 100.0),
                c2=c2,
                c2_delta=abs(c2 - anchor_c2),
                oof_acc=oof_acc,
                oof_f1=oof_f1,
                conf_ok=conf_ok,
                conf_reasons=conf_reasons,
                low_conf=diag.changed_low_confidence_frac,
                changed=int(np.sum(preds != anchor)),
            )
        )

    def rank_key(r: dict) -> tuple:
        c2_ok = 107 <= r["c2"] <= 109
        in_band = args.min_shift_pct <= r["shift"] <= args.max_shift_pct
        oof_delta = r["oof_acc"] - anchor_oof_acc
        return (
            not (r["conf_ok"] and c2_ok and in_band),
            -oof_delta,
            r["c2_delta"],
            r["shift"],
        )

    ranked = sorted(rows, key=rank_key)
    picks = [
        r
        for r in ranked
        if r["conf_ok"]
        and 107 <= r["c2"] <= 109
        and args.min_shift_pct <= r["shift"] <= args.max_shift_pct
        and r["oof_acc"] >= anchor_oof_acc - 0.0005
    ]
    picks.sort(key=lambda r: (-r["oof_acc"], r["c2_delta"], r["shift"]))

    print(f"Anchor OOF accuracy: {anchor_oof_acc:.4f}  c2={anchor_c2}")
    print(f"Unique candidates: {len(rows)}  gate-passing picks: {len(picks)}")
    print()
    print(f"{'rank':>4}  {'max_p':>5} {'mg':>4}  {'md5':>8}  {'oof':>7} {'shift':>6} {'c2':>3} {'lc':>4} conf")
    for i, r in enumerate(ranked[: args.top], start=1):
        print(
            f"{i:4d}  {r['max_p']:5.2f} {r['margin']:4.2f}  {r['md5']:>8}  "
            f"{r['oof_acc']:7.4f} {r['shift']:5.2f}% {r['c2']:3d} {r['low_conf']:4.0%} "
            f"{'PASS' if r['conf_ok'] else 'FAIL'}"
        )

    print("\n--- Top gated picks ---")
    for r in picks[:5]:
        print(
            f"  max_p={r['max_p']:.2f} margin={r['margin']:.2f} md5={r['md5']} "
            f"oof={r['oof_acc']:.4f} shift={r['shift']:.2f}% c2={r['c2']} changed={r['changed']}"
        )

    if args.dry_run or args.write_top <= 0:
        return

    test_ids = pd.Series(cache["test_ids"])
    written: list[Path] = []
    for r in picks:
        if len(written) >= args.write_top:
            break
        preds = _restricted_override(
            anchor,
            cache["test_proba"],
            xgb,
            max_confidence=r["max_p"],
            max_margin=r["margin"],
            protect_classes=PHASE2_PROTECT,
            only_from=PHASE2_ONLY_FROM,
        )
        name = f"lc1_xgb_mp{_fmt_num(r['max_p'])}_m{_fmt_num(r['margin'])}"
        notes = (
            f"Phase2 lowconf XGB from_c1 protect_c2 {name}; "
            f"oof={r['oof_acc']:.4f}; shift={r['shift']:.2f}%; "
            f"c2={r['c2']}; pred_md5={r['md5']}"
        )
        out_path = generate_submission(
            test_ids,
            preds,
            args.output_dir / f"submission_tabpfn_v3_cal_{name}.csv",
            model="TabPFN V3 calibration",
            features="91 targeted temporal + post-hoc proba + XGB low-conf override",
            notes=notes,
        )
        row = gate_submission(
            out_path,
            args.anchor,
            args.tracker,
            args.output_dir,
            min_shift_pct=args.min_shift_pct,
            max_shift_pct=args.max_shift_pct,
            baseline_proba=cache["test_proba"],
            best_public_score=args.best_public_score,
            class2_range=(107, 109),
            max_near_dup_shift_pct=0.08,
        )
        if row.block_submit:
            print(f"BLOCK {out_path.name}:")
            for reason in row.block_reasons:
                print(f"  - {reason}")
            out_path.unlink(missing_ok=True)
            continue
        written.append(out_path)
        print(f"PASS {out_path.name}  shift={row.shift_pct:.2f}%  md5={r['md5']}")

    print(f"\nWrote {len(written)} submission(s)")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
