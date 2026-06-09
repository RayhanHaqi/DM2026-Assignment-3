#!/usr/bin/env python
"""Blend boost180-calibrated TabPFN V3 probs with cached GBDT partner probs."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.prob_blend import (
    blend_two_proba,
    boost180_calibrated_proba,
    load_gbdt_partner_cache,
    search_anchor_blend,
)
from model.prob_cache import load_v3_prob_cache
from model.submission_audit import gate_submission, load_labels, matches_denylist
from model.utils import generate_submission, load_train_data
from model.validation import prediction_distribution, prediction_shift
from model.v3_probability import decode_labels
from sklearn.metrics import accuracy_score, f1_score

DEFAULT_V3_CACHE = ROOT / "output" / "prob_cache" / "tabpfn_v3_91f1.npz"
DEFAULT_PARTNER_CACHE = ROOT / "output" / "cache" / "gbdt_oof.npz"
DEFAULT_BOOST180 = ROOT / "output" / "submission_tabpfn_v3_cal_full_aggr_cut080_boost180_t06_20260605_202542_01.csv"
DEFAULT_TRACKER = ROOT / "output" / "SUBMISSIONS.md"
DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_WEIGHTS = [0.58, 0.56, 0.55, 0.54, 0.70, 0.65, 0.60, 0.50]


def _pred_md5(preds: np.ndarray) -> str:
    return hashlib.md5(np.asarray(preds, dtype=int).tobytes()).hexdigest()[:8]


def _find_existing_submissions(output_dir: Path, preds: np.ndarray) -> list[Path]:
    """Return submission CSVs with identical Label column (oldest first)."""
    expected_len = len(preds)
    matches: list[Path] = []
    for path in sorted(output_dir.glob("submission_*.csv")):
        try:
            labels = load_labels(path)
        except (ValueError, OSError):
            continue
        if len(labels) != expected_len:
            continue
        if np.array_equal(labels, preds):
            matches.append(path)
    return matches


def _row_passes_preflight(row: dict, *, class2_range: tuple[int, int], args) -> bool:
    if row["shift_pct"] < args.min_shift_pct or row["shift_pct"] > args.max_shift_pct:
        return bool(args.force)
    if row["class2_count"] < class2_range[0] or row["class2_count"] > class2_range[1]:
        return bool(args.force)
    return True


def _rank_rows(
    rows: list[dict],
    *,
    test_anchor: np.ndarray,
    test_partner: np.ndarray,
    classes: np.ndarray,
    boost180_preds: np.ndarray,
) -> list[dict]:
    enriched = []
    for row in rows:
        test_proba = blend_two_proba(test_anchor, test_partner, row["anchor_weight"])
        test_preds = decode_labels(test_proba, classes)
        shift = prediction_shift(test_preds, boost180_preds)
        dist = prediction_distribution(test_preds)
        enriched.append(
            {
                **row,
                "test_preds": test_preds,
                "test_proba": test_proba,
                "shift_pct": float(shift.percent),
                "changed": int(shift.changed),
                "class2_count": int(dist.get(2, 0)),
                "pred_md5": _pred_md5(test_preds),
            }
        )
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="boost180-calibrated V3 prob blend with GBDT partner.")
    parser.add_argument("--v3-cache", type=Path, default=DEFAULT_V3_CACHE)
    parser.add_argument("--partner-cache", type=Path, default=DEFAULT_PARTNER_CACHE)
    parser.add_argument("--partner", choices=["xgb", "lgb", "cat"], default="xgb")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BOOST180)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--anchor-weights",
        type=float,
        nargs="+",
        default=DEFAULT_WEIGHTS,
        help="Weights on boost180-calibrated V3 (partner gets 1-w)",
    )
    parser.add_argument("--write-top", type=int, default=1, help="Write up to N gated CSVs (0 = dry-run)")
    parser.add_argument("--best-public-score", type=float, default=0.7883)
    parser.add_argument("--min-shift-pct", type=float, default=1.0)
    parser.add_argument("--max-shift-pct", type=float, default=3.5)
    parser.add_argument("--class2-min", type=int, default=100)
    parser.add_argument("--class2-max", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.v3_cache.exists():
        raise SystemExit(f"V3 cache missing: {args.v3_cache}")
    if not args.partner_cache.exists():
        raise SystemExit(f"Partner cache missing: {args.partner_cache}")
    if not args.baseline.exists():
        raise SystemExit(f"Baseline missing: {args.baseline}")

    v3 = load_v3_prob_cache(args.v3_cache)
    partner = load_gbdt_partner_cache(args.partner_cache, args.partner)
    classes = np.asarray(v3["classes"], dtype=int)
    boost180_preds = load_labels(args.baseline)

    if "oof_proba" not in v3:
        raise SystemExit("V3 cache missing oof_proba; re-run scripts/cache_tabpfn_v3_probs.py")

    anchor_oof = boost180_calibrated_proba(v3["oof_proba"])
    anchor_test = boost180_calibrated_proba(v3["test_proba"])
    partner_oof = partner["oof_proba"]
    partner_test = partner["test_proba"]

    train_path = Path("data/train/train")
    if not train_path.exists():
        raise SystemExit("Need data/train/train to score OOF blends")
    _, y_train, _, _ = load_train_data(str(train_path))
    y_train = np.asarray(y_train, dtype=int)
    if len(y_train) != anchor_oof.shape[0]:
        raise SystemExit(
            f"OOF rows {anchor_oof.shape[0]} != train labels {len(y_train)}"
        )

    solo_anchor = decode_labels(anchor_oof, classes)
    solo_partner = decode_labels(partner_oof, classes)

    print("=== Solo OOF (grouped CV from caches) ===")
    print(
        f"  boost180-cal V3: acc={accuracy_score(y_train, solo_anchor):.4f} "
        f"f1={f1_score(y_train, solo_anchor, average='macro'):.4f}"
    )
    print(
        f"  {args.partner} partner: acc={accuracy_score(y_train, solo_partner):.4f} "
        f"f1={f1_score(y_train, solo_partner, average='macro'):.4f}"
    )

    rows = search_anchor_blend(anchor_oof, partner_oof, y_train, classes, args.anchor_weights)
    ranked = _rank_rows(
        rows,
        test_anchor=anchor_test,
        test_partner=partner_test,
        classes=classes,
        boost180_preds=boost180_preds,
    )

    print("\n=== Weight search (OOF accuracy, vs boost180 labels) ===")
    print(f"{'w_v3':>6} {'w_pt':>6} {'oof':>7} {'f1':>7} {'shift':>7} {'c2':>4}  md5")
    for row in ranked:
        print(
            f"{row['anchor_weight']:6.2f} {row['partner_weight']:6.2f} "
            f"{row['oof_accuracy']:7.4f} {row['oof_macro_f1']:7.4f} "
            f"{row['shift_pct']:6.2f}% {row['class2_count']:4d}  {row['pred_md5']}"
        )

    if args.write_top <= 0:
        print("\nDry run — no CSV written.")
        return

    written = []
    class2_range = (args.class2_min, args.class2_max)
    for row in ranked:
        if len(written) >= args.write_top:
            break
        if not _row_passes_preflight(row, class2_range=class2_range, args=args):
            continue

        existing_matches = _find_existing_submissions(args.output_dir, row["test_preds"])
        if existing_matches and not args.force:
            existing = existing_matches[0]
            stem = f"submission_tabpfn_v3_cal_boost180_{args.partner}_w{int(round(row['anchor_weight'] * 100))}_p{int(round(row['partner_weight'] * 100))}.csv"
            if matches_denylist(existing.name) or matches_denylist(stem):
                print(f"  REUSE BLOCK {existing.name}: filename on denylist")
                continue
            if len(existing_matches) > 1:
                extras = ", ".join(path.name for path in existing_matches[1:])
                print(f"  NOTE duplicate copies exist ({extras}); reusing {existing.name}")
            print(
                f"  REUSE {existing.name}  shift={row['shift_pct']:.2f}%  "
                f"c2={row['class2_count']}  (skipped duplicate write)"
            )
            written.append(existing)
            continue

        w_tag = int(round(row["anchor_weight"] * 100))
        p_tag = int(round(row["partner_weight"] * 100))
        name = f"tabpfn_v3_cal_boost180_{args.partner}_w{w_tag}_p{p_tag}"
        stem = f"submission_{name}.csv"
        if matches_denylist(stem) and not args.force:
            print(f"  SKIP {stem}: filename matches denylist (use --force to override)")
            continue

        path = generate_submission(
            v3["test_ids"],
            row["test_preds"],
            args.output_dir / f"submission_{name}.csv",
            model=f"TabPFN V3 cal + {args.partner} prob mix",
            features="91 temporal + post-hoc boost180 + GBDT partner",
            notes=(
                f"anchor_w={row['anchor_weight']:.2f}; partner={args.partner}; "
                f"oof={row['oof_accuracy']:.4f}; shift_vs_boost180={row['shift_pct']:.2f}%; "
                f"c2={row['class2_count']}; pred_md5={row['pred_md5']}"
            ),
        )
        try:
            audit = gate_submission(
                path,
                args.baseline,
                args.tracker,
                args.output_dir,
                best_public_score=args.best_public_score,
                min_shift_pct=args.min_shift_pct,
                max_shift_pct=args.max_shift_pct,
                class2_range=class2_range,
                baseline_proba=v3["test_proba"],
            )
        except Exception as exc:
            print(f"  GATE ERROR {path.name}: {exc}")
            continue
        if audit.block_submit and not args.force:
            print(f"  BLOCK {path.name}: {', '.join(audit.block_reasons)}")
            continue
        print(f"  PASS {path.name}  shift={audit.shift_pct:.2f}%  c2={audit.class2_count}")
        written.append(path)

    if written:
        print(f"\nReady {len(written)} submission(s)")
        for path in written:
            print(f"  {path}")
    else:
        print(
            "\nNo submissions passed gates. Re-run with --write-top 0 to inspect weights, "
            "or use --force to override."
        )


if __name__ == "__main__":
    main()
