#!/usr/bin/env python
"""Build hard-label consensus CSV from prior public-good TabPFN submissions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.hard_label_consensus import (
    DEFAULT_BASELINE_NAME,
    consensus_shift_pct,
    hard_label_majority,
    select_public_good_submissions,
    v3_disagreement_override,
)
from model.submission_audit import gate_submission, load_labels
from model.utils import generate_submission

DEFAULT_BASELINE = ROOT / "output" / DEFAULT_BASELINE_NAME
DEFAULT_TRACKER = ROOT / "output" / "SUBMISSIONS.md"
DEFAULT_OUTPUT_DIR = ROOT / "output"


def _load_ids(reference_path: Path) -> pd.Series:
    frame = pd.read_csv(reference_path)
    if "Id" not in frame.columns:
        raise ValueError(f"{reference_path} missing Id column")
    return frame["Id"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hard-label consensus from public-good TabPFN submissions (no probability blending)."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument(
        "--mode",
        choices=("majority", "v3_override"),
        default="v3_override",
        help="majority: vote across all good sources; v3_override: change V3 only on 2+ disagreements",
    )
    parser.add_argument("--min-good-score", type=float, default=0.7780)
    parser.add_argument("--min-agree", type=int, default=2, help="For v3_override mode")
    parser.add_argument("--best-score", type=float, default=0.7830)
    parser.add_argument("--min-shift-pct", type=float, default=1.0)
    parser.add_argument("--max-shift-pct", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="Print stats only; do not write CSV")
    args = parser.parse_args()

    if not args.baseline.exists():
        raise SystemExit(f"Baseline not found: {args.baseline}")
    if not args.tracker.exists():
        raise SystemExit(f"Tracker not found: {args.tracker}")

    good = select_public_good_submissions(
        args.output_dir,
        args.tracker,
        min_good_score=args.min_good_score,
        baseline_name=args.baseline.name,
    )
    print(f"Public-good sources (>={args.min_good_score:.4f}, MD5-distinct): {len(good)}")
    for item in good:
        print(f"  {item.public_score:.4f}  md5={item.md5[:8]}  {item.filename}")

    if args.mode == "majority":
        preds = hard_label_majority(good, baseline_name=args.baseline.name)
        tag = "majority"
    else:
        preds = v3_disagreement_override(
            good,
            baseline_name=args.baseline.name,
            min_agree=args.min_agree,
        )
        tag = f"v3_override_agree{args.min_agree}"

    anchor_labels = load_labels(args.baseline)
    shift = consensus_shift_pct(preds, anchor_labels)
    changed = int((preds != anchor_labels).sum())
    print(f"\nConsensus ({tag}): shift={changed}/{len(preds)} ({shift:.2f}%) vs {args.baseline.name}")

    if args.dry_run:
        print("Dry run — no CSV written.")
        return

    test_ids = _load_ids(args.baseline)
    source_names = ",".join(item.filename.replace("submission_", "").replace(".csv", "")[:24] for item in good[:5])
    notes = (
        f"hard-label {tag}; sources={len(good)}; shift_vs_v3={shift:.2f}%; "
        f"min_good={args.min_good_score:.4f}; from={source_names}"
    )
    out_path = generate_submission(
        test_ids,
        preds,
        args.output_dir / f"submission_tabpfn_hard_{tag}.csv",
        model="TabPFN hard-label consensus",
        features="prior public-good preds",
        notes=notes,
    )

    row = gate_submission(
        out_path,
        args.baseline,
        args.tracker,
        args.output_dir,
        best_public_score=args.best_score,
        min_good_score=args.min_good_score,
        min_shift_pct=args.min_shift_pct,
        max_shift_pct=args.max_shift_pct,
    )
    if row.block_submit:
        print("Phase 0 audit BLOCK — removing CSV:")
        for reason in row.block_reasons:
            print(f"  - {reason}")
        out_path.unlink(missing_ok=True)
        raise SystemExit(1)

    print(f"Phase 0 audit: PASS  md5={row.md5_prefix}  shift={row.shift_pct:.2f}%")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
