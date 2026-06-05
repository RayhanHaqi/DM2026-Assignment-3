#!/usr/bin/env python
"""Phase-0 gate check for one submission CSV before Kaggle upload."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.submission_audit import gate_submission

DEFAULT_BASELINE = ROOT / "output" / "submission_tabpfn_v3_20260601_180045_01.csv"
DEFAULT_TRACKER = ROOT / "output" / "SUBMISSIONS.md"
DEFAULT_OUTPUT_DIR = ROOT / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate one submission CSV vs TabPFN V3 baseline.")
    parser.add_argument("candidate", type=Path, help="Candidate submission CSV")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--best-score", type=float, default=0.7830)
    parser.add_argument("--min-good-score", type=float, default=0.7780)
    parser.add_argument("--min-shift-pct", type=float, default=1.0)
    parser.add_argument("--max-shift-pct", type=float, default=10.0)
    args = parser.parse_args()

    if not args.candidate.exists():
        raise SystemExit(f"Candidate not found: {args.candidate}")
    if not args.baseline.exists():
        raise SystemExit(f"Baseline not found: {args.baseline}")

    row = gate_submission(
        args.candidate,
        args.baseline,
        args.tracker,
        args.output_dir,
        best_public_score=args.best_score,
        min_good_score=args.min_good_score,
        min_shift_pct=args.min_shift_pct,
        max_shift_pct=args.max_shift_pct,
    )
    print(f"Candidate: {row.filename}")
    print(f"  md5={row.md5}  shift={row.shift_pct:.2f}%  class2={row.class2_count}")
    if row.tracker_score is not None:
        print(f"  known public={row.tracker_score:.4f}")
    if row.flags:
        print(f"  flags: {', '.join(row.flags)}")
    if row.block_submit:
        print("  RESULT: BLOCK")
        for reason in row.block_reasons:
            print(f"    - {reason}")
        raise SystemExit(1)
    print("  RESULT: PASS")


if __name__ == "__main__":
    main()
