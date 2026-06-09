#!/usr/bin/env python
"""V3 confidence diagnostics for a candidate submission CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.prob_cache import load_v3_prob_cache
from model.submission_audit import load_labels
from model.v3_diagnostics import diagnose_vs_v3, passes_confidence_shift_gate

DEFAULT_CACHE = ROOT / "output" / "prob_cache" / "tabpfn_v3_91f1.npz"
DEFAULT_BASELINE = ROOT / "output" / "submission_tabpfn_v3_20260601_180045_01.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose candidate vs TabPFN V3 baseline.")
    parser.add_argument(
        "candidate",
        nargs="?",
        type=Path,
        help="Candidate CSV (default: baseline CSV for self-check)",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--min-changed-low-conf", type=float, default=0.70)
    parser.add_argument("--max-class-delta-pp", type=float, default=1.5)
    args = parser.parse_args()

    candidate_path = args.candidate or args.baseline
    if not candidate_path.exists():
        raise SystemExit(f"Candidate not found: {candidate_path}")
    if not args.cache.exists():
        raise SystemExit(
            f"Cache not found: {args.cache}\n"
            "Run: PYTHONPATH=. python scripts/cache_tabpfn_v3_probs.py --device cuda"
        )

    cache = load_v3_prob_cache(args.cache)
    baseline_preds = load_labels(args.baseline)
    candidate_preds = load_labels(candidate_path)
    if len(candidate_preds) != len(baseline_preds):
        raise SystemExit(
            f"Length mismatch: candidate {len(candidate_preds)} vs baseline {len(baseline_preds)}"
        )

    diag = diagnose_vs_v3(baseline_preds, cache["test_proba"], candidate_preds)
    ok, reasons = passes_confidence_shift_gate(
        diag,
        min_changed_low_conf_frac=args.min_changed_low_conf,
        max_single_class_delta_pp=args.max_class_delta_pp,
    )

    print(f"Candidate: {candidate_path.name}")
    print(f"Baseline:  {args.baseline.name}")
    print(f"Shift: {diag.changed}/{diag.total} ({diag.shift_pct:.2f}%)")
    print(
        f"Changed rows low-confidence frac: {diag.changed_low_confidence_frac:.1%} "
        f"(mean max_p changed={diag.changed_mean_max_proba:.3f}, "
        f"unchanged={diag.unchanged_mean_max_proba:.3f})"
    )
    print(f"Per-class count delta (pp): {diag.per_class_delta_pp}")
    if diag.changed:
        print(f"Changed-row target counts: {diag.per_class_changed}")
    print(f"Confidence gate: {'PASS' if ok else 'BLOCK'}")
    for reason in reasons:
        print(f"  - {reason}")


if __name__ == "__main__":
    main()
