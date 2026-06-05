#!/usr/bin/env python
"""Phase 0 audit: rank local submission CSVs vs TabPFN V3 (no Kaggle slots).

Builds a do-not-submit list (known losers, MD5/prediction duplicates, class-2 drift)
and a manual-review list for unscored files that are materially different from baseline.
Does not recommend resubmitting files already known to score below the public best.
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.submission_audit import audit_all_submissions, load_labels, md5_file, shift_vs_reference
from model.validation import prediction_distribution

DEFAULT_BASELINE = ROOT / "output" / "submission_tabpfn_v3_20260601_180045_01.csv"
DEFAULT_TRACKER = ROOT / "output" / "SUBMISSIONS.md"
DEFAULT_OUTPUT_DIR = ROOT / "output"

# Known bad public results for explicit comparison (filename substring -> label)
BAD_REFERENCE_PATTERNS = {
    "mi127_infold": "mi127 infold (0.7778)",
    "v3_seed65_acc": "seed65 acc (0.7748)",
    "gbdt_cache_blend": "GBDT cache blend (0.7426)",
    "xgb_blend_95_05": "TabPFN+XGB 95/05 (0.7785)",
}


def _discover_submissions(output_dir: Path, pattern: str, tabpfn_only: bool) -> list[Path]:
    paths = sorted(output_dir.glob(pattern))
    if tabpfn_only:
        paths = [p for p in paths if "tabpfn" in p.name.lower()]
    return paths


def _find_bad_reference_files(output_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(output_dir.glob("submission_*.csv")):
        name = path.name.lower()
        for key in BAD_REFERENCE_PATTERNS:
            if key in name and key not in found:
                found[key] = path
    return found


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _write_csv_report(path: Path, rows, meta) -> None:
    fieldnames = [
        "filename",
        "md5_prefix",
        "shift_pct",
        "class2_count",
        "max_class_prop_delta_pp",
        "tracker_score",
        "nearest_file",
        "nearest_shift_pct",
        "flags",
        "block_submit",
        "block_reasons",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filename": row.filename,
                    "md5_prefix": row.md5_prefix,
                    "shift_pct": f"{row.shift_pct:.2f}",
                    "class2_count": row.class2_count,
                    "max_class_prop_delta_pp": f"{row.max_class_prop_delta_pp:.2f}",
                    "tracker_score": (
                        "" if row.tracker_score is None else f"{row.tracker_score:.4f}"
                    ),
                    "nearest_file": row.nearest_file or "",
                    "nearest_shift_pct": (
                        "" if row.nearest_shift_pct is None else f"{row.nearest_shift_pct:.2f}"
                    ),
                    "flags": ";".join(row.flags),
                    "block_submit": row.block_submit,
                    "block_reasons": "; ".join(row.block_reasons),
                }
            )
    print(f"Wrote CSV report: {path}  ({len(rows)} rows, baseline={meta['baseline_md5']})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0: audit local submissions vs TabPFN V3 (no auto-resubmit)."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Public-best baseline CSV (default: TabPFN V3)",
    )
    parser.add_argument(
        "--tracker",
        type=Path,
        default=DEFAULT_TRACKER,
        help="SUBMISSIONS.md path for MD5 and public scores",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing submission_*.csv files",
    )
    parser.add_argument(
        "--glob",
        default="submission_*.csv",
        help="Glob under --output-dir",
    )
    parser.add_argument(
        "--best-score",
        type=float,
        default=0.7830,
        help="Public best score; tracker entries below this are blocked",
    )
    parser.add_argument(
        "--min-good-score",
        type=float,
        default=0.7780,
        help="Min public score for calibrating class-2 acceptable range",
    )
    parser.add_argument(
        "--min-shift-pct",
        type=float,
        default=1.0,
        help="Minimum %% label shift vs baseline (blocks submit below this)",
    )
    parser.add_argument(
        "--max-shift-pct",
        type=float,
        default=10.0,
        help="Maximum %% label shift vs baseline (blocks submit above this)",
    )
    parser.add_argument(
        "--tabpfn-only",
        action="store_true",
        help="Only audit filenames containing 'tabpfn'",
    )
    parser.add_argument(
        "--write-csv",
        type=Path,
        default=None,
        help="Optional path for machine-readable audit table",
    )
    args = parser.parse_args()

    if not args.baseline.exists():
        raise SystemExit(f"Baseline not found: {args.baseline}")

    submission_paths = _discover_submissions(args.output_dir, args.glob, args.tabpfn_only)
    if not submission_paths:
        raise SystemExit(f"No files matched {args.glob} in {args.output_dir}")

    rows, meta, class2_range = audit_all_submissions(
        submission_paths,
        args.baseline,
        args.tracker,
        best_public_score=args.best_score,
        min_good_score=args.min_good_score,
        min_shift_pct=args.min_shift_pct,
        max_shift_pct=args.max_shift_pct,
    )
    rows_sorted = sorted(rows, key=lambda r: r.shift_pct, reverse=True)

    baseline_labels = load_labels(args.baseline)
    base_dist = prediction_distribution(baseline_labels)
    print(f"Baseline: {args.baseline.name}")
    print(f"  md5={md5_file(args.baseline)[:8]}  dist={base_dist}")
    print(f"  public best threshold: {args.best_score:.4f}")
    print(f"  class-2 acceptable range (from public >= {args.min_good_score:.4f}): {class2_range}")
    print(f"  audited {len(rows_sorted)} files (excluded baseline)")

    _print_section("Full audit (shift vs baseline, descending)")
    for row in rows_sorted:
        score_txt = "?" if row.tracker_score is None else f"{row.tracker_score:.4f}"
        flag_txt = f"  [{', '.join(row.flags)}]" if row.flags else ""
        block_txt = "  BLOCK" if row.block_submit else ""
        print(
            f"{row.shift_pct:6.2f}%  class2={row.class2_count:4d}  "
            f"public={score_txt}  md5={row.md5_prefix}  "
            f"nearest={row.nearest_shift_pct:.2f}%@{row.nearest_file}{flag_txt}{block_txt}"
        )
        print(f"         {row.filename}")
        if row.block_reasons:
            print(f"         -> {'; '.join(row.block_reasons)}")

    blocked = [r for r in rows_sorted if r.block_submit]
    _print_section(f"DO NOT SUBMIT ({len(blocked)} files)")
    if not blocked:
        print("(none)")
    for row in blocked:
        print(f"  {row.filename}")
        print(f"    {'; '.join(row.block_reasons)}")

    manual_review = [
        r
        for r in rows_sorted
        if not r.block_submit
        and r.tracker_score is None
        and "LOW_SHIFT" not in r.flags
    ]
    _print_section(
        f"Manual review only — unscored, not blocked, shift >= {args.min_shift_pct:.1f}% ({len(manual_review)})"
    )
    print(
        "These are NOT resubmit recommendations. Submit only after new training/gates and one slot at a time."
    )
    if not manual_review:
        print("(none)")
    for row in manual_review[:20]:
        print(
            f"  {row.shift_pct:6.2f}%  class2={row.class2_count}  md5={row.md5_prefix}  {row.filename}"
        )
    if len(manual_review) > 20:
        print(f"  ... and {len(manual_review) - 20} more")

    bad_refs = _find_bad_reference_files(args.output_dir)
    if bad_refs:
        _print_section("Shift vs known bad public references")
        for key, ref_path in sorted(bad_refs.items()):
            print(f"  Reference: {BAD_REFERENCE_PATTERNS[key]} ({ref_path.name})")
        for row in rows_sorted:
            parts = []
            cand_labels = load_labels(row.path)
            for key, ref_path in bad_refs.items():
                pct = shift_vs_reference(cand_labels, load_labels(ref_path))
                parts.append(f"{key}={pct:.1f}%")
            print(f"  {row.filename}: {'  '.join(parts)}")

    if args.write_csv:
        _write_csv_report(args.write_csv, rows_sorted, meta)
    else:
        print("Audit complete (read-only). Use --write-csv output/phase0_submission_audit.csv to save CSV.")


if __name__ == "__main__":
    main()
