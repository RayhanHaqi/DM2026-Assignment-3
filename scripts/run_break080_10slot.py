#!/usr/bin/env python
"""Break-0.80: finetune + coarse calibration + up to N Kaggle submits."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR = (
    ROOT / "output" / "submission_tabpfn_v3_cal_ft_c08_c1092_b18_t058_20260609_094157_01.csv"
)
COMPETITION = "nycu-data-mining-assignment-3"


def _existing_submissions(output_dir: Path) -> set[Path]:
    return set(output_dir.glob("submission_*.csv"))


def _run(cmd: list[str], *, cwd: Path) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def _submit(path: Path, message: str) -> bool:
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-f",
        str(path),
        "-m",
        message,
    ]
    print("\n>>>", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Break 0.80 pipeline with slot budget.")
    parser.add_argument("--max-slots", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--best-score", type=float, default=0.7897)
    parser.add_argument("--dry-run", action="store_true", help="Generate CSVs only; no Kaggle upload")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-cal", action="store_true")
    args = parser.parse_args()

    if not args.anchor.exists():
        raise SystemExit(f"Anchor not found: {args.anchor}")

    py = sys.executable
    output_dir = ROOT / "output"
    before = _existing_submissions(output_dir)

    if not args.skip_finetune:
        for metric in ("log_loss", "roc_auc"):
            _run(
                [
                    py,
                    "scripts/run_tabpfn_finetune.py",
                    "--device",
                    args.device,
                    "--baseline-csv",
                    str(args.anchor),
                    "--best-score",
                    str(args.best_score),
                    "--min-shift-pct",
                    "0.3",
                    "--min-oof-margin",
                    "0.0",
                    "--finetune-metric",
                    metric,
                    "--force",
                ],
                cwd=ROOT,
            )

    if not args.skip_cal:
        cal_slots = max(1, args.max_slots - 2)
        _run(
            [
                py,
                "scripts/run_tabpfn_v3_calibration.py",
                "--baseline",
                str(args.anchor),
                "--best-public-score",
                str(args.best_score),
                "--finetune-boost180",
                "--finetune-cuts",
                "--max-submissions",
                str(cal_slots),
                "--min-shift-pct",
                "0.15",
                "--max-shift-pct",
                "3.0",
                "--relax-class2",
                "--exclude-md5-prefix",
                "5a474150",
                "--exclude-md5-prefix",
                "61be51f5",
            ],
            cwd=ROOT,
        )

    after = _existing_submissions(output_dir)
    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"\nNew submission CSVs: {len(new_files)}")
    for path in new_files:
        print(f"  {path.name}")

    if args.dry_run:
        print("\nDry run — no Kaggle uploads.")
        return

    submitted = 0
    for path in new_files:
        if submitted >= args.max_slots:
            break
        msg = f"break080 {path.name}"
        if _submit(path, msg):
            submitted += 1
            print(f"  OK slot {submitted}/{args.max_slots}")
        else:
            print(f"  FAIL {path.name}")

    print(f"\nDone. Submitted {submitted}/{args.max_slots} new file(s).")


if __name__ == "__main__":
    main()
