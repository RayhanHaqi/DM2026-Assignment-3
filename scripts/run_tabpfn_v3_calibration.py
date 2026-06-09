#!/usr/bin/env python
"""V3 probability calibration / prior sweep with confidence gates."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.prob_cache import load_v3_prob_cache
from model.submission_audit import gate_submission, load_labels
from model.utils import generate_submission
from model.v3_diagnostics import diagnose_vs_v3, passes_confidence_shift_gate
from model.v3_probability import (
    apply_class_priors,
    apply_low_confidence_override,
    apply_temperature,
    decode_labels,
    decode_selective_calibration,
)

DEFAULT_CACHE = ROOT / "output" / "prob_cache" / "tabpfn_v3_91f1.npz"
DEFAULT_BASELINE = ROOT / "output" / "submission_tabpfn_v3_20260601_180045_01.csv"
DEFAULT_TRACKER = ROOT / "output" / "SUBMISSIONS.md"
DEFAULT_OUTPUT_DIR = ROOT / "output"

# Class order: 0,1,2,3,4,5
MODERATE_PRIORS: dict[str, list[float]] = {
    "neutral": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "boost_2_110": [1.0, 1.0, 1.10, 1.0, 1.0, 1.0],
    "boost_245_110": [1.0, 1.0, 1.10, 1.0, 1.10, 1.10],
    "boost_245_115": [1.0, 1.0, 1.15, 1.0, 1.15, 1.15],
    "cut_1_095": [0.95, 1.0, 1.0, 1.0, 1.0, 1.0],
    "boost_2_cut1": [0.95, 1.0, 1.15, 1.0, 1.0, 1.0],
}

AGGRESSIVE_PRIORS: dict[str, list[float]] = {
    "aggr_cut085_boost150": [0.85, 0.95, 1.50, 1.0, 1.50, 1.50],
    "aggr_cut085_boost160": [0.85, 0.95, 1.60, 1.0, 1.60, 1.60],
    "aggr_cut080_boost180": [0.80, 0.90, 1.80, 1.0, 1.80, 1.80],
    "aggr_cut085_boost200": [0.85, 0.95, 2.00, 1.0, 2.00, 2.00],
}

PRIOR_PRESETS: dict[str, list[float]] = {**MODERATE_PRIORS, **AGGRESSIVE_PRIORS}

MODERATE_TEMPS = [0.85, 0.92, 1.0, 1.08, 1.15]
AGGRESSIVE_TEMPS = [0.60, 0.65, 0.70, 0.75]
TEMPERATURES = sorted(set(MODERATE_TEMPS + AGGRESSIVE_TEMPS))

SELECTIVE_THRESHOLDS: list[tuple[float, float]] = [
    (0.70, 0.15),
    (0.75, 0.15),
    (0.80, 0.15),
    (0.85, 0.15),
    (0.85, 0.20),
]

LOWCONF_MAX_P = [0.40, 0.45, 0.48, 0.50, 0.55, 0.60, 0.65]

FINETUNE_BOOST180 = [1.75, 1.775, 1.80, 1.825, 1.85]
FINETUNE_TEMPS = [0.58, 0.59, 0.60, 0.61, 0.62]
FINETUNE_CUT0 = [0.78, 0.80, 0.82]
FINETUNE_CUT1 = [0.88, 0.90, 0.92]
WINNER_BOOST180_MD5_PREFIX = "9a02efc8"


@dataclass
class CalibrationCandidate:
    name: str
    preds: np.ndarray
    diagnostic: object
    conf_ok: bool
    conf_reasons: list[str]


def _candidate(
    name: str,
    preds: np.ndarray,
    baseline_preds: np.ndarray,
    test_proba: np.ndarray,
) -> CalibrationCandidate:
    diag = diagnose_vs_v3(baseline_preds, test_proba, preds)
    conf_ok, conf_reasons = passes_confidence_shift_gate(diag)
    return CalibrationCandidate(
        name=name,
        preds=preds,
        diagnostic=diag,
        conf_ok=conf_ok,
        conf_reasons=conf_reasons,
    )


def _build_candidates(
    cache: dict,
    baseline_preds: np.ndarray,
    *,
    alt_sources: list[tuple[str, np.ndarray]],
    moderate_prior_names: list[str],
    aggressive_prior_names: list[str],
    include_selective: bool,
    include_aggressive_full: bool,
    low_conf_margin: float,
) -> list[CalibrationCandidate]:
    test_proba = cache["test_proba"]
    classes = cache["classes"]
    out: list[CalibrationCandidate] = []

    for temp, prior_name in itertools.product(MODERATE_TEMPS, moderate_prior_names):
        if prior_name not in MODERATE_PRIORS:
            continue
        mult = np.asarray(MODERATE_PRIORS[prior_name], dtype=float)
        adjusted = apply_temperature(apply_class_priors(test_proba, mult), temp)
        preds = decode_labels(adjusted, classes)
        out.append(_candidate(f"full_{prior_name}_t{_fmt_temp(temp)}", preds, baseline_preds, test_proba))

    if include_aggressive_full:
        for temp, prior_name in itertools.product(AGGRESSIVE_TEMPS, aggressive_prior_names):
            if prior_name not in AGGRESSIVE_PRIORS:
                continue
            mult = np.asarray(AGGRESSIVE_PRIORS[prior_name], dtype=float)
            adjusted = apply_temperature(apply_class_priors(test_proba, mult), temp)
            preds = decode_labels(adjusted, classes)
            out.append(
                _candidate(f"full_{prior_name}_t{_fmt_temp(temp)}", preds, baseline_preds, test_proba)
            )

    if include_selective:
        for (max_p, margin), temp, prior_name in itertools.product(
            SELECTIVE_THRESHOLDS, AGGRESSIVE_TEMPS, aggressive_prior_names
        ):
            if prior_name not in AGGRESSIVE_PRIORS:
                continue
            mult = np.asarray(AGGRESSIVE_PRIORS[prior_name], dtype=float)
            preds = decode_selective_calibration(
                baseline_preds,
                test_proba,
                classes,
                mult,
                temp,
                max_confidence=max_p,
                max_margin=margin,
            )
            out.append(
                _candidate(
                    f"sel_{prior_name}_mp{_fmt_temp(max_p)}_m{_fmt_temp(margin)}_t{_fmt_temp(temp)}",
                    preds,
                    baseline_preds,
                    test_proba,
                )
            )

    for alt_name, alt_preds in alt_sources:
        for max_p in LOWCONF_MAX_P:
            preds = apply_low_confidence_override(
                baseline_preds,
                test_proba,
                alt_preds,
                max_confidence=max_p,
                max_margin=low_conf_margin,
            )
            out.append(
                _candidate(
                    f"lowconf_{alt_name}_maxp{_fmt_temp(max_p)}",
                    preds,
                    baseline_preds,
                    test_proba,
                )
            )

    return out


def _build_finetune_boost180_candidates(
    cache: dict,
    baseline_preds: np.ndarray,
    *,
    boost_values: list[float],
    temperatures: list[float],
    cut0_values: list[float],
    cut1_values: list[float],
) -> list[CalibrationCandidate]:
    """Narrow prior/temp grid around public-best boost180 (deduped by prediction MD5)."""
    test_proba = cache["test_proba"]
    classes = cache["classes"]
    seen_md5: set[str] = set()
    out: list[CalibrationCandidate] = []

    for cut0, cut1, boost, temp in itertools.product(
        cut0_values, cut1_values, boost_values, temperatures
    ):
        mult = np.array([cut0, cut1, boost, 1.0, boost, boost], dtype=float)
        adjusted = apply_temperature(apply_class_priors(test_proba, mult), temp)
        preds = decode_labels(adjusted, classes)
        pred_md5 = hashlib.md5(preds.astype(int).tobytes()).hexdigest()
        if pred_md5 in seen_md5:
            continue
        seen_md5.add(pred_md5)

        boost_tag = str(boost).replace(".", "")
        name = f"ft_c{str(cut0).replace('.', '')}_c1{str(cut1).replace('.', '')}_b{boost_tag}_t{_fmt_temp(temp)}"
        out.append(_candidate(name, preds, baseline_preds, test_proba))

    return out


def _finetune_rank_key(item: CalibrationCandidate, target_shift: float = 1.97):
    """Prefer unique preds near the 0.7883 winner shift (~1.97%)."""
    d = item.diagnostic
    return (
        item.conf_ok,
        -abs(d.shift_pct - target_shift),
        d.shift_pct,
    )


def _fmt_temp(value: float) -> str:
    return str(value).replace(".", "")


def _in_shift_band(shift_pct: float, lo: float, hi: float) -> bool:
    return lo <= shift_pct <= hi


def _rank_key(item: CalibrationCandidate, target_min: float, target_max: float):
    d = item.diagnostic
    in_band = _in_shift_band(d.shift_pct, target_min, target_max)
    mid = 0.5 * (target_min + target_max)
    span = max(target_max - target_min, 1e-6)
    if in_band:
        # Prefer mid-band shifts (e.g. ~3% when target is 1-5%).
        band_score = 1.0 - abs(d.shift_pct - mid) / span
    elif d.shift_pct < target_min:
        band_score = d.shift_pct - target_min
    else:
        band_score = target_max - d.shift_pct
    return (
        item.conf_ok,
        in_band,
        band_score,
        d.changed_low_confidence_frac if item.conf_ok else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TabPFN V3 calibration / prior sweep.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--alt-csv",
        action="append",
        default=[],
        help="Alternate hard labels for low-confidence override (repeatable)",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-submissions", type=int, default=2)
    parser.add_argument("--min-shift-pct", type=float, default=1.0)
    parser.add_argument("--max-shift-pct", type=float, default=5.0)
    parser.add_argument("--target-shift-min", type=float, default=1.0)
    parser.add_argument("--target-shift-max", type=float, default=5.0)
    parser.add_argument("--min-changed-low-conf", type=float, default=0.70)
    parser.add_argument(
        "--moderate-priors",
        default=",".join(MODERATE_PRIORS.keys()),
    )
    parser.add_argument(
        "--aggressive-priors",
        default=",".join(AGGRESSIVE_PRIORS.keys()),
    )
    parser.add_argument("--no-selective", action="store_true")
    parser.add_argument("--no-aggressive-full", action="store_true")
    parser.add_argument(
        "--finetune-boost180",
        action="store_true",
        help="Narrow grid: boost 1.75–1.85, temps 0.58–0.62, optional cut sweep",
    )
    parser.add_argument(
        "--finetune-cuts",
        action="store_true",
        help="With --finetune-boost180, also sweep cut0/cut1 (0.78–0.82 / 0.88–0.92)",
    )
    parser.add_argument(
        "--exclude-md5-prefix",
        action="append",
        default=[],
        help="Skip candidates whose prediction MD5 starts with this prefix (repeatable)",
    )
    parser.add_argument(
        "--relax-class2",
        action="store_true",
        help="Disable class-2 count gate (boost180 family often drifts class-2)",
    )
    parser.add_argument("--best-public-score", type=float, default=0.7883)
    args = parser.parse_args()

    if args.rebuild_cache or not args.cache.exists():
        print("Building probability cache...")
        import subprocess

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "cache_tabpfn_v3_probs.py"),
            "--output-path",
            str(args.cache),
            "--device",
            args.device,
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)

    if not args.baseline.exists():
        raise SystemExit(f"Baseline not found: {args.baseline}")

    cache = load_v3_prob_cache(args.cache)
    baseline_preds = load_labels(args.baseline)
    test_ids = pd.Series(cache["test_ids"])

    default_alts = [
        ROOT / "output" / "submission_tabpfn_fitmode_low_memory_20260601_113838_01.csv",
        ROOT / "output" / "submission_tabpfn_optuna_best_20260601_173045_01.csv",
        ROOT / "output" / "submission_tabpfn_mi_top80_20260601_180550_01.csv",
    ]
    alt_paths = [Path(p) for p in args.alt_csv] if args.alt_csv else default_alts
    alt_sources: list[tuple[str, np.ndarray]] = []
    for path in alt_paths:
        if not path.exists():
            print(f"  Skip missing alt: {path.name}")
            continue
        alt_preds = load_labels(path)
        if len(alt_preds) != len(baseline_preds):
            raise SystemExit(f"alt-csv length mismatch: {path}")
        tag = path.stem.replace("submission_tabpfn_", "")[:20]
        alt_sources.append((tag, alt_preds))

    moderate_names = [p.strip() for p in args.moderate_priors.split(",") if p.strip()]
    aggressive_names = [p.strip() for p in args.aggressive_priors.split(",") if p.strip()]

    if args.finetune_boost180:
        cut0_values = FINETUNE_CUT0 if args.finetune_cuts else [0.80]
        cut1_values = FINETUNE_CUT1 if args.finetune_cuts else [0.90]
        candidates = _build_finetune_boost180_candidates(
            cache,
            baseline_preds,
            boost_values=FINETUNE_BOOST180,
            temperatures=FINETUNE_TEMPS,
            cut0_values=cut0_values,
            cut1_values=cut1_values,
        )
        ranked = sorted(candidates, key=_finetune_rank_key, reverse=True)
        exclude_prefixes = set(args.exclude_md5_prefix) | {WINNER_BOOST180_MD5_PREFIX}
        print(
            f"Finetune boost180: {len(candidates)} unique preds "
            f"(boost={FINETUNE_BOOST180}, temps={FINETUNE_TEMPS}, "
            f"cut0={cut0_values}, cut1={cut1_values})"
        )
    else:
        candidates = _build_candidates(
            cache,
            baseline_preds,
            alt_sources=alt_sources,
            moderate_prior_names=moderate_names,
            aggressive_prior_names=aggressive_names,
            include_selective=not args.no_selective,
            include_aggressive_full=not args.no_aggressive_full,
            low_conf_margin=0.15,
        )
        ranked = sorted(
            candidates,
            key=lambda item: _rank_key(item, args.target_shift_min, args.target_shift_max),
            reverse=True,
        )
        exclude_prefixes = set(args.exclude_md5_prefix)

    in_band = [
        c
        for c in ranked
        if c.conf_ok
        and _in_shift_band(c.diagnostic.shift_pct, args.target_shift_min, args.target_shift_max)
    ]

    print(f"Evaluated {len(ranked)} calibration candidates")
    print(
        f"In target band {args.target_shift_min:.1f}-{args.target_shift_max:.1f}% "
        f"with confidence PASS: {len(in_band)}"
    )

    print("\n--- In-band by shift (confidence PASS, highest first) ---")
    for item in sorted(in_band, key=lambda c: -c.diagnostic.shift_pct)[:10]:
        d = item.diagnostic
        print(
            f"  {item.name:40s} shift={d.shift_pct:5.2f}%  "
            f"low_conf={d.changed_low_confidence_frac:.0%}"
        )

    print("\n--- In-band ranked for write (prefers ~mid-band shift) ---")
    for item in in_band[:10]:
        d = item.diagnostic
        print(
            f"  {item.name:40s} shift={d.shift_pct:5.2f}%  "
            f"low_conf={d.changed_low_confidence_frac:.0%}"
        )

    print("\n--- Top 10 overall ---")
    for item in ranked[:10]:
        d = item.diagnostic
        pred_md5 = hashlib.md5(item.preds.astype(int).tobytes()).hexdigest()[:8]
        print(
            f"  {item.name:40s} shift={d.shift_pct:5.2f}%  "
            f"c2={(item.preds == 2).sum():3d}  md5={pred_md5}  "
            f"low_conf={d.changed_low_confidence_frac:.0%}  "
            f"conf={'PASS' if item.conf_ok else 'FAIL'}"
        )
        if item.conf_reasons:
            print(f"    {'; '.join(item.conf_reasons)}")

    if args.finetune_boost180:
        print("\n--- Finetune unique (excl. winner md5) ---")
        shown = 0
        for item in ranked:
            pred_md5 = hashlib.md5(item.preds.astype(int).tobytes()).hexdigest()[:8]
            if any(pred_md5.startswith(p) for p in exclude_prefixes):
                continue
            d = item.diagnostic
            print(
                f"  {item.name:40s} shift={d.shift_pct:5.2f}%  "
                f"c2={(item.preds == 2).sum():3d}  md5={pred_md5}"
            )
            shown += 1
            if shown >= 15:
                break

    if args.dry_run:
        print("\nDry run — no CSV written.")
        return

    written: list[Path] = []
    seen_pred_md5: set[str] = set()
    for item in ranked:
        if len(written) >= args.max_submissions:
            break
        if not args.finetune_boost180 and not item.conf_ok:
            continue
        d = item.diagnostic
        if d.shift_pct < args.min_shift_pct or d.shift_pct > args.max_shift_pct:
            continue
        pred_md5 = hashlib.md5(item.preds.astype(int).tobytes()).hexdigest()
        pred_prefix = pred_md5[:8]
        if any(pred_prefix.startswith(p) for p in exclude_prefixes):
            continue
        if pred_md5 in seen_pred_md5:
            continue
        seen_pred_md5.add(pred_md5)

        notes = (
            f"V3 cal {item.name}; shift={d.shift_pct:.2f}%; "
            f"changed_low_conf={d.changed_low_confidence_frac:.2f}"
        )
        out_path = generate_submission(
            test_ids,
            item.preds,
            args.output_dir / f"submission_tabpfn_v3_cal_{item.name}.csv",
            model="TabPFN V3 calibration",
            features="91 targeted temporal + post-hoc proba",
            notes=notes,
        )
        gate_kwargs: dict = dict(
            min_shift_pct=args.min_shift_pct,
            max_shift_pct=args.max_shift_pct,
            baseline_proba=cache["test_proba"],
            min_changed_low_conf_frac=args.min_changed_low_conf,
            best_public_score=args.best_public_score,
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
        print(f"PASS {out_path.name}  shift={row.shift_pct:.2f}%  md5={row.md5_prefix}")

    print(f"\nWrote {len(written)} submission(s)")
    for path in written:
        print(f"  {path}")
    if not written:
        print("No candidates passed confidence + phase-0 gates.")


if __name__ == "__main__":
    main()
