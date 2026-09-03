#!/usr/bin/env python
"""Paired A/B comparison of two Stage-1 runs' raw_plus_graph arms.

A paired comparison is only valid when the two runs scored the SAME rows
under the SAME protocol, so mismatches here are errors, not warnings:

  1. test_predictions_raw_plus_graph.parquet joins 1:1 on transaction_id --
     no missing, no extra, no duplicate rows on either side.
  2. The label for every joined row is identical. A label mismatch means the
     two runs read different fact tables, and nothing downstream is
     comparable.
  3. If both artifact directories hold a feature_pipeline_manifest.json, the
     split row/fraud counts and total_transactions must match. Different
     splits mean different builds served the rows, and the comparison would
     be between experiments, not arms.

XGBoost with hist/threads is not bit-deterministic, so the instrument is the
paired CI, never point equality. READ THE VERDICT AGAINST THE PLAN: the
acceptance rule is "the CI must not show B significantly worse", i.e. the
interval's upper bound above 0 OR a lower bound within one single-arm
interval width of it. See docs and the projection-removal plan.

Usage:
    python scripts/compare_stage1_ab.py ARTIFACTS_A ARTIFACTS_B \
        [--variant raw_plus_graph] [--resamples 2000] [--level 0.95] [--seed 42]

A is the baseline (projection arm), B the candidate (bipartite arm).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baseline import metrics
from tfgnn.features_meta import FACT_ID, LABEL


def _load_predictions(directory: Path, variant: str) -> pd.DataFrame:
    path = directory / f"test_predictions_{variant}.parquet"
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Run `python -m baseline.train_xgboost "
            f"--variant all` with stage1.output_dir={directory} first."
        )
    frame = pd.read_parquet(path)
    missing = {FACT_ID, LABEL, "prediction"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns {sorted(missing)}")
    if frame[FACT_ID].duplicated().any():
        raise SystemExit(
            f"{path} holds duplicate {FACT_ID} rows; the join would not be 1:1"
        )
    return frame


def _manifest_facts(directory: Path) -> dict[str, object] | None:
    path = directory / "feature_pipeline_manifest.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "total_transactions": payload.get("total_transactions"),
        "split_rows": payload.get("split_rows"),
        "split_frauds": payload.get("split_frauds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap of two Stage-1 runs' raw_plus_graph arms"
    )
    _ = parser.add_argument("baseline_dir", type=Path, help="arm A (projection)")
    _ = parser.add_argument("candidate_dir", type=Path, help="arm B (bipartite)")
    _ = parser.add_argument("--variant", default="raw_plus_graph")
    _ = parser.add_argument("--resamples", type=int, default=2000)
    _ = parser.add_argument("--level", type=float, default=0.95)
    _ = parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    a_dir: Path = args.baseline_dir
    b_dir: Path = args.candidate_dir

    # ---- protocol identity ------------------------------------------------
    a_facts = _manifest_facts(a_dir)
    b_facts = _manifest_facts(b_dir)
    if a_facts is not None and b_facts is not None:
        if a_facts != b_facts:
            raise SystemExit(
                "the two runs' pipeline manifests disagree on split rows / "
                f"frauds / totals:\n  A: {a_facts}\n  B: {b_facts}\n"
                "Different splits mean different builds served the rows; this "
                "would compare experiments, not arms."
            )
        print("manifests: split rows, frauds and totals identical")
    else:
        print(
            "manifests: not found in both directories; relying on the "
            "row-level join checks below"
        )

    a = _load_predictions(a_dir, args.variant)
    b = _load_predictions(b_dir, args.variant)
    if len(a) != len(b):
        raise SystemExit(
            f"row counts differ: A has {len(a):,}, B has {len(b):,}. The two "
            "runs did not score the same test set."
        )

    joined = a.merge(b, on=FACT_ID, how="inner", suffixes=("_a", "_b"), validate="1:1")
    if len(joined) != len(a):
        raise SystemExit(
            f"only {len(joined):,} of {len(a):,} rows share a {FACT_ID}. The "
            "two runs did not score the same test set."
        )
    if not (joined[f"{LABEL}_a"] == joined[f"{LABEL}_b"]).all():
        raise SystemExit(
            "labels disagree between the runs for the same transaction_id. "
            "The two runs read different fact tables; nothing is comparable."
        )
    print(f"rows: {len(joined):,} test rows joined 1:1, labels identical")

    # ---- the paired comparison -------------------------------------------
    labels = joined[f"{LABEL}_a"].to_numpy(dtype="float64")
    baseline_scores = joined["prediction_a"].to_numpy(dtype="float64")
    candidate_scores = joined["prediction_b"].to_numpy(dtype="float64")

    comparison = metrics.paired_bootstrap(
        labels,
        baseline_scores,
        candidate_scores,
        resamples=args.resamples,
        level=args.level,
        seed=args.seed,
    )

    print(f"\n{args.variant}: candidate (B) minus baseline (A), paired")
    print(f"  A test AUCPR   {comparison.baseline_aucpr:.4f}")
    print(f"  B test AUCPR   {comparison.candidate_aucpr:.4f}")
    print(f"  difference     {comparison.difference:+.4f}")
    if comparison.interval is not None:
        lower = comparison.interval["lower"]
        upper = comparison.interval["upper"]
        print(
            f"  {comparison.interval['level']:.0%} CI       "
            f"[{lower:+.4f}, {upper:+.4f}]  ({comparison.resamples} resamples)"
        )
        print(f"  P(B better)    {comparison.probability_candidate_better:.3f}")
        if lower > 0:
            verdict = "B is significantly BETTER than A"
        elif upper < 0:
            verdict = "B is significantly WORSE than A"
        else:
            verdict = "no significant difference"
        print(f"  verdict        {verdict}")

    # Context: each arm's own stored metrics, so the paired number is read
    # next to the single-arm intervals it must be judged against.
    for tag, directory in (("A", a_dir), ("B", b_dir)):
        path = directory / f"metrics_{args.variant}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        interval = payload.get("test_report", {}).get("aucpr_interval")
        if interval:
            print(
                f"  {tag} single-arm {interval['level']:.0%} interval "
                f"[{interval['lower']:.4f}, {interval['upper']:.4f}]"
            )


if __name__ == "__main__":
    main()
