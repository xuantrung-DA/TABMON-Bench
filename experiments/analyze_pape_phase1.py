"""Cluster-aware PAPE summaries and paired comparisons with AC/COTT."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analyze_schema_v7 import benjamini_hochberg
from experiments.analyze_v8_step9 import two_way_cell_bootstrap_mean
from src.cache.stream_cache import BATCH_INDEX, atomic_json, atomic_parquet


COMPARATORS = ("ac", "cott")
TOLERANCES = (0.02, 0.05, 0.10)
SIGN_TOLERANCE = 0.02
OUTCOMES = (
    "mean_absolute_error",
    "failure_rate_tau_0_02",
    "failure_rate_tau_0_05",
    "failure_rate_tau_0_1",
    "sign_accuracy",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pape-dir", type=Path, required=True)
    parser.add_argument("--phase5-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260929)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 1000:
        parser.error("Use at least 1,000 cluster-bootstrap replicates")
    return args


def _comparator_scenarios(batch_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = batch_metrics[batch_metrics["method"].isin(COMPARATORS)].copy()
    frame = frame[
        frame["shift"].eq("no_shift") | frame["shift_fraction"].gt(0)
    ].copy()
    if set(frame["method"]) != set(COMPARATORS):
        raise ValueError("Phase 5 batch metrics are missing AC or COTT")
    for tolerance in TOLERANCES:
        suffix = str(tolerance).replace(".", "_")
        frame[f"failure_tau_{suffix}"] = frame["absolute_error"].gt(tolerance)
    frame["sign_eligible"] = frame["true_excess_error"].abs().gt(SIGN_TOLERANCE)
    frame["sign_correct"] = frame["sign_eligible"] & (
        np.sign(frame["estimated_excess_error"])
        == np.sign(frame["true_excess_error"])
    )
    descriptors = [
        "predictor_stream_id",
        "dataset",
        "model",
        "shift",
        "severity",
        "mode",
        "seed",
        "method",
    ]
    aggregations: dict[str, tuple[str, Any]] = {
        "mean_absolute_error": ("absolute_error", "mean"),
        "sign_eligible_batches": ("sign_eligible", "sum"),
        "sign_correct_batches": ("sign_correct", "sum"),
    }
    for tolerance in TOLERANCES:
        suffix = str(tolerance).replace(".", "_")
        aggregations[f"failure_rate_tau_{suffix}"] = (
            f"failure_tau_{suffix}",
            "mean",
        )
    scenarios = frame.groupby(descriptors, observed=True, sort=True).agg(
        **aggregations
    ).reset_index()
    scenarios["sign_accuracy"] = np.where(
        scenarios["sign_eligible_batches"].gt(0),
        scenarios["sign_correct_batches"] / scenarios["sign_eligible_batches"],
        np.nan,
    )
    return scenarios


def _scopes(frame: pd.DataFrame) -> list[tuple[str, str | None, pd.DataFrame]]:
    scopes: list[tuple[str, str | None, pd.DataFrame]] = [("overall", None, frame)]
    if "benchmark_assumption_regime" in frame:
        for value in sorted(frame["benchmark_assumption_regime"].unique()):
            scopes.append(
                (
                    "assumption_regime",
                    str(value),
                    frame[frame["benchmark_assumption_regime"].eq(value)],
                )
            )
    scopes.extend(
        ("shift", str(value), frame[frame["shift"].eq(value)])
        for value in sorted(frame["shift"].unique())
    )
    return scopes


def pape_summaries(
    scenarios: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for scope, value, subset in _scopes(scenarios):
        for outcome in OUTCOMES:
            valid = subset.dropna(subset=[outcome])
            if valid.empty:
                continue
            estimate, lower, upper, _, _ = two_way_cell_bootstrap_mean(
                valid, outcome, replicates, 0.95, rng
            )
            records.append(
                {
                    "scope": scope,
                    "scope_value": value,
                    "outcome": outcome,
                    "estimate": estimate,
                    "cluster_ci_lower": lower,
                    "cluster_ci_upper": upper,
                    "n_streams": len(valid),
                    "n_datasets": valid["dataset"].nunique(),
                    "n_seeds": valid["seed"].nunique(),
                }
            )
    return pd.DataFrame(records)


def paired_comparisons(
    pape: pd.DataFrame,
    comparators: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    descriptors = [
        "predictor_stream_id",
        "dataset",
        "model",
        "shift",
        "severity",
        "mode",
        "seed",
        "benchmark_assumption_regime",
    ]
    records: list[dict[str, Any]] = []
    for scope, value, pape_subset in _scopes(pape):
        for comparator in COMPARATORS:
            other = comparators[comparators["method"].eq(comparator)]
            for outcome in OUTCOMES:
                paired = pape_subset[descriptors + [outcome]].merge(
                    other[["predictor_stream_id", outcome]],
                    on="predictor_stream_id",
                    how="inner",
                    suffixes=("_pape", "_comparator"),
                    validate="one_to_one",
                ).dropna(subset=[f"{outcome}_pape", f"{outcome}_comparator"])
                if paired.empty:
                    continue
                paired["difference"] = (
                    paired[f"{outcome}_pape"]
                    - paired[f"{outcome}_comparator"]
                )
                estimate, lower, upper, p_value, _ = two_way_cell_bootstrap_mean(
                    paired, "difference", replicates, 0.95, rng
                )
                records.append(
                    {
                        "scope": scope,
                        "scope_value": value,
                        "outcome": outcome,
                        "method_a": "pape",
                        "method_b": comparator,
                        "difference_a_minus_b": estimate,
                        "cluster_ci_lower": lower,
                        "cluster_ci_upper": upper,
                        "bootstrap_p_value": p_value,
                        "n_paired_streams": len(paired),
                        "n_datasets": paired["dataset"].nunique(),
                        "n_seeds": paired["seed"].nunique(),
                    }
                )
    result = pd.DataFrame(records)
    result["q_value_bh"] = benjamini_hochberg(result["bootstrap_p_value"])
    return result


def _report(summaries: pd.DataFrame, comparisons: pd.DataFrame) -> str:
    lines = [
        "# PAPE Phase-1 statistical report",
        "",
        "Intervals use a two-way dataset--seed cluster bootstrap. Paired",
        "differences are PAPE minus the named comparator; negative MAE/failure",
        "differences favor PAPE, while positive sign-accuracy differences favor",
        "PAPE.",
        "",
        "## PAPE summaries",
        "",
        "| Scope | Value | Outcome | Estimate | 95% CI | n |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summaries.itertuples(index=False):
        lines.append(
            f"| {row.scope} | {row.scope_value or 'all'} | {row.outcome} | "
            f"{row.estimate:.4f} | [{row.cluster_ci_lower:.4f}, "
            f"{row.cluster_ci_upper:.4f}] | {row.n_streams} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Scope | Value | Outcome | Comparison | Difference | 95% CI | p | q | n |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.scope} | {row.scope_value or 'all'} | {row.outcome} | "
            f"PAPE--{row.method_b.upper()} | {row.difference_a_minus_b:.4f} | "
            f"[{row.cluster_ci_lower:.4f}, {row.cluster_ci_upper:.4f}] | "
            f"{row.bootstrap_p_value:.4g} | {row.q_value_bh:.4g} | "
            f"{row.n_paired_streams} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pape_dir = args.pape_dir.resolve()
    phase5_dir = args.phase5_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status = json.loads((pape_dir / "pape_status.json").read_text())
    if status.get("complete") is not True:
        raise ValueError("Statistical analysis requires complete PAPE output")
    pape = pd.read_parquet(pape_dir / "scenario_metrics.parquet")
    phase5_batches = pd.read_parquet(phase5_dir / "batch_metrics.parquet")
    comparators = _comparator_scenarios(phase5_batches)
    comparator_ids = set(comparators["predictor_stream_id"])
    if not set(pape["predictor_stream_id"]).issubset(comparator_ids):
        raise ValueError("AC/COTT results do not cover every PAPE stream")

    rng = np.random.default_rng(args.random_seed)
    summaries = pape_summaries(pape, args.bootstrap_replicates, rng)
    comparisons = paired_comparisons(
        pape, comparators, args.bootstrap_replicates, rng
    )
    atomic_parquet(summaries, output_dir / "pape_cluster_summaries.parquet")
    atomic_parquet(comparisons, output_dir / "pape_paired_comparisons.parquet")
    summaries.to_csv(output_dir / "pape_cluster_summaries.csv", index=False)
    comparisons.to_csv(output_dir / "pape_paired_comparisons.csv", index=False)
    (output_dir / "PAPE_PHASE1_REPORT.md").write_text(
        _report(summaries, comparisons), encoding="utf-8"
    )
    atomic_json(
        {
            "complete": True,
            "bootstrap_replicates": args.bootstrap_replicates,
            "random_seed": args.random_seed,
            "primary_uncertainty": "two-way dataset-seed cluster bootstrap",
            "multiple_testing": "Benjamini-Hochberg across PAPE comparisons",
            "pape_streams": len(pape),
            "summary_rows": len(summaries),
            "paired_rows": len(comparisons),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output_dir / "pape_analysis_status.json",
    )
    print((output_dir / "PAPE_PHASE1_REPORT.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
