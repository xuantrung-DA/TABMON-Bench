"""Final cluster-aware analysis for binary and multiclass Schema-13 results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analyze_schema_v7 import benjamini_hochberg
from experiments.analyze_v8_step9 import (
    fit_binomial_gee,
    fit_binomial_mixed_model,
    fit_linear_mixed_model,
    two_way_cell_bootstrap_mean,
)
from src.cache.stream_cache import BATCH_INDEX, atomic_json, atomic_parquet


METHODS = ("ac", "doc", "atc", "cot", "cott")
OUTCOMES = ("mean_absolute_error", "risk_failure_rate")
SHIFT_ORDER = (
    "no_shift",
    "covariate",
    "correlated",
    "concept",
    "pipeline",
    "support",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--multiclass-dir", type=Path, required=True)
    parser.add_argument("--shd-sensitivity-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260927)
    parser.add_argument("--skip-mixed-models", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 1000:
        parser.error("Use at least 1,000 bootstrap replicates")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_schema13(root: Path) -> dict[str, Any]:
    root = root.resolve()
    status = json.loads((root / "step12_status.json").read_text("utf-8"))
    manifest = json.loads((root / "step12_manifest.json").read_text("utf-8"))
    audit = json.loads((root / "step12_audit.json").read_text("utf-8"))
    if status.get("complete") is not True or int(status.get("schema_version", 0)) != 13:
        raise ValueError(f"Incomplete/non-Schema-13 result: {root}")
    if audit.get("pass") is not True or audit.get("complete") is not True:
        raise ValueError(f"Schema-13 audit has not passed: {root}")
    return {
        "root": root,
        "status": status,
        "manifest": manifest,
        "audit": audit,
        "scenarios": pd.read_parquet(root / "scenario_metrics.parquet"),
        "batches": pd.read_parquet(root / "batch_metrics.parquet"),
        "xpe": pd.read_parquet(root / "xpe_metrics.parquet"),
        "shd_calibration": pd.read_parquet(root / "shd_calibration.parquet"),
    }


def _classification(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[
        frame["endpoint"].eq("classification_error")
        & frame["method"].isin(METHODS)
    ].copy()
    counts = result.groupby("predictor_stream_id")["method"].nunique()
    if not counts.eq(len(METHODS)).all():
        raise ValueError("Classification estimators are not fully paired")
    truth = result.pivot(
        index="predictor_stream_id", columns="method", values="mean_true_value"
    )
    if (truth.max(axis=1) - truth.min(axis=1)).max() > 1e-12:
        raise ValueError("Paired methods do not share an oracle endpoint")
    return result


def _bootstrap(
    frame: pd.DataFrame,
    value: str,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    estimate, lower, upper, p_value, _ = two_way_cell_bootstrap_mean(
        frame, value, replicates, 0.95, rng
    )
    return estimate, lower, upper, p_value


def method_summaries(
    classification: pd.DataFrame,
    task: str,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    scopes = [("overall", None, classification)] + [
        ("shift", shift, classification[classification["shift"].eq(shift)])
        for shift in SHIFT_ORDER
    ]
    for scope, shift, frame in scopes:
        for method in METHODS:
            subset = frame[frame["method"].eq(method)]
            if subset.empty:
                continue
            for outcome in OUTCOMES:
                estimate, lower, upper, _ = _bootstrap(
                    subset, outcome, replicates, rng
                )
                records.append(
                    {
                        "task": task,
                        "scope": scope,
                        "shift": shift,
                        "method": method,
                        "outcome": outcome,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci_design": (
                            "dataset-seed crossed cluster bootstrap"
                            if subset["dataset"].nunique() > 1
                            else "seed cluster bootstrap; one dataset"
                        ),
                        "n_streams": len(subset),
                        "n_datasets": subset["dataset"].nunique(),
                        "n_seeds": subset["seed"].nunique(),
                    }
                )
    return pd.DataFrame(records)


def paired_comparisons(
    classification: pd.DataFrame,
    task: str,
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
    ]
    records: list[dict[str, Any]] = []
    scopes = [("overall", None, classification)] + [
        ("shift", shift, classification[classification["shift"].eq(shift)])
        for shift in SHIFT_ORDER
    ]
    for scope, shift, frame in scopes:
        if frame.empty:
            continue
        base = frame[descriptors].drop_duplicates("predictor_stream_id")
        for outcome in OUTCOMES:
            wide = frame.pivot(
                index="predictor_stream_id", columns="method", values=outcome
            )
            if wide.isna().any().any() or set(wide.columns) != set(METHODS):
                raise ValueError(f"Incomplete paired matrix: {task}/{scope}/{outcome}")
            for method_a, method_b in combinations(METHODS, 2):
                difference = (wide[method_a] - wide[method_b]).rename("difference")
                paired = base.merge(
                    difference,
                    left_on="predictor_stream_id",
                    right_index=True,
                    validate="one_to_one",
                )
                estimate, lower, upper, p_value = _bootstrap(
                    paired, "difference", replicates, rng
                )
                records.append(
                    {
                        "task": task,
                        "scope": scope,
                        "shift": shift,
                        "outcome": outcome,
                        "method_a": method_a,
                        "method_b": method_b,
                        "difference_a_minus_b": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "bootstrap_p_value": p_value,
                        "ci_winner": (
                            method_a
                            if upper < 0
                            else method_b
                            if lower > 0
                            else "no_clear_winner"
                        ),
                        "n_paired_streams": len(paired),
                    }
                )
    result = pd.DataFrame(records)
    result["q_value_bh_scope"] = np.nan
    group_keys = ["task", "scope", "shift", "outcome"]
    for _, positions in result.groupby(group_keys, dropna=False).groups.items():
        result.loc[positions, "q_value_bh_scope"] = benjamini_hochberg(
            result.loc[positions, "bootstrap_p_value"]
        )
    result["q_value_bh_global_task_outcome"] = np.nan
    for _, positions in result.groupby(["task", "outcome"]).groups.items():
        result.loc[positions, "q_value_bh_global_task_outcome"] = (
            benjamini_hochberg(result.loc[positions, "bootstrap_p_value"])
        )
    result["globally_significant"] = (
        result["q_value_bh_global_task_outcome"].lt(0.05)
        & ~result["ci_winner"].eq("no_clear_winner")
    )
    return result


def separate_endpoint_summaries(
    scenarios: pd.DataFrame,
    task: str,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    specs = {
        "confidence_log_loss": [
            "mean_true_value",
            "mean_estimated_value",
            "mean_absolute_error",
            "risk_failure_rate",
        ],
        "shd_quantile_phi2_pm_eb": [
            "false_alarm_rate",
            "power",
            "detection_delay",
            "missed_alarm",
        ],
    }
    records = []
    for method, metrics in specs.items():
        source = scenarios[scenarios["method"].eq(method)]
        for scope, shift, frame in [("overall", None, source)] + [
            ("shift", value, source[source["shift"].eq(value)])
            for value in SHIFT_ORDER
        ]:
            for metric in metrics:
                valid = frame[pd.to_numeric(frame[metric], errors="coerce").notna()]
                if valid.empty:
                    continue
                estimate, lower, upper, _ = _bootstrap(
                    valid, metric, replicates, rng
                )
                records.append(
                    {
                        "task": task,
                        "scope": scope,
                        "shift": shift,
                        "method": method,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "n_scenarios": len(valid),
                    }
                )
    return pd.DataFrame(records)


def xpe_summaries(
    xpe: pd.DataFrame,
    batches: pd.DataFrame,
    task: str,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if xpe.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = xpe.copy()
    frame["risk_failure"] = frame["absolute_risk_error"].gt(0.05).astype(float)
    if "sign_accuracy" not in frame:
        frame["sign_accuracy"] = (
            np.sign(frame["estimated_loss_change"])
            == np.sign(frame["oracle_excess_log_loss"])
        ).astype(float)
    if "false_attribution_mass" not in frame:
        frame["false_attribution_mass"] = np.where(
            frame["shift"].eq("no_shift"),
            frame["attribution_json"].map(
                lambda value: sum(abs(v) for v in json.loads(value).values())
            ),
            np.nan,
        )
    metrics = [
        "absolute_risk_error",
        "risk_failure",
        "sign_accuracy",
        "top3_recall",
        "ndcg@3",
        "attribution_failure",
        "oracle_label_transfer_accuracy",
        "false_attribution_mass",
    ]
    records = []
    for scope, shift, subset in [("overall", None, frame)] + [
        ("shift", value, frame[frame["shift"].eq(value)])
        for value in SHIFT_ORDER
    ]:
        for metric in metrics:
            valid = subset[pd.to_numeric(subset[metric], errors="coerce").notna()]
            if valid.empty:
                continue
            estimate, lower, upper, _ = _bootstrap(valid, metric, replicates, rng)
            records.append(
                {
                    "task": task,
                    "scope": scope,
                    "shift": shift,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_streams": len(valid),
                }
            )

    confidence = batches[batches["method"].eq("confidence_log_loss")].copy()
    last = confidence.loc[
        confidence.groupby("predictor_stream_id")[BATCH_INDEX].idxmax(),
        [
            "predictor_stream_id",
            "dataset",
            "model",
            "shift",
            "severity",
            "mode",
            "seed",
            "absolute_error",
        ],
    ].rename(columns={"absolute_error": "confidence_absolute_error"})
    paired = frame.merge(last, on=[
        "predictor_stream_id",
        "dataset",
        "model",
        "shift",
        "severity",
        "mode",
        "seed",
    ], validate="one_to_one")
    paired["xpe_minus_confidence_absolute_error"] = (
        paired["absolute_risk_error"] - paired["confidence_absolute_error"]
    )
    paired["xpe_better"] = paired["xpe_minus_confidence_absolute_error"].lt(0).astype(float)
    pair_records = []
    for scope, shift, subset in [("overall", None, paired)] + [
        ("shift", value, paired[paired["shift"].eq(value)])
        for value in SHIFT_ORDER
    ]:
        if subset.empty:
            continue
        for metric in ["xpe_minus_confidence_absolute_error", "xpe_better"]:
            estimate, lower, upper, p_value = _bootstrap(
                subset, metric, replicates, rng
            )
            pair_records.append(
                {
                    "task": task,
                    "scope": scope,
                    "shift": shift,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "bootstrap_p_value": p_value,
                    "n_paired_streams": len(subset),
                }
            )
    paired_summary = pd.DataFrame(pair_records)
    paired_summary["q_value_bh"] = benjamini_hochberg(
        paired_summary["bootstrap_p_value"]
    )
    return pd.DataFrame(records), paired_summary


def shd_applicability(
    calibration: pd.DataFrame,
    scenarios: pd.DataFrame,
    task: str,
) -> pd.DataFrame:
    shd = scenarios[scenarios["method"].eq("shd_quantile_phi2_pm_eb")]
    feasible = calibration[calibration["selector_feasible"]]
    feasible_cells = feasible[["dataset", "model"]]
    evaluated = shd.merge(feasible_cells, on=["dataset", "model"], how="inner")
    return pd.DataFrame(
        [
            {
                "task": task,
                "source_cells": len(calibration),
                "feasible_source_cells": int(calibration["selector_feasible"].sum()),
                "feasible_fraction": float(calibration["selector_feasible"].mean()),
                "evaluated_streams": len(evaluated),
                "event_streams": int(evaluated["event_batch"].notna().sum()),
                "detected_event_streams": int(evaluated["power"].fillna(0).sum()),
                "any_alarm_streams": int(evaluated["first_alarm_batch"].notna().sum()),
                "mean_minimum_grid_fdp": float(calibration["minimum_grid_fdp"].mean()),
                "mean_error_estimator_r2": float(calibration["error_estimator_r2"].mean()),
            }
        ]
    )


def shd_sensitivity_summary(root: Path | None) -> pd.DataFrame:
    if root is None:
        return pd.DataFrame()
    root = root.resolve()
    status = json.loads(
        (root / "shd_sensitivity_status.json").read_text("utf-8")
    )
    audit = json.loads(
        (root / "shd_sensitivity_audit.json").read_text("utf-8")
    )
    if status.get("complete") is not True or audit.get("pass") is not True:
        raise ValueError("SHD sensitivity must be complete and audited")
    metrics = pd.read_parquet(root / "shd_sensitivity_metrics.parquet")
    records = []
    for (alpha, epsilon), group in metrics.groupby(["alpha", "epsilon"]):
        evaluated = group[group["evaluated"]]
        event = evaluated[evaluated["stream_any_event"].astype(bool)]
        records.append(
            {
                "alpha": alpha,
                "epsilon": epsilon,
                "all_streams": len(group),
                "applicable_streams": len(evaluated),
                "event_streams": len(event),
                "alarm_streams": int(evaluated["stream_any_alarm"].sum()),
                "power": float(event["stream_any_alarm"].mean()) if len(event) else np.nan,
                "false_alarm_fraction": float(
                    evaluated.loc[
                        ~evaluated["stream_any_event"].astype(bool),
                        "stream_any_alarm",
                    ].mean()
                ),
                "mean_delay_detected": float(
                    evaluated["detection_delay"].dropna().mean()
                ),
                "positive_assumption_gap_fraction": float(
                    evaluated["max_oracle_assumption_4_1_gap"].gt(0).mean()
                ),
            }
        )
    return pd.DataFrame(records)


def _write_report(
    output: Path,
    methods: pd.DataFrame,
    xpe: pd.DataFrame,
    applicability: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    lines = [
        "# TABMON Schema-13 final statistical report",
        "",
        "Classification-error estimators are compared only on their shared "
        "0--1 error endpoint. Confidence, XPE, and SHD retain their declared "
        "separate estimands.",
        "",
        "## Overall classification-error MAE",
        "",
        "| Task | Method | Estimate | 95% cluster CI |",
        "|---|---|---:|---:|",
    ]
    overall = methods[
        methods["scope"].eq("overall")
        & methods["outcome"].eq("mean_absolute_error")
    ].sort_values(["task", "estimate"])
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.method.upper()} | {row.estimate:.4f} | "
            f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}] |"
        )
    lines.extend(["", "## SHD applicability", ""])
    for row in applicability.itertuples(index=False):
        lines.append(
            f"- {row.task}: {row.feasible_source_cells}/{row.source_cells} "
            f"source cells feasible; {row.detected_event_streams}/"
            f"{row.event_streams} event streams detected."
        )
    if not sensitivity.empty:
        lines.extend(
            [
                "",
                "## SHD inference sensitivity",
                "",
                "The source error regressor and quantile selector are fixed; "
                "only alpha and epsilon change.",
                "",
                "| Alpha | Epsilon | Event streams | Alarm streams | Power |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sensitivity.itertuples(index=False):
            lines.append(
                f"| {row.alpha:.2f} | {row.epsilon:.2f} | {row.event_streams} | "
                f"{row.alarm_streams} | {row.power:.3f} |"
            )
    if not xpe.empty:
        overall_xpe = xpe[xpe["scope"].eq("overall")]
        lines.extend(["", "## XPE overall", ""])
        for row in overall_xpe.itertuples(index=False):
            lines.append(
                f"- {row.task} `{row.metric}`: {row.estimate:.4f} "
                f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}]."
            )
    lines.extend(
        [
            "",
            "## Statistical boundary",
            "",
            "Binary intervals resample datasets and seeds as crossed blocks. "
            "The multiclass extension has one dataset, so its intervals "
            "resample seeds and cannot support cross-dataset generalization.",
        ]
    )
    (output / "SCHEMA13_STATISTICAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    binary = _load_schema13(args.binary_dir)
    multiclass = _load_schema13(args.multiclass_dir)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    all_methods = []
    all_pairs = []
    all_endpoints = []
    all_xpe = []
    all_xpe_pairs = []
    all_applicability = []
    classifications: dict[str, pd.DataFrame] = {}
    for task, bundle in [("binary", binary), ("multiclass", multiclass)]:
        classification = _classification(bundle["scenarios"])
        classifications[task] = classification
        all_methods.append(
            method_summaries(
                classification, task, args.bootstrap_replicates, rng
            )
        )
        all_pairs.append(
            paired_comparisons(
                classification, task, args.bootstrap_replicates, rng
            )
        )
        all_endpoints.append(
            separate_endpoint_summaries(
                bundle["scenarios"], task, args.bootstrap_replicates, rng
            )
        )
        xpe_summary, xpe_pair = xpe_summaries(
            bundle["xpe"],
            bundle["batches"],
            task,
            args.bootstrap_replicates,
            rng,
        )
        if not xpe_summary.empty:
            all_xpe.append(xpe_summary)
            all_xpe_pairs.append(xpe_pair)
        all_applicability.append(
            shd_applicability(
                bundle["shd_calibration"], bundle["scenarios"], task
            )
        )

    method_table = pd.concat(all_methods, ignore_index=True)
    pair_table = pd.concat(all_pairs, ignore_index=True)
    endpoint_table = pd.concat(all_endpoints, ignore_index=True)
    xpe_table = pd.concat(all_xpe, ignore_index=True) if all_xpe else pd.DataFrame()
    xpe_pair_table = (
        pd.concat(all_xpe_pairs, ignore_index=True)
        if all_xpe_pairs
        else pd.DataFrame()
    )
    applicability = pd.concat(all_applicability, ignore_index=True)
    sensitivity = shd_sensitivity_summary(args.shd_sensitivity_dir)

    atomic_parquet(method_table, output / "performance_method_summaries.parquet")
    method_table.to_csv(output / "performance_method_summaries.csv", index=False)
    atomic_parquet(pair_table, output / "paired_method_comparisons.parquet")
    pair_table.to_csv(output / "paired_method_comparisons.csv", index=False)
    endpoint_table.to_csv(output / "separate_endpoint_summaries.csv", index=False)
    xpe_table.to_csv(output / "xpe_summaries.csv", index=False)
    xpe_pair_table.to_csv(output / "xpe_paired_confidence.csv", index=False)
    applicability.to_csv(output / "shd_applicability.csv", index=False)
    sensitivity.to_csv(output / "shd_sensitivity_summary.csv", index=False)

    mixed_metadata: list[dict[str, Any]] = []
    if not args.skip_mixed_models:
        binary_classification = classifications["binary"]
        linear_mae, meta = fit_linear_mixed_model(
            binary_classification, "mean_absolute_error"
        )
        linear_mae.to_csv(output / "mixed_linear_mae.csv", index=False)
        mixed_metadata.append(meta)
        linear_failure, meta = fit_linear_mixed_model(
            binary_classification, "risk_failure_rate"
        )
        linear_failure.to_csv(
            output / "mixed_linear_failure_rate.csv", index=False
        )
        mixed_metadata.append(meta)
        binomial, variance, meta = fit_binomial_mixed_model(binary_classification)
        binomial.to_csv(output / "mixed_binomial_fixed.csv", index=False)
        variance.to_csv(output / "mixed_binomial_variance.csv", index=False)
        mixed_metadata.append(meta)
        gee, meta = fit_binomial_gee(binary_classification)
        gee.to_csv(output / "gee_majority_failure.csv", index=False)
        mixed_metadata.append(meta)
    atomic_json(mixed_metadata, output / "mixed_model_metadata.json")

    _write_report(output, method_table, xpe_table, applicability, sensitivity)
    inputs = {}
    for task, bundle in [("binary", binary), ("multiclass", multiclass)]:
        inputs[task] = {
            name: _sha256(bundle["root"] / name)
            for name in [
                "step12_status.json",
                "step12_manifest.json",
                "step12_audit.json",
                "scenario_metrics.parquet",
                "batch_metrics.parquet",
                "xpe_metrics.parquet",
                "shd_calibration.parquet",
            ]
        }
    atomic_json(
        {
            "schema_version": 1,
            "complete": True,
            "binary_predictor_streams": binary["status"][
                "completed_predictor_streams"
            ],
            "multiclass_predictor_streams": multiclass["status"][
                "completed_predictor_streams"
            ],
            "bootstrap_replicates": args.bootstrap_replicates,
            "random_seed": args.random_seed,
            "mixed_models_run": not args.skip_mixed_models,
            "endpoint_separation": True,
            "inputs_sha256": inputs,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output / "schema13_analysis_manifest.json",
    )
    print("=== TABMON SCHEMA-13 FINAL ANALYSIS ===")
    print(f"Performance summaries: {len(method_table)}")
    print(f"Paired comparisons: {len(pair_table)}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
