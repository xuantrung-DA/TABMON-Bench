"""Post-hoc Phase-1 audit for the frozen TABMON-Bench schema-v7 output.

This analysis does not rerun models, shifts, or monitors.  It addresses three
review risks using the saved batch trajectories:

1. localize the Random-Forest pipeline-corruption anomaly;
2. verify the attribution Recall@k/failure semantics;
3. test sensitivity to risk, attribution, harm, and alarm thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


RISK_TOLERANCES = (0.02, 0.05, 0.10)
ATTRIBUTION_THRESHOLDS = (0.25, 0.50, 0.75)
ATTRIBUTION_K_VALUES = (1, 3, 5)
HARM_THRESHOLDS = (0.02, 0.05, 0.10)
ALARM_ALPHAS = (0.005, 0.01, 0.025, 0.05)
RNG_SEED = 17_021


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zip_member(archive: zipfile.ZipFile, filename: str) -> str:
    matches = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir() and Path(info.filename).name == filename
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename}; found {matches}")
    return matches[0]


def load_v7_results(path: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Load the manifest, status, and batch table from a directory or zip."""
    if path.is_dir():
        manifest = json.loads(
            (path / "benchmark_manifest.json").read_text(encoding="utf-8")
        )
        status = json.loads(
            (path / "benchmark_status.json").read_text(encoding="utf-8")
        )
        batches = pd.read_parquet(path / "batch_metrics.parquet")
    elif path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(
                archive.read(_zip_member(archive, "benchmark_manifest.json"))
            )
            status = json.loads(
                archive.read(_zip_member(archive, "benchmark_status.json"))
            )
            batches = pd.read_parquet(
                io.BytesIO(archive.read(_zip_member(archive, "batch_metrics.parquet")))
            )
    else:
        raise FileNotFoundError(path)

    if int(manifest.get("schema_version", -1)) != 7:
        raise ValueError("Phase 1 requires schema-v7 results")
    if not status.get("complete") or int(status.get("errors_this_run", -1)) != 0:
        raise ValueError("Phase 1 requires a complete, error-free run")
    if len(batches) != 62_000 or batches["scenario_id"].nunique() != 6_200:
        raise ValueError(
            f"Unexpected v7 batch dimensions: rows={len(batches)}, "
            f"scenarios={batches['scenario_id'].nunique()}"
        )
    return manifest, status, batches


def _cluster_codes(
    frame: pd.DataFrame, cluster_columns: Iterable[str]
) -> list[tuple[np.ndarray, int]]:
    result = []
    for column in cluster_columns:
        codes, levels = pd.factorize(frame[column], sort=True)
        result.append((codes, len(levels)))
    return result


def cluster_bootstrap_ratio(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
    cluster_columns: tuple[str, ...],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Ratio-of-sums interval while keeping cluster blocks together."""
    numerator_values = pd.to_numeric(frame[numerator], errors="coerce").to_numpy(
        dtype=float
    )
    denominator_values = pd.to_numeric(
        frame[denominator], errors="coerce"
    ).to_numpy(dtype=float)
    denominator_total = float(np.nansum(denominator_values))
    if denominator_total <= 0:
        return np.nan, np.nan, np.nan
    estimate = float(np.nansum(numerator_values) / denominator_total)
    codes = _cluster_codes(frame, cluster_columns)
    draws = []
    for _ in range(replicates):
        weights = np.ones(len(frame), dtype=float)
        for level_codes, n_levels in codes:
            counts = rng.multinomial(n_levels, np.full(n_levels, 1.0 / n_levels))
            weights *= counts[level_codes]
        weighted_denominator = float(np.sum(weights * denominator_values))
        if weighted_denominator > 0:
            draws.append(
                float(np.sum(weights * numerator_values) / weighted_denominator)
            )
    if not draws:
        return estimate, np.nan, np.nan
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return estimate, float(lower), float(upper)


def _pipeline_scenario_counts(
    batches: pd.DataFrame, harm_threshold: float
) -> pd.DataFrame:
    frame = batches[
        batches["monitor"].eq("confidence")
        & batches["shift"].eq("pipeline")
        & pd.to_numeric(batches["shift_fraction"], errors="coerce").gt(0.0)
    ].copy()
    true_risk = pd.to_numeric(frame["true_excess_risk"], errors="coerce")
    estimated = pd.to_numeric(frame["estimated_risk_change"], errors="coerce")
    frame["harmful"] = true_risk.gt(harm_threshold)
    frame["sign_reversal"] = frame["harmful"] & estimated.lt(0.0)
    frame["harmful_confidence_increase"] = frame["harmful"] & pd.to_numeric(
        frame["prediction_confidence_shift"], errors="coerce"
    ).gt(0.0)
    keys = [
        "scenario_id",
        "dataset",
        "model",
        "severity",
        "mode",
        "seed",
    ]
    return (
        frame.groupby(keys, dropna=False)
        .agg(
            post_batches=("batch_index", "size"),
            harmful_batches=("harmful", "sum"),
            reversed_batches=("sign_reversal", "sum"),
            harmful_confidence_increase_batches=(
                "harmful_confidence_increase",
                "sum",
            ),
            true_excess_risk_mean=("true_excess_risk", "mean"),
            estimated_risk_change_mean=("estimated_risk_change", "mean"),
            prediction_confidence_shift_mean=(
                "prediction_confidence_shift",
                "mean",
            ),
            prediction_entropy_shift_mean=("prediction_entropy_shift", "mean"),
        )
        .reset_index()
    )


def _pipeline_group_table(
    scenarios: pd.DataFrame,
    group_columns: list[str],
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    records = []
    iterator = scenarios.groupby(group_columns, dropna=False, sort=True)
    for group_value, frame in iterator:
        if not isinstance(group_value, tuple):
            group_value = (group_value,)
        labels = dict(zip(group_columns, group_value))
        clusters = tuple(
            column for column in ("dataset", "seed") if frame[column].nunique() > 1
        )
        if not clusters:
            clusters = ("seed",)
        estimate, lower, upper = cluster_bootstrap_ratio(
            frame,
            "reversed_batches",
            "harmful_batches",
            clusters,
            replicates,
            rng,
        )
        conf_estimate, conf_lower, conf_upper = cluster_bootstrap_ratio(
            frame,
            "harmful_confidence_increase_batches",
            "harmful_batches",
            clusters,
            replicates,
            rng,
        )
        records.append(
            {
                **labels,
                "post_batches": int(frame["post_batches"].sum()),
                "harmful_batches": int(frame["harmful_batches"].sum()),
                "reversed_batches": int(frame["reversed_batches"].sum()),
                "sign_reversal_rate": estimate,
                "sign_reversal_ci_lower": lower,
                "sign_reversal_ci_upper": upper,
                "harmful_confidence_increase_rate": conf_estimate,
                "harmful_confidence_increase_ci_lower": conf_lower,
                "harmful_confidence_increase_ci_upper": conf_upper,
                "true_excess_risk_mean": frame["true_excess_risk_mean"].mean(),
                "estimated_risk_change_mean": frame[
                    "estimated_risk_change_mean"
                ].mean(),
                "prediction_confidence_shift_mean": frame[
                    "prediction_confidence_shift_mean"
                ].mean(),
                "prediction_entropy_shift_mean": frame[
                    "prediction_entropy_shift_mean"
                ].mean(),
                "n_scenarios": len(frame),
                "n_datasets": frame["dataset"].nunique(),
                "n_seeds": frame["seed"].nunique(),
            }
        )
    return pd.DataFrame(records)


def pipeline_anomaly_analysis(
    batches: pd.DataFrame, replicates: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = _pipeline_scenario_counts(batches, harm_threshold=0.05)
    by_model = _pipeline_group_table(scenarios, ["model"], replicates, rng)
    by_cell = _pipeline_group_table(
        scenarios, ["model", "dataset", "severity"], replicates, rng
    )
    return by_model, by_cell


def attribution_recall_from_json(
    ground_truth_json: str, predicted_json: str, k: int
) -> float:
    ground_truth = json.loads(ground_truth_json)
    predicted = json.loads(predicted_json)
    active = {name for name, value in ground_truth.items() if abs(float(value)) > 0.0}
    if not active:
        return np.nan
    ranking = sorted(
        predicted,
        key=lambda name: (-abs(float(predicted[name])), str(name)),
    )
    selected = set(ranking[: min(k, len(ranking))])
    return len(active & selected) / len(active)


def attribution_metric_audit(
    batches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = batches[
        batches["monitor"].eq("drift_shap")
        & pd.to_numeric(batches["shift_fraction"], errors="coerce").gt(0.0)
        & batches["top3_recall"].notna()
    ].copy()
    for k in ATTRIBUTION_K_VALUES:
        frame[f"recall_at_{k}_recomputed"] = [
            attribution_recall_from_json(truth, prediction, k)
            for truth, prediction in zip(
                frame["ground_truth_attribution_json"],
                frame["predicted_attribution_json"],
            )
        ]
    if not np.allclose(
        pd.to_numeric(frame["top3_recall"], errors="raise"),
        frame["recall_at_3_recomputed"],
        atol=1e-12,
    ):
        raise RuntimeError("Stored top3_recall does not match JSON recomputation")
    expected_failure = frame["recall_at_3_recomputed"].lt(0.5)
    if not np.array_equal(
        frame["attribution_failure"].astype(bool).to_numpy(),
        expected_failure.to_numpy(),
    ):
        raise RuntimeError("Stored attribution_failure violates Recall@3 < 0.5")

    batch_summary = (
        frame.groupby("shift", sort=True)
        .agg(
            recall_at_1=("recall_at_1_recomputed", "mean"),
            recall_at_3=("recall_at_3_recomputed", "mean"),
            recall_at_5=("recall_at_5_recomputed", "mean"),
            failure_rate=("attribution_failure", "mean"),
            n_batches=("batch_index", "size"),
        )
        .reset_index()
    )
    distribution = (
        frame.groupby(["shift", "recall_at_3_recomputed"], sort=True)
        .size()
        .rename("n_batches")
        .reset_index()
    )
    distribution["fraction"] = distribution["n_batches"] / distribution.groupby(
        "shift"
    )["n_batches"].transform("sum")

    scenario = (
        frame.groupby(
            [
                "scenario_id",
                "dataset",
                "model",
                "shift",
                "severity",
                "mode",
                "seed",
            ],
            dropna=False,
        )
        .agg(
            recall_at_3=("recall_at_3_recomputed", "mean"),
            failure_rate=("attribution_failure", "mean"),
        )
        .reset_index()
    )
    scenario_summary = (
        scenario.groupby("shift", sort=True)
        .agg(
            scenario_mean_recall_at_3=("recall_at_3", "mean"),
            scenario_mean_failure_rate=("failure_rate", "mean"),
            n_scenarios=("scenario_id", "size"),
        )
        .reset_index()
    )
    return batch_summary, distribution, scenario_summary


def threshold_sensitivity(
    batches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase = batches[
        batches["shift"].eq("no_shift")
        | pd.to_numeric(batches["shift_fraction"], errors="coerce").gt(0.0)
    ].copy()
    confidence = phase[phase["monitor"].eq("confidence")].copy()
    risk_records = []
    for tolerance in RISK_TOLERANCES:
        confidence["sensitive_failure"] = pd.to_numeric(
            confidence["risk_abs_error"], errors="coerce"
        ).gt(tolerance)
        scenario = (
            confidence.groupby(["scenario_id", "shift"], dropna=False)[
                "sensitive_failure"
            ]
            .mean()
            .reset_index()
        )
        for shift, frame in scenario.groupby("shift", sort=True):
            risk_records.append(
                {
                    "risk_failure_tolerance": tolerance,
                    "shift": shift,
                    "failure_rate": frame["sensitive_failure"].mean(),
                    "n_scenarios": len(frame),
                }
            )

    attribution = phase[
        phase["monitor"].eq("drift_shap") & phase["top3_recall"].notna()
    ].copy()
    attribution_records = []
    for k in ATTRIBUTION_K_VALUES:
        recalls = np.asarray(
            [
                attribution_recall_from_json(truth, prediction, k)
                for truth, prediction in zip(
                    attribution["ground_truth_attribution_json"],
                    attribution["predicted_attribution_json"],
                )
            ],
            dtype=float,
        )
        for threshold in ATTRIBUTION_THRESHOLDS:
            working = attribution[["scenario_id", "shift"]].copy()
            working["recall"] = recalls
            working["failure"] = recalls < threshold
            scenario = (
                working.groupby(["scenario_id", "shift"], dropna=False)
                .agg(recall=("recall", "mean"), failure=("failure", "mean"))
                .reset_index()
            )
            for shift, frame in scenario.groupby("shift", sort=True):
                attribution_records.append(
                    {
                        "k": k,
                        "attribution_failure_threshold": threshold,
                        "shift": shift,
                        "mean_recall": frame["recall"].mean(),
                        "failure_rate": frame["failure"].mean(),
                        "n_scenarios": len(frame),
                    }
                )

    harm_records = []
    for threshold in HARM_THRESHOLDS:
        scenario = _pipeline_scenario_counts(batches, threshold)
        for model, frame in scenario.groupby("model", sort=True):
            harmful = int(frame["harmful_batches"].sum())
            reversed_count = int(frame["reversed_batches"].sum())
            harm_records.append(
                {
                    "harm_threshold": threshold,
                    "model": model,
                    "harmful_batches": harmful,
                    "reversed_batches": reversed_count,
                    "sign_reversal_rate": (
                        reversed_count / harmful if harmful else np.nan
                    ),
                }
            )

    null = batches[batches["shift"].eq("no_shift")].copy()
    alarm_records = []
    for alpha in ALARM_ALPHAS:
        null["sensitive_alarm"] = pd.to_numeric(
            null["alarm_p_value"], errors="coerce"
        ).le(alpha)
        scenario = (
            null.groupby(["scenario_id", "monitor"], dropna=False)[
                "sensitive_alarm"
            ]
            .agg(["mean", "any"])
            .reset_index()
        )
        for monitor, frame in scenario.groupby("monitor", sort=True):
            p_values = pd.to_numeric(
                null.loc[null["monitor"].eq(monitor), "alarm_p_value"],
                errors="coerce",
            )
            alarm_records.append(
                {
                    "alpha": alpha,
                    "monitor": monitor,
                    "null_batch_alarm_rate": frame["mean"].mean(),
                    "null_scenario_any_alarm_rate": frame["any"].mean(),
                    "minimum_observed_p_value": p_values.min(),
                    "n_null_scenarios": len(frame),
                }
            )

    return (
        pd.DataFrame(risk_records),
        pd.DataFrame(attribution_records),
        pd.DataFrame(harm_records),
        pd.DataFrame(alarm_records),
    )


def _fmt(value: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(float(value)) else f"{value:.{digits}f}"


def write_report(
    output_dir: Path,
    model_summary: pd.DataFrame,
    cell_summary: pd.DataFrame,
    attribution_summary: pd.DataFrame,
    attribution_distribution: pd.DataFrame,
    risk_sensitivity: pd.DataFrame,
    harm_sensitivity: pd.DataFrame,
    alarm_sensitivity: pd.DataFrame,
) -> None:
    rf = model_summary.set_index("model").loc["rf"]
    others = model_summary[~model_summary["model"].eq("rf")]
    rf_cells = cell_summary[cell_summary["model"].eq("rf")]
    rf_harmful_datasets = int(
        rf_cells.groupby("dataset")["harmful_batches"].sum().gt(0).sum()
    )
    correlated = attribution_distribution[
        attribution_distribution["shift"].eq("correlated")
    ].set_index("recall_at_3_recomputed")
    half_recall = (
        float(correlated.loc[0.5, "fraction"]) if 0.5 in correlated.index else 0.0
    )
    concept_sensitivity = risk_sensitivity[
        risk_sensitivity["shift"].eq("concept")
    ]
    pipeline_sensitivity = harm_sensitivity.groupby("harm_threshold").agg(
        harmful=("harmful_batches", "sum"),
        reversed=("reversed_batches", "sum"),
    )
    pipeline_sensitivity["rate"] = (
        pipeline_sensitivity["reversed"] / pipeline_sensitivity["harmful"]
    )
    alpha_one = alarm_sensitivity[np.isclose(alarm_sensitivity["alpha"], 0.01)]

    report = f"""# TABMON-Bench schema-v7 Phase-1 audit

## Scope

This is a post-hoc audit of the frozen v7 batch trajectories. No model was
retrained, no shift was regenerated, and no monitor score was recomputed.

## 1. Random-Forest pipeline anomaly

- RF sign reversal among harmful pipeline batches is
  **{_fmt(rf['sign_reversal_rate'] * 100, 1)}%**
  (95% cluster interval {_fmt(rf['sign_reversal_ci_lower'] * 100, 1)}--
  {_fmt(rf['sign_reversal_ci_upper'] * 100, 1)}%;
  {int(rf['reversed_batches'])}/{int(rf['harmful_batches'])} batches).
- The other model-level rates span
  **{_fmt(others['sign_reversal_rate'].min() * 100, 1)}%--
  {_fmt(others['sign_reversal_rate'].max() * 100, 1)}%**.
- RF has harmful batches in {rf_harmful_datasets}/5 datasets. Its exception is
  therefore heterogeneous rather than a domain-invariant protection. The
  dataset/severity table identifies which domains drive the aggregate.
- Prediction-confidence and entropy shifts are reported beside risk response;
  these are mechanism indicators, not causal proof about the sentinel.

## 2. Attribution metric audit

- Stored Recall@3 and `attribution_failure` exactly match recomputation from
  the saved attribution JSON.
- For correlated shifts, **{_fmt(half_recall * 100, 1)}%** of scored batches
  have Recall@3 = 0.5. Because failure is defined as Recall@3 **< 0.5**, these
  partial recoveries lower mean recall but are not failures. This explains why
  correlated-shift mean recall can be below the covariate value while its
  binary failure rate is also lower.
- This is a threshold-semantics effect, not a data inconsistency. The sensitivity
  table reports k in {ATTRIBUTION_K_VALUES} and thresholds in
  {ATTRIBUTION_THRESHOLDS}.

## 3. Threshold sensitivity

- Concept-shift risk failure ranges from
  **{_fmt(concept_sensitivity['failure_rate'].min() * 100, 1)}% to
  {_fmt(concept_sensitivity['failure_rate'].max() * 100, 1)}%** over risk-error
  tolerances {RISK_TOLERANCES}.
- Overall pipeline sign reversal ranges from
  **{_fmt(pipeline_sensitivity['rate'].min() * 100, 1)}% to
  {_fmt(pipeline_sensitivity['rate'].max() * 100, 1)}%** over harmful-risk
  thresholds {HARM_THRESHOLDS}.
- At alpha=0.01, null batch alarm rates are:
  {', '.join(f"{row.monitor}={row.null_batch_alarm_rate:.3f}" for row in alpha_one.itertuples())}.
  The saved run used only 100 null calibration batches, so the smallest
  attainable conformal p-value is approximately 1/101. Alarm recalibration is
  a separate next-stage experiment, not silently repaired in this audit.

## Decision

Phase 1 validates the saved metric implementation and localizes the RF result,
but it does not justify a model-wide robustness claim for RF. The paper-safe
claim remains that pipeline sign reversal is strong overall and highly
model/domain dependent. Threshold-sensitive quantities must be reported with
their definitions or sensitivity ranges.
"""
    (output_dir / "phase1_audit_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.results.resolve()
    output_dir = args.output_dir.resolve()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, status, batches = load_v7_results(source)
    rng = np.random.default_rng(RNG_SEED)
    model_summary, cell_summary = pipeline_anomaly_analysis(
        batches, args.bootstrap_replicates, rng
    )
    attribution_summary, attribution_distribution, scenario_summary = (
        attribution_metric_audit(batches)
    )
    (
        risk_sensitivity,
        attribution_sensitivity,
        harm_sensitivity,
        alarm_sensitivity,
    ) = threshold_sensitivity(batches)

    outputs = {
        "pipeline_anomaly_by_model.csv": model_summary,
        "pipeline_anomaly_by_model_dataset_severity.csv": cell_summary,
        "attribution_metric_audit_batch.csv": attribution_summary,
        "attribution_recall_distribution.csv": attribution_distribution,
        "attribution_metric_audit_scenario.csv": scenario_summary,
        "risk_failure_threshold_sensitivity.csv": risk_sensitivity,
        "attribution_threshold_sensitivity.csv": attribution_sensitivity,
        "pipeline_harm_threshold_sensitivity.csv": harm_sensitivity,
        "null_alarm_alpha_sensitivity.csv": alarm_sensitivity,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    analysis_manifest = {
        "analysis": "schema-v7 Phase-1 post-hoc audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": sha256_file(source) if source.is_file() else None,
        "source_schema_version": manifest["schema_version"],
        "source_complete": status["complete"],
        "source_monitor_scenario_evaluations": batches["scenario_id"].nunique(),
        "source_monitor_batch_records": len(batches),
        "bootstrap_replicates": args.bootstrap_replicates,
        "random_seed": RNG_SEED,
        "risk_tolerances": RISK_TOLERANCES,
        "attribution_thresholds": ATTRIBUTION_THRESHOLDS,
        "attribution_k_values": ATTRIBUTION_K_VALUES,
        "harm_thresholds": HARM_THRESHOLDS,
        "alarm_alphas": ALARM_ALPHAS,
        "new_model_training": False,
        "new_shift_generation": False,
        "figures_generated": False,
    }
    manifest_path = output_dir / "phase1_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir,
        model_summary,
        cell_summary,
        attribution_summary,
        attribution_distribution,
        risk_sensitivity,
        harm_sensitivity,
        alarm_sensitivity,
    )
    print("=== TABMON SCHEMA-V7 PHASE 1 ===")
    print(f"Source evaluations: {batches['scenario_id'].nunique()}")
    print(f"Source batches: {len(batches)}")
    print(f"Saved at: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
