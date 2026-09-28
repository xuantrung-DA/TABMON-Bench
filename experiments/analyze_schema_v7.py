"""Paper-level statistical analysis for a completed TABMON-Bench schema-v7 run.

The analysis unit is a scenario/seed, not an individual batch.  Shifted
scenarios use post-shift batches only; null scenarios use their complete
trajectory.  Cross-domain confidence intervals use a two-way cluster
bootstrap over datasets and seeds so the repeated model/shift measurements are
kept together.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t


SCHEMA_VERSION = 7
DESCRIPTOR_COLUMNS = [
    "scenario_id",
    "dataset",
    "model",
    "monitor",
    "shift",
    "severity",
    "mode",
    "seed",
]
ANALYSIS_METRICS = [
    "true_excess_risk",
    "estimated_risk_change",
    "risk_abs_error",
    "risk_failure",
    "coverage",
    "interval_width",
    "alarm",
    "alarm_failure",
    "monitor_failure",
    "monitor_score",
    "domain_classifier_auc",
    "effective_sample_size_fraction",
    "prediction_confidence_shift",
    "prediction_entropy_shift",
    "attribution_failure",
    "top3_recall",
    "ndcg@3",
    "attribution_abs_mass",
    "realized_wasserstein_std",
    "realized_support_violation_rate",
]
SEVERITY_METRICS = [
    "true_excess_risk",
    "estimated_risk_change",
    "risk_abs_error",
    "risk_failure",
    "alarm",
    "alarm_failure",
    "monitor_failure",
    "monitor_score",
    "domain_classifier_auc",
    "prediction_confidence_shift",
    "attribution_failure",
    "top3_recall",
    "realized_wasserstein_std",
    "realized_support_violation_rate",
]
SEVERITY_ORDER = {"low": 1.0, "medium": 2.0, "high": 3.0}


@dataclass(frozen=True)
class HeadlineSpec:
    name: str
    monitor: str
    shift: str
    metric: str
    description: str


HEADLINE_SPECS = [
    HeadlineSpec(
        "null_confidence_risk_mae",
        "confidence",
        "no_shift",
        "risk_abs_error",
        "Confidence absolute risk error under the null control",
    ),
    HeadlineSpec(
        "null_confidence_risk_failure",
        "confidence",
        "no_shift",
        "risk_failure",
        "Confidence risk-failure rate under the null control",
    ),
    HeadlineSpec(
        "null_confidence_coverage",
        "confidence",
        "no_shift",
        "coverage",
        "Confidence batch-risk interval coverage under the null control",
    ),
    HeadlineSpec(
        "null_confidence_alarm_rate",
        "confidence",
        "no_shift",
        "alarm",
        "Confidence alarm rate under the null control",
    ),
    HeadlineSpec(
        "null_drift_shap_alarm_rate",
        "drift_shap",
        "no_shift",
        "alarm",
        "DriftSHAP alarm rate under the null control",
    ),
    HeadlineSpec(
        "concept_true_risk_change",
        "confidence",
        "concept",
        "true_excess_risk",
        "Oracle excess log loss under post-shift concept batches",
    ),
    HeadlineSpec(
        "concept_confidence_estimated_risk_change",
        "confidence",
        "concept",
        "estimated_risk_change",
        "Confidence estimated risk change under concept shift",
    ),
    HeadlineSpec(
        "concept_confidence_risk_failure",
        "confidence",
        "concept",
        "risk_failure",
        "Confidence risk-failure rate under concept shift",
    ),
    HeadlineSpec(
        "concept_confidence_alarm_rate",
        "confidence",
        "concept",
        "alarm",
        "Confidence alarm rate under concept shift",
    ),
    HeadlineSpec(
        "concept_drift_shap_domain_auc",
        "drift_shap",
        "concept",
        "domain_classifier_auc",
        "Observable domain-classifier AUC under concept shift",
    ),
    HeadlineSpec(
        "concept_drift_shap_monitor_failure",
        "drift_shap",
        "concept",
        "monitor_failure",
        "DriftSHAP combined failure rate under concept shift",
    ),
    HeadlineSpec(
        "pipeline_true_risk_change",
        "confidence",
        "pipeline",
        "true_excess_risk",
        "Oracle excess log loss under pipeline corruption",
    ),
    HeadlineSpec(
        "pipeline_confidence_estimated_risk_change",
        "confidence",
        "pipeline",
        "estimated_risk_change",
        "Confidence estimated risk change under pipeline corruption",
    ),
    HeadlineSpec(
        "pipeline_confidence_risk_mae",
        "confidence",
        "pipeline",
        "risk_abs_error",
        "Confidence absolute risk error under pipeline corruption",
    ),
    HeadlineSpec(
        "pipeline_confidence_risk_failure",
        "confidence",
        "pipeline",
        "risk_failure",
        "Confidence risk-failure rate under pipeline corruption",
    ),
    HeadlineSpec(
        "pipeline_confidence_alarm_rate",
        "confidence",
        "pipeline",
        "alarm",
        "Confidence alarm rate under pipeline corruption",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-level uncertainty analyses from schema v7."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--variance-bootstrap-replicates", type=int, default=500)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=17_021)
    return parser.parse_args()


def _validate_inputs(results_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = results_dir / "benchmark_manifest.json"
    status_path = results_dir / "benchmark_status.json"
    batch_path = results_dir / "batch_metrics.parquet"
    for path in (manifest_path, status_path, batch_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Statistical analysis requires a schema-v7 manifest")
    if not status.get("complete") or int(status.get("errors_this_run", -1)) != 0:
        raise ValueError("Benchmark must be complete and error-free")
    batches = pd.read_parquet(batch_path)
    missing = set(DESCRIPTOR_COLUMNS + ["shift_fraction"]) - set(batches.columns)
    if missing:
        raise ValueError(f"Missing batch columns: {sorted(missing)}")
    if batches.duplicated(["scenario_id", "batch_index"]).any():
        raise ValueError("Duplicate scenario/batch rows")
    if not batches.groupby("scenario_id").size().eq(10).all():
        raise ValueError("Every scenario must contain exactly ten batches")
    return manifest, batches


def _numeric_mean(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    result: dict[str, float] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        result[column] = float(values.mean()) if values.notna().any() else np.nan
    return pd.Series(result)


def build_scenario_metrics(batches: pd.DataFrame) -> pd.DataFrame:
    """Collapse batch trajectories without treating batches as replicates."""
    analysis_rows = batches[
        batches["shift"].eq("no_shift")
        | pd.to_numeric(batches["shift_fraction"], errors="coerce").gt(0.0)
    ].copy()
    available = [column for column in ANALYSIS_METRICS if column in analysis_rows]
    grouped = analysis_rows.groupby(DESCRIPTOR_COLUMNS, dropna=False, sort=False)
    metrics = grouped.apply(
        lambda frame: _numeric_mean(frame, available),
        include_groups=False,
    ).reset_index()
    counts = grouped.size().rename("analysis_batch_count").reset_index()
    metrics = metrics.merge(counts, on=DESCRIPTOR_COLUMNS, validate="one_to_one")
    if metrics["scenario_id"].duplicated().any():
        raise RuntimeError("Scenario aggregation produced duplicate IDs")
    return metrics


def student_t_interval(
    values: Iterable[float], confidence_level: float = 0.95
) -> tuple[float, float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = float(np.mean(array))
    if n == 1:
        return mean, np.nan, np.nan, np.nan, 1
    standard_deviation = float(np.std(array, ddof=1))
    standard_error = standard_deviation / math.sqrt(n)
    critical = float(t.ppf((1.0 + confidence_level) / 2.0, df=n - 1))
    return (
        mean,
        standard_deviation,
        mean - critical * standard_error,
        mean + critical * standard_error,
        n,
    )


def seed_level_intervals(
    scenarios: pd.DataFrame, confidence_level: float
) -> pd.DataFrame:
    keys = ["dataset", "model", "monitor", "shift", "severity", "mode"]
    records: list[dict[str, Any]] = []
    metrics = [column for column in ANALYSIS_METRICS if column in scenarios]
    for group_values, frame in scenarios.groupby(keys, dropna=False, sort=True):
        descriptors = dict(zip(keys, group_values))
        for metric in metrics:
            mean, std, lower, upper, n = student_t_interval(
                pd.to_numeric(frame[metric], errors="coerce"), confidence_level
            )
            if n == 0:
                continue
            records.append(
                {
                    **descriptors,
                    "metric": metric,
                    "estimate": mean,
                    "seed_std": std,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_seeds": n,
                    "ci_method": "Student-t across scenario seeds",
                }
            )
    return pd.DataFrame(records)


def _cluster_code_arrays(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    dataset_codes, datasets = pd.factorize(frame["dataset"], sort=True)
    seed_codes, seeds = pd.factorize(frame["seed"], sort=True)
    return dataset_codes, seed_codes, len(datasets), len(seeds)


def _cluster_weights_from_codes(
    dataset_codes: np.ndarray,
    seed_codes: np.ndarray,
    n_datasets: int,
    n_seeds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dataset_counts = rng.multinomial(
        n_datasets, np.full(n_datasets, 1.0 / n_datasets)
    )
    seed_counts = rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds))
    return dataset_counts[dataset_codes] * seed_counts[seed_codes]


def _cluster_weights(
    frame: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Return one two-way dataset/seed bootstrap weight vector."""
    codes = _cluster_code_arrays(frame)
    return _cluster_weights_from_codes(*codes, rng)


def two_way_cluster_bootstrap_mean(
    frame: pd.DataFrame,
    metric: str,
    replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    working = frame.loc[valid, ["dataset", "seed"]].copy()
    values = values[valid]
    if not len(values):
        return np.nan, np.nan, np.nan, 0
    estimate = float(np.mean(values))
    draws = np.empty(replicates, dtype=float)
    codes = _cluster_code_arrays(working)
    for index in range(replicates):
        weights = _cluster_weights_from_codes(*codes, rng)
        draws[index] = np.average(values, weights=weights)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return estimate, float(lower), float(upper), len(values)


def headline_estimates(
    scenarios: pd.DataFrame,
    replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    records = []
    for spec in HEADLINE_SPECS:
        frame = scenarios[
            scenarios["monitor"].eq(spec.monitor)
            & scenarios["shift"].eq(spec.shift)
        ].copy()
        estimate, lower, upper, n = two_way_cluster_bootstrap_mean(
            frame, spec.metric, replicates, confidence_level, rng
        )
        seed_means = frame.groupby("seed")[spec.metric].mean()
        _, seed_std, seed_lower, seed_upper, n_seeds = student_t_interval(
            seed_means, confidence_level
        )
        records.append(
            {
                "finding": spec.name,
                "description": spec.description,
                "monitor": spec.monitor,
                "shift": spec.shift,
                "metric": spec.metric,
                "estimate": estimate,
                "cluster_ci_lower": lower,
                "cluster_ci_upper": upper,
                "seed_std": seed_std,
                "seed_t_ci_lower": seed_lower,
                "seed_t_ci_upper": seed_upper,
                "n_scenarios": n,
                "n_datasets": frame["dataset"].nunique(),
                "n_seeds": n_seeds,
                "cluster_bootstrap_replicates": replicates,
            }
        )
    return pd.DataFrame(records)


def _two_way_cluster_bootstrap_ratio(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
    replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    numerator_values = frame[numerator].to_numpy(dtype=float)
    denominator_values = frame[denominator].to_numpy(dtype=float)
    estimate = float(numerator_values.sum() / denominator_values.sum())
    draws = np.empty(replicates, dtype=float)
    codes = _cluster_code_arrays(frame)
    for index in range(replicates):
        weights = _cluster_weights_from_codes(*codes, rng)
        draws[index] = float(
            np.sum(weights * numerator_values)
            / np.sum(weights * denominator_values)
        )
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return estimate, float(lower), float(upper)


def pipeline_sign_reversal(
    batches: pd.DataFrame,
    replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    frame = batches[
        batches["monitor"].eq("confidence")
        & batches["shift"].eq("pipeline")
        & pd.to_numeric(batches["shift_fraction"], errors="coerce").gt(0.0)
    ].copy()
    threshold = pd.to_numeric(frame["risk_event_threshold"], errors="coerce")
    true_risk = pd.to_numeric(frame["true_excess_risk"], errors="coerce")
    estimated_risk = pd.to_numeric(
        frame["estimated_risk_change"], errors="coerce"
    )
    frame["harmful"] = true_risk.gt(threshold)
    frame["sign_reversal"] = frame["harmful"] & estimated_risk.lt(0.0)
    scenario_keys = DESCRIPTOR_COLUMNS
    counts = (
        frame.groupby(scenario_keys, dropna=False)
        .agg(
            harmful_batches=("harmful", "sum"),
            reversed_batches=("sign_reversal", "sum"),
        )
        .reset_index()
    )
    counts = counts[counts["harmful_batches"].gt(0)].copy()

    scopes: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("dataset", ["dataset"]),
        ("model", ["model"]),
        ("dataset_model", ["dataset", "model"]),
    ]
    records: list[dict[str, Any]] = []
    for scope, columns in scopes:
        iterator = (
            [((), counts)]
            if not columns
            else counts.groupby(columns, dropna=False, sort=True)
        )
        for group_value, group in iterator:
            if columns and not isinstance(group_value, tuple):
                group_value = (group_value,)
            labels = dict(zip(columns, group_value)) if columns else {}
            estimate, lower, upper = _two_way_cluster_bootstrap_ratio(
                group,
                "reversed_batches",
                "harmful_batches",
                replicates,
                confidence_level,
                rng,
            )
            records.append(
                {
                    "scope": scope,
                    "dataset": labels.get("dataset", "all"),
                    "model": labels.get("model", "all"),
                    "sign_reversal_rate": estimate,
                    "cluster_ci_lower": lower,
                    "cluster_ci_upper": upper,
                    "harmful_batches": int(group["harmful_batches"].sum()),
                    "reversed_batches": int(group["reversed_batches"].sum()),
                    "n_scenarios": len(group),
                }
            )
    return pd.DataFrame(records)


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if not len(valid_positions):
        return result
    valid = values[valid_positions]
    order = np.argsort(valid)
    ranked = valid[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[valid_positions] = restored
    return result


def severity_correlations(
    scenarios: pd.DataFrame,
    replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = scenarios[~scenarios["shift"].eq("no_shift")].copy()
    frame["severity_ordinal"] = frame["severity"].map(SEVERITY_ORDER)
    keys = ["dataset", "model", "monitor", "shift", "mode", "seed"]
    records: list[dict[str, Any]] = []
    for group_values, group in frame.groupby(keys, dropna=False, sort=True):
        if set(group["severity"]) != set(SEVERITY_ORDER):
            continue
        descriptors = dict(zip(keys, group_values))
        for metric in SEVERITY_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = values.notna() & group["severity_ordinal"].notna()
            if valid.sum() != 3:
                continue
            is_constant = values[valid].nunique() < 2
            correlation = (
                None
                if is_constant
                else spearmanr(
                    group.loc[valid, "severity_ordinal"], values[valid]
                )
            )
            records.append(
                {
                    **descriptors,
                    "metric": metric,
                    "spearman_rho": (
                        0.0 if is_constant else float(correlation.statistic)
                    ),
                    "p_value": 1.0 if is_constant else float(correlation.pvalue),
                    "constant_across_severity": bool(is_constant),
                }
            )
    by_group = pd.DataFrame(records)
    if by_group.empty:
        return by_group, pd.DataFrame()
    by_group["q_value_bh"] = benjamini_hochberg(by_group["p_value"])

    summary_records = []
    summary_keys = ["monitor", "shift", "metric"]
    for group_values, group in by_group.groupby(summary_keys, sort=True):
        estimate, lower, upper, n = two_way_cluster_bootstrap_mean(
            group.rename(columns={"spearman_rho": "value"}),
            "value",
            replicates,
            confidence_level,
            rng,
        )
        summary_records.append(
            {
                **dict(zip(summary_keys, group_values)),
                "mean_spearman_rho": estimate,
                "cluster_ci_lower": lower,
                "cluster_ci_upper": upper,
                "positive_fraction": float(group["spearman_rho"].gt(0).mean()),
                "negative_fraction": float(group["spearman_rho"].lt(0).mean()),
                "constant_fraction": float(
                    group["constant_across_severity"].mean()
                ),
                "n_groups": n,
                "n_datasets": group["dataset"].nunique(),
                "n_seeds": group["seed"].nunique(),
            }
        )
    return by_group, pd.DataFrame(summary_records)


def balanced_variance_components(
    frame: pd.DataFrame,
    outcome: str,
    factors: list[str],
    interactions: list[tuple[str, str]],
) -> pd.DataFrame:
    columns = factors + [outcome]
    working = frame[columns].copy()
    working[outcome] = pd.to_numeric(working[outcome], errors="coerce")
    working = working.dropna(subset=[outcome])
    if working.empty:
        return pd.DataFrame()
    cell_counts = working.groupby(factors, dropna=False).size()
    if cell_counts.nunique() != 1:
        raise ValueError(f"Variance design is not balanced for {outcome}")
    values = working[outcome].to_numpy(dtype=float)
    grand_mean = float(values.mean())
    total_ss = float(np.sum((values - grand_mean) ** 2))
    if total_ss <= 0:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    main_means: dict[str, pd.Series] = {}
    for factor in factors:
        grouped = working.groupby(factor, dropna=False)[outcome].agg(
            ["mean", "size"]
        )
        main_means[factor] = grouped["mean"]
        sum_squares = float(
            np.sum(grouped["size"] * (grouped["mean"] - grand_mean) ** 2)
        )
        records.append(
            {
                "term": factor,
                "term_type": "main_effect",
                "sum_squares": sum_squares,
                "variance_fraction": sum_squares / total_ss,
            }
        )

    for first, second in interactions:
        grouped = working.groupby([first, second], dropna=False)[outcome].agg(
            ["mean", "size"]
        )
        effect = []
        for (first_level, second_level), row in grouped.iterrows():
            interaction_mean = (
                row["mean"]
                - main_means[first].loc[first_level]
                - main_means[second].loc[second_level]
                + grand_mean
            )
            effect.append(row["size"] * interaction_mean**2)
        sum_squares = float(np.sum(effect))
        records.append(
            {
                "term": f"{first}:{second}",
                "term_type": "two_way_interaction",
                "sum_squares": sum_squares,
                "variance_fraction": sum_squares / total_ss,
            }
        )

    explained = float(sum(record["sum_squares"] for record in records))
    residual = max(total_ss - explained, 0.0)
    records.append(
        {
            "term": "residual_and_higher_order",
            "term_type": "residual",
            "sum_squares": residual,
            "variance_fraction": residual / total_ss,
        }
    )
    result = pd.DataFrame(records)
    result["total_sum_squares"] = total_ss
    return result


def _hierarchical_resample(
    frame: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    datasets = np.asarray(sorted(frame["dataset"].unique()))
    seeds = np.asarray(sorted(frame["seed"].unique()))
    dataset_draws = rng.choice(datasets, size=len(datasets), replace=True)
    seed_draws = rng.choice(seeds, size=len(seeds), replace=True)
    parts = []
    for dataset_index, dataset in enumerate(dataset_draws):
        for seed_index, seed in enumerate(seed_draws):
            part = frame[
                frame["dataset"].eq(dataset) & frame["seed"].eq(seed)
            ].copy()
            part["dataset"] = f"bootstrap_dataset_{dataset_index}"
            part["seed"] = f"bootstrap_seed_{seed_index}"
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def variance_decomposition(
    scenarios: pd.DataFrame,
    replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    non_null = scenarios[~scenarios["shift"].eq("no_shift")].copy()
    factors = ["dataset", "model", "shift", "severity", "mode", "seed"]
    interactions = [
        ("dataset", "model"),
        ("dataset", "shift"),
        ("model", "shift"),
        ("shift", "severity"),
        ("shift", "mode"),
        ("model", "severity"),
    ]
    analyses = [
        ("oracle", "confidence", "true_excess_risk"),
        ("confidence", "confidence", "risk_abs_error"),
        ("confidence", "confidence", "risk_failure"),
        ("confidence", "confidence", "alarm_failure"),
        ("confidence", "confidence", "monitor_failure"),
        ("drift_shap", "drift_shap", "attribution_failure"),
        ("drift_shap", "drift_shap", "alarm_failure"),
        ("drift_shap", "drift_shap", "monitor_failure"),
    ]
    records = []
    for analysis_name, monitor, outcome in analyses:
        frame = non_null[non_null["monitor"].eq(monitor)].copy()
        point = balanced_variance_components(
            frame, outcome, factors, interactions
        )
        if point.empty:
            continue
        bootstrap_values = {term: [] for term in point["term"]}
        for _ in range(replicates):
            sampled = _hierarchical_resample(frame, rng)
            sample_components = balanced_variance_components(
                sampled, outcome, factors, interactions
            )
            for row in sample_components.itertuples(index=False):
                bootstrap_values[row.term].append(row.variance_fraction)
        alpha = 1.0 - confidence_level
        for row in point.itertuples(index=False):
            draws = np.asarray(bootstrap_values[row.term], dtype=float)
            lower, upper = np.quantile(
                draws, [alpha / 2.0, 1.0 - alpha / 2.0]
            )
            records.append(
                {
                    "analysis": analysis_name,
                    "monitor": monitor,
                    "outcome": outcome,
                    "term": row.term,
                    "term_type": row.term_type,
                    "variance_fraction": row.variance_fraction,
                    "cluster_ci_lower": float(lower),
                    "cluster_ci_upper": float(upper),
                    "bootstrap_replicates": replicates,
                }
            )
    return pd.DataFrame(records)


def _write_summary(
    output_dir: Path,
    headlines: pd.DataFrame,
    sign_reversal: pd.DataFrame,
) -> None:
    lines = [
        "# TABMON-Bench schema-v7 statistical analysis",
        "",
        "All shifted estimates use post-shift batches. Null estimates use the",
        "complete no-shift trajectory. Scenario/seed is the analysis unit.",
        "Cluster intervals resample datasets and seeds while preserving all",
        "model/shift measurements within each selected block.",
        "",
        "## Headline estimates",
        "",
        "| Finding | Estimate | 95% cluster CI |",
        "| --- | ---: | ---: |",
    ]
    for row in headlines.itertuples(index=False):
        lines.append(
            f"| {row.finding} | {row.estimate:.4f} | "
            f"[{row.cluster_ci_lower:.4f}, {row.cluster_ci_upper:.4f}] |"
        )
    overall = sign_reversal[sign_reversal["scope"].eq("overall")].iloc[0]
    lines.extend(
        [
            "",
            "## Pipeline sign reversal",
            "",
            f"Overall rate: `{overall.sign_reversal_rate:.4f}` "
            f"(95% cluster CI "
            f"`[{overall.cluster_ci_lower:.4f}, "
            f"{overall.cluster_ci_upper:.4f}]`; "
            f"{int(overall.reversed_batches)}/"
            f"{int(overall.harmful_batches)} harmful post-shift batches).",
            "",
            "## Interpretation limits",
            "",
            "- Only five datasets and five seeds are available; cluster intervals",
            "  honestly reflect this limited number of top-level units.",
            "- Variance decomposition is descriptive balanced-factor ANOVA, not a",
            "  causal attribution or a generalized mixed-effects model.",
            "- FDR-adjusted severity-correlation p-values are exploratory because",
            "  each within-scenario correlation contains only three severities.",
        ]
    )
    (output_dir / "statistical_analysis_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    if args.variance_bootstrap_replicates < 50:
        raise ValueError("variance-bootstrap-replicates must be at least 50")
    if not 0.8 < args.confidence_level < 1.0:
        raise ValueError("confidence-level must be between 0.8 and 1.0")

    results_dir = args.results_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else results_dir / "paper_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, batches = _validate_inputs(results_dir)
    rng = np.random.default_rng(args.random_seed)

    print("[1/6] Building scenario-level post-shift table", flush=True)
    scenarios = build_scenario_metrics(batches)
    scenarios.to_parquet(output_dir / "scenario_post_metrics.parquet", index=False)
    scenarios.to_csv(output_dir / "scenario_post_metrics.csv", index=False)

    print("[2/6] Computing seed-level Student-t intervals", flush=True)
    seed_intervals = seed_level_intervals(scenarios, args.confidence_level)
    seed_intervals.to_csv(output_dir / "seed_level_intervals.csv", index=False)

    print("[3/6] Computing two-way cluster bootstrap headlines", flush=True)
    headlines = headline_estimates(
        scenarios,
        args.bootstrap_replicates,
        args.confidence_level,
        rng,
    )
    headlines.to_csv(output_dir / "headline_estimates.csv", index=False)

    print("[4/6] Computing pipeline sign-reversal intervals", flush=True)
    sign_reversal = pipeline_sign_reversal(
        batches,
        args.bootstrap_replicates,
        args.confidence_level,
        rng,
    )
    sign_reversal.to_csv(output_dir / "pipeline_sign_reversal.csv", index=False)

    print("[5/6] Computing severity correlations and FDR", flush=True)
    correlation_groups, correlation_summary = severity_correlations(
        scenarios,
        args.bootstrap_replicates,
        args.confidence_level,
        rng,
    )
    correlation_groups.to_csv(
        output_dir / "severity_correlations_by_group.csv", index=False
    )
    correlation_summary.to_csv(
        output_dir / "severity_correlation_summary.csv", index=False
    )

    print("[6/6] Computing balanced-factor variance decomposition", flush=True)
    variance = variance_decomposition(
        scenarios,
        args.variance_bootstrap_replicates,
        args.confidence_level,
        rng,
    )
    variance.to_csv(output_dir / "variance_decomposition.csv", index=False)
    _write_summary(output_dir, headlines, sign_reversal)

    analysis_manifest = {
        "analysis_schema_version": 1,
        "benchmark_schema_version": SCHEMA_VERSION,
        "source_benchmark_manifest": manifest,
        "random_seed": args.random_seed,
        "confidence_level": args.confidence_level,
        "bootstrap_replicates": args.bootstrap_replicates,
        "variance_bootstrap_replicates": args.variance_bootstrap_replicates,
        "analysis_unit": "scenario_seed",
        "shifted_phase": "shift_fraction > 0",
        "null_phase": "complete no_shift trajectory",
        "cluster_bootstrap": "two-way dataset and seed block bootstrap",
        "multiple_testing": "Benjamini-Hochberg for group severity correlations",
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=== STATISTICAL ANALYSIS COMPLETE ===")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Seed-level intervals: {len(seed_intervals)}")
    print(f"Headline estimates: {len(headlines)}")
    print(f"Severity correlations: {len(correlation_groups)}")
    print(f"Variance rows: {len(variance)}")
    print(f"Saved at: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
