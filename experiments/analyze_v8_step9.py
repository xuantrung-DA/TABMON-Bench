"""Step 9: paired, cluster-aware, and mixed-effects analysis of TABMON v8.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analyze_schema_v7 import benjamini_hochberg
from src.cache.stream_cache import atomic_json


METHODS = ("ac", "doc", "atc", "cot", "cott")
PAIRWISE_OUTCOMES = {
    "mean_absolute_error": "lower_is_better",
    "risk_failure_rate": "lower_is_better",
}
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
    parser.add_argument("--v8-1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260926)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 1000:
        parser.error("Use at least 1,000 two-way bootstrap replicates")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _validate_input(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    status = json.loads((root / "v8_1_status.json").read_text("utf-8"))
    manifest = json.loads((root / "v8_1_manifest.json").read_text("utf-8"))
    if status.get("complete") is not True or status.get("schema_version") != "8.1":
        raise ValueError("Step 9 requires a complete schema-v8.1 result")
    if manifest.get("probability_clip_epsilon") != 1e-7:
        raise ValueError("Step 9 refuses an unfrozen risk definition")
    scenarios = pd.read_parquet(root / "scenario_metrics.parquet")
    batches = pd.read_parquet(root / "batch_metrics.parquet")
    if len(scenarios) != 21_700 or len(batches) != 217_000:
        raise ValueError("Schema-v8.1 grid is incomplete")
    return scenarios, batches


def _cell_sum_and_count(
    frame: pd.DataFrame, value: str
) -> tuple[np.ndarray, np.ndarray]:
    working = frame[["dataset", "seed", value]].copy()
    working[value] = pd.to_numeric(working[value], errors="coerce")
    working = working.dropna(subset=[value])
    if working.empty:
        raise ValueError(f"No finite values for {value}")
    grouped = working.groupby(["dataset", "seed"])[value].agg(["sum", "size"])
    datasets = sorted(working["dataset"].unique())
    seeds = sorted(working["seed"].unique())
    full_index = pd.MultiIndex.from_product(
        [datasets, seeds], names=["dataset", "seed"]
    )
    grouped = grouped.reindex(full_index, fill_value=0.0)
    return (
        grouped["sum"].to_numpy(dtype=float).reshape(len(datasets), len(seeds)),
        grouped["size"].to_numpy(dtype=float).reshape(len(datasets), len(seeds)),
    )


def two_way_cell_bootstrap_mean(
    frame: pd.DataFrame,
    value: str,
    replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, np.ndarray]:
    """Two-way dataset/seed block bootstrap for a balanced factorial grid."""

    cell_sum, cell_count = _cell_sum_and_count(frame, value)
    n_datasets, n_seeds = cell_sum.shape
    dataset_counts = rng.multinomial(
        n_datasets,
        np.full(n_datasets, 1.0 / n_datasets),
        size=replicates,
    )
    seed_counts = rng.multinomial(
        n_seeds,
        np.full(n_seeds, 1.0 / n_seeds),
        size=replicates,
    )
    numerators = np.einsum(
        "rd,ds,rs->r", dataset_counts, cell_sum, seed_counts, optimize=True
    )
    denominators = np.einsum(
        "rd,ds,rs->r", dataset_counts, cell_count, seed_counts, optimize=True
    )
    # Sparse event-defined summaries can produce an empty resample. Redraw only
    # those replicates; do not replace them with zeros or silently drop them.
    empty = denominators <= 0
    while np.any(empty):
        count = int(np.sum(empty))
        dataset_counts[empty] = rng.multinomial(
            n_datasets, np.full(n_datasets, 1.0 / n_datasets), size=count
        )
        seed_counts[empty] = rng.multinomial(
            n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=count
        )
        numerators[empty] = np.einsum(
            "rd,ds,rs->r",
            dataset_counts[empty],
            cell_sum,
            seed_counts[empty],
            optimize=True,
        )
        denominators[empty] = np.einsum(
            "rd,ds,rs->r",
            dataset_counts[empty],
            cell_count,
            seed_counts[empty],
            optimize=True,
        )
        empty = denominators <= 0
    draws = numerators / denominators
    estimate = float(pd.to_numeric(frame[value], errors="coerce").mean())
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    deviations = draws - estimate
    p_value = float(
        (1 + np.sum(np.abs(deviations) >= abs(estimate))) / (replicates + 1)
    )
    return estimate, float(lower), float(upper), min(p_value, 1.0), draws


def _classification_frame(scenarios: pd.DataFrame) -> pd.DataFrame:
    frame = scenarios[scenarios["endpoint"].eq("classification_error")].copy()
    if set(frame["method"]) != set(METHODS):
        raise ValueError("Classification estimator grid is incomplete")
    counts = frame.groupby("predictor_stream_id")["method"].nunique()
    if not counts.eq(len(METHODS)).all():
        raise ValueError("Methods are not paired on every predictor stream")
    true_values = frame.pivot(
        index="predictor_stream_id", columns="method", values="mean_true_value"
    )
    if np.nanmax(true_values.max(axis=1) - true_values.min(axis=1)) > 1e-12:
        raise ValueError("Paired estimators do not share the same oracle endpoint")
    if len(frame) != 15_500:
        raise ValueError("Classification estimator grid is incomplete")
    return frame


def classification_summaries(
    classification: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    scopes: list[tuple[str, str | None, pd.DataFrame]] = [
        ("overall", None, classification)
    ] + [
        ("shift", shift, classification[classification["shift"].eq(shift)])
        for shift in SHIFT_ORDER
    ]
    for scope, shift, frame in scopes:
        for method in METHODS:
            method_frame = frame[frame["method"].eq(method)]
            for outcome in PAIRWISE_OUTCOMES:
                estimate, lower, upper, _, _ = two_way_cell_bootstrap_mean(
                    method_frame, outcome, replicates, 0.95, rng
                )
                records.append(
                    {
                        "scope": scope,
                        "shift": shift,
                        "method": method,
                        "outcome": outcome,
                        "estimate": estimate,
                        "cluster_ci_lower": lower,
                        "cluster_ci_upper": upper,
                        "n_streams": len(method_frame),
                        "n_datasets": method_frame["dataset"].nunique(),
                        "n_seeds": method_frame["seed"].nunique(),
                    }
                )
    return pd.DataFrame(records)


def paired_classification_comparisons(
    classification: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    descriptors = [
        "predictor_stream_id", "dataset", "model", "shift", "severity",
        "mode", "seed",
    ]
    records: list[dict[str, Any]] = []
    scopes: list[tuple[str, str | None, pd.DataFrame]] = [
        ("overall", None, classification)
    ] + [
        ("shift", shift, classification[classification["shift"].eq(shift)])
        for shift in SHIFT_ORDER
    ]
    for scope, shift, frame in scopes:
        if frame.empty:
            continue
        base = frame[descriptors].drop_duplicates("predictor_stream_id")
        for outcome in PAIRWISE_OUTCOMES:
            wide = frame.pivot(
                index="predictor_stream_id", columns="method", values=outcome
            )
            if wide.isna().any().any() or set(wide.columns) != set(METHODS):
                raise ValueError(f"Incomplete paired matrix for {scope}/{outcome}")
            for method_a, method_b in combinations(METHODS, 2):
                difference = (
                    wide[method_a] - wide[method_b]
                ).rename("paired_difference")
                paired = base.merge(
                    difference,
                    left_on="predictor_stream_id",
                    right_index=True,
                    validate="one_to_one",
                )
                estimate, lower, upper, p_value, _ = two_way_cell_bootstrap_mean(
                    paired, "paired_difference", replicates, 0.95, rng
                )
                winner = (
                    method_a if upper < 0.0 else method_b if lower > 0.0 else "no_clear_winner"
                )
                records.append(
                    {
                        "scope": scope,
                        "shift": shift,
                        "outcome": outcome,
                        "method_a": method_a,
                        "method_b": method_b,
                        "difference_a_minus_b": estimate,
                        "cluster_ci_lower": lower,
                        "cluster_ci_upper": upper,
                        "bootstrap_p_value": p_value,
                        "ci_winner": winner,
                        "n_paired_streams": len(paired),
                        "n_datasets": paired["dataset"].nunique(),
                        "n_seeds": paired["seed"].nunique(),
                    }
                )
    result = pd.DataFrame(records)
    result["q_value_bh_scope"] = np.nan
    for _, positions in result.groupby(["outcome", "scope", "shift"], dropna=False).groups.items():
        result.loc[positions, "q_value_bh_scope"] = benjamini_hochberg(
            result.loc[positions, "bootstrap_p_value"]
        )
    result["q_value_bh_global_outcome"] = np.nan
    for _, positions in result.groupby("outcome").groups.items():
        result.loc[positions, "q_value_bh_global_outcome"] = benjamini_hochberg(
            result.loc[positions, "bootstrap_p_value"]
        )
    result["globally_significant"] = (
        result["q_value_bh_global_outcome"].lt(0.05)
        & ~result["ci_winner"].eq("no_clear_winner")
    )
    return result


def endpoint_summaries(
    scenarios: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    specs = {
        "confidence_log_loss": [
            "mean_true_value", "mean_estimated_value", "mean_absolute_error",
            "risk_failure_rate", "null_stream_any_alarm", "detected_event",
            "false_alarm_rate", "power", "detection_delay",
        ],
        "drift_shap_permutation": [
            "mean_top3_recall", "mean_ndcg_at_3", "attribution_failure_rate",
            "null_stream_any_alarm", "detected_event", "false_alarm_rate",
            "power", "detection_delay",
        ],
    }
    records: list[dict[str, Any]] = []
    for method, metrics in specs.items():
        method_frame = scenarios[scenarios["method"].eq(method)]
        for shift in SHIFT_ORDER:
            frame = method_frame[method_frame["shift"].eq(shift)]
            for metric in metrics:
                valid = frame[pd.to_numeric(frame[metric], errors="coerce").notna()]
                if valid.empty:
                    continue
                estimate, lower, upper, _, _ = two_way_cell_bootstrap_mean(
                    valid, metric, replicates, 0.95, rng
                )
                records.append(
                    {
                        "method": method,
                        "shift": shift,
                        "metric": metric,
                        "estimate": estimate,
                        "cluster_ci_lower": lower,
                        "cluster_ci_upper": upper,
                        "n_scenarios": len(valid),
                    }
                )
    return pd.DataFrame(records)


def _fixed_formula(outcome: str) -> str:
    return (
        f"{outcome} ~ "
        "C(method, Treatment(reference='ac')) * "
        "C(shift, Treatment(reference='covariate')) + "
        "C(model, Treatment(reference='lr')) + "
        "C(severity, Treatment(reference='low')) + "
        "C(mode, Treatment(reference='abrupt')) + C(dataset) + C(seed)"
    )


def _design_level_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return observed design cardinalities for analysis metadata.

    The Step-9 models are reused by the final Schema-13 analysis, whose seed
    count differs from the original v8.1 run.  Deriving these values from the
    analysis frame prevents stale, schema-specific labels in exported model
    metadata.
    """

    required = {"dataset", "seed", "predictor_stream_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing design columns: {sorted(missing)}")
    dataset_seed = frame[["dataset", "seed"]].drop_duplicates()
    return {
        "datasets": int(frame["dataset"].nunique()),
        "seeds": int(frame["seed"].nunique()),
        "dataset_seed_cells": int(len(dataset_seed)),
        "predictor_streams": int(frame["predictor_stream_id"].nunique()),
    }


def _outcome_equivalent_methods(
    frame: pd.DataFrame,
    outcome: str,
    reference: str = "ac",
) -> list[str]:
    """Find methods whose stream-level outcome equals the reference exactly."""

    wide = frame.pivot(
        index="predictor_stream_id", columns="method", values=outcome
    )
    if reference not in wide:
        raise ValueError(f"Missing reference method: {reference}")
    return sorted(
        method
        for method in wide.columns
        if method != reference and wide[method].equals(wide[reference])
    )


def fit_linear_mixed_model(
    classification: pd.DataFrame, outcome: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = classification[~classification["shift"].eq("no_shift")].copy()
    formula = _fixed_formula(outcome)
    model = smf.mixedlm(
        formula,
        frame,
        groups=frame["predictor_stream_id"],
        re_formula="1",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result: MixedLMResultsWrapper = model.fit(
            reml=False, method="powell", maxiter=500, disp=False
        )
    confidence = result.conf_int().loc[result.fe_params.index]
    records = []
    for term in result.fe_params.index:
        records.append(
            {
                "model": f"linear_mixed_{outcome}",
                "term": term,
                "estimate": float(result.fe_params[term]),
                "std_error": float(result.bse_fe[term]),
                "ci_lower": float(confidence.loc[term, 0]),
                "ci_upper": float(confidence.loc[term, 1]),
                "p_value": float(result.pvalues[term]),
            }
        )
    table = pd.DataFrame(records)
    table["q_value_bh"] = benjamini_hochberg(table["p_value"])
    levels = _design_level_counts(frame)
    metadata = {
        "model": f"linear_mixed_{outcome}",
        "formula": formula,
        "converged": bool(result.converged),
        "n_observations": int(result.nobs),
        "groups": "predictor_stream_id random intercept",
        "dataset_and_seed": (
            "fixed effects "
            f"({levels['datasets']} dataset levels; {levels['seeds']} seed levels)"
        ),
        "random_intercept_variance": float(result.cov_re.iloc[0, 0]),
        "residual_variance": float(result.scale),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "optimizer": "powell",
        "warnings": [str(item.message) for item in caught],
    }
    return table, metadata


def fit_binomial_mixed_model(
    classification: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = classification[~classification["shift"].eq("no_shift")].copy()
    frame["majority_failure"] = frame["risk_failure_rate"].gt(0.5).astype(int)
    frame["dataset_seed"] = (
        frame["dataset"].astype(str) + "::" + frame["seed"].astype(str)
    )
    formula = _fixed_formula("majority_failure")
    model = BinomialBayesMixedGLM.from_formula(
        formula,
        {"dataset_seed": "0 + C(dataset_seed)"},
        frame,
        vcp_p=0.5,
        fe_p=2.0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit_vb()
    fixed = pd.DataFrame(
        {
            "model": "binomial_mixed_majority_failure",
            "term": model.exog_names,
            "posterior_mean_log_odds": result.fe_mean,
            "posterior_sd": result.fe_sd,
        }
    )
    fixed["credible_lower_log_odds"] = (
        fixed["posterior_mean_log_odds"] - 1.96 * fixed["posterior_sd"]
    )
    fixed["credible_upper_log_odds"] = (
        fixed["posterior_mean_log_odds"] + 1.96 * fixed["posterior_sd"]
    )
    fixed["odds_ratio"] = np.exp(fixed["posterior_mean_log_odds"])
    fixed["odds_ratio_lower"] = np.exp(fixed["credible_lower_log_odds"])
    fixed["odds_ratio_upper"] = np.exp(fixed["credible_upper_log_odds"])
    variance = pd.DataFrame(
        {
            "component": model.vcp_names,
            "posterior_mean_log_sd": result.vcp_mean,
            "posterior_sd": result.vcp_sd,
            "random_effect_sd": np.exp(result.vcp_mean),
        }
    )
    levels = _design_level_counts(frame)
    metadata = {
        "model": "binomial_mixed_majority_failure",
        "formula": formula,
        "outcome_definition": "risk_failure_rate > 0.5",
        "fit": "variational Bayes binomial mixed model",
        "n_observations": len(frame),
        "groups": (
            "dataset-seed cell random intercept "
            f"({levels['dataset_seed_cells']} levels)"
        ),
        "dataset_and_seed": (
            "dataset and seed fixed main effects plus random dataset-seed cell"
        ),
        "warnings": [str(item.message) for item in caught],
    }
    return fixed, variance, metadata


def fit_binomial_gee(
    classification: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Marginal logistic model with predictor-stream clustered inference."""

    frame = classification[~classification["shift"].eq("no_shift")].copy()
    frame["majority_failure"] = frame["risk_failure_rate"].gt(0.5).astype(int)
    # A method with exactly the same binary outcome as the reference has a
    # deterministic zero contrast.  Its sandwich variance is zero up to
    # floating-point error and statsmodels otherwise reports NaN standard
    # errors.  Keep that equality in the paired analysis and exclude only the
    # redundant method from this GEE sensitivity model.
    equivalent_methods = _outcome_equivalent_methods(
        frame, "majority_failure", reference="ac"
    )
    if equivalent_methods:
        frame = frame[~frame["method"].isin(equivalent_methods)].copy()
    formula = _fixed_formula("majority_failure")
    model = smf.gee(
        formula,
        groups="predictor_stream_id",
        data=frame,
        family=Binomial(),
        cov_struct=Exchangeable(),
    )
    result = model.fit(maxiter=200)
    confidence = result.conf_int()
    table = pd.DataFrame(
        {
            "model": "binomial_gee_majority_failure",
            "term": result.params.index,
            "estimate_log_odds": result.params.to_numpy(dtype=float),
            "robust_std_error": result.bse.to_numpy(dtype=float),
            "ci_lower_log_odds": confidence.iloc[:, 0].to_numpy(dtype=float),
            "ci_upper_log_odds": confidence.iloc[:, 1].to_numpy(dtype=float),
            "p_value": result.pvalues.to_numpy(dtype=float),
        }
    )
    table["q_value_bh"] = benjamini_hochberg(table["p_value"])
    table["odds_ratio"] = np.exp(table["estimate_log_odds"])
    table["odds_ratio_lower"] = np.exp(table["ci_lower_log_odds"])
    table["odds_ratio_upper"] = np.exp(table["ci_upper_log_odds"])
    levels = _design_level_counts(frame)
    metadata = {
        "model": "binomial_gee_majority_failure",
        "formula": formula,
        "outcome_definition": "risk_failure_rate > 0.5",
        "fit": "GEE logistic with exchangeable working correlation",
        "n_observations": len(frame),
        "excluded_outcome_equivalent_methods": equivalent_methods,
        "exclusion_reason": (
            "stream-level majority-failure outcome exactly equals AC; "
            "paired estimates retain these methods"
            if equivalent_methods
            else None
        ),
        "groups": (
            "predictor_stream_id robust clusters "
            f"({levels['predictor_streams']} levels)"
        ),
        "converged": bool(result.converged),
        "dependence_parameter": float(np.asarray(result.cov_struct.dep_params)),
    }
    return table, metadata


def _write_report(
    output_dir: Path,
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
    endpoints: pd.DataFrame,
    model_metadata: list[dict[str, Any]],
) -> None:
    significant = comparisons[comparisons["globally_significant"]]
    overall_significant = significant[significant["scope"].eq("overall")]
    significant_counts = (
        significant.groupby(["outcome", "shift"], dropna=False)
        .size()
        .reset_index(name="tests")
    )
    overall = summaries[
        summaries["scope"].eq("overall")
        & summaries["outcome"].eq("mean_absolute_error")
    ].sort_values("estimate")
    lines = [
        "# TABMON Step 9 statistical report",
        "",
        "All direct method comparisons are restricted to the shared "
        "classification-error endpoint and paired predictor streams.",
        "Confidence and DriftSHAP-style results are summarized separately.",
        "",
        "## Overall classification-error MAE",
        "",
        "| Method | Estimate | 95% dataset-seed cluster CI |",
        "|---|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {row.method.upper()} | {row.estimate:.4f} | "
            f"[{row.cluster_ci_lower:.4f}, {row.cluster_ci_upper:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Multiplicity-controlled paired comparisons",
            "",
            f"- Total tests: {len(comparisons)}.",
            f"- Globally significant after BH within outcome: {len(significant)}.",
            f"- Significant overall method comparisons: {len(overall_significant)}.",
            "- A positive A-minus-B difference means method A is worse.",
            "- The bootstrap resamples datasets and seeds as crossed blocks.",
            "",
            "| Outcome | Shift | Globally significant pairs |",
            "|---|---|---:|",
        ]
    )
    for row in significant_counts.itertuples(index=False):
        lines.append(f"| {row.outcome} | {row.shift} | {row.tests} |")
    lines.extend(
        [
            "",
            "## Mixed-effects models",
            "",
        ]
    )
    for metadata in model_metadata:
        lines.append(
            f"- `{metadata['model']}`: n={metadata['n_observations']}; "
            f"groups={metadata['groups']}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The paired cluster bootstrap is the primary inferential analysis. "
            "Mixed models adjust for scenario factors and within-stream pairing, "
            "but dataset and seed are treated as fixed effects because only five "
            "levels of each are available. No cross-endpoint ranking is permitted.",
        ]
    )
    (output_dir / "STEP9_REPORT.md").write_text("\n".join(lines) + "\n", "utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.v8_1_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenarios, batches = _validate_input(root)
    classification = _classification_frame(scenarios)
    rng = np.random.default_rng(args.random_seed)

    print("[1/6] Classification endpoint summaries", flush=True)
    summaries = classification_summaries(
        classification, args.bootstrap_replicates, rng
    )
    print("[2/6] Paired method comparisons", flush=True)
    comparisons = paired_classification_comparisons(
        classification, args.bootstrap_replicates, rng
    )
    print("[3/6] Separate Confidence/DriftSHAP summaries", flush=True)
    endpoints = endpoint_summaries(scenarios, args.bootstrap_replicates, rng)
    print("[4/6] Linear mixed-effects models", flush=True)
    lmm_mae, meta_mae = fit_linear_mixed_model(
        classification, "mean_absolute_error"
    )
    lmm_failure, meta_failure = fit_linear_mixed_model(
        classification, "risk_failure_rate"
    )
    print("[5/6] Binomial mixed-effects and paired-stream GEE models", flush=True)
    glmm_fixed, glmm_variance, meta_glmm = fit_binomial_mixed_model(classification)
    gee_fixed, meta_gee = fit_binomial_gee(classification)

    summaries.to_csv(output / "classification_method_summary.csv", index=False)
    comparisons.to_csv(
        output / "paired_classification_comparisons.csv", index=False
    )
    endpoints.to_csv(output / "separate_endpoint_summary.csv", index=False)
    pd.concat([lmm_mae, lmm_failure], ignore_index=True).to_csv(
        output / "linear_mixed_effects.csv", index=False
    )
    glmm_fixed.to_csv(output / "binomial_mixed_effects.csv", index=False)
    glmm_variance.to_csv(
        output / "binomial_random_effects.csv", index=False
    )
    gee_fixed.to_csv(output / "binomial_gee.csv", index=False)
    model_metadata = [meta_mae, meta_failure, meta_glmm, meta_gee]
    atomic_json(
        {"models": model_metadata}, output / "mixed_model_metadata.json"
    )
    _write_report(output, summaries, comparisons, endpoints, model_metadata)
    print("[6/6] Writing manifest and status", flush=True)
    manifest = {
        "step": 9,
        "analysis_unit": "method-predictor-stream scenario summary",
        "classification_endpoint": "0-1 classification error",
        "direct_comparison_methods": list(METHODS),
        "paired_design": True,
        "cross_endpoint_ranking_prohibited": True,
        "primary_uncertainty": "two-way dataset-seed cluster bootstrap",
        "bootstrap_replicates": args.bootstrap_replicates,
        "multiple_comparison_correction": (
            "Benjamini-Hochberg within each outcome across all overall and "
            "shift-specific pairwise tests"
        ),
        "mixed_effects": model_metadata,
        "input_sha256": {
            name: _sha256(root / name)
            for name in (
                "batch_metrics.parquet",
                "scenario_metrics.parquet",
                "v8_1_manifest.json",
            )
        },
        "random_seed": args.random_seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(manifest, output / "step9_manifest.json")
    atomic_json(
        {
            "complete": True,
            "step": 9,
            "classification_summaries": len(summaries),
            "paired_tests": len(comparisons),
            "separate_endpoint_summaries": len(endpoints),
            "linear_mixed_coefficients": len(lmm_mae) + len(lmm_failure),
            "binomial_mixed_coefficients": len(glmm_fixed),
            "binomial_gee_coefficients": len(gee_fixed),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output / "step9_status.json",
    )
    print("=== TABMON STEP 9 STATISTICAL ANALYSIS ===")
    print(f"Paired tests: {len(comparisons)}")
    print(f"Globally significant: {int(comparisons['globally_significant'].sum())}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
