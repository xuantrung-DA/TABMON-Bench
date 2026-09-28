"""Generate publication figures for the TABMON-Bench manuscript.

Every figure is saved as a vector PDF and a 300-dpi PNG preview.  The script
uses only schema-v7 benchmark artifacts and the derived statistical-analysis
tables; no values are manually embedded in the plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analyze_schema_v7 import two_way_cluster_bootstrap_mean


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
YELLOW = "#E69F00"
SKY = "#56B4E9"
BLACK = "#222222"
GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
SHIFT_COLORS = {
    "concept": PURPLE,
    "correlated": SKY,
    "covariate": BLUE,
    "pipeline": ORANGE,
    "support": GREEN,
}
MODEL_COLORS = {"lr": BLUE, "rf": GREEN, "xgb": ORANGE, "mlp": PURPLE}
DATASET_LABELS = {
    "acs_income": "ACS Income",
    "adult": "Adult",
    "bank_marketing": "Bank Marketing",
    "covertype": "Covertype",
    "diabetes_hospitals": "Diabetes Hospitals",
}
MODEL_LABELS = {"lr": "LR", "rf": "RF", "xgb": "XGB", "mlp": "MLP"}
SHIFT_LABELS = {
    "no_shift": "No shift",
    "covariate": "Covariate",
    "correlated": "Correlated",
    "pipeline": "Pipeline",
    "support": "Support",
    "concept": "Concept",
}
SEVERITY_ORDER = ["low", "medium", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--random-seed", type=int, default=17_021)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.titleweight": "bold",
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    text_color: str = "white",
    fontsize: float = 9.5,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight="bold",
        linespacing=1.25,
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    style: str = "-",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            linestyle=style,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def figure_protocol(output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(13.2, 5.0))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    _box(axis, (0.02, 0.64), 0.16, 0.22, "Training data\n(labels available)", BLUE)
    _box(axis, (0.22, 0.64), 0.16, 0.22, "Frozen predictor\nLR / RF / XGB / MLP", GREEN)
    _arrow(axis, (0.18, 0.75), (0.22, 0.75))

    _box(axis, (0.02, 0.18), 0.16, 0.24, "Calibration data\nmonitor fitting +\nnull thresholds", SKY, text_color=BLACK)
    _box(axis, (0.22, 0.18), 0.18, 0.24, "Controlled stream\n5 shifts × 3 severities\nabrupt / gradual", YELLOW, text_color=BLACK)
    _arrow(axis, (0.18, 0.30), (0.22, 0.30))
    _arrow(axis, (0.30, 0.64), (0.31, 0.42))

    _box(axis, (0.46, 0.59), 0.19, 0.27, "Label-free monitor\nobserves X and f(X)\nnever target Y", PURPLE)
    _arrow(axis, (0.40, 0.34), (0.46, 0.67))
    _arrow(axis, (0.38, 0.75), (0.46, 0.75))

    _box(axis, (0.46, 0.14), 0.19, 0.25, "Offline oracle\nhidden target labels\ntrue log-loss + events", ORANGE)
    _arrow(axis, (0.40, 0.28), (0.46, 0.26), color=ORANGE, style="--")

    _box(axis, (0.72, 0.35), 0.25, 0.32, "Reliability evaluator\nrisk error • attribution\nfalse alarms • delay\nobservable failure diagnostics", BLACK)
    _arrow(axis, (0.65, 0.72), (0.72, 0.58), color=PURPLE)
    _arrow(axis, (0.65, 0.26), (0.72, 0.43), color=ORANGE)
    axis.text(
        0.845,
        0.18,
        "Output: reliability boundary + failure map",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=BLACK,
    )
    axis.set_title(
        "TABMON-Bench separates monitoring-time observables from offline oracle labels",
        fontsize=13,
        pad=8,
    )
    save_figure(fig, output_dir, "fig1_benchmark_protocol")


def _severity_summary(
    scenarios: pd.DataFrame,
    monitor: str,
    metric: str,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    records = []
    frame = scenarios[
        scenarios["shift"].eq("concept") & scenarios["monitor"].eq(monitor)
    ]
    for severity in SEVERITY_ORDER:
        subset = frame[frame["severity"].eq(severity)]
        estimate, lower, upper, _ = two_way_cluster_bootstrap_mean(
            subset, metric, replicates, 0.95, rng
        )
        records.append(
            {
                "severity": severity,
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
            }
        )
    return pd.DataFrame(records)


def figure_concept_blindness(
    scenarios: pd.DataFrame,
    correlations: pd.DataFrame,
    output_dir: Path,
    replicates: int,
    rng: np.random.Generator,
) -> None:
    specs = [
        ("confidence", "true_excess_risk", "Oracle excess log loss", ORANGE),
        ("confidence", "estimated_risk_change", "Confidence risk estimate", BLUE),
        ("drift_shap", "monitor_score", "DriftSHAP score", PURPLE),
        ("drift_shap", "domain_classifier_auc", "Domain-classifier AUC", GREEN),
    ]
    fig, axes_grid = plt.subplots(2, 2, figsize=(10.6, 7.2), constrained_layout=True)
    axes = axes_grid.ravel()
    x = np.arange(3)
    for index, (monitor, metric, title, color) in enumerate(specs):
        axis = axes[index]
        summary = _severity_summary(
            scenarios, monitor, metric, replicates, rng
        )
        y = summary["estimate"].to_numpy()
        error = np.vstack([y - summary["lower"], summary["upper"] - y])
        axis.errorbar(
            x,
            y,
            yerr=error,
            marker="o",
            markersize=5.5,
            linewidth=2,
            capsize=3,
            color=color,
        )
        rho_row = correlations[
            correlations["monitor"].eq(monitor)
            & correlations["shift"].eq("concept")
            & correlations["metric"].eq(metric)
        ]
        rho = float(rho_row["mean_spearman_rho"].iloc[0])
        constant = float(rho_row["constant_fraction"].iloc[0])
        subtitle = rf"Mean Spearman $\rho={rho:.2f}$"
        if constant > 0:
            subtitle += f"; flat groups={constant:.0%}"
        axis.set_title(title + "\n" + subtitle, fontsize=9.5)
        axis.set_xticks(x, ["Low", "Medium", "High"])
        axis.set_xlabel("Concept-shift severity")
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
        panel_label(axis, chr(ord("A") + index))
        if metric == "estimated_risk_change":
            axis.axhline(0, color=GRAY, linewidth=1, linestyle="--")
        if metric == "domain_classifier_auc":
            axis.axhline(0.5, color=GRAY, linewidth=1, linestyle="--")
            axis.set_ylim(0.47, 0.53)
    fig.suptitle(
        "Concept severity increases oracle risk while observable monitor signals remain flat",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "fig2_concept_blindness")


def _annotated_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    fmt: str = ".0%",
    cmap: str = "viridis",
) -> matplotlib.image.AxesImage:
    image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                label = "—"
            else:
                label = format(value, fmt)
            normalized = (value - vmin) / max(vmax - vmin, 1e-12)
            color = "white" if normalized < 0.25 or normalized > 0.72 else BLACK
            axis.text(column, row, label, ha="center", va="center", fontsize=6.7, color=color)
    return image


def _dataset_model_order(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    datasets = [name for name in DATASET_LABELS if name in set(frame["dataset"])]
    models = [name for name in ["lr", "rf", "xgb", "mlp"] if name in set(frame["model"])]
    keys = [f"{dataset}|{model}" for dataset in datasets for model in models]
    labels = [
        f"{DATASET_LABELS[dataset]} · {MODEL_LABELS[model]}"
        for dataset in datasets
        for model in models
    ]
    return keys, labels


def figure_failure_map(scenarios: pd.DataFrame, output_dir: Path) -> None:
    shifts = ["no_shift", "covariate", "correlated", "pipeline", "support", "concept"]
    frame = scenarios.copy()
    frame["row_key"] = frame["dataset"] + "|" + frame["model"]
    row_keys, row_labels = _dataset_model_order(frame)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 8.2), constrained_layout=True)
    for index, monitor in enumerate(["confidence", "drift_shap"]):
        pivot = (
            frame[frame["monitor"].eq(monitor)]
            .pivot_table(index="row_key", columns="shift", values="monitor_failure", aggfunc="mean")
            .reindex(index=row_keys, columns=shifts)
        )
        image = _annotated_heatmap(
            axes[index],
            pivot.to_numpy(dtype=float),
            row_labels,
            [SHIFT_LABELS[s] for s in shifts],
            "Confidence monitor" if monitor == "confidence" else "DriftSHAP monitor",
        )
        panel_label(axes[index], chr(ord("A") + index))
        if index == 1:
            axes[index].set_yticklabels([])
    colorbar = fig.colorbar(image, ax=axes, shrink=0.72, pad=0.02)
    colorbar.set_label("Post-shift monitor-failure rate")
    fig.suptitle(
        "Monitoring reliability is jointly determined by dataset, model, and shift family",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "fig3_reliability_failure_map")


def figure_pipeline_sign_reversal(
    scenarios: pd.DataFrame,
    sign_reversal: pd.DataFrame,
    output_dir: Path,
) -> None:
    pipeline = scenarios[
        scenarios["monitor"].eq("confidence")
        & scenarios["shift"].eq("pipeline")
    ].copy()
    means = (
        pipeline.groupby(["dataset", "model"])[
            ["true_excess_risk", "estimated_risk_change"]
        ]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    axis = axes[0]
    for model in ["lr", "rf", "xgb", "mlp"]:
        raw = pipeline[pipeline["model"].eq(model)]
        axis.scatter(
            raw["true_excess_risk"],
            raw["estimated_risk_change"],
            s=11,
            alpha=0.12,
            color=MODEL_COLORS[model],
            edgecolors="none",
        )
        mean = means[means["model"].eq(model)]
        axis.scatter(
            mean["true_excess_risk"],
            mean["estimated_risk_change"],
            s=62,
            color=MODEL_COLORS[model],
            edgecolors="white",
            linewidth=0.8,
            label=MODEL_LABELS[model],
            zorder=3,
        )
    axis.axhline(0, color=BLACK, linewidth=0.9)
    axis.axvline(0.05, color=BLACK, linewidth=0.9, linestyle=":")
    x_values = pipeline["true_excess_risk"].to_numpy(dtype=float)
    y_values = pipeline["estimated_risk_change"].to_numpy(dtype=float)
    axis.set_xlim(min(-0.25, np.nanmin(x_values) - 0.1), np.nanmax(x_values) * 1.04)
    y_padding = max((np.nanmax(y_values) - np.nanmin(y_values)) * 0.08, 0.01)
    axis.set_ylim(np.nanmin(y_values) - y_padding, np.nanmax(y_values) + y_padding)
    axis.set_xlabel("Oracle excess log loss")
    axis.set_ylabel("Confidence-estimated risk change")
    axis.set_title("Risk estimates often move in the wrong direction")
    axis.legend(ncol=2, frameon=False, loc="lower right")
    axis.grid(color=LIGHT_GRAY, linewidth=0.7)
    panel_label(axis, "A")

    axis = axes[1]
    forest = pd.concat(
        [
            sign_reversal[sign_reversal["scope"].eq("dataset")].assign(
                label=lambda x: x["dataset"].map(DATASET_LABELS)
            ),
            sign_reversal[sign_reversal["scope"].eq("model")].assign(
                label=lambda x: x["model"].map(MODEL_LABELS)
            ),
        ],
        ignore_index=True,
    )
    forest["category"] = np.where(forest["scope"].eq("dataset"), "Dataset", "Model")
    forest = forest.sort_values(["category", "sign_reversal_rate"])
    y = np.arange(len(forest))
    colors = [BLUE if category == "Dataset" else ORANGE for category in forest["category"]]
    estimates = forest["sign_reversal_rate"].to_numpy(dtype=float)
    errors = np.vstack(
        [
            estimates - forest["cluster_ci_lower"].to_numpy(dtype=float),
            forest["cluster_ci_upper"].to_numpy(dtype=float) - estimates,
        ]
    )
    for pos, estimate, error, color in zip(y, estimates, errors.T, colors):
        axis.errorbar(
            estimate,
            pos,
            xerr=error.reshape(2, 1),
            fmt="o",
            color=color,
            capsize=3,
            markersize=5,
        )
    overall = sign_reversal[sign_reversal["scope"].eq("overall")].iloc[0]
    axis.axvline(overall["sign_reversal_rate"], color=BLACK, linestyle="--", linewidth=1.2)
    axis.text(
        0.98,
        0.04,
        f"Overall {overall['sign_reversal_rate']:.1%}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )
    axis.set_yticks(y, forest["label"])
    axis.set_xlim(0, 1.02)
    axis.set_xlabel("Sign-reversal rate among harmful batches")
    axis.set_title("Failure persists across datasets and models")
    axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.7)
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, label="Dataset"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, label="Model"),
        ],
        frameon=False,
        loc="lower left",
    )
    panel_label(axis, "B")
    fig.suptitle(
        "Pipeline corruption produces systematic risk-estimate sign reversal",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "fig4_pipeline_sign_reversal")


def figure_drift_vs_harm(scenarios: pd.DataFrame, output_dir: Path) -> None:
    frame = scenarios[
        scenarios["monitor"].eq("drift_shap")
        & ~scenarios["shift"].eq("no_shift")
    ].copy()
    frame = (
        frame.groupby(["dataset", "model", "shift", "severity", "mode"])[
            ["domain_classifier_auc", "true_excess_risk"]
        ]
        .mean()
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    for shift in ["covariate", "correlated", "pipeline", "support", "concept"]:
        subset = frame[frame["shift"].eq(shift)]
        axis.scatter(
            subset["domain_classifier_auc"],
            subset["true_excess_risk"],
            s=26,
            alpha=0.65,
            color=SHIFT_COLORS[shift],
            edgecolors="white",
            linewidth=0.35,
            label=SHIFT_LABELS[shift],
        )
    axis.axvline(0.7, color=GRAY, linestyle="--", linewidth=1)
    axis.axhline(0.05, color=GRAY, linestyle="--", linewidth=1)
    high_drift = frame["domain_classifier_auc"].gt(0.7)
    harmful = frame["true_excess_risk"].gt(0.05)
    false_drift = int((high_drift & ~harmful).sum())
    missed_harm = int((~high_drift & harmful).sum())
    axis.text(
        0.98,
        0.04,
        f"Detectable but not harmful: {false_drift}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=BLACK,
        bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "pad": 3},
    )
    axis.text(
        0.02,
        0.96,
        f"Harmful but weakly detectable: {missed_harm}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=BLACK,
        bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "pad": 3},
    )
    axis.set_xlabel("Observable domain-classifier AUC")
    axis.set_ylabel("Oracle excess log loss")
    axis.set_title("Observable distribution drift is not equivalent to harmful model degradation")
    axis.legend(ncol=3, frameon=False, loc="upper center")
    axis.grid(color=LIGHT_GRAY, linewidth=0.7)
    save_figure(fig, output_dir, "fig5_drift_vs_harm")


def figure_attribution_map(scenarios: pd.DataFrame, output_dir: Path) -> None:
    shifts = ["covariate", "correlated", "pipeline", "support"]
    frame = scenarios[scenarios["monitor"].eq("drift_shap")].copy()
    frame["row_key"] = frame["dataset"] + "|" + frame["model"]
    row_keys, row_labels = _dataset_model_order(frame)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 8.2), constrained_layout=True)
    metrics = [
        ("top3_recall", "Top-3 intervention recall", "viridis"),
        ("attribution_failure", "Attribution-failure rate", "magma"),
    ]
    images = []
    for index, (metric, title, cmap) in enumerate(metrics):
        pivot = (
            frame.pivot_table(index="row_key", columns="shift", values=metric, aggfunc="mean")
            .reindex(index=row_keys, columns=shifts)
        )
        images.append(
            _annotated_heatmap(
                axes[index],
                pivot.to_numpy(dtype=float),
                row_labels,
                [SHIFT_LABELS[s] for s in shifts],
                title,
                cmap=cmap,
            )
        )
        panel_label(axes[index], chr(ord("A") + index))
        if index == 1:
            axes[index].set_yticklabels([])
    for axis, image, label in zip(axes, images, ["Recall", "Failure rate"]):
        colorbar = fig.colorbar(image, ax=axis, shrink=0.58, pad=0.02)
        colorbar.set_label(label)
    fig.suptitle(
        "DriftSHAP attribution reliability varies substantially across domains and predictors",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "fig6_attribution_reliability")


def _variance_category(term: str) -> str:
    if term in {"dataset", "model", "shift"}:
        return term.title()
    if term == "dataset:shift":
        return "Dataset × shift"
    if term == "model:shift":
        return "Model × shift"
    if term == "dataset:model":
        return "Dataset × model"
    if "severity" in term or "mode" in term or term == "seed":
        return "Severity / mode / seed"
    if term == "residual_and_higher_order":
        return "Residual / higher-order"
    return "Other interactions"


def figure_variance_decomposition(variance: pd.DataFrame, output_dir: Path) -> None:
    selections = [
        ("confidence", "risk_failure", "Confidence risk failure"),
        ("confidence", "alarm_failure", "Confidence alarm failure"),
        ("drift_shap", "attribution_failure", "DriftSHAP attribution failure"),
        ("drift_shap", "monitor_failure", "DriftSHAP combined failure"),
    ]
    records = []
    for monitor, outcome, label in selections:
        subset = variance[
            variance["monitor"].eq(monitor) & variance["outcome"].eq(outcome)
        ].copy()
        subset["category"] = subset["term"].map(_variance_category)
        for category, value in subset.groupby("category")["variance_fraction"].sum().items():
            records.append({"outcome": label, "category": category, "value": value})
    data = pd.DataFrame(records)
    categories = [
        "Dataset",
        "Model",
        "Shift",
        "Dataset × shift",
        "Model × shift",
        "Dataset × model",
        "Severity / mode / seed",
        "Other interactions",
        "Residual / higher-order",
    ]
    colors = [BLUE, GREEN, ORANGE, PURPLE, SKY, YELLOW, "#8C564B", "#7F7F7F", "#D1D5DB"]
    fig, axis = plt.subplots(figsize=(10.5, 4.7), constrained_layout=True)
    outcomes = [label for _, _, label in selections]
    left = np.zeros(len(outcomes))
    for category, color in zip(categories, colors):
        values = np.asarray(
            [
                data.loc[
                    data["outcome"].eq(outcome) & data["category"].eq(category),
                    "value",
                ].sum()
                for outcome in outcomes
            ]
        )
        axis.barh(outcomes, values, left=left, color=color, edgecolor="white", linewidth=0.5, label=category)
        for row, (start, value) in enumerate(zip(left, values)):
            if value >= 0.075:
                text_color = "white" if color not in {"#D1D5DB", YELLOW, SKY} else BLACK
                axis.text(start + value / 2, row, f"{value:.0%}", ha="center", va="center", fontsize=8, color=text_color, fontweight="bold")
        left += values
    axis.set_xlim(0, 1)
    axis.set_xlabel("Fraction of total outcome variance")
    axis.set_title("Failure variance is dominated by shift, dataset, and their interactions")
    axis.invert_yaxis()
    axis.legend(ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.18), loc="upper center")
    axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.7)
    save_figure(fig, output_dir, "fig7_variance_decomposition")


def figure_null_controls(scenarios: pd.DataFrame, output_dir: Path) -> None:
    frame = scenarios[scenarios["shift"].eq("no_shift")].copy()
    frame["row_key"] = frame["dataset"] + "|" + frame["model"]
    row_keys, row_labels = _dataset_model_order(frame)
    specs = [
        ("confidence", "risk_abs_error", "Confidence risk MAE", 0.0, 0.04, ".3f", "viridis"),
        ("confidence", "coverage", "Confidence interval coverage", 0.8, 1.0, ".0%", "viridis"),
        ("confidence", "alarm", "Confidence alarm rate", 0.0, 0.15, ".1%", "magma"),
        ("drift_shap", "alarm", "DriftSHAP alarm rate", 0.0, 0.15, ".1%", "magma"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 7.8), constrained_layout=True)
    for index, (monitor, metric, title, vmin, vmax, fmt, cmap) in enumerate(specs):
        values = (
            frame[frame["monitor"].eq(monitor)]
            .groupby("row_key")[metric]
            .mean()
            .reindex(row_keys)
            .to_numpy(dtype=float)
            .reshape(-1, 1)
        )
        image = _annotated_heatmap(
            axes[index], values, row_labels, ["Null"], title, vmin=vmin, vmax=vmax, fmt=fmt, cmap=cmap
        )
        if index > 0:
            axes[index].set_yticklabels([])
        colorbar = fig.colorbar(image, ax=axes[index], shrink=0.55, pad=0.04)
        panel_label(axes[index], chr(ord("A") + index))
    fig.suptitle("Null controls expose heterogeneous calibration despite low average error", fontsize=13, fontweight="bold")
    save_figure(fig, output_dir, "figA1_null_controls")


def figure_failure_prediction(
    grouped_auc: pd.DataFrame, seed_auc: pd.DataFrame, output_dir: Path
) -> None:
    seed = seed_auc.rename(columns={"held_out_seed": "held_out_group"}).copy()
    seed["held_out_dimension"] = "seed"
    frame = pd.concat([seed, grouped_auc], ignore_index=True)
    dimensions = ["seed", "model", "shift", "dataset"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    offsets = {"confidence": -0.12, "drift_shap": 0.12}
    for monitor, color, label in [
        ("confidence", BLUE, "Confidence"),
        ("drift_shap", ORANGE, "DriftSHAP"),
    ]:
        for index, dimension in enumerate(dimensions):
            values = frame[
                frame["monitor"].eq(monitor)
                & frame["held_out_dimension"].eq(dimension)
            ]
            x = index + offsets[monitor]
            axes[0].scatter(np.full(len(values), x), values["roc_auc"], s=24, alpha=0.65, color=color)
            axes[0].plot([x - 0.08, x + 0.08], [values["roc_auc"].mean()] * 2, color=BLACK, linewidth=2)
            lift = values["average_precision"] - values["failure_prevalence"]
            axes[1].scatter(np.full(len(values), x), lift, s=24, alpha=0.65, color=color)
            axes[1].plot([x - 0.08, x + 0.08], [lift.mean()] * 2, color=BLACK, linewidth=2)
    for axis in axes:
        axis.set_xticks(np.arange(len(dimensions)), ["Seed", "Model", "Shift", "Dataset"])
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)
    axes[0].axhline(0.5, color=GRAY, linestyle="--", linewidth=1)
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_title("Failure-prediction discrimination")
    axes[1].axhline(0.0, color=GRAY, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Average precision − failure prevalence")
    axes[1].set_title("Precision gain over the prevalence baseline")
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, label="Confidence"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, label="DriftSHAP"),
        ],
        frameon=False,
        loc="lower left",
    )
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    fig.suptitle("Observable failure diagnostics transfer poorly to unseen datasets", fontsize=13, fontweight="bold")
    save_figure(fig, output_dir, "figA2_failure_prediction")


def figure_temporal_detection(aggregate: pd.DataFrame, output_dir: Path) -> None:
    frame = aggregate[
        ~aggregate["shift"].eq("no_shift")
        & aggregate["mode"].isin(["abrupt", "gradual"])
    ].copy()
    records = []
    for monitor, power_column, delay_column in [
        ("confidence", "risk_alarm_power", "risk_alarm_detection_delay"),
        ("drift_shap", "shift_alarm_power", "shift_alarm_detection_delay"),
    ]:
        subset = frame[frame["monitor"].eq(monitor)]
        for (shift, mode), group in subset.groupby(["shift", "mode"]):
            records.append(
                {
                    "monitor": monitor,
                    "shift": shift,
                    "mode": mode,
                    "power": pd.to_numeric(group[power_column], errors="coerce").mean(),
                    "delay": pd.to_numeric(group[delay_column], errors="coerce").mean(),
                }
            )
    data = pd.DataFrame(records)
    shifts = ["covariate", "correlated", "pipeline", "support", "concept"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), constrained_layout=True)
    for row, monitor in enumerate(["confidence", "drift_shap"]):
        subset = data[data["monitor"].eq(monitor)]
        x = np.arange(len(shifts))
        width = 0.35
        for mode, offset, color in [("abrupt", -width / 2, BLUE), ("gradual", width / 2, ORANGE)]:
            mode_data = subset[subset["mode"].eq(mode)].set_index("shift").reindex(shifts)
            axes[row, 0].bar(x + offset, mode_data["power"], width, color=color, label=mode.title())
            axes[row, 1].bar(x + offset, mode_data["delay"], width, color=color, label=mode.title())
        axes[row, 0].set_ylim(0, 1)
        axes[row, 0].set_ylabel("Detection power")
        axes[row, 1].set_ylabel("Mean delay (batches)")
        for column in range(2):
            axes[row, column].set_xticks(x, [SHIFT_LABELS[s] for s in shifts], rotation=30, ha="right")
            axes[row, column].grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)
            axes[row, column].set_title(("Confidence risk alarms" if monitor == "confidence" else "DriftSHAP shift alarms") + (" — power" if column == 0 else " — delay"))
            panel_label(axes[row, column], chr(ord("A") + row * 2 + column))
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    fig.suptitle("Abrupt and gradual streams expose different sequential-detection behavior", fontsize=13, fontweight="bold")
    save_figure(fig, output_dir, "figA3_temporal_detection")


def figure_severity_correlations(correlations: pd.DataFrame, output_dir: Path) -> None:
    shifts = ["covariate", "correlated", "pipeline", "support", "concept"]
    metrics_by_monitor = {
        "confidence": ["true_excess_risk", "estimated_risk_change", "domain_classifier_auc", "risk_failure", "alarm"],
        "drift_shap": ["true_excess_risk", "monitor_score", "domain_classifier_auc", "attribution_failure", "alarm"],
    }
    metric_labels = {
        "true_excess_risk": "Oracle risk",
        "estimated_risk_change": "Estimated risk",
        "monitor_score": "Monitor score",
        "domain_classifier_auc": "Domain AUC",
        "risk_failure": "Risk failure",
        "attribution_failure": "Attr. failure",
        "alarm": "Alarm rate",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for index, monitor in enumerate(["confidence", "drift_shap"]):
        metrics = metrics_by_monitor[monitor]
        pivot = (
            correlations[correlations["monitor"].eq(monitor)]
            .pivot_table(index="metric", columns="shift", values="mean_spearman_rho", aggfunc="mean")
            .reindex(index=metrics, columns=shifts)
        )
        image = _annotated_heatmap(
            axes[index],
            pivot.to_numpy(dtype=float),
            [metric_labels[m] for m in metrics],
            [SHIFT_LABELS[s] for s in shifts],
            "Confidence" if monitor == "confidence" else "DriftSHAP",
            vmin=-1,
            vmax=1,
            fmt=".2f",
            cmap="coolwarm",
        )
        panel_label(axes[index], chr(ord("A") + index))
    colorbar = fig.colorbar(image, ax=axes, shrink=0.72, pad=0.02)
    colorbar.set_label("Mean Spearman correlation with severity")
    fig.suptitle("Severity–response relationships depend on both shift and monitoring task", fontsize=13, fontweight="bold")
    save_figure(fig, output_dir, "figA4_severity_correlations")


def main() -> int:
    args = parse_args()
    configure_style()
    results_dir = args.results_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    scenarios = pd.read_parquet(analysis_dir / "scenario_post_metrics.parquet")
    correlations = pd.read_csv(analysis_dir / "severity_correlation_summary.csv")
    sign_reversal = pd.read_csv(analysis_dir / "pipeline_sign_reversal.csv")
    variance = pd.read_csv(analysis_dir / "variance_decomposition.csv")
    aggregate = pd.read_parquet(results_dir / "aggregate_metrics.parquet")
    grouped_auc = pd.read_csv(results_dir / "reliability_diagnostic_grouped_auc.csv")
    seed_auc = pd.read_csv(results_dir / "reliability_diagnostic_auc.csv")
    rng = np.random.default_rng(args.random_seed)

    jobs = [
        ("1/11", "protocol schematic", lambda: figure_protocol(output_dir)),
        ("2/11", "concept blindness", lambda: figure_concept_blindness(scenarios, correlations, output_dir, args.bootstrap_replicates, rng)),
        ("3/11", "failure map", lambda: figure_failure_map(scenarios, output_dir)),
        ("4/11", "pipeline sign reversal", lambda: figure_pipeline_sign_reversal(scenarios, sign_reversal, output_dir)),
        ("5/11", "drift versus harm", lambda: figure_drift_vs_harm(scenarios, output_dir)),
        ("6/11", "attribution map", lambda: figure_attribution_map(scenarios, output_dir)),
        ("7/11", "variance decomposition", lambda: figure_variance_decomposition(variance, output_dir)),
        ("8/11", "null controls", lambda: figure_null_controls(scenarios, output_dir)),
        ("9/11", "failure prediction", lambda: figure_failure_prediction(grouped_auc, seed_auc, output_dir)),
        ("10/11", "temporal detection", lambda: figure_temporal_detection(aggregate, output_dir)),
        ("11/11", "severity correlations", lambda: figure_severity_correlations(correlations, output_dir)),
    ]
    for progress, name, job in jobs:
        print(f"[{progress}] {name}", flush=True)
        job()
    print(f"Saved 11 PDF and 11 PNG figures at: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
