"""Compare TABMON's independent PAPE implementation with the official code.

The official repository is loaded from a user-supplied checkout at runtime;
none of its CC BY-NC-SA source is copied into the MIT-licensed TABMON tree.
This audit uses a deterministic synthetic binary problem and compares both
density-ratio weights and the final expected-accuracy estimate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.pape import (
    PAPEDensityRatioEstimator,
    PAPEClassificationErrorEstimator,
)


def _problem() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260929)
    reference = pd.DataFrame(
        {
            "category": rng.choice(["a", "b", "c"], size=240),
            "x1": rng.normal(0.0, 1.0, size=240),
            "x2": rng.normal(0.0, 1.0, size=240),
        }
    )
    target = pd.DataFrame(
        {
            "category": rng.choice(
                ["a", "b", "c"], size=96, p=[0.15, 0.30, 0.55]
            ),
            "x1": rng.normal(0.65, 1.1, size=96),
            "x2": rng.normal(-0.25, 0.9, size=96),
        }
    )
    category_effect = reference["category"].map({"a": -0.4, "b": 0.1, "c": 0.5})
    reference_probability = 1.0 / (
        1.0 + np.exp(-(0.9 * reference["x1"] - 0.4 * reference["x2"] + category_effect))
    )
    target_category_effect = target["category"].map(
        {"a": -0.4, "b": 0.1, "c": 0.5}
    )
    target_probability = 1.0 / (
        1.0 + np.exp(-(0.9 * target["x1"] - 0.4 * target["x2"] + target_category_effect))
    )
    reference["model_probability"] = reference_probability
    reference["model_prediction"] = (reference_probability >= 0.5).astype(int)
    reference["label"] = rng.binomial(1, reference_probability)
    target["model_probability"] = target_probability
    target["model_prediction"] = (target_probability >= 0.5).astype(int)
    return reference, target


def run_audit(official_repo: Path, tolerance: float = 1e-10) -> dict[str, object]:
    official_repo = official_repo.resolve()
    if not (official_repo / "methods" / "PAPE.py").is_file():
        raise FileNotFoundError(
            f"Official PAPE checkout not found at {official_repo}"
        )
    sys.path.insert(0, str(official_repo))
    try:
        official_module = importlib.import_module("methods.PAPE")
    finally:
        sys.path.pop(0)

    reference, target = _problem()
    feature_columns = ["category", "x1", "x2"]
    official = official_module.PAPE(
        "model_probability",
        "model_prediction",
        "label",
        ["x1", "x2"],
        ["category"],
        ["accuracy"],
    )
    official.fit(reference)
    official_domain_model = official.fit_DRE_model(
        target[feature_columns], reference[feature_columns]
    )
    official_weights = official.get_chunk_weights_on_ref_data(
        official_domain_model,
        reference[feature_columns],
        target[feature_columns],
    )
    official_accuracy = float(official.estimate(target)[0])

    local_density = PAPEDensityRatioEstimator(n_jobs=None).estimate_weights(
        reference[feature_columns], target[feature_columns]
    )
    reference_predictions = pd.DataFrame(
        {
            "probability_class_0": 1.0 - reference["model_probability"],
            "probability_class_1": reference["model_probability"],
            "predicted_class": reference["model_prediction"],
        }
    )
    target_predictions = pd.DataFrame(
        {
            "probability_class_0": 1.0 - target["model_probability"],
            "probability_class_1": target["model_probability"],
            "predicted_class": target["model_prediction"],
        }
    )
    local_estimator = PAPEClassificationErrorEstimator(
        reference_predictions,
        reference["label"],
        [0, 1],
        n_jobs=None,
    )
    local_accuracy = local_estimator.estimate(
        target_predictions, local_density.weights
    ).estimated_accuracy

    weight_difference = float(
        np.max(np.abs(np.asarray(official_weights) - local_density.weights))
    )
    accuracy_difference = abs(official_accuracy - local_accuracy)
    passed = weight_difference <= tolerance and accuracy_difference <= tolerance
    return {
        "passed": passed,
        "tolerance": tolerance,
        "official_repository": str(official_repo),
        "reference_rows": len(reference),
        "target_rows": len(target),
        "maximum_absolute_weight_difference": weight_difference,
        "official_estimated_accuracy": official_accuracy,
        "tabmon_estimated_accuracy": local_accuracy,
        "absolute_accuracy_difference": accuracy_difference,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args(argv)
    if args.tolerance <= 0:
        parser.error("tolerance must be positive")
    report = run_audit(args.official_repo, args.tolerance)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    print("=== TABMON PAPE FIDELITY AUDIT ===")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
