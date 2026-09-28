"""Compare scalable grouped-permutation XPE with official-style KernelSHAP.

The audit uses the same equal-mass EMD matches and sampled rows for both
backends. It is intentionally restricted to a predeclared stress subset:
high-severity abrupt pipeline corruption, the earliest seed, LR and XGB, and
every available dataset. A failed audit forbids presenting the scalable
backend as an XPE result; the KernelSHAP backend must then be used directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.xpe import (
    IMPLEMENTATION_VERSION,
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    XPEExplainer,
)
from src.cache.stream_cache import ObservableCacheReader, atomic_json, atomic_parquet


def _seed(*values: object) -> int:
    payload = "|".join(str(value) for value in values)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def _index(cache: Path) -> pd.DataFrame:
    streams = pd.read_parquet(cache / "control" / "stream_index.parquet")
    predictors = pd.read_parquet(cache / "control" / "predictor_stream_index.parquet")
    frame = predictors.merge(
        streams[["stream_id", "shift", "severity", "mode", "seed", "batch_size"]],
        on="stream_id",
        validate="many_to_one",
    )
    earliest = int(frame["seed"].min())
    chosen = frame[
        frame["model"].isin(["lr", "xgb"])
        & (frame["seed"] == earliest)
        & (frame["shift"] == "pipeline")
        & (frame["severity"] == "high")
        & (frame["mode"] == "abrupt")
    ].copy()
    expected = 2 * chosen["dataset"].nunique()
    if len(chosen) != expected:
        raise ValueError("Could not construct the predeclared XPE fidelity subset")
    return chosen.sort_values(["dataset", "model"]).reset_index(drop=True)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, left[0]) and np.allclose(right, right[0]):
        return 1.0 if np.allclose(left, right) else 0.0
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=1024)
    parser.add_argument("--kernel-nsamples", type=int, default=3000)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if args.sample_size < 2 or args.permutations < 1 or args.kernel_nsamples < 1:
        parser.error("Invalid XPE fidelity budget")

    cache = args.cache_dir.resolve()
    models = args.base_models_dir.resolve()
    output = args.output_dir.resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    index = _index(cache)
    if args.smoke_test:
        index = index.iloc[:1].copy()
    observable = ObservableCacheReader(cache)

    for position, row in enumerate(index.itertuples(index=False), start=1):
        checkpoint = checkpoints / f"{row.predictor_stream_id}.json"
        if checkpoint.is_file():
            prior = json.loads(checkpoint.read_text("utf-8"))
            reusable = (
                prior.get("implementation_version") == IMPLEMENTATION_VERSION
                and prior.get("sample_size") == args.sample_size
                and prior.get("permutations") == args.permutations
                and prior.get("kernel_nsamples") == args.kernel_nsamples
            )
            if reusable:
                continue
            checkpoint.unlink()
        reference_X = observable.load_reference_features(row.dataset)
        reference_y = observable.load_source_calibration_labels(row.dataset)
        target = observable.load_target_features(row.stream_id)
        target = target.iloc[-int(row.batch_size) :].reset_index(drop=True)
        model = joblib.load(
            models / "models" / row.dataset / f"{row.model}_calibrated.pkl"
        )
        explainer = XPEExplainer(reference_X, reference_y)
        seed = _seed("xpe-fidelity", row.predictor_stream_id)
        scalable = explainer.explain(
            model,
            target,
            sample_size=args.sample_size,
            attribution_backend="grouped_permutation",
            permutations=args.permutations,
            random_seed=seed,
        )
        official = explainer.explain(
            model,
            target,
            sample_size=args.sample_size,
            attribution_backend="kernel_shap",
            kernel_nsamples=args.kernel_nsamples,
            random_seed=seed,
        )
        left = scalable.per_target_attribution.reshape(-1)
        right = official.per_target_attribution.reshape(-1)
        left_feature = np.asarray(list(scalable.attribution.values()))
        right_feature = np.asarray(list(official.attribution.values()))
        relative_l1 = float(
            np.sum(np.abs(left_feature - right_feature))
            / max(np.sum(np.abs(right_feature)), 1e-12)
        )
        record = {
            "implementation_version": IMPLEMENTATION_VERSION,
            "predictor_stream_id": row.predictor_stream_id,
            "dataset": row.dataset,
            "model": row.model,
            "seed": int(row.seed),
            "sample_size": args.sample_size,
            "permutations": args.permutations,
            "kernel_nsamples": args.kernel_nsamples,
            "per_target_spearman": _correlation(left, right),
            "feature_spearman": _correlation(left_feature, right_feature),
            "feature_relative_l1": relative_l1,
            "per_target_mae": float(np.mean(np.abs(left - right))),
            "estimated_loss_change_delta": abs(
                scalable.estimated_loss_change - official.estimated_loss_change
            ),
            "scalable_efficiency_error": scalable.shapley_efficiency_max_abs_error,
            "kernel_efficiency_error": official.shapley_efficiency_max_abs_error,
            "coupling_retained_mass_fraction": scalable.coupling_retained_mass_fraction,
            "coupling_one_to_one_fraction": scalable.coupling_one_to_one_fraction,
        }
        atomic_json(record, checkpoint)
        print(f"[{position}/{len(index)}] {row.dataset}/{row.model}", flush=True)

    records = [
        json.loads(path.read_text("utf-8"))
        for path in sorted(checkpoints.glob("*.json"))
    ]
    frame = pd.DataFrame(records)
    atomic_parquet(frame, output / "xpe_fidelity_metrics.parquet")
    complete = len(frame) == len(index)
    thresholds = {
        "median_feature_spearman_min": 0.90,
        "median_feature_relative_l1_max": 0.15,
        "max_loss_change_delta": 1e-10,
        "max_efficiency_error": 1e-5,
        "min_coupling_mass": 1.0 - 1e-6,
    }
    passed = bool(
        complete
        and frame["feature_spearman"].median()
        >= thresholds["median_feature_spearman_min"]
        and frame["feature_relative_l1"].median()
        <= thresholds["median_feature_relative_l1_max"]
        and frame["estimated_loss_change_delta"].max()
        <= thresholds["max_loss_change_delta"]
        and frame[
            ["scalable_efficiency_error", "kernel_efficiency_error"]
        ].to_numpy().max()
        <= thresholds["max_efficiency_error"]
        and frame["coupling_retained_mass_fraction"].min()
        >= thresholds["min_coupling_mass"]
    )
    summary = {
        "complete": complete,
        "pass": passed,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "implementation_version": IMPLEMENTATION_VERSION,
        "subset": "earliest seed; pipeline/high/abrupt; LR and XGB; all datasets",
        "evaluated_streams": len(frame),
        "thresholds": thresholds,
        "median_feature_spearman": float(frame["feature_spearman"].median()),
        "median_feature_relative_l1": float(
            frame["feature_relative_l1"].median()
        ),
        "median_per_target_spearman": float(
            frame["per_target_spearman"].median()
        ),
        "max_efficiency_error": float(
            frame[["scalable_efficiency_error", "kernel_efficiency_error"]]
            .to_numpy()
            .max()
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(summary, output / "xpe_fidelity_summary.json")
    print(json.dumps(summary, indent=2))
    return 0 if passed or args.smoke_test else 1


if __name__ == "__main__":
    raise SystemExit(main())
