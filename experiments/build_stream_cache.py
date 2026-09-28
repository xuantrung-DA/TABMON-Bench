"""Build the reusable observable/oracle cache for TABMON-Bench v8.

Each shifted feature stream is generated once and shared by all model
probability caches.  Hidden target labels and oracle losses are written under
a physically separate ``oracle`` directory.  The monitor-facing reader only
exposes ``observable`` files.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_config import (
    DATASETS,
    DATASET_CONFIG,
    MODELS,
    MODES,
    SEVERITIES,
    SHIFT_FAMILIES,
)
from src.cache.stream_cache import (
    BATCH_INDEX,
    CACHE_SCHEMA_VERSION,
    KEY_COLUMNS,
    ROW_INDEX,
    SOURCE_LABEL,
    PredictorStreamSpec,
    atomic_json,
    atomic_parquet,
    build_stream_specs,
    losses_for_labels,
    materialize_stream,
    oracle_batch_risk,
    predict_frame,
)


CACHE_DATASET_CONFIG = {
    **DATASET_CONFIG,
    "covertype_multiclass": {
        "target": "Cover_Type",
        "shift_feature": "Elevation",
        "corr_features": ["Elevation", "Slope"],
    },
}
CACHE_DATASETS = [*DATASETS, "covertype_multiclass"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=CACHE_DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument(
        "--shifts", nargs="+", choices=SHIFT_FAMILIES, default=SHIFT_FAMILIES
    )
    parser.add_argument(
        "--severities", nargs="+", choices=SEVERITIES, default=SEVERITIES
    )
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=["abrupt", "gradual"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--base-models-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "base_models",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "stream_cache_v1",
    )
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if args.num_batches <= 0 or args.batch_size <= 0:
        parser.error("num-batches and batch-size must be positive")
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("max-hours must be positive")
    if args.smoke_test:
        args.datasets = ["adult"]
        args.models = ["lr"]
        args.shifts = ["covariate"]
        args.severities = ["high"]
        args.modes = ["abrupt"]
        args.seeds = [42]
        args.num_batches = 2
        args.batch_size = min(args.batch_size, 32)
    return args


def _validate_inputs(args: argparse.Namespace) -> None:
    missing = []
    for dataset in args.datasets:
        for filename in ("calibration.parquet", "test_pool.parquet"):
            path = args.data_dir / dataset / filename
            if not path.is_file():
                missing.append(path)
        for model in args.models:
            path = (
                args.base_models_dir
                / "models"
                / dataset
                / f"{model}_calibrated.pkl"
            )
            if not path.is_file():
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing cache inputs:\n" + "\n".join(f"  - {path}" for path in missing)
        )


def _reference_cache(
    cache_dir: Path,
    dataset: str,
    reference: pd.DataFrame,
    target_column: str,
    models: dict[str, Any],
) -> tuple[dict[str, float], dict[str, list[Any]]]:
    observable_reference = cache_dir / "observable" / "references" / dataset
    reference_X = reference.drop(columns=[target_column]).reset_index(drop=True)
    reference_keys = pd.DataFrame(
        {"_tabmon_reference_row": np.arange(len(reference), dtype=np.int32)}
    )
    feature_path = observable_reference / "features.parquet"
    source_label_path = observable_reference / "source_labels.parquet"
    if not feature_path.is_file():
        atomic_parquet(pd.concat([reference_keys, reference_X], axis=1), feature_path)
    if not source_label_path.is_file():
        atomic_parquet(
            pd.concat(
                [
                    reference_keys,
                    reference[target_column]
                    .reset_index(drop=True)
                    .rename(SOURCE_LABEL),
                ],
                axis=1,
            ),
            source_label_path,
        )

    risks: dict[str, float] = {}
    class_labels: dict[str, list[Any]] = {}
    keyed_reference = pd.concat(
        [
            reference_keys.rename(
                columns={"_tabmon_reference_row": BATCH_INDEX}
            ).assign(**{ROW_INDEX: np.arange(len(reference), dtype=np.int32)}),
            reference_X,
        ],
        axis=1,
    )
    # The temporary batch key is removed from the persisted reference file.
    keyed_reference[BATCH_INDEX] = 0
    for model_name, model in models.items():
        prediction, probabilities, classes = predict_frame(model, keyed_reference)
        prediction = prediction.drop(columns=[BATCH_INDEX]).rename(
            columns={ROW_INDEX: "_tabmon_reference_row"}
        )
        path = (
            cache_dir
            / "observable"
            / "reference_predictions"
            / dataset
            / f"{model_name}.parquet"
        )
        if not path.is_file():
            atomic_parquet(prediction, path)
        losses = losses_for_labels(
            probabilities, classes, reference[target_column].reset_index(drop=True)
        )
        risks[model_name] = float(np.mean(losses))
        class_labels[model_name] = [
            value.item() if isinstance(value, np.generic) else value
            for value in classes.tolist()
        ]
        atomic_json(
            {
                "dataset": dataset,
                "model": model_name,
                "reference_log_loss": risks[model_name],
                "n_reference_rows": len(reference),
                "label_role": "labeled source calibration only",
            },
            cache_dir
            / "oracle"
            / "reference_risk"
            / dataset
            / f"{model_name}.json",
        )
    return risks, class_labels


def _completed_stream(
    cache_dir: Path, stream_id: str, predictor_ids: list[str]
) -> bool:
    stream_paths = [
        cache_dir / "observable" / "streams" / f"{stream_id}.parquet",
        cache_dir / "oracle" / "targets" / f"{stream_id}.parquet",
        cache_dir / "oracle" / "interventions" / f"{stream_id}.parquet",
    ]
    predictor_paths = []
    for predictor_id in predictor_ids:
        predictor_paths.extend(
            [
                cache_dir
                / "observable"
                / "predictions"
                / f"{predictor_id}.parquet",
                cache_dir / "oracle" / "batch_risk" / f"{predictor_id}.parquet",
            ]
        )
    return all(path.is_file() for path in stream_paths + predictor_paths)


def _write_manifests(
    args: argparse.Namespace,
    stream_records: list[dict[str, Any]],
    predictor_records: list[dict[str, Any]],
    requested_streams: int,
    stopped_for_time: bool,
) -> None:
    cache_dir = args.cache_dir
    atomic_parquet(
        pd.DataFrame(stream_records), cache_dir / "control" / "stream_index.parquet"
    )
    atomic_parquet(
        pd.DataFrame(predictor_records),
        cache_dir / "control" / "predictor_stream_index.parquet",
    )
    complete = len(stream_records) == requested_streams and not stopped_for_time
    status = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_revision": 1,
        "complete": complete,
        "requested_streams": requested_streams,
        "completed_streams": len(stream_records),
        "completed_predictor_streams": len(predictor_records),
        "stopped_for_time": stopped_for_time,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(status, cache_dir / "cache_status.json")
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_revision": 1,
        "purpose": "reusable streams and model probabilities for v8 monitors",
        "configuration": {
            "datasets": args.datasets,
            "models": args.models,
            "shifts": args.shifts,
            "severities": args.severities,
            "modes": args.modes,
            "seeds": args.seeds,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
        },
        "security_boundary": {
            "observable_contains_target_labels": False,
            "observable_contains_oracle_risk": False,
            "observable_contains_failure_labels": False,
            "source_calibration_labels_are_permitted": True,
            "target_labels_location": "oracle/targets only",
            "oracle_reader_is_separate": True,
            "generator_shift_fraction_is_monitor_input": False,
            "intervention_ground_truth_is_monitor_input": False,
        },
        "storage": {
            "target_features": "observable/streams",
            "target_probabilities": "observable/predictions",
            "source_reference": "observable/references",
            "target_labels": "oracle/targets",
            "oracle_batch_risk": "oracle/batch_risk",
            "intervention_targets": "oracle/interventions",
            "scenario_control_index": "control",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": metadata.version("xgboost"),
        },
    }
    atomic_json(manifest, cache_dir / "cache_manifest.json")
    observable_manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "allowed_contents": [
            "unlabeled target features",
            "model probabilities and predictions",
            "labeled source calibration data",
        ],
        "forbidden_contents": [
            "target labels",
            "oracle target risk",
            "failure labels",
            "intervention ground truth",
            "generator shift fraction",
        ],
        "monitor_reader": "src.cache.stream_cache.ObservableCacheReader",
    }
    atomic_json(observable_manifest, cache_dir / "observable" / "manifest.json")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.data_dir = args.data_dir.resolve()
    args.base_models_dir = args.base_models_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    _validate_inputs(args)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    specs = build_stream_specs(
        args.datasets,
        args.shifts,
        args.severities,
        args.modes,
        args.seeds,
        args.num_batches,
        args.batch_size,
    )
    started = time.monotonic()
    stream_records: list[dict[str, Any]] = []
    predictor_records: list[dict[str, Any]] = []
    stopped_for_time = False
    by_dataset = {dataset: [] for dataset in args.datasets}
    for spec in specs:
        by_dataset[spec.dataset].append(spec)

    completed = 0
    for dataset in args.datasets:
        config = CACHE_DATASET_CONFIG[dataset]
        target = config["target"]
        reference = pd.read_parquet(
            args.data_dir / dataset / "calibration.parquet"
        )
        test_pool = pd.read_parquet(args.data_dir / dataset / "test_pool.parquet")
        models = {
            model_name: joblib.load(
                args.base_models_dir
                / "models"
                / dataset
                / f"{model_name}_calibrated.pkl"
            )
            for model_name in args.models
        }
        reference_risk, class_labels = _reference_cache(
            args.cache_dir, dataset, reference, target, models
        )

        for spec in by_dataset[dataset]:
            predictor_specs = [
                PredictorStreamSpec(spec.stream_id, dataset, model_name)
                for model_name in args.models
            ]
            predictor_ids = [item.predictor_stream_id for item in predictor_specs]
            if _completed_stream(args.cache_dir, spec.stream_id, predictor_ids):
                observable = pd.read_parquet(
                    args.cache_dir
                    / "observable"
                    / "streams"
                    / f"{spec.stream_id}.parquet"
                )
                oracle_targets = pd.read_parquet(
                    args.cache_dir
                    / "oracle"
                    / "targets"
                    / f"{spec.stream_id}.parquet"
                )
            else:
                observable, oracle_targets, interventions = materialize_stream(
                    spec, test_pool, target, config
                )
                atomic_parquet(
                    observable,
                    args.cache_dir
                    / "observable"
                    / "streams"
                    / f"{spec.stream_id}.parquet",
                )
                atomic_parquet(
                    oracle_targets,
                    args.cache_dir
                    / "oracle"
                    / "targets"
                    / f"{spec.stream_id}.parquet",
                )
                atomic_parquet(
                    interventions,
                    args.cache_dir
                    / "oracle"
                    / "interventions"
                    / f"{spec.stream_id}.parquet",
                )

            stream_records.append(
                {
                    **spec.__dict__,
                    "stream_id": spec.stream_id,
                    "n_rows": len(observable),
                    "observable_path": f"observable/streams/{spec.stream_id}.parquet",
                }
            )
            for predictor_spec in predictor_specs:
                model_name = predictor_spec.model
                predictor_id = predictor_spec.predictor_stream_id
                prediction_path = (
                    args.cache_dir
                    / "observable"
                    / "predictions"
                    / f"{predictor_id}.parquet"
                )
                risk_path = (
                    args.cache_dir
                    / "oracle"
                    / "batch_risk"
                    / f"{predictor_id}.parquet"
                )
                if not prediction_path.is_file() or not risk_path.is_file():
                    prediction, probabilities, classes = predict_frame(
                        models[model_name], observable
                    )
                    risk = oracle_batch_risk(
                        oracle_targets,
                        probabilities,
                        classes,
                        reference_risk[model_name],
                    )
                    atomic_parquet(prediction, prediction_path)
                    atomic_parquet(risk, risk_path)
                predictor_records.append(
                    {
                        **predictor_spec.__dict__,
                        "predictor_stream_id": predictor_id,
                        "n_rows": len(observable),
                        "classes_json": json.dumps(class_labels[model_name]),
                        "prediction_path": (
                            f"observable/predictions/{predictor_id}.parquet"
                        ),
                    }
                )

            completed += 1
            print(
                f"[{completed}/{len(specs)}] {dataset}/{spec.shift}/"
                f"{spec.severity}/{spec.mode}/seed={spec.seed}",
                flush=True,
            )
            if args.max_hours is not None:
                elapsed_hours = (time.monotonic() - started) / 3600.0
                if elapsed_hours >= args.max_hours:
                    stopped_for_time = True
                    break
        if stopped_for_time:
            break

    _write_manifests(
        args,
        stream_records,
        predictor_records,
        len(specs),
        stopped_for_time,
    )
    print("=== TABMON STREAM CACHE ===")
    print(f"Streams: {len(stream_records)}/{len(specs)}")
    print(f"Predictor streams: {len(predictor_records)}")
    print(f"Cache: {args.cache_dir}")
    return 0 if not stopped_for_time else 2


if __name__ == "__main__":
    raise SystemExit(main())
