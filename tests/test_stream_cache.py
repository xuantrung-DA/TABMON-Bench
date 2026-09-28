import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from experiments.audit_stream_cache_archive import audit_archive
from experiments.build_stream_cache import main as build_stream_cache
from experiments.repair_stream_cache_v1 import repair_archive
from src.cache.stream_cache import (
    BATCH_INDEX,
    FORBIDDEN_OBSERVABLE_COLUMNS,
    ObservableCacheReader,
    OracleCacheReader,
    ROW_INDEX,
    assert_observable_columns,
    build_stream_specs,
    predict_frame,
)


class NearTieModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, X):
        del X
        return np.asarray([[0.5 - 1e-9, 0.5 + 1e-9]], dtype=np.float64)


class StreamCacheTest(unittest.TestCase):
    def test_prediction_class_uses_persisted_float32_probabilities(self):
        observable = pd.DataFrame(
            {BATCH_INDEX: [0], ROW_INDEX: [0], "feature": [1.0]}
        )
        result, original_probabilities, classes = predict_frame(
            NearTieModel(), observable
        )

        stored = result[["probability_class_0", "probability_class_1"]].to_numpy()
        expected = classes[np.argmax(stored, axis=1)]
        self.assertEqual(int(np.argmax(original_probabilities, axis=1)[0]), 1)
        self.assertEqual(result.loc[0, "predicted_class"], expected[0])
        self.assertEqual(result.loc[0, "predicted_class"], 0)

    def test_repair_archive_changes_only_inconsistent_prediction_class(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_zip = root / "cache_v1.zip"
            output_zip = root / "cache_v1_1.zip"
            prediction_path = "observable/predictions/example.parquet"
            prediction_frame = pd.DataFrame(
                {
                    BATCH_INDEX: [0],
                    ROW_INDEX: [0],
                    "probability_class_0": np.asarray([0.5], dtype=np.float32),
                    "probability_class_1": np.asarray([0.5], dtype=np.float32),
                    "predicted_class": [1],
                }
            )
            predictor_index = pd.DataFrame(
                {
                    "prediction_path": [prediction_path],
                    "classes_json": [json.dumps([0, 1])],
                }
            )

            def parquet_bytes(frame):
                buffer = io.BytesIO()
                frame.to_parquet(buffer, index=False)
                return buffer.getvalue()

            with zipfile.ZipFile(input_zip, "w") as archive:
                archive.writestr(
                    "cache/cache_status.json",
                    json.dumps({"cache_schema_version": 1, "complete": True}),
                )
                archive.writestr("cache/cache_manifest.json", "{}")
                archive.writestr(
                    "cache/control/predictor_stream_index.parquet",
                    parquet_bytes(predictor_index),
                )
                archive.writestr(
                    "cache/" + prediction_path,
                    parquet_bytes(prediction_frame),
                )
                archive.writestr("cache/unchanged.txt", b"unchanged")

            report = repair_archive(input_zip, output_zip)
            self.assertEqual(report["prediction_files_repaired"], 1)
            self.assertEqual(report["prediction_rows_repaired"], 1)
            self.assertFalse(report["probabilities_modified"])
            self.assertFalse(report["oracle_values_modified"])

            with zipfile.ZipFile(output_zip) as archive:
                repaired = pd.read_parquet(
                    io.BytesIO(archive.read("cache/" + prediction_path))
                )
                status = json.loads(archive.read("cache/cache_status.json"))
                embedded = json.loads(archive.read("cache/repair_report.json"))
                self.assertEqual(archive.read("cache/unchanged.txt"), b"unchanged")

            self.assertEqual(repaired.loc[0, "predicted_class"], 0)
            self.assertEqual(repaired.loc[0, "probability_class_0"], 0.5)
            self.assertEqual(repaired.loc[0, "probability_class_1"], 0.5)
            self.assertEqual(status["cache_revision"], 1)
            self.assertEqual(embedded["prediction_rows_repaired"], 1)

            audit = audit_archive(output_zip, compare_source=input_zip)
            self.assertEqual(audit["predicted_class_argmax_mismatches"], 0)
            self.assertEqual(audit["unexpected_changed_members"], 0)
            self.assertTrue(audit["repaired_file_nonclass_columns_unchanged"])

    def test_full_controlled_design_has_775_shared_streams(self):
        specs = build_stream_specs(
            datasets=[
                "adult",
                "bank_marketing",
                "acs_income",
                "covertype",
                "diabetes_hospitals",
            ],
            shifts=[
                "no_shift",
                "covariate",
                "correlated",
                "support",
                "pipeline",
                "concept",
            ],
            severities=["low", "medium", "high"],
            modes=["abrupt", "gradual"],
            seeds=[42, 43, 44, 45, 46],
            num_batches=10,
            batch_size=1000,
        )
        self.assertEqual(len(specs), 775)
        self.assertEqual(len({spec.stream_id for spec in specs}), 775)
        self.assertEqual(sum(spec.shift == "no_shift" for spec in specs), 25)

    def test_observable_allowlist_rejects_oracle_columns(self):
        for column in sorted(FORBIDDEN_OBSERVABLE_COLUMNS):
            with self.subTest(column=column):
                with self.assertRaises(ValueError):
                    assert_observable_columns(["age", column])

    def test_cache_builder_physically_separates_target_oracle(self):
        rng = np.random.default_rng(11)
        frame = pd.DataFrame(
            {
                "age": rng.normal(40, 10, 180),
                "education-num": rng.integers(1, 16, 180),
            }
        )
        frame["income"] = (
            frame["age"] + 1.5 * frame["education-num"] > 55
        ).astype(int)

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            data_dir = root / "data" / "adult"
            model_dir = root / "models" / "models" / "adult"
            cache_dir = root / "cache"
            data_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            calibration = frame.iloc[:80].reset_index(drop=True)
            test_pool = frame.iloc[80:].reset_index(drop=True)
            calibration.to_parquet(data_dir / "calibration.parquet", index=False)
            test_pool.to_parquet(data_dir / "test_pool.parquet", index=False)

            model = LogisticRegression(max_iter=500).fit(
                calibration[["age", "education-num"]], calibration["income"]
            )
            joblib.dump(model, model_dir / "lr_calibrated.pkl")
            joblib.dump(model, model_dir / "rf_calibrated.pkl")

            exit_code = build_stream_cache(
                [
                    "--datasets",
                    "adult",
                    "--models",
                    "lr",
                    "rf",
                    "--shifts",
                    "covariate",
                    "--severities",
                    "high",
                    "--modes",
                    "abrupt",
                    "--seeds",
                    "42",
                    "--num-batches",
                    "2",
                    "--batch-size",
                    "16",
                    "--data-dir",
                    str(root / "data"),
                    "--base-models-dir",
                    str(root / "models"),
                    "--cache-dir",
                    str(cache_dir),
                ]
            )
            self.assertEqual(exit_code, 0)

            status = json.loads((cache_dir / "cache_status.json").read_text())
            self.assertTrue(status["complete"])
            self.assertEqual(status["completed_streams"], 1)
            self.assertEqual(status["completed_predictor_streams"], 2)

            stream_index = pd.read_parquet(
                cache_dir / "control" / "stream_index.parquet"
            )
            predictor_index = pd.read_parquet(
                cache_dir / "control" / "predictor_stream_index.parquet"
            )
            self.assertEqual(len(stream_index), 1)
            self.assertEqual(len(predictor_index), 2)
            self.assertEqual(predictor_index["stream_id"].nunique(), 1)

            observable_reader = ObservableCacheReader(cache_dir)
            oracle_reader = OracleCacheReader(cache_dir)
            stream_id = stream_index.loc[0, "stream_id"]
            predictor_id = predictor_index.loc[0, "predictor_stream_id"]
            target_X = observable_reader.load_target_features(stream_id)
            probabilities = observable_reader.load_target_probabilities(
                predictor_id
            )
            targets = oracle_reader.load_target_labels(stream_id)
            batch_risk = oracle_reader.load_batch_risk(predictor_id)

            self.assertEqual(len(target_X), 32)
            self.assertEqual(len(probabilities), 32)
            self.assertEqual(len(targets), 32)
            self.assertEqual(len(batch_risk), 2)
            self.assertNotIn("income", target_X.columns)
            self.assertTrue(
                FORBIDDEN_OBSERVABLE_COLUMNS.isdisjoint(target_X.columns)
            )
            self.assertTrue(
                FORBIDDEN_OBSERVABLE_COLUMNS.isdisjoint(probabilities.columns)
            )
            self.assertIn("target_label", targets.columns)
            self.assertIn("true_excess_risk", batch_risk.columns)
            self.assertFalse(hasattr(observable_reader, "load_target_labels"))

            source_y = observable_reader.load_source_calibration_labels("adult")
            self.assertEqual(len(source_y), len(calibration))


if __name__ == "__main__":
    unittest.main()
