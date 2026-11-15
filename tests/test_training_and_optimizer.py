from __future__ import annotations

import joblib
import numpy as np
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from python.cache_optimizer import generate_candidates, recommend_cache_configuration
from python.config import CacheConfig, DATASET_DIR
from python.evaluate_model import evaluate_model
from python.train_model import (
    EXPECTED_PLAN_ID,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    _aggregate_repeated_experiments,
    _evaluation_status,
    _scores,
    _split_by_trace,
    _validate_plan_dataset_scenarios,
    _validate_training_dataset,
    _workload_stratified_folds,
    cross_validate_model,
    train_model,
)


class _FixedPredictor:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.values[: len(frame)], dtype=float)


def _valid_training_row(plan_id: str = EXPECTED_PLAN_ID) -> dict:
    return {
        "trace_id": "trace-1",
        "plan_id": plan_id,
        "workload_type": "cnn",
        "cache_size_bytes": 4096,
        "associativity": 1,
        "replacement_policy": "LRU",
        "block_size_bytes": 64,
        "read_write_ratio": 4.0,
        "num_accesses": 100,
        "temporal_locality": 0.5,
        "spatial_locality": 0.75,
        "hit_rate": 0.75,
        "miss_rate": 0.25,
        "hit_count": 75,
        "miss_count": 25,
        "access_count": 100,
        "expected_access_count": 100,
        "actual_access_count": 100,
        "measurement_source": "vivado_xsim",
        "synthetic": False,
        "validated": True,
    }


class TrainingAndOptimizerTests(unittest.TestCase):
    def test_optimizer_rejects_nonfinite_and_out_of_range_predictions(self) -> None:
        candidate = CacheConfig(4096, 1, "LRU", 64)
        invalid_values = (
            (float("nan"), "NaN or infinite"),
            (float("inf"), "NaN or infinite"),
            (float("-inf"), "NaN or infinite"),
            (-0.01, "outside"),
            (1.01, "outside"),
        )
        with tempfile.TemporaryDirectory() as temp:
            for value, message in invalid_values:
                bundle = {
                    "training_report": {"status": "evaluated"},
                    "models": {"hit_rate": _FixedPredictor([value])},
                    "feature_columns": ["workload_type"],
                }
                with self.subTest(value=value), patch(
                    "python.cache_optimizer.joblib.load", return_value=bundle
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        recommend_cache_configuration(
                            {"workload_type": "cnn"},
                            Path(temp) / "model.joblib",
                            Path(temp) / "recommendation.json",
                            candidates=[candidate],
                        )

    def test_optimizer_rejects_illegal_custom_candidates(self) -> None:
        bundle = {
            "training_report": {"status": "evaluated"},
            "models": {"hit_rate": _FixedPredictor([0.75])},
            "feature_columns": ["workload_type"],
        }
        with tempfile.TemporaryDirectory() as temp:
            for policy in ("FIFO", "RANDOM"):
                with self.subTest(policy=policy), patch(
                    "python.cache_optimizer.joblib.load", return_value=bundle
                ):
                    with self.assertRaisesRegex(ValueError, "direct-mapped.*LRU"):
                        recommend_cache_configuration(
                            {"workload_type": "cnn"},
                            Path(temp) / "model.joblib",
                            candidates=[CacheConfig(8192, 1, policy, 64)],
                        )

            with patch("python.cache_optimizer.joblib.load", return_value=bundle):
                with self.assertRaisesRegex(ValueError, "must be a CacheConfig"):
                    recommend_cache_configuration(
                        {"workload_type": "cnn"},
                        Path(temp) / "model.joblib",
                        candidates=[object()],
                    )

    def test_optimizer_candidates_cover_supported_geometry_and_policies(self) -> None:
        candidates = generate_candidates()
        self.assertEqual(len(candidates), 84)
        self.assertEqual(
            {candidate.cache_size_bytes for candidate in candidates},
            {4096, 8192, 16384, 32768},
        )
        self.assertEqual({candidate.associativity for candidate in candidates}, {1, 2, 4})
        self.assertEqual({candidate.block_size_bytes for candidate in candidates}, {32, 64, 128})
        self.assertEqual(
            {candidate.replacement_policy for candidate in candidates},
            {"LRU", "FIFO", "RANDOM"},
        )
        self.assertIn(CacheConfig(8192, 4, "RANDOM", 32), candidates)
        self.assertNotIn(CacheConfig(8192, 1, "FIFO", 64), candidates)

    def test_training_splits_by_trace_and_exports_recommendation(self) -> None:
        rows = []
        for trace_index in range(10):
            for cache_size, associativity in ((4096, 1), (8192, 2), (16384, 4)):
                miss_rate = min(0.9, 0.7 - trace_index * 0.01 - associativity * 0.02)
                access_count = 1000 + trace_index
                miss_count = round(miss_rate * access_count)
                hit_count = access_count - miss_count
                actual_hit_rate = hit_count / access_count
                actual_miss_rate = miss_count / access_count
                rows.append(
                    {
                        "trace_id": f"trace-{trace_index}",
                        "workload_type": ("cnn", "transformer", "matmul")[trace_index % 3],
                        "cache_size_bytes": cache_size,
                        "associativity": associativity,
                        "replacement_policy": "LRU",
                        "block_size_bytes": 64,
                        "read_write_ratio": 4.0,
                        "num_accesses": access_count,
                        "temporal_locality": 0.3 + trace_index / 100,
                        "spatial_locality": 0.6,
                        "hit_rate": actual_hit_rate,
                        "miss_rate": actual_miss_rate,
                        "estimated_amat": 1.0 + actual_miss_rate * 20,
                        "hit_count": hit_count,
                        "miss_count": miss_count,
                        "access_count": access_count,
                        "expected_access_count": access_count,
                        "actual_access_count": access_count,
                        "measurement_source": "vivado_xsim",
                        "synthetic": False,
                        "validated": True,
                    }
                )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_path = root / "data.csv"
            model_path = root / "model.joblib"
            pd.DataFrame(rows).to_csv(data_path, index=False)
            report = train_model(
                data_path,
                model_path,
                root / "training.json",
                allow_unplanned_dataset=True,
            )
            self.assertEqual(report["status"], "evaluated")
            self.assertEqual(report["targets"], ["hit_rate"])
            train_ids = set(report["training_trace_ids"])
            test_ids = set(report["test_trace_ids"])
            self.assertTrue(train_ids)
            self.assertTrue(test_ids)
            self.assertTrue(train_ids.isdisjoint(test_ids))
            self.assertEqual(report["training_distinct_trace_count"], len(train_ids))
            self.assertEqual(report["test_distinct_trace_count"], len(test_ids))
            self.assertGreaterEqual(report["test_distinct_trace_count"], 2)
            evaluation = evaluate_model(data_path, model_path, root / "evaluation.json")
            self.assertEqual(set(evaluation["evaluated_trace_ids"]), test_ids)
            self.assertIn("derived_miss_rate", evaluation["metrics"])
            self.assertEqual(evaluation["training_sample_count"], report["training_record_count"])
            self.assertEqual(evaluation["testing_sample_count"], report["test_record_count"])
            self.assertEqual(evaluation["training_trace_group_count"], len(train_ids))
            self.assertEqual(evaluation["testing_trace_group_count"], len(test_ids))
            self.assertTrue(evaluation["train_test_trace_ids_disjoint"])

            workload = {
                "workload_type": "cnn",
                "cache_size_bytes": 4096,
                "associativity": 1,
                "replacement_policy": "LRU",
                "block_size_bytes": 64,
                "read_write_ratio": 4.0,
                "num_accesses": 1000,
                "temporal_locality": 0.5,
                "spatial_locality": 0.6,
            }
            result = recommend_cache_configuration(
                workload,
                model_path,
                root / "recommendation.json",
                max_cache_size_bytes=8192,
                max_associativity=2,
            )
            self.assertEqual(result["objective"], "hit_rate")
            self.assertTrue(all("predicted_miss_rate" in row for row in result["ranking"]))
            self.assertTrue((root / "recommendation.cfg").is_file())
            self.assertLessEqual(result["recommendation"]["cache_size_bytes"], 8192)
            self.assertLessEqual(result["recommendation"]["associativity"], 2)

    def test_repeated_trace_configuration_rows_have_one_training_weight(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "trace_id": "trace-a",
                    "cache_size_bytes": 4096,
                    "associativity": 2,
                    "block_size_bytes": 64,
                    "replacement_policy": "LRU",
                    "hit_count": 8,
                    "miss_count": 2,
                    "access_count": 10,
                    "expected_access_count": 10,
                    "actual_access_count": 10,
                    "hit_rate": 0.8,
                    "miss_rate": 0.2,
                },
                {
                    "trace_id": "trace-a",
                    "cache_size_bytes": 4096,
                    "associativity": 2,
                    "block_size_bytes": 64,
                    "replacement_policy": "LRU",
                    "hit_count": 12,
                    "miss_count": 8,
                    "access_count": 20,
                    "expected_access_count": 20,
                    "actual_access_count": 20,
                    "hit_rate": 0.6,
                    "miss_rate": 0.4,
                },
            ]
        )
        training_rows = _aggregate_repeated_experiments(rows)

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(training_rows), 1)
        self.assertAlmostEqual(training_rows.iloc[0]["hit_rate"], 20 / 30)
        self.assertAlmostEqual(training_rows.iloc[0]["miss_rate"], 10 / 30)
        self.assertEqual(training_rows.iloc[0]["expected_access_count"], 30)
        self.assertEqual(training_rows.iloc[0]["actual_access_count"], 30)

    def test_training_target_and_holdout_group_requirements(self) -> None:
        self.assertEqual(TARGET_COLUMNS, ["hit_rate"])
        self.assertEqual(_evaluation_status(5, 1), "insufficient_data")
        self.assertEqual(_evaluation_status(5, 2), "evaluated")

    def test_training_dataset_requires_valid_provenance_and_plan_id(self) -> None:
        missing_source = _valid_training_row()
        missing_source.pop("measurement_source")
        with self.assertRaisesRegex(ValueError, "missing required"):
            _validate_training_dataset(pd.DataFrame([missing_source]))

        invalid_source = _valid_training_row()
        invalid_source["measurement_source"] = "test_fixture"
        with self.assertRaisesRegex(ValueError, "vivado_xsim"):
            _validate_training_dataset(pd.DataFrame([invalid_source]))

        missing_plan = _valid_training_row()
        missing_plan.pop("plan_id")
        with self.assertRaisesRegex(ValueError, "missing required"):
            _validate_training_dataset(pd.DataFrame([missing_plan]))

        wrong_plan = _valid_training_row("wrong-plan")
        with self.assertRaisesRegex(ValueError, "expected plan_id"):
            _validate_training_dataset(pd.DataFrame([wrong_plan]))

    def test_training_dataset_rejects_synthetic_and_unvalidated_records(self) -> None:
        for column, value, message in (
            ("synthetic", True, "synthetic"),
            ("validated", False, "validated=True"),
        ):
            row = _valid_training_row()
            row[column] = value
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_training_dataset(pd.DataFrame([row]))

    def test_training_dataset_rejects_invalid_hit_rate_targets(self) -> None:
        for value in (-0.01, 1.01, float("nan")):
            row = _valid_training_row()
            row["hit_rate"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "hit_rate values"):
                    _validate_training_dataset(pd.DataFrame([row]))

    def test_training_dataset_rejects_counter_and_rate_inconsistency(self) -> None:
        invalid_counters = _valid_training_row()
        invalid_counters["hit_count"] = 74
        with self.assertRaisesRegex(ValueError, "inconsistent hit/miss counters"):
            _validate_training_dataset(pd.DataFrame([invalid_counters]))

        invalid_rates = _valid_training_row()
        invalid_rates["hit_rate"] = 0.7
        invalid_rates["miss_rate"] = 0.3
        with self.assertRaisesRegex(ValueError, "do not agree with the validated counters"):
            _validate_training_dataset(pd.DataFrame([invalid_rates]))

    def test_scores_calculate_mae_rmse_and_r2(self) -> None:
        scores = _scores(pd.Series([0.0, 1.0]), np.asarray([0.0, 0.5]))
        self.assertAlmostEqual(scores["mae"], 0.25)
        self.assertAlmostEqual(scores["rmse"], np.sqrt(0.125))
        self.assertAlmostEqual(scores["r2"], 0.5)

    def test_clean_plan_dataset_split_is_reproducible_without_fitting(self) -> None:
        dataset_path = DATASET_DIR / "cache_dataset_plan_396.csv"
        frame = _validate_training_dataset(pd.read_csv(dataset_path))
        scenarios = _aggregate_repeated_experiments(frame)
        _validate_plan_dataset_scenarios(scenarios)

        first_train, first_test = _split_by_trace(scenarios, 0.2)
        second_train, second_test = _split_by_trace(scenarios, 0.2)
        first_train_ids = set(scenarios.iloc[first_train]["trace_id"].astype(str))
        first_test_ids = set(scenarios.iloc[first_test]["trace_id"].astype(str))
        second_train_ids = set(scenarios.iloc[second_train]["trace_id"].astype(str))
        second_test_ids = set(scenarios.iloc[second_test]["trace_id"].astype(str))

        self.assertEqual(first_train_ids, second_train_ids)
        self.assertEqual(first_test_ids, second_test_ids)
        self.assertTrue(first_train_ids.isdisjoint(first_test_ids))
        self.assertEqual((len(first_train), len(first_test)), (315, 81))
        self.assertEqual((len(first_train_ids), len(first_test_ids)), (12, 3))

    def test_workload_stratified_folds_are_balanced_separated_and_reproducible(self) -> None:
        dataset_path = DATASET_DIR / "cache_dataset_plan_396.csv"
        frame = _validate_training_dataset(pd.read_csv(dataset_path))
        scenarios = _aggregate_repeated_experiments(frame)
        _validate_plan_dataset_scenarios(scenarios)

        first = _workload_stratified_folds(scenarios, seed=42)
        second = _workload_stratified_folds(scenarios, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

        all_trace_ids = set(scenarios["trace_id"].astype(str))
        tested_trace_ids: list[str] = []
        for fold in first:
            test_trace_ids = set(fold)
            train_trace_ids = all_trace_ids - test_trace_ids
            self.assertEqual(len(test_trace_ids), 3)
            self.assertEqual(len(train_trace_ids), 12)
            self.assertTrue(train_trace_ids.isdisjoint(test_trace_ids))
            workload_counts = (
                scenarios.loc[
                    scenarios["trace_id"].astype(str).isin(test_trace_ids),
                    ["trace_id", "workload_type"],
                ]
                .drop_duplicates()["workload_type"]
                .value_counts()
                .to_dict()
            )
            self.assertEqual(
                workload_counts, {"cnn": 1, "transformer": 1, "matmul": 1}
            )
            tested_trace_ids.extend(fold)

        self.assertEqual(len(tested_trace_ids), 15)
        self.assertEqual(set(tested_trace_ids), all_trace_ids)
        self.assertEqual(len(tested_trace_ids), len(set(tested_trace_ids)))

    def test_cross_validation_generates_complete_oof_predictions_and_metrics(self) -> None:
        dataset_path = DATASET_DIR / "cache_dataset_plan_396.csv"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = root / "cv_report.json"
            predictions_path = root / "oof_predictions.csv"
            report = cross_validate_model(
                dataset_path,
                report_path=report_path,
                predictions_path=predictions_path,
                seed=42,
            )
            predictions = pd.read_csv(predictions_path)

            self.assertEqual(report["plan_id"], EXPECTED_PLAN_ID)
            self.assertEqual(report["scenario_count"], 396)
            self.assertEqual(report["out_of_fold_prediction_count"], 396)
            self.assertTrue(report["out_of_fold_predictions_complete"])
            self.assertTrue(report["all_traces_tested_exactly_once"])
            self.assertTrue(report["all_folds_train_test_trace_ids_disjoint"])
            self.assertEqual(len(predictions), 396)
            self.assertEqual(
                predictions[
                    ["trace_id", "cache_size_bytes", "associativity",
                     "block_size_bytes", "replacement_policy"]
                ].drop_duplicates().shape[0],
                396,
            )
            self.assertTrue(
                np.isfinite(predictions["predicted_hit_rate"].to_numpy()).all()
            )
            self.assertTrue(predictions["predicted_hit_rate"].between(0, 1).all())

            tested_trace_ids: list[str] = []
            for fold in report["folds"]:
                train_ids = set(fold["training_trace_ids"])
                test_ids = set(fold["testing_trace_ids"])
                self.assertEqual(len(train_ids), 12)
                self.assertEqual(len(test_ids), 3)
                self.assertTrue(train_ids.isdisjoint(test_ids))
                self.assertTrue(fold["train_test_trace_ids_disjoint"])
                self.assertEqual(fold["training_row_count"] + fold["testing_row_count"], 396)
                tested_trace_ids.extend(test_ids)
            self.assertEqual(len(tested_trace_ids), 15)
            self.assertEqual(len(tested_trace_ids), len(set(tested_trace_ids)))

            self.assertEqual(
                {
                    workload: metrics["trace_count"]
                    for workload, metrics in report["per_workload_metrics"].items()
                },
                {"cnn": 5, "matmul": 5, "transformer": 5},
            )
            self.assertEqual(
                {
                    workload: metrics["scenario_count"]
                    for workload, metrics in report["per_workload_metrics"].items()
                },
                {"cnn": 132, "matmul": 132, "transformer": 132},
            )
            self.assertEqual(len(report["per_trace_metrics"]), 15)

            actual = predictions["actual_hit_rate"].to_numpy()
            predicted = predictions["predicted_hit_rate"].to_numpy()
            self.assertAlmostEqual(
                report["pooled_metrics"]["mae"], float(np.mean(np.abs(actual - predicted)))
            )
            self.assertAlmostEqual(
                report["pooled_metrics"]["rmse"],
                float(np.sqrt(np.mean(np.square(actual - predicted)))),
            )
            self.assertAlmostEqual(
                report["pooled_metrics"]["r2"],
                float(1 - np.sum(np.square(actual - predicted))
                      / np.sum(np.square(actual - np.mean(actual)))),
            )
            self.assertTrue(report_path.is_file())

    def test_cross_validation_cannot_overwrite_dataset_or_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                cross_validate_model(
                    DATASET_DIR / "cache_dataset_plan_396.csv",
                    report_path=DATASET_DIR / "cache_dataset_plan_396.csv",
                    predictions_path=Path(temp) / "predictions.csv",
                )
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                cross_validate_model(
                    DATASET_DIR / "cache_dataset_plan_396.csv",
                    report_path=Path(temp) / "report.json",
                    predictions_path=Path("training") / "training_metrics.json",
                )

    def test_evaluation_requires_two_held_out_trace_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_path = root / "dataset.csv"
            dataset_path.write_text("trace_id\ntrace-a\n", encoding="utf-8")
            model_path = root / "model.joblib"
            joblib.dump({"test_trace_ids": ["trace-a"]}, model_path)

            with self.assertRaisesRegex(ValueError, "at least two held-out trace IDs"):
                evaluate_model(dataset_path, model_path)

    def test_evaluation_rejects_overlapping_trace_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_path = root / "dataset.csv"
            pd.DataFrame({"trace_id": ["trace-a", "trace-b"]}).to_csv(
                dataset_path, index=False
            )
            bundle = {
                "training_trace_ids": ["trace-a"],
                "test_trace_ids": ["trace-a", "trace-b"],
                "training_report": {},
            }
            with patch("python.evaluate_model.joblib.load", return_value=bundle):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    evaluate_model(dataset_path, root / "model.joblib")

    def test_optimizer_rejects_insufficient_training_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model_path = Path(temp) / "insufficient.joblib"
            joblib.dump(
                {"training_report": {"status": "insufficient_data"}, "models": {}},
                model_path,
            )
            with self.assertRaisesRegex(ValueError, "training status 'evaluated'"):
                recommend_cache_configuration({}, model_path, Path(temp) / "rec.json")


if __name__ == "__main__":
    unittest.main()
