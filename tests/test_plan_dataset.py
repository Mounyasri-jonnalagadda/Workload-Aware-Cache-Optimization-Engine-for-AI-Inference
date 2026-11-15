from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from python.dataset_builder import (
    _plan_record_rejection_reason,
    _pool_count_metrics,
    build_plan_dataset,
)
from python.experiment_plan import create_experiment_plan
from python.train_model import FEATURE_COLUMNS


def _result_fixture(
    experiment: dict,
    run_id: str,
    hits: int = 6000,
    plan_id: str | None = None,
) -> dict:
    accesses = 10_000
    result = {
        "run_id": run_id,
        "status": "success",
        "synthetic": False,
        "measurement_source": "vivado_xsim",
        "validated": True,
        "validation": {"passed": True},
        "expected_access_count": accesses,
        "trace_id": experiment["trace_id"],
        "workload_features": {
            "workload_type": experiment["workload_type"],
            "seed": experiment["seed"],
            "num_accesses": accesses,
            "read_write_ratio": 4.0,
            "temporal_locality": 0.5,
            "spatial_locality": 0.75,
        },
        "cache_config": experiment["cache_config"],
        "metrics": {
            "access_count": accesses,
            "hit_count": hits,
            "miss_count": accesses - hits,
            "hit_rate": hits / accesses,
            "miss_rate": (accesses - hits) / accesses,
            "estimated_amat": None,
        },
    }
    if plan_id is not None:
        result["plan_id"] = plan_id
    return result


class PlanDatasetTests(unittest.TestCase):
    def test_count_pooling_recomputes_all_counters_and_rates(self) -> None:
        pooled = _pool_count_metrics(
            pd.DataFrame(
                [
                    {
                        "hit_count": 8,
                        "miss_count": 2,
                        "access_count": 10,
                        "expected_access_count": 10,
                        "actual_access_count": 10,
                    },
                    {
                        "hit_count": 15,
                        "miss_count": 5,
                        "access_count": 20,
                        "expected_access_count": 20,
                        "actual_access_count": 20,
                    },
                ]
            )
        )
        self.assertEqual(pooled["hit_count"], 23)
        self.assertEqual(pooled["miss_count"], 7)
        self.assertEqual(pooled["access_count"], 30)
        self.assertEqual(pooled["expected_access_count"], 30)
        self.assertEqual(pooled["actual_access_count"], 30)
        self.assertAlmostEqual(pooled["hit_rate"], 23 / 30)
        self.assertAlmostEqual(pooled["miss_rate"], 7 / 30)

    def test_plan_dataset_filters_pools_and_records_provenance(self) -> None:
        plan = create_experiment_plan(accesses=10_000)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_path = root / "plan.json"
            results_dir = root / "simulation_results"
            results_dir.mkdir()
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            for index, experiment in enumerate(plan["experiments"]):
                result = _result_fixture(
                    experiment,
                    f"run-{index}",
                    plan_id=plan["plan_id"] if index == 0 else None,
                )
                (results_dir / f"result-{index}.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )

            first_experiment = plan["experiments"][0]
            repeated = _result_fixture(first_experiment, "repeat-run", hits=5000)
            (results_dir / "repeat.json").write_text(
                json.dumps(repeated), encoding="utf-8"
            )
            foreign_plan = _result_fixture(
                first_experiment, "foreign-plan-run", plan_id="another-plan"
            )
            (results_dir / "foreign-plan.json").write_text(
                json.dumps(foreign_plan), encoding="utf-8"
            )
            synthetic = _result_fixture(first_experiment, "synthetic-run")
            synthetic["synthetic"] = True
            (results_dir / "synthetic.json").write_text(
                json.dumps(synthetic), encoding="utf-8"
            )

            output_path = root / "cache_dataset_plan_396.csv"
            quality_path = root / "cache_dataset_plan_396.quality.json"
            frame = build_plan_dataset(
                plan_path,
                results_dir,
                output_path,
                quality_path,
            )

            self.assertEqual(len(frame), 396)
            self.assertEqual(frame["trace_id"].nunique(), 15)
            self.assertEqual(
                frame.groupby("workload_type").size().to_dict(),
                {"cnn": 132, "matmul": 132, "transformer": 132},
            )
            self.assertEqual(
                frame[
                    [
                        "cache_size_bytes",
                        "associativity",
                        "block_size_bytes",
                        "replacement_policy",
                    ]
                ]
                .drop_duplicates()
                .shape[0],
                84,
            )
            self.assertFalse(frame.duplicated(["trace_id", "cache_size_bytes", "associativity", "block_size_bytes", "replacement_policy"]).any())

            pooled = frame.iloc[0]
            self.assertEqual(pooled["source_run_count"], 2)
            self.assertEqual(pooled["access_count"], 20_000)
            self.assertEqual(pooled["expected_access_count"], 20_000)
            self.assertEqual(pooled["actual_access_count"], 20_000)
            self.assertEqual(pooled["hit_count"], 11_000)
            self.assertEqual(pooled["miss_count"], 9_000)
            self.assertAlmostEqual(pooled["hit_rate"], 0.55)
            self.assertAlmostEqual(pooled["miss_rate"], 0.45)
            self.assertEqual(pooled["plan_id_provenance"], "mixed")
            self.assertEqual(pooled["plan_id"], plan["plan_id"])
            self.assertEqual(frame["num_accesses"].unique().tolist(), [10_000])
            self.assertEqual(set(frame["measurement_source"]), {"vivado_xsim"})
            self.assertTrue(frame["validated"].all())
            self.assertFalse(frame["synthetic"].any())
            self.assertNotIn("source_run_count", FEATURE_COLUMNS)
            self.assertNotIn("plan_id", FEATURE_COLUMNS)

            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            self.assertEqual(quality["plan_id"], plan["plan_id"])
            self.assertEqual(quality["prepared_scenario_count"], 396)
            self.assertEqual(quality["accepted_source_run_count"], 397)
            self.assertEqual(quality["excluded_by_reason"]["different_plan_id"], 1)
            self.assertEqual(quality["excluded_by_reason"]["synthetic_or_unmarked"], 1)
            self.assertEqual(quality["plan_id_provenance"]["explicit_plan_id_runs"], 1)
            self.assertEqual(quality["plan_id_provenance"]["legacy_exact_key_match_runs"], 396)
            self.assertTrue(output_path.is_file())

    def test_invalid_source_record_fails_plan_validation(self) -> None:
        record = _result_fixture(
            {
                "trace_id": "trace",
                "workload_type": "cnn",
                "seed": 42,
                "cache_config": {},
            },
            "invalid",
        )
        record["metrics"]["miss_rate"] = 0.3
        self.assertEqual(
            _plan_record_rejection_reason(record, "cnn", 42, 10_000),
            "inconsistent_rates",
        )

    def test_plan_dataset_refuses_original_dataset_paths(self) -> None:
        plan = create_experiment_plan(accesses=10_000)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            original_dataset = root / "cache_dataset.csv"
            original_report = root / "cache_dataset.quality.json"
            original_dataset.write_text("original", encoding="utf-8")
            original_report.write_text("original report", encoding="utf-8")
            with patch("python.dataset_builder.DATASET_DIR", root):
                with self.assertRaisesRegex(ValueError, "must not overwrite"):
                    build_plan_dataset(
                        plan_path,
                        root / "missing-results",
                        original_dataset,
                        root / "new-quality.json",
                    )
                with self.assertRaisesRegex(ValueError, "must not overwrite"):
                    build_plan_dataset(
                        plan_path,
                        root / "missing-results",
                        root / "new-dataset.csv",
                        original_report,
                    )
            self.assertEqual(original_dataset.read_text(encoding="utf-8"), "original")
            self.assertEqual(original_report.read_text(encoding="utf-8"), "original report")


if __name__ == "__main__":
    unittest.main()
