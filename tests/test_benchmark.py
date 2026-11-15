from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from python.benchmark import (
    BASELINE_CONFIG,
    _validate_measurement,
    run_benchmark,
)
from python.cache_optimizer import generate_candidates
from python.config import CacheConfig, DATASET_DIR
from python.feature_engineering import extract_workload_features, trace_identity
from python.trace_exporter import export_trace


RECOMMENDED_CONFIG = CacheConfig(8192, 2, "LRU", 64)


def _config_key(config: CacheConfig) -> tuple[int, int, int, str]:
    return (
        config.cache_size_bytes,
        config.associativity,
        config.block_size_bytes,
        config.replacement_policy,
    )


class BenchmarkTests(unittest.TestCase):
    def _create_trace(self, root: Path) -> tuple[Path, Path, dict]:
        frame = pd.DataFrame(
            {
                "address": [0, 4, 8, 4],
                "operation": ["R", "W", "R", "R"],
                "workload_type": ["cnn"] * 4,
                "sequence": [0, 1, 2, 3],
            }
        )
        csv_path, mem_path = export_trace(
            frame,
            csv_path=root / "workload.csv",
            memory_path=root / "workload.mem",
        )
        csv_frame = pd.read_csv(csv_path)
        features = extract_workload_features(csv_frame)
        features["trace_id"] = trace_identity(csv_frame)
        return csv_path, mem_path, features

    def _simulation_result(
        self,
        config: CacheConfig,
        overrides: dict | None = None,
    ) -> dict:
        hit_count = 2 if config == BASELINE_CONFIG else 3
        miss_count = 4 - hit_count
        result = {
            "run_id": f"run-{config.associativity}",
            "status": "success",
            "synthetic": False,
            "validated": True,
            "measurement_source": "vivado_xsim",
            "expected_access_count": 4,
            "validation": {"passed": True, "actual_access_count": 4},
            "metrics": {
                "access_count": 4,
                "hit_count": hit_count,
                "miss_count": miss_count,
                "hit_rate": hit_count / 4,
                "miss_rate": miss_count / 4,
                "estimated_amat": None,
            },
            "errors": [],
            "result_file": f"simulation_results/{config.associativity}.json",
        }
        if overrides:
            for name, value in overrides.items():
                if name == "metrics":
                    if value is None:
                        result["metrics"] = None
                    else:
                        result["metrics"].update(value)
                elif name == "validation":
                    if value is None:
                        result["validation"] = None
                    else:
                        result["validation"].update(value)
                else:
                    result[name] = value
        return result

    def _run(
        self,
        root: Path,
        recommended_config: CacheConfig = RECOMMENDED_CONFIG,
        result_overrides: dict | None = None,
        candidate_configs: list[CacheConfig] | None = None,
        simulation_exceptions: dict | None = None,
    ):
        csv_path, mem_path, features = self._create_trace(root)

        def fake_recommendation(_features, _model, output_path=None, candidates=None):
            ranking = []
            for config in candidates:
                predicted = 0.85 if config == recommended_config else 0.7
                ranking.append(
                    {
                        **config.to_dict(),
                        "predicted_hit_rate": predicted,
                        "predicted_miss_rate": 1.0 - predicted,
                    }
                )
            ranking.sort(key=lambda row: row["predicted_hit_rate"], reverse=True)
            return {"recommendation": ranking[0], "ranking": ranking}

        def fake_simulation(**kwargs):
            config = kwargs["cache_config"]
            exception = (
                simulation_exceptions.get(_config_key(config))
                if simulation_exceptions
                else None
            )
            if exception is not None:
                raise exception
            overrides = (
                result_overrides.get(_config_key(config), {})
                if result_overrides
                else None
            )
            return self._simulation_result(config, overrides)

        candidate_generator = (
            patch(
                "python.benchmark.generate_candidates",
                return_value=candidate_configs,
            )
            if candidate_configs is not None
            else patch(
                "python.benchmark.generate_candidates",
                wraps=generate_candidates,
            )
        )
        with (
            candidate_generator as candidate_generation_call,
            patch(
                "python.benchmark.recommend_cache_configuration",
                side_effect=fake_recommendation,
            ) as recommendation_call,
            patch(
                "python.benchmark.run_simulation",
                side_effect=fake_simulation,
            ) as simulation_call,
        ):
            frame = run_benchmark(
                mem_path,
                features,
                rtl_dir=root / "rtl",
                model_path=root / "model.joblib",
                result_dir=root / "simulation_results",
                output_path=root / "metrics" / "benchmark.csv",
                workload_csv_path=csv_path,
            )
        return (
            frame,
            simulation_call,
            recommendation_call,
            csv_path,
            mem_path,
            candidate_generation_call,
        )

    def test_baseline_and_recommendation_are_identified_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame, simulation_call, recommendation_call, _, _, generator_call = (
                self._run(root)
            )
            self.assertEqual(simulation_call.call_count, 2)
            self.assertEqual(generator_call.call_count, 1)
            self.assertGreater(
                len(recommendation_call.call_args.kwargs["candidates"]), 2
            )
            simulated_keys = {
                _config_key(call.kwargs["cache_config"])
                for call in simulation_call.call_args_list
            }
            self.assertEqual(
                simulated_keys,
                {_config_key(BASELINE_CONFIG), _config_key(RECOMMENDED_CONFIG)},
            )
            self.assertEqual(
                len(
                    {
                        (
                            call.kwargs["trace_file"],
                            call.kwargs["trace_id"],
                        )
                        for call in simulation_call.call_args_list
                    }
                ),
                1,
            )
            self.assertEqual(int(frame["is_baseline"].sum()), 1)
            self.assertEqual(int(frame["is_ml_recommended"].sum()), 1)
            baseline = frame.loc[frame["is_baseline"]].iloc[0]
            recommended = frame.loc[frame["is_ml_recommended"]].iloc[0]
            self.assertEqual(
                (
                    baseline["cache_size_bytes"],
                    baseline["associativity"],
                    baseline["block_size_bytes"],
                    baseline["replacement_policy"],
                ),
                (8192, 1, 64, "LRU"),
            )
            self.assertEqual(recommended["predicted_hit_rate"], 0.85)
            self.assertEqual(recommended["actual_hit_rate"], 0.75)
            self.assertEqual(recommended["actual_miss_rate"], 0.25)
            self.assertEqual(recommended["access_count"], 4)
            self.assertEqual(recommended["hit_count"], 3)
            self.assertEqual(recommended["miss_count"], 1)
            self.assertTrue(recommended["validated"])
            self.assertEqual(recommended["measurement_source"], "vivado_xsim")
            self.assertTrue(recommended["measurement_valid"])
            self.assertTrue(recommended["trace_hash"])

    def test_recommendation_equal_to_baseline_runs_once_and_marks_both(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            frame, simulation_call, recommendation_call, _, _, generator_call = self._run(
                Path(temp),
                recommended_config=BASELINE_CONFIG,
            )
            self.assertEqual(generator_call.call_count, 1)
            self.assertGreater(
                len(recommendation_call.call_args.kwargs["candidates"]), 2
            )
            self.assertEqual(simulation_call.call_count, 1)
            self.assertEqual(len(frame), 1)
            self.assertTrue(frame.iloc[0]["is_baseline"])
            self.assertTrue(frame.iloc[0]["is_ml_recommended"])
            comparison = json.loads(
                Path(frame.attrs["comparison_report_path"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(comparison["comparison_valid"])
            self.assertEqual(comparison["absolute_hit_rate_improvement"], 0.0)
            self.assertFalse(comparison["improvement_claimed"])

    def test_only_top_ranked_configuration_is_simulated_with_the_baseline(self) -> None:
        selected_config = CacheConfig(16384, 2, "FIFO", 64)
        with tempfile.TemporaryDirectory() as temp:
            frame, simulation_call, recommendation_call, _, _, _ = self._run(
                Path(temp),
                recommended_config=selected_config,
            )
            candidates = recommendation_call.call_args.kwargs["candidates"]
            self.assertGreater(len(candidates), 2)
            self.assertIn(selected_config, candidates)
            self.assertEqual(simulation_call.call_count, 2)
            self.assertEqual(
                {
                    _config_key(call.kwargs["cache_config"])
                    for call in simulation_call.call_args_list
                },
                {_config_key(BASELINE_CONFIG), _config_key(selected_config)},
            )
            selected = frame.loc[frame["is_ml_recommended"]].iloc[0]
            self.assertEqual(
                _config_key(
                    CacheConfig(
                        selected["cache_size_bytes"],
                        selected["associativity"],
                        selected["replacement_policy"],
                        selected["block_size_bytes"],
                    )
                ),
                _config_key(selected_config),
            )

    def test_expected_and_actual_access_counts_require_strict_integer_types(self) -> None:
        valid_result = self._simulation_result(BASELINE_CONFIG)
        self.assertTrue(_validate_measurement(valid_result, 4)[0])

        for invalid_count in (4.0, True):
            with self.subTest(field="result expected_access_count", value=invalid_count):
                result = self._simulation_result(BASELINE_CONFIG)
                result["expected_access_count"] = invalid_count
                valid, errors, _ = _validate_measurement(result, 4)
                self.assertFalse(valid)
                self.assertTrue(
                    any("result expected_access_count" in error for error in errors)
                )

            with self.subTest(field="validation actual_access_count", value=invalid_count):
                result = self._simulation_result(BASELINE_CONFIG)
                result["validation"]["actual_access_count"] = invalid_count
                valid, errors, _ = _validate_measurement(result, 4)
                self.assertFalse(valid)
                self.assertTrue(
                    any("actual_access_count is not a non-negative integer" in error
                        for error in errors)
                )

            with self.subTest(field="input expected_access_count", value=invalid_count):
                result = self._simulation_result(BASELINE_CONFIG)
                valid, errors, _ = _validate_measurement(result, invalid_count)
                self.assertFalse(valid)
                self.assertTrue(
                    any("input expected_access_count" in error for error in errors)
                )

    def test_csv_mem_mismatch_is_rejected_before_recommendation_or_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path, mem_path, features = self._create_trace(root)
            lines = mem_path.read_text(encoding="ascii").splitlines()
            lines[1] = "0 0000000C"
            mem_path.write_text("\n".join(lines) + "\n", encoding="ascii")
            with (
                patch("python.benchmark.recommend_cache_configuration") as recommend,
                patch("python.benchmark.run_simulation") as simulate,
            ):
                with self.assertRaisesRegex(ValueError, "differ at access 1"):
                    run_benchmark(
                        mem_path,
                        features,
                        rtl_dir=root / "rtl",
                        model_path=root / "model.joblib",
                        result_dir=root / "results",
                        workload_csv_path=csv_path,
                    )
            recommend.assert_not_called()
            simulate.assert_not_called()

    def test_invalid_counters_and_measurements_do_not_claim_improvement(self) -> None:
        invalid_results = (
            {"metrics": {"miss_count": 0}},
            {"metrics": {"access_count": 5}},
            {"metrics": {"hit_rate": 0.1}},
            {"metrics": None},
            {"validation": None},
            {"status": "failed"},
            {"synthetic": True},
            {"validated": False},
            {"measurement_source": "synthetic_dry_run"},
        )
        with tempfile.TemporaryDirectory() as temp:
            for index, override in enumerate(invalid_results):
                with self.subTest(override=override):
                    root = Path(temp) / str(index)
                    root.mkdir()
                    invalid = {_config_key(RECOMMENDED_CONFIG): override}
                    frame, _, _, _, _, _ = self._run(root, result_overrides=invalid)
                    recommended = frame.loc[frame["is_ml_recommended"]].iloc[0]
                    self.assertFalse(recommended["measurement_valid"])
                    comparison = json.loads(
                        Path(frame.attrs["comparison_report_path"]).read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(comparison["status"], "unavailable")
                    self.assertIsNone(comparison["absolute_hit_rate_improvement"])
                    self.assertFalse(comparison["improvement_claimed"])

    def test_simulator_exceptions_are_recorded_and_other_run_continues(self) -> None:
        for failed_config in (BASELINE_CONFIG, RECOMMENDED_CONFIG):
            with self.subTest(configuration=_config_key(failed_config)):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    frame, simulation_call, _, _, _, _ = self._run(
                        root,
                        simulation_exceptions={
                            _config_key(failed_config): RuntimeError(
                                "simulator process failed"
                            )
                        },
                    )
                    self.assertEqual(simulation_call.call_count, 2)
                    failed = frame.loc[
                        frame.apply(
                            lambda row: _config_key(
                                CacheConfig(
                                    row["cache_size_bytes"],
                                    row["associativity"],
                                    row["replacement_policy"],
                                    row["block_size_bytes"],
                                )
                            )
                            == _config_key(failed_config),
                            axis=1,
                        )
                    ].iloc[0]
                    self.assertEqual(failed["status"], "exception")
                    self.assertIn("RuntimeError: simulator process failed", failed["errors"])
                    self.assertFalse(failed["measurement_valid"])
                    self.assertTrue(failed["is_baseline"] or failed["is_ml_recommended"])

                    records_path = Path(frame.attrs["records_report_path"])
                    comparison_path = Path(frame.attrs["comparison_report_path"])
                    self.assertTrue(records_path.is_file())
                    self.assertTrue(comparison_path.is_file())
                    records = json.loads(records_path.read_text(encoding="utf-8"))
                    self.assertEqual(records["status"], "partial")
                    self.assertTrue(
                        any(
                            record["status"] == "exception"
                            and "simulator process failed" in record["errors"]
                            for record in records["records"]
                        )
                    )
                    comparison = json.loads(
                        comparison_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(comparison["status"], "unavailable")
                    self.assertFalse(comparison["improvement_claimed"])
                    self.assertIsNone(comparison["absolute_hit_rate_improvement"])
                    self.assertEqual(
                        int(frame["measurement_valid"].sum()),
                        1,
                    )

    def test_comparison_improvements_are_calculated_from_valid_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame, _, _, _, _, _ = self._run(root)
            comparison = json.loads(
                Path(frame.attrs["comparison_report_path"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(comparison["status"], "complete")
            self.assertEqual(comparison["baseline_actual_hit_rate"], 0.5)
            self.assertEqual(comparison["recommended_actual_hit_rate"], 0.75)
            self.assertEqual(comparison["predicted_recommended_hit_rate"], 0.85)
            self.assertEqual(comparison["absolute_hit_rate_improvement"], 0.25)
            self.assertEqual(
                comparison["relative_hit_rate_improvement_percentage"], 50.0
            )
            self.assertEqual(comparison["miss_rate_reduction"], 0.25)
            self.assertEqual(
                comparison["relative_miss_rate_reduction_percentage"], 50.0
            )
            self.assertTrue(comparison["improvement_claimed"])

    def test_outputs_are_unique_and_protected_datasets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, _, _, _, _, _ = self._run(root)
            first_csv = Path(first.attrs["csv_path"])
            first_contents = first_csv.read_bytes()
            second, _, _, _, _, _ = self._run(root)
            self.assertNotEqual(first.attrs["csv_path"], second.attrs["csv_path"])
            self.assertNotEqual(
                first.attrs["simulation_results_dir"],
                second.attrs["simulation_results_dir"],
            )
            self.assertEqual(first_csv.read_bytes(), first_contents)
            self.assertIn("_", first_csv.stem)

            csv_path, mem_path, features = self._create_trace(root / "protected")
            with patch("python.benchmark.run_simulation") as simulate:
                with self.assertRaisesRegex(ValueError, "separate output paths"):
                    run_benchmark(
                        mem_path,
                        features,
                        rtl_dir=root / "rtl",
                        model_path=root / "model.joblib",
                        result_dir=root / "separate_results",
                        output_path=root / "metrics" / "collision.csv",
                        comparison_path=root / "metrics" / "collision.json",
                        workload_csv_path=csv_path,
                    )
                simulate.assert_not_called()
                with self.assertRaisesRegex(ValueError, "protected model or dataset"):
                    run_benchmark(
                        mem_path,
                        features,
                        rtl_dir=root / "rtl",
                        model_path=root / "model.joblib",
                        result_dir=root / "safe_results",
                        output_path=DATASET_DIR / "cache_dataset_plan_396.csv",
                        workload_csv_path=csv_path,
                    )
            simulate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
