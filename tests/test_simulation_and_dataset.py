from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from python.config import CacheConfig, load_cache_config
from python.dataset_builder import build_dataset
from python.simulation_parser import parse_simulation_log
from python.vivado_runner import (
    _build_subprocess_command,
    count_trace_records,
    run_simulation,
    validate_simulation_metrics,
)


class SimulationAndDatasetTests(unittest.TestCase):
    def test_parser_extracts_counters_and_rates(self) -> None:
        parsed = parse_simulation_log(
            "ACCESS_COUNT=100\nHIT_COUNT=75\nMISS_COUNT=25\nESTIMATED_AMAT=4.5\n"
        )
        self.assertEqual(parsed["metrics"]["access_count"], 100)
        self.assertEqual(parsed["metrics"]["hit_rate"], 0.75)
        self.assertEqual(parsed["metrics"]["miss_rate"], 0.25)
        self.assertEqual(parsed["metrics"]["estimated_amat"], 4.5)

    def test_dataset_excludes_synthetic_and_incomplete_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            results = Path(temp) / "results"
            results.mkdir()
            base = {
                "status": "success",
                "synthetic": False,
                "measurement_source": "vivado_xsim",
                "validated": True,
                "expected_access_count": 100,
                "validation": {"passed": True},
                "trace_id": "trace-1",
                "workload_features": {
                    "workload_type": "cnn",
                    "read_write_ratio": 4.0,
                    "num_accesses": 100,
                    "temporal_locality": 0.4,
                    "spatial_locality": 0.8,
                },
                "cache_config": {
                    "cache_size_bytes": 4096,
                    "associativity": 1,
                    "replacement_policy": "LRU",
                    "block_size_bytes": 64,
                },
                "metrics": {
                    "access_count": 100,
                    "hit_count": 75,
                    "miss_count": 25,
                    "hit_rate": 0.75,
                    "miss_rate": 0.25,
                },
            }
            (results / "actual.json").write_text(json.dumps(base), encoding="utf-8")
            synthetic = {**base, "synthetic": True, "trace_id": "synthetic"}
            (results / "synthetic.json").write_text(json.dumps(synthetic), encoding="utf-8")
            incomplete = {**base, "status": "failed", "trace_id": "failed"}
            (results / "failed.json").write_text(json.dumps(incomplete), encoding="utf-8")
            invalid = {
                **base,
                "trace_id": "invalid",
                "expected_access_count": 101,
                "validation": {"passed": False},
            }
            (results / "invalid.json").write_text(json.dumps(invalid), encoding="utf-8")
            test_record = {
                **base,
                "trace_id": "test-fixture",
                "measurement_source": "test_fixture",
            }
            (results / "test_fixture.json").write_text(
                json.dumps(test_record), encoding="utf-8"
            )
            data = build_dataset(
                results,
                Path(temp) / "dataset.csv",
                hit_time_cycles=1.0,
                miss_penalty_cycles=10.0,
            )
            self.assertEqual(len(data), 1)
            self.assertEqual(data.iloc[0]["trace_id"], "trace-1")
            self.assertAlmostEqual(data.iloc[0]["estimated_amat"], 3.5)
            self.assertEqual(data.iloc[0]["access_count"], 100)
            self.assertEqual(data.iloc[0]["expected_access_count"], 100)
            self.assertEqual(data.iloc[0]["actual_access_count"], 100)
            self.assertEqual(data.iloc[0]["hit_count"], 75)
            self.assertEqual(data.iloc[0]["miss_count"], 25)

    def test_dataset_preserves_duplicate_measurements_and_reports_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            results.mkdir()
            base = {
                "run_id": "run-1",
                "status": "success",
                "synthetic": False,
                "measurement_source": "vivado_xsim",
                "validated": True,
                "expected_access_count": 10,
                "validation": {"passed": True},
                "trace_id": "trace-a",
                "workload_features": {
                    "workload_type": "cnn",
                    "seed": 42,
                    "read_write_ratio": 4.0,
                    "num_accesses": 10,
                    "temporal_locality": 0.4,
                    "spatial_locality": 0.8,
                },
                "cache_config": {
                    "cache_size_bytes": 4096,
                    "associativity": 2,
                    "replacement_policy": "LRU",
                    "block_size_bytes": 64,
                },
                "metrics": {
                    "access_count": 10,
                    "hit_count": 7,
                    "miss_count": 3,
                    "hit_rate": 0.7,
                    "miss_rate": 0.3,
                },
            }
            (results / "run-1.json").write_text(json.dumps(base), encoding="utf-8")
            repeated = {**base, "run_id": "run-2"}
            (results / "run-2.json").write_text(json.dumps(repeated), encoding="utf-8")
            invalid = {
                **base,
                "run_id": "invalid",
                "expected_access_count": 11,
            }
            (results / "invalid.json").write_text(json.dumps(invalid), encoding="utf-8")

            data = build_dataset(results, root / "dataset.csv")

            self.assertEqual(len(data), 2)
            self.assertEqual(set(data["trace_id"]), {"trace-a"})
            self.assertEqual(set(data["seed"]), {42})
            self.assertTrue(data["validated"].all())
            quality = data.attrs["quality_report"]
            self.assertEqual(quality["included_validated_records"], 2)
            self.assertEqual(quality["excluded_by_reason"]["inconsistent_counters"], 1)
            self.assertEqual(quality["distinct_trace_configuration_count"], 1)
            self.assertEqual(quality["exact_duplicate_extra_rows"], 1)
            self.assertEqual(quality["repeated_experiment_extra_rows"], 1)
            self.assertEqual(quality["count_consistency"]["invalid_rows"], 0)
            self.assertTrue(Path(data.attrs["quality_report_path"]).is_file())

    def test_trace_record_counter_validates_exported_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "trace.mem"
            trace.write_text("0 00000000\n1 00000004\n\n", encoding="ascii")
            self.assertEqual(count_trace_records(trace), 2)

    def test_trace_record_counter_rejects_malformed_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "trace.mem"
            trace.write_text("2 00000000\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "Malformed trace record"):
                count_trace_records(trace)

    def test_successful_counter_validation(self) -> None:
        metrics = {"access_count": 10, "hit_count": 6, "miss_count": 4}
        self.assertEqual(validate_simulation_metrics(metrics, 10), [])

    def test_access_count_mismatch_fails_validation(self) -> None:
        metrics = {"access_count": 9, "hit_count": 6, "miss_count": 3}
        errors = validate_simulation_metrics(metrics, 10)
        self.assertTrue(any("does not match exported trace count" in error for error in errors))

    def test_missing_counters_fail_validation(self) -> None:
        errors = validate_simulation_metrics(
            {"access_count": 10, "hit_count": 6}, 10
        )
        self.assertTrue(any("miss_count is missing" in error for error in errors))

    def test_hit_miss_inconsistency_fails_validation(self) -> None:
        metrics = {"access_count": 10, "hit_count": 6, "miss_count": 3}
        errors = validate_simulation_metrics(metrics, 10)
        self.assertTrue(any("HIT_COUNT + MISS_COUNT" in error for error in errors))

    def test_dry_run_has_no_performance_metrics_and_is_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "empty.mem"
            trace.write_text("0 00000000\n", encoding="ascii")
            result = run_simulation(
                trace,
                CacheConfig(4096, 1),
                result_dir=Path(temp) / "results",
                dry_run=True,
                plan_id="plan-42",
            )
            self.assertEqual(result["status"], "dry_run")
            self.assertTrue(result["synthetic"])
            self.assertEqual(result["metrics"], {})
            saved_result = json.loads(
                Path(result["result_file"]).read_text(encoding="utf-8")
            )
            self.assertEqual(saved_result["plan_id"], "plan-42")
            self.assertIn("no RTL simulation", Path(result["log_file"]).read_text())

            legacy_result = run_simulation(
                trace,
                CacheConfig(4096, 1),
                result_dir=Path(temp) / "results",
                dry_run=True,
            )
            saved_legacy_result = json.loads(
                Path(legacy_result["result_file"]).read_text(encoding="utf-8")
            )
            self.assertNotIn("plan_id", saved_legacy_result)

    def test_missing_rtl_is_reported_without_simulation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = root / "trace.mem"
            trace.write_text("0 00000000\n", encoding="ascii")
            rtl_dir = root / "rtl"
            rtl_dir.mkdir()
            result = run_simulation(
                trace,
                CacheConfig(4096, 1),
                rtl_dir=rtl_dir,
                result_dir=root / "results",
            )
            self.assertEqual(result["status"], "unavailable")
            self.assertFalse(result["synthetic"])
            self.assertEqual(result["metrics"], {})
            self.assertTrue(result["errors"])
            self.assertTrue(Path(result["log_file"]).is_file())

    def test_plusarg_config_can_be_loaded_by_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config_file = Path(temp) / "recommendation.cfg"
            config_file.write_text(
                "CACHE_SIZE=8192\nASSOCIATIVITY=2\nBLOCK_SIZE=64\nREPLACEMENT_POLICY=FIFO\n",
                encoding="ascii",
            )
            config = load_cache_config(config_file)
            self.assertEqual(config, CacheConfig(8192, 2, "FIFO", 64))

    def test_runner_accepts_supported_set_associativity_in_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "trace.mem"
            trace.write_text("0 00000000\n", encoding="ascii")
            result = run_simulation(
                trace,
                CacheConfig(4096, 2, "FIFO", 32),
                result_dir=Path(temp) / "results",
                dry_run=True,
            )
            self.assertEqual(result["status"], "dry_run")

    def test_cache_config_rejects_unsupported_geometry_and_policy(self) -> None:
        invalid_configs = (
            (2048, 1, "LRU", 64),
            (8192, 3, "LRU", 64),
            (8192, 1, "LRU", 16),
            (8192, 2, "PLRU", 64),
        )
        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    CacheConfig(*values)

    def test_windows_batch_command_quotes_paths_with_spaces(self) -> None:
        command = [
            r"C:\vivado\2025.2\Vivado\bin\xvlog.bat",
            "-d",
            "CACHE_SIZE_BYTES_CFG=8192",
            "-d",
            "BLOCK_SIZE_BYTES_CFG=64",
            r"C:\Users\mouny\coa project\rtl\cache_controller.v",
        ]
        with patch.dict("os.environ", {"COMSPEC": "cmd.exe"}):
            invocation, executable = _build_subprocess_command(command, windows=True)

        self.assertIsInstance(invocation, str)
        self.assertEqual(executable, "cmd.exe")
        self.assertIn(' /d /c ', invocation)
        self.assertIn('"CACHE_SIZE_BYTES_CFG=8192"', invocation)
        self.assertIn('"BLOCK_SIZE_BYTES_CFG=64"', invocation)
        self.assertIn(
            '"C:\\Users\\mouny\\coa project\\rtl\\cache_controller.v"',
            invocation,
        )
        plusarg_invocation, _ = _build_subprocess_command(
            ["xsim.bat", "-testplusarg", "TRACE_FILE=trace.mem"], windows=True
        )
        self.assertIsInstance(plusarg_invocation, str)
        self.assertIn('"TRACE_FILE=trace.mem"', plusarg_invocation)

    def test_windows_batch_execution_stops_on_error_and_logs_both_streams(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vivado test ") as temp:
            root = Path(temp) / "Project With Spaces"
            rtl_dir = root / "rtl"
            testbench_dir = root / "testbench"
            rtl_dir.mkdir(parents=True)
            testbench_dir.mkdir()
            (rtl_dir / "cache_controller.v").write_text("module cache_controller; endmodule\n")
            (testbench_dir / "cache_tb.v").write_text("module cache_tb; endmodule\n")
            trace = root / "trace.mem"
            trace.write_text("0 00000000\n", encoding="ascii")
            tool_paths = {
                name: str(root / "Vivado 2025.2" / "bin" / f"{name}.bat")
                for name in ("xvlog", "xelab", "xsim")
            }
            process_results = [
                SimpleNamespace(returncode=0, stdout="xvlog stdout", stderr="xvlog stderr"),
                SimpleNamespace(returncode=1, stdout="xelab stdout", stderr="xelab stderr"),
            ]

            with (
                patch("python.vivado_runner.shutil.which", side_effect=tool_paths.get),
                patch("python.vivado_runner._is_windows", return_value=True),
                patch(
                    "python.vivado_runner.subprocess.run",
                    side_effect=process_results,
                ) as run,
            ):
                result = run_simulation(
                    trace,
                    CacheConfig(4096, 1),
                    rtl_dir=rtl_dir,
                    testbench_dir=testbench_dir,
                    result_dir=root / "results",
                )

            self.assertEqual(result["status"], "failed")
            self.assertEqual(run.call_count, 2)
            invocations = [entry.args[0] for entry in run.call_args_list]
            self.assertEqual(
                [
                    ("xvlog.bat" in invocation, "xelab.bat" in invocation)
                    for invocation in invocations
                ],
                [(True, False), (False, True)],
            )
            self.assertTrue(
                all(
                    Path(entry.kwargs["executable"]).name.lower() == "cmd.exe"
                    for entry in run.call_args_list
                )
            )
            self.assertTrue(all(entry.kwargs["cwd"].is_dir() for entry in run.call_args_list))
            log = Path(result["log_file"]).read_text(encoding="utf-8")
            for output in ("xvlog stdout", "xvlog stderr", "xelab stdout", "xelab stderr"):
                self.assertIn(output, log)
            self.assertIn(
                subprocess.list2cmdline([str(rtl_dir / "cache_controller.v")]), log
            )
            self.assertNotIn("xsim.bat", log)


if __name__ == "__main__":
    unittest.main()
