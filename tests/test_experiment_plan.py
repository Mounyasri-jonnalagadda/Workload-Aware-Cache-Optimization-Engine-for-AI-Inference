from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from python.cache_optimizer import generate_candidates
from python.experiment_plan import (
    ANCHOR_CONFIGS,
    DEFAULT_EXPERIMENT_SEEDS,
    create_experiment_plan,
    execute_experiment_plan,
)
from python.workload_generator import WORKLOAD_TYPES


class ExperimentPlanTests(unittest.TestCase):
    def test_balanced_plan_covers_all_candidates_per_workload(self) -> None:
        plan = create_experiment_plan(accesses=128)
        candidates = generate_candidates()
        candidate_keys = {
            (
                config.cache_size_bytes,
                config.associativity,
                config.block_size_bytes,
                config.replacement_policy,
            )
            for config in candidates
        }

        self.assertEqual(plan["expected_simulation_count"], 396)
        self.assertEqual(plan["workload_trace_count"], 15)
        self.assertEqual(plan["candidate_configuration_count"], 84)
        self.assertEqual(plan["workload_distribution"], {
            "cnn": 132,
            "transformer": 132,
            "matmul": 132,
        })
        self.assertEqual(set(plan["seeds"]), set(DEFAULT_EXPERIMENT_SEEDS))
        global_configuration_counts = Counter(
            (
                item["cache_config"]["cache_size_bytes"],
                item["cache_config"]["associativity"],
                item["cache_config"]["block_size_bytes"],
                item["cache_config"]["replacement_policy"],
            )
            for item in plan["experiments"]
        )
        self.assertEqual(
            sorted(global_configuration_counts.values()),
            [3] * 72 + [15] * 12,
        )

        for workload_type in WORKLOAD_TYPES:
            workload_experiments = [
                item for item in plan["experiments"]
                if item["workload_type"] == workload_type
            ]
            observed = {
                (
                    item["cache_config"]["cache_size_bytes"],
                    item["cache_config"]["associativity"],
                    item["cache_config"]["block_size_bytes"],
                    item["cache_config"]["replacement_policy"],
                )
                for item in workload_experiments
            }
            self.assertEqual(observed, candidate_keys)
            runs_per_seed = Counter(item["seed"] for item in workload_experiments)
            self.assertEqual(sorted(runs_per_seed.values()), [26, 26, 26, 27, 27])

            per_config = Counter(
                (
                    item["cache_config"]["cache_size_bytes"],
                    item["cache_config"]["associativity"],
                    item["cache_config"]["block_size_bytes"],
                    item["cache_config"]["replacement_policy"],
                )
                for item in workload_experiments
            )
            self.assertEqual(sorted(per_config.values()), [1] * 72 + [5] * 12)

    def test_anchor_set_covers_levels_and_reports_pairwise_gaps(self) -> None:
        plan = create_experiment_plan(accesses=64)
        anchors = ANCHOR_CONFIGS
        coverage = plan["anchor_pairwise_coverage"]

        self.assertEqual(len(anchors), 12)
        self.assertEqual(
            {config.cache_size_bytes for config in anchors},
            {4096, 8192, 16384, 32768},
        )
        self.assertEqual({config.associativity for config in anchors}, {1, 2, 4})
        self.assertEqual({config.block_size_bytes for config in anchors}, {32, 64, 128})
        self.assertEqual(
            {config.replacement_policy for config in anchors if config.associativity > 1},
            {"LRU", "FIFO", "RANDOM"},
        )
        self.assertTrue(
            all(
                config.replacement_policy == "LRU"
                for config in anchors
                if config.associativity == 1
            )
        )
        self.assertEqual(coverage["covered_pair_count"], 59)
        self.assertEqual(coverage["legal_pair_count"], 61)
        self.assertEqual(
            {
                (
                    tuple(pair["parameters"]),
                    pair[pair["parameters"][0]],
                    pair[pair["parameters"][1]],
                )
                for pair in coverage["uncovered_pairs"]
            },
            {
                (("cache_size_bytes", "replacement_policy"), 4096, "FIFO"),
                (("cache_size_bytes", "replacement_policy"), 32768, "RANDOM"),
            },
        )

    def test_plan_and_trace_ids_are_reproducible_and_content_based(self) -> None:
        first = create_experiment_plan(accesses=128)
        second = create_experiment_plan(accesses=128)

        self.assertEqual(first["plan_id"], second["plan_id"])
        self.assertEqual(first["experiments"], second["experiments"])
        for workload_type in WORKLOAD_TYPES:
            ids = {
                item["trace_id"]
                for item in first["traces"]
                if item["workload_type"] == workload_type
            }
            self.assertEqual(len(ids), len(DEFAULT_EXPERIMENT_SEEDS))

    def test_plan_executor_passes_plan_id_to_every_result_run(self) -> None:
        plan = create_experiment_plan(accesses=8)

        class DatasetResult:
            def __init__(self, quality_path: str) -> None:
                self.attrs = {"quality_report_path": quality_path}

            def __len__(self) -> int:
                return 396

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with (
                patch("python.experiment_plan.WORKLOAD_DIR", root / "workloads"),
                patch(
                    "python.experiment_plan.export_trace",
                    return_value=(root / "trace.csv", root / "trace.mem"),
                ),
                patch(
                    "python.experiment_plan.run_simulation",
                    return_value={"status": "success", "validated": True},
                ) as run_simulation,
                patch(
                    "python.experiment_plan.build_dataset",
                    return_value=DatasetResult(str(root / "quality.json")),
                ),
            ):
                summary = execute_experiment_plan(
                    plan_path,
                    results_dir=root / "results",
                    dataset_path=root / "dataset.csv",
                )

        self.assertEqual(summary["status"], "success")
        self.assertEqual(run_simulation.call_count, 396)
        self.assertTrue(
            all(
                call.kwargs["plan_id"] == plan["plan_id"]
                for call in run_simulation.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
