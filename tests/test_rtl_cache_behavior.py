from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from python.config import SIMULATION_RESULTS_DIR, CacheConfig
from python.vivado_runner import run_simulation


class RTLCacheBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [tool for tool in ("xvlog", "xelab", "xsim") if not shutil.which(tool)]
        if missing:
            raise unittest.SkipTest(
                "Vivado Simulator tools are unavailable: " + ", ".join(missing)
            )

    def simulate(self, records: list[tuple[int, int]], config: CacheConfig) -> dict[str, int]:
        with tempfile.TemporaryDirectory(prefix="cache rtl test ") as temp:
            result = self._run_and_validate(records, config, Path(temp) / "results")
            return result["metrics"]

    def simulate_persisted(
        self, records: list[tuple[int, int]], config: CacheConfig, run_name: str
    ) -> dict[str, Any]:
        result_dir = SIMULATION_RESULTS_DIR / "rtl_policy_regression" / run_name
        return self._run_and_validate(records, config, result_dir)

    def _run_and_validate(
        self, records: list[tuple[int, int]], config: CacheConfig, result_dir: Path
    ) -> dict[str, Any]:
        result_dir.mkdir(parents=True, exist_ok=True)
        trace = result_dir / "trace.mem"
        trace.write_text(
            "".join(f"{operation} {address:08X}\n" for operation, address in records),
            encoding="ascii",
        )
        result = run_simulation(
            trace_file=trace,
            cache_config=config,
            result_dir=result_dir,
        )
        log = Path(result["log_file"]).read_text(encoding="utf-8")
        self.assertEqual(
            result["status"],
            "success",
            f"Vivado run failed: {result['errors']}\n{log}",
        )
        self.assertTrue(result["validated"])
        return result

    def test_direct_mapped_two_way_and_four_way_hit_miss_behavior(self) -> None:
        direct_sizes = [
            self.simulate(
                [(0, 0), (0, 0)],
                CacheConfig(size, 1, "LRU", 32),
            )
            for size in (4096, 8192, 16384, 32768)
        ]
        direct_conflict = self.simulate(
            [(0, 0), (0, 0), (0, 4096), (0, 0)],
            CacheConfig(4096, 1, "LRU", 32),
        )
        two_way = self.simulate(
            [(0, 0), (0, 4096), (0, 0)],
            CacheConfig(4096, 2, "LRU", 64),
        )
        four_way = self.simulate(
            [(0, 0), (0, 4096), (0, 8192), (0, 12288), (0, 0)],
            CacheConfig(4096, 4, "LRU", 64),
        )

        for metrics in direct_sizes:
            self.assertEqual((metrics["hit_count"], metrics["miss_count"]), (1, 1))
        self.assertEqual(
            (direct_conflict["hit_count"], direct_conflict["miss_count"]), (1, 3)
        )
        self.assertEqual((two_way["hit_count"], two_way["miss_count"]), (1, 2))
        self.assertEqual((four_way["hit_count"], four_way["miss_count"]), (1, 4))

    def test_lru_fifo_and_random_have_real_deterministic_victim_behavior(self) -> None:
        conflict_trace = [(0, 0), (0, 4096), (0, 0), (0, 8192), (0, 4096)]
        lru = self.simulate(conflict_trace, CacheConfig(4096, 2, "LRU", 64))
        fifo = self.simulate(conflict_trace, CacheConfig(4096, 2, "FIFO", 64))
        random_first = self.simulate(
            conflict_trace, CacheConfig(4096, 2, "RANDOM", 64)
        )
        random_second = self.simulate(
            conflict_trace, CacheConfig(4096, 2, "RANDOM", 64)
        )

        self.assertEqual((lru["hit_count"], lru["miss_count"]), (1, 4))
        self.assertEqual((fifo["hit_count"], fifo["miss_count"]), (2, 3))
        self.assertEqual((random_first["hit_count"], random_first["miss_count"]), (2, 3))
        self.assertEqual(
            (random_first["hit_count"], random_first["miss_count"]),
            (random_second["hit_count"], random_second["miss_count"]),
        )

    def test_six_access_trace_distinguishes_all_replacement_policies(self) -> None:
        conflict_trace = [
            (0, 0x00000000),
            (0, 0x00001000),
            (0, 0x00000000),
            (0, 0x00002000),
            (0, 0x00000000),
            (0, 0x00002000),
        ]
        config_by_policy = {
            "LRU": CacheConfig(8192, 2, "LRU", 64),
            "FIFO": CacheConfig(8192, 2, "FIFO", 64),
            "RANDOM": CacheConfig(8192, 2, "RANDOM", 64),
        }
        expected_counts = {
            "LRU": (3, 3),
            "FIFO": (2, 4),
            "RANDOM": (1, 5),
        }
        results = {
            policy: self.simulate_persisted(
                conflict_trace, config, f"six_access_{policy.lower()}"
            )
            for policy, config in config_by_policy.items()
        }
        results["RANDOM_REPEAT"] = self.simulate_persisted(
            conflict_trace, config_by_policy["RANDOM"], "six_access_random_repeat"
        )

        for policy, result in results.items():
            metrics = result["metrics"]
            self.assertEqual(result["expected_access_count"], 6)
            self.assertEqual(metrics["access_count"], 6)
            self.assertEqual(result["validation"]["actual_access_count"], 6)
            self.assertEqual(
                (metrics["hit_count"], metrics["miss_count"]),
                expected_counts["RANDOM" if policy == "RANDOM_REPEAT" else policy],
            )
            self.assertTrue(Path(result["log_file"]).is_file())
            self.assertTrue(Path(result["result_file"]).is_file())

        self.assertEqual(
            results["RANDOM"]["metrics"],
            results["RANDOM_REPEAT"]["metrics"],
        )

    def test_write_allocate_and_write_through_readback_correctness(self) -> None:
        metrics = self.simulate(
            [
                (1, 0),
                (0, 0),
                (1, 4),
                (0, 0),
                (0, 4),
            ],
            CacheConfig(4096, 4, "FIFO", 128),
        )
        self.assertEqual(metrics["access_count"], 5)
        self.assertEqual((metrics["hit_count"], metrics["miss_count"]), (4, 1))


if __name__ == "__main__":
    unittest.main()
