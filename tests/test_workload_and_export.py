from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from python.feature_engineering import trace_identity
from python.trace_exporter import export_trace
from python.workload_generator import generate_workload


class WorkloadAndExportTests(unittest.TestCase):
    def test_trace_identity_is_content_based(self) -> None:
        original = generate_workload("cnn", num_accesses=256, seed=7)
        same_trace = generate_workload("cnn", num_accesses=256, seed=7)
        different_trace = generate_workload("cnn", num_accesses=256, seed=8)

        self.assertEqual(trace_identity(original), trace_identity(same_trace))
        self.assertNotEqual(trace_identity(original), trace_identity(different_trace))

    def test_generation_is_reproducible_and_has_required_schema(self) -> None:
        for kind in ("cnn", "transformer", "matmul"):
            first = generate_workload(kind, num_accesses=256, seed=7)
            second = generate_workload(kind, num_accesses=256, seed=7)
            self.assertTrue(first.equals(second))
            self.assertEqual(
                list(first.columns),
                ["address", "operation", "workload_type", "sequence"],
            )
            self.assertEqual(set(first["operation"]), {"R", "W"})
            self.assertEqual(set(first["workload_type"]), {kind})

    def test_export_encodes_operations_and_applies_trace_limit(self) -> None:
        trace = generate_workload("cnn", num_accesses=8, seed=2)
        with tempfile.TemporaryDirectory() as temp:
            csv_path, mem_path = export_trace(
                trace,
                Path(temp) / "trace.csv",
                Path(temp) / "trace.mem",
                address_width=32,
                max_accesses=3,
            )
            self.assertEqual(len(csv_path.read_text(encoding="utf-8").splitlines()), 4)
            lines = mem_path.read_text(encoding="ascii").splitlines()
            self.assertEqual(len(lines), 3)
            for index, line in enumerate(lines):
                op, address = line.split()
                self.assertEqual(op, "0" if trace.iloc[index]["operation"] == "R" else "1")
                self.assertEqual(int(address, 16), int(trace.iloc[index]["address"]))

    def test_export_rejects_addresses_that_do_not_fit(self) -> None:
        trace = generate_workload("matmul", num_accesses=1, seed=1)
        trace.loc[0, "address"] = 256
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "fits in 8 bits"):
                export_trace(
                    trace,
                    Path(temp) / "trace.csv",
                    Path(temp) / "trace.mem",
                    address_width=8,
                )


if __name__ == "__main__":
    unittest.main()
