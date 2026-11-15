"""Feature extraction shared by dataset construction and optimization."""

from __future__ import annotations

import hashlib

import pandas as pd

from .workload_generator import TRACE_COLUMNS


def trace_identity(trace: pd.DataFrame) -> str:
    """Return a stable ID for the normalized contents of a workload trace."""
    normalized = trace.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def extract_workload_features(trace: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize access count, read/write balance, and address locality."""
    missing = set(TRACE_COLUMNS) - set(trace.columns)
    if missing:
        raise ValueError(f"Trace is missing columns: {sorted(missing)}")
    if trace.empty:
        raise ValueError("Cannot extract features from an empty trace.")
    addresses = pd.to_numeric(trace["address"], errors="raise").to_numpy(dtype="uint64")
    operations = trace["operation"].astype(str).str.upper()
    if not operations.isin(["R", "W"]).all():
        raise ValueError("Trace operations must be R or W.")
    unique_count = len(set(int(address) for address in addresses))
    temporal = 1.0 - unique_count / len(addresses)
    if len(addresses) > 1:
        deltas = [abs(int(right) - int(left)) for left, right in zip(addresses, addresses[1:])]
        spatial = sum(delta <= 64 for delta in deltas) / len(deltas)
    else:
        spatial = 0.0
    reads = int((operations == "R").sum())
    writes = len(trace) - reads
    ratio = reads / writes if writes else float(reads)
    kinds = trace["workload_type"].astype(str).str.lower().unique()
    if len(kinds) != 1:
        raise ValueError("A trace must contain exactly one workload_type.")
    return {
        "workload_type": str(kinds[0]),
        "read_write_ratio": float(ratio),
        "num_accesses": int(len(trace)),
        "temporal_locality": float(temporal),
        "spatial_locality": float(spatial),
    }
