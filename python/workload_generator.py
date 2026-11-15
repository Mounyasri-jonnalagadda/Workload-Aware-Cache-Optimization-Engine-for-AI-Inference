"""Reproducible, locality-oriented AI workload trace generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import WORKLOAD_DIR, ensure_project_directories

WORKLOAD_TYPES = ("cnn", "transformer", "matmul")
TRACE_COLUMNS = ("address", "operation", "workload_type", "sequence")


def generate_workload(
    workload_type: str,
    num_accesses: int = 10_000,
    seed: int = 42,
    parameters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate an address trace with repeatable workload-specific locality."""
    kind = workload_type.lower()
    if kind not in WORKLOAD_TYPES:
        raise ValueError(f"workload_type must be one of {WORKLOAD_TYPES}.")
    if num_accesses <= 0:
        raise ValueError("num_accesses must be positive.")
    parameters = parameters or {}
    rng = np.random.default_rng(seed)
    addresses = np.empty(num_accesses, dtype=np.uint64)

    if kind == "cnn":
        _generate_cnn(addresses, rng, parameters)
    elif kind == "transformer":
        _generate_transformer(addresses, rng, parameters)
    else:
        _generate_matmul(addresses, rng, parameters)

    write_probability = float(parameters.get("write_probability", 0.15))
    if not 0 <= write_probability <= 1:
        raise ValueError("write_probability must be between 0 and 1.")
    operations = np.where(rng.random(num_accesses) < write_probability, "W", "R")
    return pd.DataFrame(
        {
            "address": addresses,
            "operation": operations,
            "workload_type": kind,
            "sequence": np.arange(num_accesses, dtype=np.int64),
        },
        columns=TRACE_COLUMNS,
    )


def _generate_cnn(
    output: np.ndarray, rng: np.random.Generator, parameters: dict[str, Any]
) -> None:
    region = _positive_int(parameters, "region_bytes", 65_536)
    weights = _positive_int(parameters, "weight_region_bytes", 8_192)
    for index in range(len(output)):
        if index % 4 == 0:
            output[index] = rng.integers(0, weights // 4 + 1) * 4
        else:
            # Adjacent pixels and overlapping convolution windows stay nearby.
            base = ((index // 32) * 128 + (index % 32) * 4) % region
            output[index] = base + int(rng.integers(0, 4)) * 4


def _generate_transformer(
    output: np.ndarray, rng: np.random.Generator, parameters: dict[str, Any]
) -> None:
    embedding = _positive_int(parameters, "embedding_region_bytes", 131_072)
    attention = _positive_int(parameters, "attention_region_bytes", 262_144)
    for index in range(len(output)):
        phase = (index // 64) % 3
        if phase == 0:
            token = (index // 8) % max(1, embedding // 64)
            output[index] = token * 64 + int(rng.integers(0, 16)) * 4
        elif phase == 1:
            # Revisit key/value vectors while traversing a bounded attention window.
            token = ((index // 16) + int(rng.integers(0, 4))) % max(1, attention // 64)
            output[index] = embedding + token * 64 + int(rng.integers(0, 16)) * 4
        else:
            output[index] = (
                embedding
                + attention
                + ((index * 64) % max(64, embedding))
                + int(rng.integers(0, 16)) * 4
            )


def _generate_matmul(
    output: np.ndarray, rng: np.random.Generator, parameters: dict[str, Any]
) -> None:
    matrix_bytes = _positive_int(parameters, "matrix_bytes", 65_536)
    rows = _positive_int(parameters, "row_bytes", 256)
    for index in range(len(output)):
        phase = (index // 16) % 3
        if phase == 0:
            output[index] = ((index // 4) * 4) % matrix_bytes
        elif phase == 1:
            column = ((index // 4) * 64) % matrix_bytes
            output[index] = column + int(rng.integers(0, max(1, rows // 4))) * 4
        else:
            output[index] = rng.integers(0, matrix_bytes // 4 + 1) * 4


def _positive_int(parameters: dict[str, Any], key: str, default: int) -> int:
    value = int(parameters.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be positive.")
    return value


def save_workload(
    trace: pd.DataFrame, output_path: str | Path | None = None
) -> Path:
    """Persist a generated trace as a CSV file."""
    required = set(TRACE_COLUMNS)
    if not required.issubset(trace.columns):
        raise ValueError(f"Trace is missing columns: {sorted(required - set(trace.columns))}")
    ensure_project_directories()
    path = Path(output_path) if output_path else WORKLOAD_DIR / "workload.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(path, index=False)
    return path
