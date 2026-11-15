"""Export Python CSV traces to the explicit text interface used by RTL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import WORKLOAD_DIR
from .workload_generator import TRACE_COLUMNS

OPERATION_ENCODING = {"R": 0, "W": 1}


def export_trace(
    trace: pd.DataFrame | str | Path,
    csv_path: str | Path | None = None,
    memory_path: str | Path | None = None,
    address_width: int = 32,
    max_accesses: int | None = None,
) -> tuple[Path, Path]:
    """Write the canonical CSV and two-column ``op address`` hex trace.

    Each .mem line is ``0 <address>`` for a read or ``1 <address>`` for a write.
    This is a plain-text testbench contract, not an automatic CSV import.
    """
    if address_width <= 0:
        raise ValueError("address_width must be positive.")
    if max_accesses is not None and max_accesses <= 0:
        raise ValueError("max_accesses must be positive when provided.")
    frame = pd.read_csv(trace) if isinstance(trace, (str, Path)) else trace.copy()
    missing = set(TRACE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Trace is missing columns: {sorted(missing)}")
    if max_accesses is not None:
        frame = frame.iloc[:max_accesses].copy()
    if frame.empty:
        raise ValueError("Cannot export an empty trace.")
    if not frame["operation"].astype(str).str.upper().isin(OPERATION_ENCODING).all():
        raise ValueError("Trace operations must be R or W.")

    encoded_rows: list[tuple[int, int]] = []
    maximum = (1 << address_width) - 1
    for operation, address in zip(frame["operation"], frame["address"]):
        value = int(address)
        if value < 0 or value > maximum or value != address:
            raise ValueError(
                f"Address {address} must be an integer that fits in {address_width} bits."
            )
        encoded_rows.append((OPERATION_ENCODING[str(operation).upper()], value))

    csv_file = Path(csv_path) if csv_path else WORKLOAD_DIR / "workload.csv"
    mem_file = Path(memory_path) if memory_path else csv_file.with_suffix(".mem")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    mem_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_file, index=False)
    hex_width = (address_width + 3) // 4
    with mem_file.open("w", encoding="ascii", newline="\n") as output:
        for operation, value in encoded_rows:
            output.write(f"{operation} {value:0{hex_width}X}\n")
    return csv_file, mem_file
