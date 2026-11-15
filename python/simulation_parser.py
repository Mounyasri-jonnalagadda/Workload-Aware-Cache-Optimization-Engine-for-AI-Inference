"""Extract cache metrics and failure indicators from simulator output."""

from __future__ import annotations

import re
from typing import Any

METRIC_NAMES = {
    "hit_count": ("HIT_COUNT", "HITS"),
    "miss_count": ("MISS_COUNT", "MISSES"),
    "access_count": ("ACCESS_COUNT", "TOTAL_ACCESSES"),
    "estimated_amat": ("ESTIMATED_AMAT", "AMAT"),
    "cycles": ("TOTAL_CYCLES", "CYCLES"),
}


def parse_simulation_log(text: str) -> dict[str, Any]:
    """Parse common ``NAME=number`` testbench counters and fatal messages."""
    metrics: dict[str, int | float | None] = {key: None for key in METRIC_NAMES}
    for metric, labels in METRIC_NAMES.items():
        for label in labels:
            match = re.search(
                rf"(?im)^\s*{label}\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*$", text
            )
            if match:
                number = float(match.group(1))
                metrics[metric] = int(number) if number.is_integer() else number
                break
    hits, misses = metrics["hit_count"], metrics["miss_count"]
    if isinstance(hits, int) and isinstance(misses, int) and hits + misses > 0:
        metrics["hit_rate"] = hits / (hits + misses)
        metrics["miss_rate"] = misses / (hits + misses)
    else:
        metrics["hit_rate"] = None
        metrics["miss_rate"] = None
    failure_lines = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"\b(FATAL|ERROR)\b", line, flags=re.IGNORECASE)
    ]
    return {"metrics": metrics, "errors": failure_lines}
