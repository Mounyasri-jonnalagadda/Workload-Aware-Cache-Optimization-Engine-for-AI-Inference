"""Shared paths and cache configuration definitions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_DIR = PROJECT_ROOT / "workloads"
DATASET_DIR = PROJECT_ROOT / "datasets"
SIMULATION_RESULTS_DIR = PROJECT_ROOT / "simulation_results"
TRAINING_DIR = PROJECT_ROOT / "training"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
RTL_DIR = PROJECT_ROOT / "rtl"
TESTBENCH_DIR = PROJECT_ROOT / "testbench"
BLOCK_SIZE_BYTES = 64
SUPPORTED_CACHE_SIZES_BYTES = (4096, 8192, 16384, 32768)
SUPPORTED_ASSOCIATIVITIES = (1, 2, 4)
SUPPORTED_BLOCK_SIZES_BYTES = (32, 64, 128)
SUPPORTED_REPLACEMENT_POLICIES = ("LRU", "FIFO", "RANDOM")
REPLACEMENT_POLICY_CODES = {"LRU": 0, "FIFO": 1, "RANDOM": 2}

# Supply both values to train and optimize against an AMAT target.
HIT_TIME_CYCLES: float | None = None
MISS_PENALTY_CYCLES: float | None = None


@dataclass(frozen=True)
class CacheConfig:
    """A cache geometry and replacement policy supported by the testbench."""

    cache_size_bytes: int
    associativity: int
    replacement_policy: str = "LRU"
    block_size_bytes: int = BLOCK_SIZE_BYTES

    def __post_init__(self) -> None:
        policy = self.replacement_policy.upper()
        object.__setattr__(self, "replacement_policy", policy)
        if self.cache_size_bytes not in SUPPORTED_CACHE_SIZES_BYTES:
            raise ValueError(
                f"Cache size must be one of {SUPPORTED_CACHE_SIZES_BYTES} bytes."
            )
        if self.associativity not in SUPPORTED_ASSOCIATIVITIES:
            raise ValueError(
                f"Associativity must be one of {SUPPORTED_ASSOCIATIVITIES} ways."
            )
        if self.block_size_bytes not in SUPPORTED_BLOCK_SIZES_BYTES:
            raise ValueError(
                f"Block size must be one of {SUPPORTED_BLOCK_SIZES_BYTES} bytes."
            )
        if policy not in SUPPORTED_REPLACEMENT_POLICIES:
            raise ValueError(
                f"Replacement policy must be one of {SUPPORTED_REPLACEMENT_POLICIES}."
            )
        if self.cache_size_bytes % (self.associativity * self.block_size_bytes):
            raise ValueError("Cache size must be divisible by associativity × block size.")
        num_sets = self.cache_size_bytes // (self.associativity * self.block_size_bytes)
        if num_sets < 1 or num_sets & (num_sets - 1):
            raise ValueError("The number of cache sets must be a positive power of two.")

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def load_cache_config(path: str | Path) -> CacheConfig:
    """Load a cache configuration JSON document or KEY=VALUE plusarg file."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        data = data.get("recommendation", data.get("cache_config", data))
    else:
        data: dict[str, str] = {}
        for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid config line {line_number}: expected KEY=VALUE.")
            key, value = line.split("=", 1)
            data[key.strip().upper()] = value.strip()
        data = {
            "cache_size_bytes": data.get("CACHE_SIZE"),
            "associativity": data.get("ASSOCIATIVITY"),
            "replacement_policy": data.get("REPLACEMENT_POLICY", "LRU"),
            "block_size_bytes": data.get("BLOCK_SIZE", BLOCK_SIZE_BYTES),
        }
    size = data.get("cache_size_bytes", data.get("CACHE_SIZE"))
    ways = data.get("associativity", data.get("ASSOCIATIVITY"))
    if size is None or ways is None:
        raise ValueError("Configuration must specify CACHE_SIZE and ASSOCIATIVITY.")
    return CacheConfig(
        cache_size_bytes=int(size),
        associativity=int(ways),
        replacement_policy=str(data.get("replacement_policy", data.get("REPLACEMENT_POLICY", "LRU"))),
        block_size_bytes=int(data.get("block_size_bytes", data.get("BLOCK_SIZE", BLOCK_SIZE_BYTES))),
    )


def ensure_project_directories() -> None:
    """Create the standard output directories when an operation needs them."""
    for path in (
        WORKLOAD_DIR,
        DATASET_DIR,
        SIMULATION_RESULTS_DIR,
        TRAINING_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
