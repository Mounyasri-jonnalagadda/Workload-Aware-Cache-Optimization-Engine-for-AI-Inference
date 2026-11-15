"""Reproducible, balanced workload/configuration experiment plans."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .config import (
    DATASET_DIR,
    REPORTS_DIR,
    SIMULATION_RESULTS_DIR,
    WORKLOAD_DIR,
    CacheConfig,
)
from .dataset_builder import build_dataset
from .feature_engineering import extract_workload_features, trace_identity
from .trace_exporter import export_trace
from .vivado_runner import run_simulation
from .workload_generator import WORKLOAD_TYPES, generate_workload
from .cache_optimizer import generate_candidates

DEFAULT_EXPERIMENT_SEEDS = (42, 100, 200, 300, 400)
DESIGN_NAME = "balanced-396"
CONFIGURATION_FACTORS = (
    "cache_size_bytes",
    "associativity",
    "block_size_bytes",
    "replacement_policy",
)

ANCHOR_CONFIGS = (
    CacheConfig(4096, 1, "LRU", 128),
    CacheConfig(4096, 2, "RANDOM", 64),
    CacheConfig(4096, 4, "LRU", 32),
    CacheConfig(8192, 1, "LRU", 64),
    CacheConfig(8192, 2, "FIFO", 32),
    CacheConfig(8192, 4, "RANDOM", 128),
    CacheConfig(16384, 1, "LRU", 64),
    CacheConfig(16384, 2, "FIFO", 128),
    CacheConfig(16384, 4, "RANDOM", 32),
    CacheConfig(32768, 1, "LRU", 32),
    CacheConfig(32768, 2, "LRU", 128),
    CacheConfig(32768, 4, "FIFO", 64),
)


def _config_key(config: CacheConfig) -> tuple[int, int, int, str]:
    return (
        config.cache_size_bytes,
        config.associativity,
        config.block_size_bytes,
        config.replacement_policy,
    )


def pairwise_coverage(
    selected: Iterable[CacheConfig],
    candidates: Iterable[CacheConfig] | None = None,
) -> dict[str, Any]:
    """Report observed and uncovered legal value-pairs for each config factor pair."""
    candidate_list = list(candidates) if candidates is not None else generate_candidates()
    selected_list = list(selected)
    legal_pairs: dict[tuple[str, str], set[tuple[Any, Any]]] = {}
    observed_pairs: dict[tuple[str, str], set[tuple[Any, Any]]] = {}
    for left, right in combinations(CONFIGURATION_FACTORS, 2):
        legal_pairs[(left, right)] = set()
        observed_pairs[(left, right)] = set()

    def values(config: CacheConfig) -> dict[str, Any]:
        return {
            "cache_size_bytes": config.cache_size_bytes,
            "associativity": config.associativity,
            "block_size_bytes": config.block_size_bytes,
            "replacement_policy": config.replacement_policy,
        }

    for config in candidate_list:
        config_values = values(config)
        for pair in legal_pairs:
            legal_pairs[pair].add((config_values[pair[0]], config_values[pair[1]]))
    for config in selected_list:
        config_values = values(config)
        for pair in observed_pairs:
            observed_pairs[pair].add((config_values[pair[0]], config_values[pair[1]]))

    pair_reports = []
    uncovered_pairs = []
    for pair in legal_pairs:
        missing = sorted(legal_pairs[pair] - observed_pairs[pair], key=str)
        pair_reports.append(
            {
                "parameters": list(pair),
                "covered_pair_count": len(observed_pairs[pair]),
                "legal_pair_count": len(legal_pairs[pair]),
                "uncovered_pairs": [
                    {pair[0]: left, pair[1]: right} for left, right in missing
                ],
            }
        )
        uncovered_pairs.extend(
            {"parameters": list(pair), pair[0]: left, pair[1]: right}
            for left, right in missing
        )
    return {
        "covered_pair_count": sum(item["covered_pair_count"] for item in pair_reports),
        "legal_pair_count": sum(item["legal_pair_count"] for item in pair_reports),
        "pair_reports": pair_reports,
        "uncovered_pairs": uncovered_pairs,
    }


def _validate_anchor_configs(candidates: list[CacheConfig]) -> dict[str, Any]:
    candidate_keys = {_config_key(config) for config in candidates}
    anchor_keys = {_config_key(config) for config in ANCHOR_CONFIGS}
    if len(ANCHOR_CONFIGS) != 12 or len(anchor_keys) != 12:
        raise ValueError("The balanced experiment design requires 12 unique anchors.")
    if not anchor_keys.issubset(candidate_keys):
        raise ValueError("An anchor configuration is not a supported optimizer candidate.")

    required_values = {
        "cache_size_bytes": {config.cache_size_bytes for config in candidates},
        "associativity": {config.associativity for config in candidates},
        "block_size_bytes": {config.block_size_bytes for config in candidates},
        "replacement_policy": {
            config.replacement_policy for config in candidates if config.associativity > 1
        },
    }
    for parameter, values in required_values.items():
        present = {
            getattr(config, parameter)
            for config in ANCHOR_CONFIGS
            if parameter != "replacement_policy" or config.associativity > 1
        }
        if not values.issubset(present):
            raise ValueError(
                f"Anchor configurations do not cover all required {parameter} values."
            )
    if any(
        config.associativity == 1 and config.replacement_policy != "LRU"
        for config in ANCHOR_CONFIGS
    ):
        raise ValueError("Direct-mapped anchors must use canonical LRU.")
    return pairwise_coverage(ANCHOR_CONFIGS, candidates)


def create_experiment_plan(
    accesses: int = 10_000,
    seeds: Iterable[int] = DEFAULT_EXPERIMENT_SEEDS,
    design_seed: int = 42,
) -> dict[str, Any]:
    """Create the deterministic 396-run plan without writing files or running RTL."""
    seed_values = tuple(int(seed) for seed in seeds)
    if accesses <= 0:
        raise ValueError("accesses must be positive.")
    if len(seed_values) != 5 or len(set(seed_values)) != 5 or any(seed < 0 for seed in seed_values):
        raise ValueError("Exactly five distinct non-negative workload seeds are required.")

    candidates = sorted(generate_candidates(), key=_config_key)
    anchor_coverage = _validate_anchor_configs(candidates)
    anchor_keys = {_config_key(config) for config in ANCHOR_CONFIGS}
    remaining = [config for config in candidates if _config_key(config) not in anchor_keys]

    trace_rows: dict[tuple[str, int], dict[str, Any]] = {}
    trace_manifest: list[dict[str, Any]] = []
    for workload_type in WORKLOAD_TYPES:
        for seed in seed_values:
            trace = generate_workload(workload_type, accesses, seed)
            trace_id = trace_identity(trace)
            trace_rows[(workload_type, seed)] = {
                "trace_id": trace_id,
                "trace": trace,
            }
            trace_manifest.append(
                {
                    "workload_type": workload_type,
                    "seed": seed,
                    "trace_id": trace_id,
                    "num_accesses": len(trace),
                }
            )

    assignments: list[dict[str, Any]] = []
    for workload_index, workload_type in enumerate(WORKLOAD_TYPES):
        shuffled = list(remaining)
        random.Random(design_seed + workload_index).shuffle(shuffled)
        seed_buckets: dict[int, list[CacheConfig]] = {seed: [] for seed in seed_values}
        for index, config in enumerate(shuffled):
            seed_buckets[seed_values[index % len(seed_values)]].append(config)

        for seed in seed_values:
            trace_id = trace_rows[(workload_type, seed)]["trace_id"]
            assigned = [*ANCHOR_CONFIGS, *seed_buckets[seed]]
            for config in assigned:
                assignments.append(
                    {
                        "workload_type": workload_type,
                        "seed": seed,
                        "trace_id": trace_id,
                        "cache_config": config.to_dict(),
                    }
                )

    per_workload = {
        workload: sum(item["workload_type"] == workload for item in assignments)
        for workload in WORKLOAD_TYPES
    }
    per_trace = Counter(
        (item["workload_type"], item["seed"]) for item in assignments
    )
    configuration_counts = Counter(
        (
            item["cache_config"]["cache_size_bytes"],
            item["cache_config"]["associativity"],
            item["cache_config"]["block_size_bytes"],
            item["cache_config"]["replacement_policy"],
        )
        for item in assignments
    )
    for workload_type in WORKLOAD_TYPES:
        configs_for_workload = {
            (
                item["cache_config"]["cache_size_bytes"],
                item["cache_config"]["associativity"],
                item["cache_config"]["block_size_bytes"],
                item["cache_config"]["replacement_policy"],
            )
            for item in assignments
            if item["workload_type"] == workload_type
        }
        if configs_for_workload != {_config_key(config) for config in candidates}:
            raise RuntimeError(f"The plan does not cover every candidate for {workload_type}.")

    body = {
        "schema_version": 1,
        "design": DESIGN_NAME,
        "accesses": accesses,
        "seeds": list(seed_values),
        "design_seed": design_seed,
        "workload_types": list(WORKLOAD_TYPES),
        "candidate_configuration_count": len(candidates),
        "anchor_configuration_count": len(ANCHOR_CONFIGS),
        "expected_simulation_count": len(assignments),
        "workload_trace_count": len(trace_manifest),
        "workload_distribution": per_workload,
        "runs_per_workload_seed": {
            f"{workload}:{seed}": per_trace[(workload, seed)]
            for workload in WORKLOAD_TYPES
            for seed in seed_values
        },
        "configuration_run_distribution": [
            {
                "cache_size_bytes": key[0],
                "associativity": key[1],
                "block_size_bytes": key[2],
                "replacement_policy": key[3],
                "run_count": count,
            }
            for key, count in sorted(configuration_counts.items())
        ],
        "anchors": [config.to_dict() for config in ANCHOR_CONFIGS],
        "anchor_pairwise_coverage": anchor_coverage,
        "traces": trace_manifest,
        "experiments": assignments,
    }
    plan_hash = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"plan_id": plan_hash, **body}


def write_experiment_plan(plan: dict[str, Any], output_path: str | Path) -> Path:
    """Persist a generated plan manifest."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return destination


def validate_experiment_plan(plan: dict[str, Any]) -> None:
    """Verify a manifest against its deterministic generator and plan ID."""
    try:
        expected_plan = create_experiment_plan(
            accesses=int(plan["accesses"]),
            seeds=plan["seeds"],
            design_seed=int(plan["design_seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Experiment plan is missing valid generation parameters.") from exc
    if plan != expected_plan:
        raise ValueError("Experiment plan does not match its reproducible manifest.")


def execute_experiment_plan(
    plan_path: str | Path,
    results_dir: str | Path = SIMULATION_RESULTS_DIR,
    dataset_path: str | Path = DATASET_DIR / "cache_dataset.csv",
    rtl_dir: str | Path | None = None,
    testbench_dir: str | Path | None = None,
    top_module: str = "cache_tb",
) -> dict[str, Any]:
    """Execute a reviewed manifest, retaining each actual simulation result."""
    from .config import RTL_DIR, TESTBENCH_DIR

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    validate_experiment_plan(plan)

    plan_workload_dir = WORKLOAD_DIR / "experiment_plans" / plan["plan_id"]
    traces: dict[tuple[str, int], tuple[Path, Path, dict[str, Any]]] = {}
    for trace_info in plan["traces"]:
        workload_type = trace_info["workload_type"]
        seed = int(trace_info["seed"])
        trace = generate_workload(workload_type, int(plan["accesses"]), seed)
        trace_id = trace_identity(trace)
        if trace_id != trace_info["trace_id"]:
            raise ValueError(f"Trace identity changed for {workload_type} seed {seed}.")
        trace_name = f"{workload_type}_seed{seed}_{trace_id[:12]}"
        csv_path, mem_path = export_trace(
            trace,
            csv_path=plan_workload_dir / f"{trace_name}.csv",
            memory_path=plan_workload_dir / f"{trace_name}.mem",
        )
        features = extract_workload_features(trace)
        features.update({"trace_id": trace_id, "seed": seed})
        traces[(workload_type, seed)] = (csv_path, mem_path, features)

    results: list[dict[str, Any]] = []
    for experiment in plan["experiments"]:
        trace_key = (experiment["workload_type"], int(experiment["seed"]))
        _, mem_path, features = traces[trace_key]
        cache_config = CacheConfig(**experiment["cache_config"])
        result = run_simulation(
            trace_file=mem_path,
            cache_config=cache_config,
            rtl_dir=rtl_dir or RTL_DIR,
            top_module=top_module,
            result_dir=results_dir,
            workload_features=features,
            trace_id=experiment["trace_id"],
            testbench_dir=testbench_dir or TESTBENCH_DIR,
            plan_id=plan["plan_id"],
        )
        results.append(result)
        if result.get("status") != "success" or result.get("validated") is not True:
            return {
                "status": "failed",
                "completed_simulations": len(results),
                "expected_simulations": plan["expected_simulation_count"],
                "failed_result": result,
                "dataset_path": str(dataset_path),
            }

    dataset = build_dataset(results_dir=results_dir, output_path=dataset_path)
    report_path = Path(dataset.attrs["quality_report_path"])
    return {
        "status": "success",
        "completed_simulations": len(results),
        "expected_simulations": plan["expected_simulation_count"],
        "dataset_path": str(dataset_path),
        "dataset_record_count": len(dataset),
        "quality_report_path": str(report_path),
    }
