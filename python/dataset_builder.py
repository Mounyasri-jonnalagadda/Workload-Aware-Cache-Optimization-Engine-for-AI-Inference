"""Build a training table exclusively from successful real simulations."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .config import DATASET_DIR, SIMULATION_RESULTS_DIR

MEASUREMENT_COLUMNS = [
    "trace_id",
    "workload_type",
    "cache_size_bytes",
    "associativity",
    "replacement_policy",
    "block_size_bytes",
    "read_write_ratio",
    "num_accesses",
    "temporal_locality",
    "spatial_locality",
    "hit_rate",
    "miss_rate",
    "estimated_amat",
    "hit_count",
    "miss_count",
    "access_count",
    "expected_access_count",
    "actual_access_count",
]

PROVENANCE_COLUMNS = [
    "run_id",
    "seed",
    "measurement_source",
    "synthetic",
    "validated",
    "result_file",
]

DATASET_COLUMNS = MEASUREMENT_COLUMNS + PROVENANCE_COLUMNS
PLAN_DATASET_PATH = DATASET_DIR / "cache_dataset_plan_396.csv"
PLAN_QUALITY_REPORT_PATH = DATASET_DIR / "cache_dataset_plan_396.quality.json"
PLAN_PROVENANCE_COLUMNS = [
    "plan_id",
    "accesses_per_run",
    "source_run_count",
    "source_run_ids",
    "source_result_files",
    "plan_id_provenance",
    "legacy_source_run_count",
]
PLAN_DATASET_COLUMNS = DATASET_COLUMNS + PLAN_PROVENANCE_COLUMNS
CONFIGURATION_COLUMNS = [
    "cache_size_bytes",
    "associativity",
    "block_size_bytes",
    "replacement_policy",
]
EXPERIMENT_KEY_COLUMNS = ["trace_id", *CONFIGURATION_COLUMNS]


def _record_rejection_reason(record: dict[str, Any]) -> str | None:
    if record.get("status") != "success":
        return "unsuccessful"
    if record.get("synthetic") is not False:
        return "synthetic_or_unmarked"
    if record.get("measurement_source") != "vivado_xsim":
        return "non_vivado_measurement"
    if record.get("validated") is not True:
        return "not_validated"
    if record.get("validation", {}).get("passed") is not True:
        return "validation_failed"

    metrics = record.get("metrics", {})
    expected = record.get("expected_access_count")
    actual = metrics.get("access_count")
    hits = metrics.get("hit_count")
    misses = metrics.get("miss_count")
    if any(type(value) is not int for value in (expected, actual, hits, misses)):
        return "missing_or_invalid_counters"
    if expected <= 0 or actual != expected or hits + misses != actual:
        return "inconsistent_counters"

    hit_rate = metrics.get("hit_rate")
    miss_rate = metrics.get("miss_rate")
    if (
        not isinstance(hit_rate, (int, float))
        or not isinstance(miss_rate, (int, float))
        or not math.isfinite(hit_rate)
        or not math.isfinite(miss_rate)
        or not 0 <= hit_rate <= 1
        or not 0 <= miss_rate <= 1
        or not math.isclose(hit_rate, hits / actual, rel_tol=1e-9, abs_tol=1e-12)
        or not math.isclose(miss_rate, misses / actual, rel_tol=1e-9, abs_tol=1e-12)
    ):
        return "inconsistent_rates"
    return None


def _pool_count_metrics(records: pd.DataFrame) -> dict[str, int | float]:
    """Pool validated transaction counters and recalculate both rates."""
    required = ("hit_count", "miss_count", "access_count")
    missing = set(required) - set(records.columns)
    if missing:
        raise ValueError(f"Cannot pool measurements without counters: {sorted(missing)}")
    if records.empty:
        raise ValueError("Cannot pool an empty measurement group.")

    totals: dict[str, int | float] = {}
    values_by_column: dict[str, list[int]] = {}
    optional = ("expected_access_count", "actual_access_count")
    for column in (*required, *(name for name in optional if name in records.columns)):
        numeric = pd.to_numeric(records[column], errors="coerce")
        if numeric.isna().any() or any(
            not math.isfinite(float(value)) or float(value) != int(value)
            for value in numeric
        ):
            raise ValueError(f"Measurement counter {column} contains invalid values.")
        values = [int(value) for value in numeric]
        if any(value < 0 for value in values):
            raise ValueError(f"Measurement counter {column} cannot be negative.")
        values_by_column[column] = values
        totals[column] = sum(values)

    hits = values_by_column["hit_count"]
    misses = values_by_column["miss_count"]
    accesses = values_by_column["access_count"]
    if any(access <= 0 for access in accesses):
        raise ValueError("Every source measurement must contain accesses.")
    if any(hit + miss != access for hit, miss, access in zip(hits, misses, accesses)):
        raise ValueError("A source measurement has inconsistent hit/miss counters.")
    if "actual_access_count" in values_by_column and (
        values_by_column["actual_access_count"] != accesses
    ):
        raise ValueError("actual_access_count does not match access_count.")
    if "expected_access_count" in values_by_column and (
        values_by_column["expected_access_count"] != accesses
    ):
        raise ValueError("expected_access_count does not match access_count.")

    total_accesses = totals["access_count"]
    totals["hit_rate"] = totals["hit_count"] / total_accesses
    totals["miss_rate"] = totals["miss_count"] / total_accesses
    return totals


def _native(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _duplicate_groups(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    counts = frame.groupby(list(columns), dropna=False).size().reset_index(name="record_count")
    duplicates = counts.loc[counts["record_count"] > 1]
    return [
        {
            **{column: _native(row[column]) for column in columns},
            "record_count": int(row["record_count"]),
            "extra_records": int(row["record_count"]) - 1,
        }
        for _, row in duplicates.iterrows()
    ]


def dataset_quality_report(
    frame: pd.DataFrame,
    source_record_count: int,
    excluded_by_reason: dict[str, int],
) -> dict[str, Any]:
    """Summarize row validity, duplicate experiments, and configuration coverage."""
    exact_duplicates = _duplicate_groups(frame, MEASUREMENT_COLUMNS)
    repeated_experiments = _duplicate_groups(frame, EXPERIMENT_KEY_COLUMNS)

    distributions: dict[str, dict[str, int]] = {}
    for column in (
        "workload_type",
        "cache_size_bytes",
        "associativity",
        "block_size_bytes",
        "replacement_policy",
    ):
        counts = frame[column].value_counts(dropna=False)
        distributions[column] = {
            "<null>" if pd.isna(value) else str(value): int(count)
            for value, count in counts.items()
        }

    unique_configurations = (
        int(frame[CONFIGURATION_COLUMNS].drop_duplicates().shape[0])
        if not frame.empty
        else 0
    )
    unique_experiments = (
        int(frame[EXPERIMENT_KEY_COLUMNS].drop_duplicates().shape[0])
        if not frame.empty
        else 0
    )
    return {
        "status": "complete",
        "source_record_count": source_record_count,
        "included_validated_records": int(len(frame)),
        "excluded_record_count": int(sum(excluded_by_reason.values())),
        "excluded_by_reason": excluded_by_reason,
        "distinct_trace_count": int(frame["trace_id"].nunique(dropna=True)),
        "distinct_configuration_count": unique_configurations,
        "distinct_trace_configuration_count": unique_experiments,
        "exact_duplicate_groups": exact_duplicates,
        "exact_duplicate_extra_rows": sum(group["extra_records"] for group in exact_duplicates),
        "repeated_trace_configuration_groups": repeated_experiments,
        "repeated_experiment_extra_rows": sum(
            group["extra_records"] for group in repeated_experiments
        ),
        "missing_values_by_column": {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if count
        },
        "distributions": distributions,
        "count_consistency": {
            "valid_rows": int(len(frame)),
            "invalid_rows": 0,
            "checks": [
                "expected_access_count equals access_count",
                "hit_count plus miss_count equals access_count",
                "hit_rate and miss_rate agree with measured counts",
            ],
        },
    }


def build_dataset(
    results_dir: str | Path = SIMULATION_RESULTS_DIR,
    output_path: str | Path | None = None,
    hit_time_cycles: float | None = None,
    miss_penalty_cycles: float | None = None,
    quality_report_path: str | Path | None = None,
) -> pd.DataFrame:
    """Preserve each successful measurement and write an accompanying quality report."""
    if (hit_time_cycles is None) != (miss_penalty_cycles is None):
        raise ValueError("Provide both timing assumptions or neither.")
    if hit_time_cycles is not None and (hit_time_cycles < 0 or miss_penalty_cycles < 0):
        raise ValueError("Timing assumptions must be non-negative.")

    rows: list[dict[str, Any]] = []
    excluded_by_reason: dict[str, int] = {}
    result_files = sorted(Path(results_dir).glob("*.json"))
    for path in result_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        rejection_reason = _record_rejection_reason(record)
        if rejection_reason:
            excluded_by_reason[rejection_reason] = (
                excluded_by_reason.get(rejection_reason, 0) + 1
            )
            continue

        config = record.get("cache_config", {})
        features = record.get("workload_features", {})
        metrics = record.get("metrics", {})
        hit_rate = metrics["hit_rate"]
        miss_rate = metrics["miss_rate"]
        row = {
            "trace_id": record.get("trace_id"),
            "workload_type": features.get("workload_type"),
            "cache_size_bytes": config.get("cache_size_bytes"),
            "associativity": config.get("associativity"),
            "replacement_policy": config.get("replacement_policy"),
            "block_size_bytes": config.get("block_size_bytes"),
            "read_write_ratio": features.get("read_write_ratio"),
            "num_accesses": features.get("num_accesses"),
            "temporal_locality": features.get("temporal_locality"),
            "spatial_locality": features.get("spatial_locality"),
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "estimated_amat": metrics.get("estimated_amat"),
            "hit_count": metrics["hit_count"],
            "miss_count": metrics["miss_count"],
            "access_count": metrics["access_count"],
            "expected_access_count": record["expected_access_count"],
            "actual_access_count": metrics["access_count"],
            "run_id": record.get("run_id"),
            "seed": features.get("seed", record.get("seed")),
            "measurement_source": record.get("measurement_source"),
            "synthetic": record.get("synthetic"),
            "validated": record.get("validated"),
            "result_file": record.get("result_file") or str(path),
        }
        if row["estimated_amat"] is None and hit_time_cycles is not None:
            row["estimated_amat"] = hit_time_cycles + miss_rate * miss_penalty_cycles
        rows.append(row)

    frame = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    if output_path is None:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        destination = DATASET_DIR / "cache_dataset.csv"
    else:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)

    quality = dataset_quality_report(frame, len(result_files), excluded_by_reason)
    quality_path = (
        Path(quality_report_path)
        if quality_report_path is not None
        else destination.with_name(f"{destination.stem}.quality.json")
    )
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    frame.attrs["quality_report"] = quality
    frame.attrs["quality_report_path"] = str(quality_path)
    return frame


def _plan_scenario_key(trace_id: Any, config: Any) -> tuple[Any, ...] | None:
    if not isinstance(trace_id, str) or not trace_id or not isinstance(config, dict):
        return None
    fields = (
        "cache_size_bytes",
        "associativity",
        "replacement_policy",
        "block_size_bytes",
    )
    if any(field not in config for field in fields):
        return None
    numeric_fields = (
        config["cache_size_bytes"],
        config["associativity"],
        config["block_size_bytes"],
    )
    if any(type(value) is not int for value in numeric_fields):
        return None
    if not isinstance(config["replacement_policy"], str):
        return None
    return (
        trace_id,
        config["cache_size_bytes"],
        config["associativity"],
        config["replacement_policy"].upper(),
        config["block_size_bytes"],
    )


def _plan_record_rejection_reason(
    record: dict[str, Any],
    expected_workload: str,
    expected_seed: int,
    expected_accesses: int,
) -> str | None:
    rejection = _record_rejection_reason(record)
    if rejection:
        return rejection

    features = record.get("workload_features")
    if not isinstance(features, dict):
        return "missing_workload_features"
    if features.get("workload_type") != expected_workload:
        return "workload_type_mismatch"
    if features.get("num_accesses") != expected_accesses:
        return "inconsistent_workload_access_count"

    seed = features.get("seed")
    if seed is None:
        seed = record.get("seed")
    if seed not in (None, ""):
        try:
            if int(seed) != expected_seed:
                return "workload_seed_mismatch"
        except (TypeError, ValueError):
            return "invalid_workload_seed"

    for name in ("read_write_ratio", "temporal_locality", "spatial_locality"):
        value = features.get(name)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            return "invalid_workload_features"
        if name != "read_write_ratio" and value > 1:
            return "invalid_workload_features"
    return None


def build_plan_dataset(
    plan_path: str | Path,
    results_dir: str | Path = SIMULATION_RESULTS_DIR,
    output_path: str | Path = PLAN_DATASET_PATH,
    quality_report_path: str | Path = PLAN_QUALITY_REPORT_PATH,
) -> pd.DataFrame:
    """Build one pooled, validated training row per assignment in a plan."""
    from .experiment_plan import validate_experiment_plan
    from .workload_generator import WORKLOAD_TYPES

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    validate_experiment_plan(plan)
    if plan.get("expected_simulation_count") != 396:
        raise ValueError("Plan-specific dataset preparation requires a 396-run plan.")
    if plan.get("candidate_configuration_count") != 84:
        raise ValueError("Plan must cover all 84 legal cache configurations.")
    if plan.get("accesses") != 10_000:
        raise ValueError("Plan-specific training data requires 10,000 accesses per run.")

    trace_metadata = {
        trace["trace_id"]: (trace["workload_type"], trace["seed"], trace["num_accesses"])
        for trace in plan["traces"]
    }
    if (
        len(trace_metadata) != 15
        or {item[0] for item in trace_metadata.values()} != set(WORKLOAD_TYPES)
        or any(item[2] != 10_000 for item in trace_metadata.values())
    ):
        raise ValueError("Plan must contain 15 unique 10,000-access workload traces.")

    planned: dict[tuple[Any, ...], dict[str, Any]] = {}
    for experiment in plan["experiments"]:
        config = experiment.get("cache_config")
        key = _plan_scenario_key(experiment.get("trace_id"), config)
        if key is None or key in planned:
            raise ValueError("Plan contains a missing or duplicate trace/configuration key.")
        expected_trace = trace_metadata.get(experiment["trace_id"])
        if expected_trace is None or expected_trace[:2] != (
            experiment.get("workload_type"),
            experiment.get("seed"),
        ):
            raise ValueError("Plan assignment does not match its trace metadata.")
        if config["associativity"] == 1 and config["replacement_policy"] != "LRU":
            raise ValueError("Direct-mapped plan assignments must use canonical LRU.")
        planned[key] = experiment

    workload_keys = {
        workload: {
            key
            for key, experiment in planned.items()
            if experiment["workload_type"] == workload
        }
        for workload in WORKLOAD_TYPES
    }
    configuration_keys = {key[1:] for key in planned}
    if (
        len(planned) != 396
        or any(len(keys) != 132 for keys in workload_keys.values())
        or len(configuration_keys) != 84
        or any(
            {key[1:] for key in workload_keys[workload]} != configuration_keys
            for workload in WORKLOAD_TYPES
        )
    ):
        raise ValueError("Plan assignments do not satisfy the 396-scenario coverage.")

    destination = Path(output_path)
    quality_path = Path(quality_report_path)
    protected_paths = {
        (DATASET_DIR / "cache_dataset.csv").resolve(),
        (DATASET_DIR / "cache_dataset.quality.json").resolve(),
    }
    if destination.resolve() in protected_paths or quality_path.resolve() in protected_paths:
        raise ValueError("Plan preparation must not overwrite the original dataset or report.")
    if destination.resolve() == quality_path.resolve():
        raise ValueError("Dataset CSV and quality report must use different paths.")

    excluded_by_reason: dict[str, int] = {}
    records_by_scenario: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = {}
    matching_result_count = 0
    result_files = sorted(Path(results_dir).rglob("*.json"))
    for path in result_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        key = _plan_scenario_key(record.get("trace_id"), record.get("cache_config"))
        if key not in planned:
            excluded_by_reason["not_in_plan"] = excluded_by_reason.get("not_in_plan", 0) + 1
            continue
        matching_result_count += 1

        record_plan_id = record.get("plan_id")
        if record_plan_id not in (None, "") and record_plan_id != plan["plan_id"]:
            reason = "different_plan_id"
        else:
            experiment = planned[key]
            reason = _plan_record_rejection_reason(
                record,
                experiment["workload_type"],
                experiment["seed"],
                plan["accesses"],
            )
        if reason:
            excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1
            continue
        records_by_scenario.setdefault(key, []).append((path, record))

    missing_keys = set(planned) - set(records_by_scenario)
    if missing_keys:
        examples = sorted(missing_keys, key=str)[:5]
        raise ValueError(
            f"No successful validated real result for {len(missing_keys)} planned "
            f"scenarios; examples: {examples}"
        )

    rows: list[dict[str, Any]] = []
    legacy_source_count = 0
    source_run_count_distribution: Counter[int] = Counter()
    for key, experiment in planned.items():
        sources = sorted(records_by_scenario[key], key=lambda item: str(item[1].get("run_id", item[0])))
        run_ids = [str(record.get("run_id") or path.stem) for path, record in sources]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError(f"Duplicate run IDs matched planned scenario {key}.")
        feature_signatures = {
            tuple(
                record.get("workload_features", {}).get(name)
                for name in (
                    "read_write_ratio",
                    "num_accesses",
                    "temporal_locality",
                    "spatial_locality",
                )
            )
            for _, record in sources
        }
        if len(feature_signatures) != 1:
            raise ValueError(f"Source workload features disagree for planned scenario {key}.")

        count_frame = pd.DataFrame(
            [
                {
                    "hit_count": record["metrics"]["hit_count"],
                    "miss_count": record["metrics"]["miss_count"],
                    "access_count": record["metrics"]["access_count"],
                    "expected_access_count": record["expected_access_count"],
                    "actual_access_count": record["metrics"]["access_count"],
                }
                for _, record in sources
            ]
        )
        pooled = _pool_count_metrics(count_frame)
        first_path, first_record = sources[0]
        features = first_record["workload_features"]
        explicit_source_count = sum(
            record.get("plan_id") == plan["plan_id"] for _, record in sources
        )
        legacy_count = len(sources) - explicit_source_count
        legacy_source_count += legacy_count
        source_run_count_distribution[len(sources)] += 1

        rows.append(
            {
                "trace_id": experiment["trace_id"],
                "workload_type": experiment["workload_type"],
                "cache_size_bytes": experiment["cache_config"]["cache_size_bytes"],
                "associativity": experiment["cache_config"]["associativity"],
                "replacement_policy": experiment["cache_config"]["replacement_policy"],
                "block_size_bytes": experiment["cache_config"]["block_size_bytes"],
                "read_write_ratio": features["read_write_ratio"],
                "num_accesses": plan["accesses"],
                "temporal_locality": features["temporal_locality"],
                "spatial_locality": features["spatial_locality"],
                "hit_rate": pooled["hit_rate"],
                "miss_rate": pooled["miss_rate"],
                "estimated_amat": first_record.get("metrics", {}).get("estimated_amat"),
                "hit_count": pooled["hit_count"],
                "miss_count": pooled["miss_count"],
                "access_count": pooled["access_count"],
                "expected_access_count": pooled["expected_access_count"],
                "actual_access_count": pooled["actual_access_count"],
                "run_id": run_ids[0],
                "seed": experiment["seed"],
                "measurement_source": "vivado_xsim",
                "synthetic": False,
                "validated": True,
                "result_file": str(first_path),
                "plan_id": plan["plan_id"],
                "accesses_per_run": plan["accesses"],
                "source_run_count": len(sources),
                "source_run_ids": json.dumps(run_ids),
                "source_result_files": json.dumps([str(path) for path, _ in sources]),
                "plan_id_provenance": (
                    "legacy_key_match"
                    if explicit_source_count == 0
                    else "mixed"
                    if legacy_count
                    else "explicit_plan_id"
                ),
                "legacy_source_run_count": legacy_count,
            }
        )

    frame = pd.DataFrame(rows, columns=PLAN_DATASET_COLUMNS)
    if len(frame) != 396 or frame.duplicated(EXPERIMENT_KEY_COLUMNS).any():
        raise ValueError("Prepared dataset is not exactly 396 unique planned scenarios.")
    if (
        set(frame["num_accesses"]) != {10_000}
        or not (frame["access_count"] == frame["actual_access_count"]).all()
        or not (frame["actual_access_count"] == frame["expected_access_count"]).all()
        or not (frame["hit_count"] + frame["miss_count"] == frame["access_count"]).all()
        or not all(
            math.isclose(row.hit_rate, row.hit_count / row.access_count)
            and math.isclose(row.miss_rate, row.miss_count / row.access_count)
            for row in frame.itertuples()
        )
        or not frame["synthetic"].eq(False).all()
        or not frame["validated"].eq(True).all()
        or set(frame["measurement_source"]) != {"vivado_xsim"}
    ):
        raise ValueError("Prepared scenario rows failed final measurement validation.")

    quality = dataset_quality_report(frame, len(result_files), excluded_by_reason)
    quality.update(
        {
            "schema_version": 1,
            "status": "complete",
            "plan_id": plan["plan_id"],
            "plan_file": str(Path(plan_path)),
            "planned_assignment_count": len(planned),
            "prepared_scenario_count": len(frame),
            "matching_result_record_count": matching_result_count,
            "accepted_source_run_count": int(frame["source_run_count"].sum()),
            "legacy_source_run_count": legacy_source_count,
            "source_run_count_distribution": {
                str(count): scenarios
                for count, scenarios in sorted(source_run_count_distribution.items())
            },
            "workload_scenario_counts": {
                workload: int(count)
                for workload, count in frame["workload_type"].value_counts().items()
            },
            "accesses_per_run": plan["accesses"],
            "aggregation": {
                "scenario_key": ["trace_id", *CONFIGURATION_COLUMNS],
                "pooled_counters": ["hit_count", "miss_count", "access_count"],
                "rates": "recalculated from pooled counters",
                "num_accesses_feature": "retained as accesses per source run",
                "source_run_count_is_model_feature": False,
            },
            "plan_id_provenance": {
                "explicit_plan_id_runs": int(
                    frame["source_run_count"].sum() - legacy_source_count
                ),
                "legacy_exact_key_match_runs": legacy_source_count,
            },
            "counter_consistency": {
                "valid_scenarios": len(frame),
                "invalid_scenarios": 0,
                "each_source_run_accesses": plan["accesses"],
                "pooled_expected_actual_access_counts_match": True,
                "hit_plus_miss_equals_pooled_access_count": True,
            },
        }
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    frame.attrs["quality_report"] = quality
    frame.attrs["quality_report_path"] = str(quality_path)
    return frame
