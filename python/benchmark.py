"""Run validated baseline-versus-ML cache benchmarks using real simulations."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .cache_optimizer import generate_candidates, recommend_cache_configuration
from .config import (
    DATASET_DIR,
    HIT_TIME_CYCLES,
    METRICS_DIR,
    MISS_PENALTY_CYCLES,
    MODELS_DIR,
    SIMULATION_RESULTS_DIR,
    CacheConfig,
)
from .feature_engineering import extract_workload_features, trace_identity
from .trace_exporter import OPERATION_ENCODING
from .vivado_runner import count_trace_records, run_simulation
from .workload_generator import TRACE_COLUMNS, WORKLOAD_TYPES

BASELINE_CONFIG = CacheConfig(8192, 1, "LRU", 64)
FEATURE_VALIDATION_KEYS = (
    "workload_type",
    "read_write_ratio",
    "num_accesses",
    "temporal_locality",
    "spatial_locality",
    "trace_id",
)


def _validate_workload_inputs(
    trace_file: str | Path,
    workload_csv_path: str | Path,
    supplied_features: dict[str, Any] | None,
) -> tuple[dict[str, Any], int, str]:
    csv_path = Path(workload_csv_path)
    frame = pd.read_csv(csv_path)
    missing_columns = set(TRACE_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Workload CSV is missing required trace columns: {sorted(missing_columns)}"
        )
    if frame.empty:
        raise ValueError("Workload CSV contains no accesses.")

    sequence = pd.to_numeric(frame["sequence"], errors="coerce")
    if (
        sequence.isna().any()
        or not sequence.map(lambda value: float(value).is_integer()).all()
        or sequence.astype("int64").tolist() != list(range(len(frame)))
    ):
        raise ValueError("Workload CSV sequence must be the ordered range 0..N-1.")

    operations = frame["operation"].astype(str).str.upper()
    if not operations.isin(OPERATION_ENCODING).all():
        raise ValueError("Workload CSV operations must be R or W.")
    workload_types = frame["workload_type"].astype(str).str.strip().str.lower()
    if (
        workload_types.eq("").any()
        or workload_types.nunique() != 1
        or workload_types.iloc[0] not in WORKLOAD_TYPES
    ):
        raise ValueError("Workload CSV must contain one supported workload type.")

    addresses = pd.to_numeric(frame["address"], errors="coerce")
    if addresses.isna().any():
        raise ValueError("Workload CSV contains an invalid address.")
    canonical_addresses: list[int] = []
    for row_index, address in enumerate(addresses):
        if not math.isfinite(float(address)):
            raise ValueError(f"Workload CSV address at sequence {row_index} is not finite.")
        value = int(address)
        if (
            value != address
            or value < 0
            or value > 0xFFFFFFFF
            or value % 4 != 0
        ):
            raise ValueError(
                f"Workload CSV address at sequence {row_index} must be a "
                "word-aligned 32-bit integer."
            )
        canonical_addresses.append(value)

    csv_accesses = [
        (OPERATION_ENCODING[operation], address)
        for operation, address in zip(operations.tolist(), canonical_addresses)
    ]
    expected_access_count = count_trace_records(trace_file)
    mem_accesses: list[tuple[int, int]] = []
    with Path(trace_file).open("r", encoding="ascii") as trace:
        for raw_line in trace:
            fields = raw_line.split()
            if fields:
                mem_accesses.append((int(fields[0]), int(fields[1], 16)))

    if len(csv_accesses) != expected_access_count:
        raise ValueError(
            "Workload CSV and MEM trace access counts differ: "
            f"{len(csv_accesses)} != {expected_access_count}."
        )
    for index, (csv_access, mem_access) in enumerate(zip(csv_accesses, mem_accesses)):
        if csv_access != mem_access:
            raise ValueError(
                f"Workload CSV and MEM trace differ at access {index}: "
                f"{csv_access} != {mem_access}."
            )

    canonical_frame = frame.copy()
    canonical_frame["operation"] = operations
    canonical_frame["workload_type"] = workload_types
    canonical_frame["address"] = canonical_addresses
    canonical_frame["sequence"] = sequence.astype("int64")
    features = extract_workload_features(canonical_frame)
    trace_id = trace_identity(frame)
    canonical_text = "\n".join(
        f"{operation} {address:08X}" for operation, address in csv_accesses
    )
    trace_hash = hashlib.sha256(canonical_text.encode("ascii")).hexdigest()
    features["trace_id"] = trace_id
    features["trace_hash"] = trace_hash

    if supplied_features:
        for key in FEATURE_VALIDATION_KEYS:
            if key not in supplied_features:
                continue
            actual_value = supplied_features[key]
            expected_value = features[key]
            if isinstance(expected_value, (int, float)) and not isinstance(
                expected_value, bool
            ):
                try:
                    matches = math.isclose(
                        float(actual_value),
                        float(expected_value),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                except (TypeError, ValueError, OverflowError):
                    matches = False
            elif key == "workload_type":
                matches = str(actual_value).strip().lower() == expected_value
            else:
                matches = str(actual_value) == str(expected_value)
            if not matches:
                raise ValueError(
                    f"Supplied workload feature {key!r} does not match the CSV trace."
                )

    return features, expected_access_count, trace_hash


def _configuration_key(config: CacheConfig) -> tuple[int, int, int, str]:
    return (
        config.cache_size_bytes,
        config.associativity,
        config.block_size_bytes,
        config.replacement_policy,
    )


def _validate_measurement(
    result: dict[str, Any], expected_access_count: int
) -> tuple[bool, list[str], dict[str, int | float | None]]:
    errors: list[str] = []
    if type(expected_access_count) is not int or expected_access_count <= 0:
        errors.append("input expected_access_count is not a positive integer")
    if not isinstance(result, dict):
        return False, ["simulation result is not an object"], {
            "access_count": None,
            "hit_count": None,
            "miss_count": None,
            "hit_rate": None,
            "miss_rate": None,
        }
    if result.get("status") != "success":
        errors.append(f"status is {result.get('status')!r}, not 'success'")
    if result.get("synthetic") is not False:
        errors.append("result is synthetic or has no synthetic=false provenance")
    if result.get("validated") is not True:
        errors.append("result is not marked validated=true")
    if result.get("measurement_source") != "vivado_xsim":
        errors.append("measurement_source is not 'vivado_xsim'")
    result_expected_count = result.get("expected_access_count")
    if type(result_expected_count) is not int or result_expected_count < 0:
        errors.append("result expected_access_count is not a non-negative integer")
    elif result_expected_count != expected_access_count:
        errors.append("result expected_access_count does not match the input trace")
    validation = result.get("validation")
    if not isinstance(validation, dict):
        validation = {}
        errors.append("runner validation report is missing or malformed")
    if validation.get("passed") is not True:
        errors.append("runner validation did not pass")
    validation_access_count = validation.get("actual_access_count")
    if type(validation_access_count) is not int or validation_access_count < 0:
        errors.append("validated actual_access_count is not a non-negative integer")
    if result.get("errors"):
        errors.append("simulation result contains errors")

    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
        errors.append("simulation metrics are missing or malformed")
    counts: dict[str, int | float | None] = {}
    for name in ("access_count", "hit_count", "miss_count"):
        value = metrics.get(name)
        counts[name] = value if type(value) is int else None
        if type(value) is not int or value < 0:
            errors.append(f"{name} is missing or is not a non-negative integer")

    access_count = counts["access_count"]
    hit_count = counts["hit_count"]
    miss_count = counts["miss_count"]
    counters_valid = all(
        type(value) is int and value >= 0
        for value in (access_count, hit_count, miss_count)
    )
    if counters_valid:
        if access_count != expected_access_count:
            errors.append("access_count does not match the input trace")
        if hit_count + miss_count != access_count:
            errors.append("hit_count + miss_count does not equal access_count")
        if (
            type(validation_access_count) is int
            and validation_access_count != access_count
        ):
            errors.append("validated actual_access_count does not match access_count")

    rates: dict[str, int | float | None] = {}
    for name, count_name in (("hit_rate", "hit_count"), ("miss_rate", "miss_count")):
        rate = metrics.get(name)
        rates[name] = rate if type(rate) in (int, float) else None
        if (
            type(rate) not in (int, float)
            or not math.isfinite(float(rate))
            or rate < 0
            or rate > 1
        ):
            errors.append(f"{name} is missing or outside [0, 1]")
        elif counters_valid and access_count:
            expected_rate = counts[count_name] / access_count
            if not math.isclose(float(rate), expected_rate, rel_tol=1e-9, abs_tol=1e-12):
                errors.append(f"{name} does not match the counters")

    return not errors, errors, {**counts, **rates}


def _timestamped_path(path: Path, token: str, suffix: str | None = None) -> Path:
    extension = suffix if suffix is not None else path.suffix
    return path.with_name(f"{path.stem}_{token}{extension}")


def _allocate_output_paths(
    output_path: str | Path | None,
    comparison_path: str | Path | None,
    result_dir: str | Path,
    protected_paths: set[Path],
) -> tuple[Path, Path, Path, Path]:
    base_csv = Path(output_path) if output_path is not None else METRICS_DIR / "benchmark.csv"
    if base_csv.resolve() in protected_paths:
        raise ValueError("Benchmark output path points to a protected model or dataset.")
    if base_csv.suffix and base_csv.suffix.lower() != ".csv":
        raise ValueError("Benchmark --output must use a .csv extension.")
    if comparison_path is not None and Path(comparison_path).resolve() in protected_paths:
        raise ValueError("Comparison output path points to a protected model or dataset.")
    base_comparison = Path(comparison_path) if comparison_path is not None else None
    if base_comparison is not None and base_comparison.suffix.lower() != ".json":
        raise ValueError("Comparison output must use a .json extension.")
    if base_comparison is not None and base_comparison.resolve() in {
        base_csv.with_suffix(".json").resolve(),
        base_csv.resolve(),
    }:
        raise ValueError("Benchmark records and comparison reports need separate output paths.")

    simulation_root = Path(result_dir)
    if simulation_root.exists() and not simulation_root.is_dir():
        raise ValueError(f"Simulation results path is not a directory: {simulation_root}")
    base_csv = base_csv if base_csv.suffix else base_csv.with_suffix(".csv")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt = 0
    while True:
        token = stamp if attempt == 0 else f"{stamp}_{attempt}"
        csv_file = _timestamped_path(base_csv, token, ".csv")
        records_file = csv_file.with_suffix(".json")
        if base_comparison is None:
            comparison_file = csv_file.with_name(f"{csv_file.stem}_comparison.json")
        else:
            comparison_file = _timestamped_path(base_comparison, token, ".json")
        simulation_dir = simulation_root / f"benchmark_{token}"
        paths = (csv_file, records_file, comparison_file)
        if (
            len({path.resolve() for path in paths}) != len(paths)
            or any(path.exists() for path in paths)
            or simulation_dir.exists()
        ):
            attempt += 1
            continue
        simulation_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            simulation_dir.mkdir(exist_ok=False)
        except FileExistsError:
            attempt += 1
            continue
        return csv_file, records_file, comparison_file, simulation_dir


def _comparison_report(
    baseline_record: dict[str, Any],
    recommended_record: dict[str, Any],
    baseline_config: CacheConfig,
    recommended_config: CacheConfig,
    trace_id: str,
    trace_hash: str,
    expected_access_count: int,
) -> dict[str, Any]:
    valid = baseline_record["measurement_valid"] and recommended_record[
        "measurement_valid"
    ]
    report: dict[str, Any] = {
        "status": "complete" if valid else "unavailable",
        "comparison_valid": bool(valid),
        "improvement_claimed": False,
        "trace_id": trace_id,
        "trace_hash": trace_hash,
        "expected_access_count": expected_access_count,
        "baseline_configuration": baseline_config.to_dict(),
        "recommended_configuration": recommended_config.to_dict(),
        "baseline_actual_hit_rate": None,
        "recommended_actual_hit_rate": None,
        "predicted_recommended_hit_rate": recommended_record.get(
            "predicted_hit_rate"
        ),
        "absolute_hit_rate_improvement": None,
        "absolute_hit_rate_improvement_percentage_points": None,
        "relative_hit_rate_improvement_percentage": None,
        "baseline_actual_miss_rate": None,
        "recommended_actual_miss_rate": None,
        "miss_rate_reduction": None,
        "relative_miss_rate_reduction_percentage": None,
        "recommended_prediction_error_hit_rate": None,
        "baseline_validation_errors": baseline_record["measurement_errors"],
        "recommended_validation_errors": recommended_record["measurement_errors"],
        "baseline_run_id": baseline_record.get("run_id"),
        "recommended_run_id": recommended_record.get("run_id"),
    }
    if not valid:
        return report

    baseline_hit = float(baseline_record["actual_hit_rate"])
    recommended_hit = float(recommended_record["actual_hit_rate"])
    baseline_miss = float(baseline_record["actual_miss_rate"])
    recommended_miss = float(recommended_record["actual_miss_rate"])
    absolute_improvement = recommended_hit - baseline_hit
    miss_reduction = baseline_miss - recommended_miss
    predicted_hit = recommended_record.get("predicted_hit_rate")

    report.update(
        {
            "baseline_actual_hit_rate": baseline_hit,
            "recommended_actual_hit_rate": recommended_hit,
            "absolute_hit_rate_improvement": absolute_improvement,
            "absolute_hit_rate_improvement_percentage_points": (
                absolute_improvement * 100
            ),
            "relative_hit_rate_improvement_percentage": (
                absolute_improvement / baseline_hit * 100
                if baseline_hit > 0
                else None
            ),
            "baseline_actual_miss_rate": baseline_miss,
            "recommended_actual_miss_rate": recommended_miss,
            "miss_rate_reduction": miss_reduction,
            "relative_miss_rate_reduction_percentage": (
                miss_reduction / baseline_miss * 100 if baseline_miss > 0 else None
            ),
            "recommended_prediction_error_hit_rate": (
                abs(float(predicted_hit) - recommended_hit)
                if isinstance(predicted_hit, (int, float))
                else None
            ),
            "improvement_claimed": absolute_improvement > 0,
        }
    )
    return report


def run_benchmark(
    trace_file: str | Path,
    workload_features: dict[str, Any],
    rtl_dir: str | Path,
    top_module: str = "cache_tb",
    model_path: str | Path | None = None,
    max_cache_size_bytes: int | None = None,
    max_associativity: int | None = None,
    result_dir: str | Path = SIMULATION_RESULTS_DIR,
    output_path: str | Path | None = None,
    hit_time_cycles: float | None = HIT_TIME_CYCLES,
    miss_penalty_cycles: float | None = MISS_PENALTY_CYCLES,
    *,
    workload_csv_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compare the baseline and top-ranked ML configuration on one matched trace."""
    if not model_path:
        raise ValueError("Baseline-versus-ML benchmarking requires a trained model path.")
    if workload_csv_path is None:
        raise ValueError("A workload CSV path is required to validate the MEM trace.")
    if (hit_time_cycles is None) != (miss_penalty_cycles is None):
        raise ValueError("Provide both timing assumptions or neither.")
    if hit_time_cycles is not None and (
        hit_time_cycles < 0 or miss_penalty_cycles < 0
    ):
        raise ValueError("Timing assumptions must be non-negative.")

    validated_features, expected_access_count, trace_hash = _validate_workload_inputs(
        trace_file, workload_csv_path, workload_features
    )
    if (
        max_cache_size_bytes is not None
        and BASELINE_CONFIG.cache_size_bytes > max_cache_size_bytes
    ):
        raise ValueError(
            "The baseline cache size exceeds max_cache_size_bytes; "
            "baseline-versus-ML comparison is not possible."
        )
    if max_associativity is not None and BASELINE_CONFIG.associativity > max_associativity:
        raise ValueError(
            "The baseline associativity exceeds max_associativity; "
            "baseline-versus-ML comparison is not possible."
        )

    candidates = generate_candidates(
        max_cache_size_bytes=max_cache_size_bytes,
        max_associativity=max_associativity,
    )
    candidate_by_key = {_configuration_key(candidate): candidate for candidate in candidates}
    baseline_key = _configuration_key(BASELINE_CONFIG)
    candidate_by_key.setdefault(baseline_key, BASELINE_CONFIG)
    candidates = list(candidate_by_key.values())

    protected_paths = {
        Path(model_path).resolve(),
        (MODELS_DIR / "cache_optimizer.joblib").resolve(),
        (DATASET_DIR / "cache_dataset.csv").resolve(),
        (DATASET_DIR / "cache_dataset_plan_396.csv").resolve(),
        Path(trace_file).resolve(),
        Path(workload_csv_path).resolve(),
    }
    csv_file, records_file, comparison_file, simulation_dir = _allocate_output_paths(
        output_path, comparison_path, result_dir, protected_paths
    )
    recommendation_path = simulation_dir / "benchmark_recommendation.json"
    recommendation = recommend_cache_configuration(
        validated_features,
        model_path,
        output_path=recommendation_path,
        candidates=candidates,
    )
    ranking = recommendation["ranking"]
    if not ranking:
        raise ValueError("The optimizer returned no ranked cache candidates.")
    selected = ranking[0]
    recommended_config = CacheConfig(
        selected["cache_size_bytes"],
        selected["associativity"],
        selected["replacement_policy"],
        selected["block_size_bytes"],
    )
    recommended_key = _configuration_key(recommended_config)
    if recommended_key not in candidate_by_key:
        raise ValueError("The model recommendation is not in the benchmark candidate set.")
    predictions: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for row in ranking:
        config = CacheConfig(
            row["cache_size_bytes"],
            row["associativity"],
            row["replacement_policy"],
            row["block_size_bytes"],
        )
        predictions[_configuration_key(config)] = row

    simulation_configs = [BASELINE_CONFIG]
    if recommended_key != baseline_key:
        simulation_configs.append(recommended_config)

    rows: list[dict[str, Any]] = []
    for config in simulation_configs:
        key = _configuration_key(config)
        try:
            result = run_simulation(
                trace_file=trace_file,
                cache_config=config,
                rtl_dir=rtl_dir,
                top_module=top_module,
                result_dir=simulation_dir,
                workload_features=validated_features,
                trace_id=str(validated_features["trace_id"]),
            )
            if not isinstance(result, dict):
                raise TypeError("Simulation runner returned a non-object result.")
        except Exception as exc:
            result = {
                "run_id": None,
                "status": "exception",
                "synthetic": None,
                "validated": False,
                "measurement_source": None,
                "expected_access_count": expected_access_count,
                "validation": {"passed": False, "actual_access_count": None},
                "metrics": {},
                "errors": [f"{type(exc).__name__}: {exc}"],
                "result_file": None,
            }
        valid, validation_errors, measured = _validate_measurement(
            result, expected_access_count
        )
        prediction = predictions.get(key, {})
        access_count = measured["access_count"]
        hit_count = measured["hit_count"]
        miss_count = measured["miss_count"]
        result_metrics = result.get("metrics")
        if not isinstance(result_metrics, dict):
            result_metrics = {}
        actual_hit_rate = (
            hit_count / access_count
            if valid and isinstance(access_count, int) and access_count > 0
            else None
        )
        actual_miss_rate = (
            miss_count / access_count
            if valid and isinstance(access_count, int) and access_count > 0
            else None
        )
        estimated_amat = result_metrics.get("estimated_amat")
        if (
            estimated_amat is None
            and actual_miss_rate is not None
            and hit_time_cycles is not None
            and miss_penalty_cycles is not None
        ):
            estimated_amat = hit_time_cycles + actual_miss_rate * miss_penalty_cycles
        validation = result.get("validation")
        if not isinstance(validation, dict):
            validation = {}
        result_errors = result.get("errors", [])
        if isinstance(result_errors, str):
            result_errors = [result_errors]
        elif not isinstance(result_errors, (list, tuple)):
            result_errors = [str(result_errors)] if result_errors else []
        rows.append(
            {
                "run_id": result.get("run_id"),
                "trace_id": validated_features["trace_id"],
                "trace_hash": trace_hash,
                "workload_type": validated_features["workload_type"],
                **config.to_dict(),
                "status": result.get("status"),
                "synthetic": result.get("synthetic"),
                "validated": result.get("validated"),
                "validation_passed": validation.get("passed"),
                "measurement_source": result.get("measurement_source"),
                "measurement_valid": valid,
                "measurement_errors": "; ".join(validation_errors),
                "is_baseline": key == baseline_key,
                "is_ml_recommended": key == recommended_key,
                "predicted_hit_rate": prediction.get("predicted_hit_rate"),
                "predicted_miss_rate": prediction.get("predicted_miss_rate"),
                "actual_hit_rate": actual_hit_rate,
                "actual_miss_rate": actual_miss_rate,
                "access_count": access_count,
                "hit_count": hit_count,
                "miss_count": miss_count,
                "estimated_amat": estimated_amat,
                "result_file": result.get("result_file"),
                "errors": "; ".join(str(error) for error in result_errors),
            }
        )

    frame = pd.DataFrame(rows)
    baseline_rows = frame.loc[frame["is_baseline"]]
    recommended_rows = frame.loc[frame["is_ml_recommended"]]
    if len(baseline_rows) != 1 or len(recommended_rows) != 1:
        raise RuntimeError("Benchmark must identify exactly one baseline and one recommendation.")
    comparison = _comparison_report(
        baseline_rows.iloc[0].to_dict(),
        recommended_rows.iloc[0].to_dict(),
        BASELINE_CONFIG,
        recommended_config,
        str(validated_features["trace_id"]),
        trace_hash,
        expected_access_count,
    )
    comparison["model_path"] = str(model_path)
    comparison["measurement_source"] = (
        "vivado_xsim" if comparison["comparison_valid"] else None
    )

    records_report = {
        "status": "complete" if frame["measurement_valid"].all() else "partial",
        "trace_id": validated_features["trace_id"],
        "trace_hash": trace_hash,
        "expected_access_count": expected_access_count,
        "baseline_configuration": BASELINE_CONFIG.to_dict(),
        "recommended_configuration": recommended_config.to_dict(),
        "records": rows,
    }
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("x", encoding="utf-8", newline="") as output:
        frame.to_csv(output, index=False)
    with records_file.open("x", encoding="utf-8") as output:
        json.dump(records_report, output, indent=2)
    with comparison_file.open("x", encoding="utf-8") as output:
        json.dump(comparison, output, indent=2)

    frame.attrs["csv_path"] = str(csv_file)
    frame.attrs["records_report_path"] = str(records_file)
    frame.attrs["comparison_report_path"] = str(comparison_file)
    frame.attrs["simulation_results_dir"] = str(simulation_dir)
    return frame
