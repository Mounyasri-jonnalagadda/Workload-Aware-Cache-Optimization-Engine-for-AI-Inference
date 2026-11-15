"""Predict and rank feasible cache configurations for a workload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .config import (
    MODELS_DIR,
    SUPPORTED_ASSOCIATIVITIES,
    SUPPORTED_BLOCK_SIZES_BYTES,
    SUPPORTED_CACHE_SIZES_BYTES,
    SUPPORTED_REPLACEMENT_POLICIES,
    CacheConfig,
)
from .train_model import FEATURE_COLUMNS


def generate_candidates(
    cache_sizes: Iterable[int] = SUPPORTED_CACHE_SIZES_BYTES,
    associativities: Iterable[int] = SUPPORTED_ASSOCIATIVITIES,
    replacement_policies: Iterable[str] = SUPPORTED_REPLACEMENT_POLICIES,
    max_cache_size_bytes: int | None = None,
    max_associativity: int | None = None,
    block_sizes: Iterable[int] = SUPPORTED_BLOCK_SIZES_BYTES,
) -> list[CacheConfig]:
    """Enumerate supported geometries and meaningful replacement policies."""
    cache_sizes = tuple(cache_sizes)
    associativities = tuple(associativities)
    replacement_policies = tuple(replacement_policies)
    block_sizes = tuple(block_sizes)
    candidates: list[CacheConfig] = []
    for size in cache_sizes:
        for ways in associativities:
            if max_cache_size_bytes is not None and size > max_cache_size_bytes:
                continue
            if max_associativity is not None and ways > max_associativity:
                continue
            policies = ("LRU",) if ways == 1 else replacement_policies
            for block_size in block_sizes:
                for policy in policies:
                    try:
                        candidates.append(CacheConfig(size, ways, policy, block_size))
                    except ValueError:
                        continue
    if not candidates:
        raise ValueError("No valid cache configurations satisfy the supplied constraints.")
    return candidates


def recommend_cache_configuration(
    workload_features: dict[str, Any],
    model_path: str | Path,
    output_path: str | Path | None = None,
    candidates: Iterable[CacheConfig] | None = None,
    max_cache_size_bytes: int | None = None,
    max_associativity: int | None = None,
) -> dict[str, Any]:
    """Rank candidates by predicted hit rate and derive the corresponding miss rate."""
    bundle = joblib.load(model_path)
    training_status = bundle.get("training_report", {}).get("status")
    if training_status != "evaluated":
        raise ValueError(
            "Cache recommendations require a model with training status 'evaluated'; "
            f"got {training_status!r}."
        )
    models = bundle["models"]
    if "hit_rate" not in models:
        raise ValueError("Model has no hit_rate predictor.")
    if candidates is None:
        options = generate_candidates(
            max_cache_size_bytes=max_cache_size_bytes,
            max_associativity=max_associativity,
        )
    else:
        options = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, CacheConfig):
                raise ValueError(
                    f"Custom candidate {index} must be a CacheConfig instance."
                )
            try:
                validated = CacheConfig(
                    candidate.cache_size_bytes,
                    candidate.associativity,
                    candidate.replacement_policy,
                    candidate.block_size_bytes,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Custom candidate {index} is not a supported cache configuration: {exc}"
                ) from exc
            if (
                validated.associativity == 1
                and validated.replacement_policy != "LRU"
            ):
                raise ValueError(
                    f"Custom candidate {index} is direct-mapped; replacement policy must be LRU."
                )
            options.append(validated)
    options = [
        item
        for item in options
        if (max_cache_size_bytes is None or item.cache_size_bytes <= max_cache_size_bytes)
        and (max_associativity is None or item.associativity <= max_associativity)
    ]
    if not options:
        raise ValueError("No candidate remains within the hardware resource constraints.")
    rows = []
    features = bundle["feature_columns"]
    for option in options:
        row = dict(workload_features)
        row.update(option.to_dict())
        missing = set(features) - set(row)
        if missing:
            raise ValueError(f"Workload features are missing: {sorted(missing)}")
        rows.append(row)
    prediction_frame = pd.DataFrame(rows)
    try:
        predicted_hit_rates = np.asarray(
            models["hit_rate"].predict(prediction_frame[features]), dtype=float
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "The hit_rate model returned non-numeric predictions; inspect or retrain the model."
        ) from exc
    if predicted_hit_rates.ndim != 1 or len(predicted_hit_rates) != len(options):
        raise ValueError(
            "The hit_rate model returned an unexpected number of predictions; "
            "inspect or retrain the model."
        )
    nonfinite_indices = np.flatnonzero(~np.isfinite(predicted_hit_rates))
    if len(nonfinite_indices):
        raise ValueError(
            "The hit_rate model returned NaN or infinite predictions for candidate "
            f"indices {nonfinite_indices.tolist()}; inspect or retrain the model."
        )
    out_of_range_indices = np.flatnonzero(
        (predicted_hit_rates < 0.0) | (predicted_hit_rates > 1.0)
    )
    if len(out_of_range_indices):
        invalid_values = predicted_hit_rates[out_of_range_indices].tolist()
        raise ValueError(
            "The hit_rate model returned predictions outside [0, 1] for candidate "
            f"indices {out_of_range_indices.tolist()}: {invalid_values}; inspect or retrain the model."
        )

    ranked = []
    for index, option in enumerate(options):
        predicted_hit_rate = float(predicted_hit_rates[index])
        ranked.append(
            {
                **option.to_dict(),
                "predicted_hit_rate": predicted_hit_rate,
                "predicted_miss_rate": 1.0 - predicted_hit_rate,
            }
        )
    ranked.sort(key=lambda item: item["predicted_hit_rate"], reverse=True)
    result = {
        "status": "recommendation",
        "objective": "hit_rate",
        "limitation": (
            "Configurations are ranked by predicted hit rate; miss rate is derived "
            "as 1 - hit rate. AMAT is not modeled."
        ),
        "hardware_constraints": {
            "max_cache_size_bytes": max_cache_size_bytes,
            "max_associativity": max_associativity,
            "resource_utilization": None,
        },
        "workload_features": workload_features,
        "ranking": ranked,
        "recommendation": ranked[0],
        "vivado_plusargs": {
            "CACHE_SIZE": ranked[0]["cache_size_bytes"],
            "ASSOCIATIVITY": ranked[0]["associativity"],
            "BLOCK_SIZE": ranked[0]["block_size_bytes"],
            "REPLACEMENT_POLICY": ranked[0]["replacement_policy"],
        },
    }
    if output_path is None:
        output = MODELS_DIR / "recommendation.json"
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = output.with_suffix(".cfg")
    config_path.write_text(
        "".join(f"{key}={value}\n" for key, value in result["vivado_plusargs"].items()),
        encoding="ascii",
    )
    result["config_file"] = str(config_path)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
