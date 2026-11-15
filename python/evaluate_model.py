"""Evaluate saved predictors on their held-out workload traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .train_model import (
    MINIMUM_HOLDOUT_TRACE_GROUPS,
    _aggregate_repeated_experiments,
    _scores,
)


def evaluate_model(
    dataset_path: str | Path,
    model_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score only the grouped holdout traces stored with the trained model."""
    bundle = joblib.load(model_path)
    frame = pd.read_csv(dataset_path)
    training_report = bundle.get("training_report", {})
    training_ids = set(
        str(trace_id)
        for trace_id in bundle.get(
            "training_trace_ids", training_report.get("training_trace_ids", [])
        )
    )
    test_ids = set(
        str(trace_id)
        for trace_id in bundle.get(
            "test_trace_ids", training_report.get("test_trace_ids", [])
        )
    )
    if len(test_ids) < MINIMUM_HOLDOUT_TRACE_GROUPS:
        raise ValueError(
            "Model needs at least two held-out trace IDs for grouped evaluation."
        )
    if not training_ids:
        raise ValueError("Model bundle is missing training trace IDs.")
    overlap = training_ids & test_ids
    if overlap:
        raise ValueError(
            "Training and testing trace IDs overlap: " + ", ".join(sorted(overlap))
        )

    trace_ids = frame["trace_id"].astype(str)
    training_data = _aggregate_repeated_experiments(
        frame.loc[trace_ids.isin(training_ids)]
    )
    holdout = _aggregate_repeated_experiments(frame.loc[trace_ids.isin(test_ids)])
    if set(training_data["trace_id"].astype(str)) != training_ids:
        raise ValueError("Dataset is missing one or more training trace groups.")
    if set(holdout["trace_id"].astype(str)) != test_ids:
        raise ValueError("Dataset is missing one or more held-out trace groups.")

    recorded_training_count = training_report.get("training_record_count")
    if (
        recorded_training_count is not None
        and int(recorded_training_count) != len(training_data)
    ):
        raise ValueError("Training sample count does not match the model bundle.")
    recorded_test_count = training_report.get("test_record_count")
    if recorded_test_count is not None and int(recorded_test_count) != len(holdout):
        raise ValueError("Testing sample count does not match the model bundle.")
    scores: dict[str, dict[str, float | None]] = {}
    hit_model = bundle["models"].get("hit_rate")
    if hit_model is not None:
        valid = holdout["hit_rate"].notna()
        if valid.any():
            actual_hit = holdout.loc[valid, "hit_rate"]
            predicted_hit = hit_model.predict(
                holdout.loc[valid, bundle["feature_columns"]]
            )
            scores["hit_rate"] = _scores(actual_hit, predicted_hit)
            scores["derived_miss_rate"] = _scores(1.0 - actual_hit, 1.0 - predicted_hit)
    report = {
        "status": training_report["status"],
        "distinct_workload_traces": training_report["distinct_workload_traces"],
        "distinct_holdout_traces": training_report.get(
            "distinct_holdout_traces", len(test_ids)
        ),
        "training_sample_count": len(training_data),
        "testing_sample_count": len(holdout),
        "training_trace_group_count": len(training_ids),
        "testing_trace_group_count": len(test_ids),
        "train_test_trace_ids_disjoint": True,
        "training_distinct_configuration_count": training_report.get(
            "training_distinct_configuration_count"
        ),
        "test_distinct_configuration_count": training_report.get(
            "test_distinct_configuration_count"
        ),
        "metrics": scores,
        "training_trace_ids": sorted(training_ids),
        "evaluated_trace_ids": sorted(test_ids),
        "note": training_report["evaluation_note"],
    }
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
