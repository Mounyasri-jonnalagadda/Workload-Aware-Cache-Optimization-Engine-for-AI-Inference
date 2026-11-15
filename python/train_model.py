"""Grouped train/test evaluation for random-forest cache predictors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    MODELS_DIR,
    SUPPORTED_ASSOCIATIVITIES,
    SUPPORTED_BLOCK_SIZES_BYTES,
    SUPPORTED_CACHE_SIZES_BYTES,
    SUPPORTED_REPLACEMENT_POLICIES,
    TRAINING_DIR,
    CacheConfig,
)
from .dataset_builder import _pool_count_metrics

EXPECTED_PLAN_ID = "6c753f966dd564b8e2e27d0b2a6e935ee4f3404d506ac635843401b2e6bc4205"
FEATURE_COLUMNS = [
    "workload_type",
    "cache_size_bytes",
    "associativity",
    "replacement_policy",
    "block_size_bytes",
    "read_write_ratio",
    "num_accesses",
    "temporal_locality",
    "spatial_locality",
]
TARGET_COLUMNS = ["hit_rate"]
MINIMUM_EVALUATION_GROUPS = 5
MINIMUM_HOLDOUT_TRACE_GROUPS = 2
CONFIGURATION_COLUMNS = [
    "cache_size_bytes",
    "associativity",
    "block_size_bytes",
    "replacement_policy",
]
TRAINING_PROVENANCE_COLUMNS = [
    "measurement_source",
    "synthetic",
    "validated",
    "hit_rate",
    "miss_rate",
    "hit_count",
    "miss_count",
    "access_count",
    "expected_access_count",
    "actual_access_count",
]


def _validate_training_dataset(
    frame: pd.DataFrame,
    expected_plan_id: str | None = EXPECTED_PLAN_ID,
) -> pd.DataFrame:
    """Reject datasets without real-measurement provenance and consistent metrics."""
    required = set(FEATURE_COLUMNS + ["trace_id", *TRAINING_PROVENANCE_COLUMNS])
    if expected_plan_id is not None:
        required.add("plan_id")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required training/provenance columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Training dataset contains no records.")
    if frame["trace_id"].isna().any() or frame["trace_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Training dataset contains missing or empty trace_id values.")

    if expected_plan_id is not None:
        if frame["plan_id"].isna().any() or not frame["plan_id"].astype(str).eq(expected_plan_id).all():
            raise ValueError(
                f"Training dataset must contain only the expected plan_id {expected_plan_id}."
            )
    elif "plan_id" in frame.columns and (
        frame["plan_id"].isna().any()
        or frame["plan_id"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Training dataset contains missing plan_id provenance.")

    if not frame["measurement_source"].eq("vivado_xsim").all():
        raise ValueError("Every training row must have measurement_source 'vivado_xsim'.")
    if frame["synthetic"].isna().any() or not frame["synthetic"].eq(False).all():
        raise ValueError("Training dataset must not contain synthetic or unmarked records.")
    if frame["validated"].isna().any() or not frame["validated"].eq(True).all():
        raise ValueError("Every training row must be marked validated=True.")

    rates: dict[str, pd.Series] = {}
    for column in ("hit_rate", "miss_rate"):
        rate = pd.to_numeric(frame[column], errors="coerce")
        if (
            rate.isna().any()
            or not np.isfinite(rate.to_numpy(dtype=float)).all()
            or ((rate < 0) | (rate > 1)).any()
        ):
            raise ValueError(f"{column} values must be finite and within [0, 1].")
        rates[column] = rate
        frame[column] = rate

    _pool_count_metrics(frame)
    counts = {
        column: pd.to_numeric(frame[column], errors="coerce")
        for column in ("hit_count", "miss_count", "access_count")
    }
    if not np.allclose(
        (rates["hit_rate"] + rates["miss_rate"]).to_numpy(dtype=float),
        1.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("hit_rate plus miss_rate must equal 1.")
    if not np.allclose(
        rates["hit_rate"].to_numpy(dtype=float),
        (counts["hit_count"] / counts["access_count"]).to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-12,
    ) or not np.allclose(
        rates["miss_rate"].to_numpy(dtype=float),
        (counts["miss_count"] / counts["access_count"]).to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError("Hit/miss rates do not agree with the validated counters.")
    return frame


def _validate_plan_dataset_scenarios(frame: pd.DataFrame) -> None:
    """Check that the expected plan still has its complete legal assignment set."""
    if "accesses_per_run" not in frame.columns or (
        pd.to_numeric(frame["num_accesses"], errors="coerce").ne(10_000).any()
        or pd.to_numeric(frame["accesses_per_run"], errors="coerce").ne(10_000).any()
    ):
        raise ValueError("Plan-specific training rows must retain 10,000 accesses per run.")

    legal_configurations: set[tuple[int, int, int, str]] = set()
    for size in SUPPORTED_CACHE_SIZES_BYTES:
        for associativity in SUPPORTED_ASSOCIATIVITIES:
            policies = ("LRU",) if associativity == 1 else SUPPORTED_REPLACEMENT_POLICIES
            for block_size in SUPPORTED_BLOCK_SIZES_BYTES:
                for policy in policies:
                    try:
                        config = CacheConfig(size, associativity, policy, block_size)
                    except ValueError:
                        continue
                    legal_configurations.add(
                        (
                            config.cache_size_bytes,
                            config.associativity,
                            config.block_size_bytes,
                            config.replacement_policy,
                        )
                    )

    if len(frame) != 396 or frame["trace_id"].nunique() != 15:
        raise ValueError("Plan-specific training data must contain 396 scenarios across 15 traces.")
    workload_counts = frame["workload_type"].value_counts().to_dict()
    if workload_counts != {"cnn": 132, "transformer": 132, "matmul": 132}:
        raise ValueError("Plan-specific training data must contain 132 scenarios per workload.")

    observed_global: set[tuple[int, int, int, str]] = set()
    for values in frame[CONFIGURATION_COLUMNS].drop_duplicates().itertuples(
        index=False, name=None
    ):
        try:
            config = CacheConfig(
                int(values[0]), int(values[1]), str(values[3]), int(values[2])
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Plan dataset contains an invalid cache configuration: {values}") from exc
        key = (
            config.cache_size_bytes,
            config.associativity,
            config.block_size_bytes,
            config.replacement_policy,
        )
        if config.associativity == 1 and config.replacement_policy != "LRU":
            raise ValueError("Direct-mapped plan configurations must use canonical LRU.")
        observed_global.add(key)
    if observed_global != legal_configurations:
        raise ValueError("Plan dataset does not cover the complete legal cache configuration space.")

    for workload in ("cnn", "transformer", "matmul"):
        observed = {
            (
                int(row.cache_size_bytes),
                int(row.associativity),
                int(row.block_size_bytes),
                str(row.replacement_policy),
            )
            for row in frame.loc[frame["workload_type"] == workload, CONFIGURATION_COLUMNS]
            .drop_duplicates()
            .itertuples(index=False)
        }
        if observed != legal_configurations:
            raise ValueError(f"Plan dataset does not cover all configurations for {workload}.")


def _split_by_trace(
    frame: pd.DataFrame, test_size: float
) -> tuple[np.ndarray, np.ndarray]:
    groups = frame["trace_id"].astype(str)
    if groups.nunique() < 2:
        raise ValueError("At least two distinct workload traces are required for grouped evaluation.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    return next(splitter.split(frame, groups=groups))


def _aggregate_repeated_experiments(frame: pd.DataFrame) -> pd.DataFrame:
    """Give each trace/configuration scenario one training row."""
    scenario_columns = ["trace_id", *CONFIGURATION_COLUMNS]
    grouped = frame.groupby(scenario_columns, dropna=False, sort=False)
    rows = []
    for _, repeated in grouped:
        row = repeated.iloc[0].copy()
        if {"hit_count", "miss_count", "access_count"}.issubset(repeated.columns):
            row.update(_pool_count_metrics(repeated))
        else:
            row["hit_rate"] = float(pd.to_numeric(repeated["hit_rate"]).mean())
            if "miss_rate" in repeated.columns:
                row["miss_rate"] = 1.0 - row["hit_rate"]
        rows.append(row)
    return pd.DataFrame(rows, columns=frame.columns).reset_index(drop=True)


def _distinct_configuration_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame[CONFIGURATION_COLUMNS].drop_duplicates().shape[0])


def _evaluation_status(distinct_trace_count: int, distinct_test_trace_count: int) -> str:
    if (
        distinct_trace_count >= MINIMUM_EVALUATION_GROUPS
        and distinct_test_trace_count >= MINIMUM_HOLDOUT_TRACE_GROUPS
    ):
        return "evaluated"
    return "insufficient_data"


def _make_model(frame: pd.DataFrame, features: list[str]) -> Pipeline:
    categorical = [
        name for name in features if not pd.api.types.is_numeric_dtype(frame[name])
    ]
    numeric = [name for name in features if name not in categorical]
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    forest = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=1
    )
    return Pipeline([("preprocess", preprocess), ("regressor", forest)])


def _scores(actual: pd.Series, predicted: np.ndarray) -> dict[str, float | None]:
    if len(actual) == 0:
        return {"mae": None, "rmse": None, "r2": None}
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": rmse,
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def _workload_stratified_folds(frame: pd.DataFrame, seed: int) -> list[list[str]]:
    """Assign one trace from each workload to every fold."""
    trace_workloads = frame[["trace_id", "workload_type"]].copy()
    trace_workloads["trace_id"] = trace_workloads["trace_id"].astype(str)
    if trace_workloads.groupby("trace_id")["workload_type"].nunique().gt(1).any():
        raise ValueError("Each trace_id must belong to exactly one workload.")
    trace_workloads = trace_workloads.drop_duplicates()
    expected_workload_counts = {"cnn": 5, "transformer": 5, "matmul": 5}
    workload_counts = trace_workloads["workload_type"].value_counts().to_dict()
    if workload_counts != expected_workload_counts:
        raise ValueError(
            "Five-fold workload-stratified CV requires five traces for each of "
            "cnn, transformer, and matmul."
        )

    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(5)]
    for workload in ("cnn", "transformer", "matmul"):
        trace_ids = sorted(
            trace_workloads.loc[
                trace_workloads["workload_type"] == workload, "trace_id"
            ].tolist()
        )
        for fold_index, trace_id in enumerate(rng.permutation(trace_ids)):
            folds[fold_index].append(str(trace_id))
    return [sorted(fold) for fold in folds]


def cross_validate_model(
    dataset_path: str | Path,
    report_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate fixed Random Forest settings with workload-stratified grouped CV."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    dataset_file = Path(dataset_path)
    report_file = (
        Path(report_path)
        if report_path is not None
        else TRAINING_DIR / "grouped_cv_report.json"
    )
    predictions_file = (
        Path(predictions_path)
        if predictions_path is not None
        else TRAINING_DIR / "grouped_cv_oof_predictions.csv"
    )
    protected_paths = {
        dataset_file.resolve(),
        (MODELS_DIR / "cache_optimizer.joblib").resolve(),
        (TRAINING_DIR / "training_metrics.json").resolve(),
        (TRAINING_DIR / "feature_importance.csv").resolve(),
    }
    if report_file.resolve() == predictions_file.resolve():
        raise ValueError("CV report and predictions must use separate output paths.")
    if report_file.resolve() in protected_paths or predictions_file.resolve() in protected_paths:
        raise ValueError(
            "CV outputs must not overwrite the input dataset, trained model, or "
            "existing training outputs."
        )

    frame = pd.read_csv(dataset_file)
    frame = _validate_training_dataset(frame, EXPECTED_PLAN_ID).reset_index(drop=True)
    frame["trace_id"] = frame["trace_id"].astype(str)
    frame = _aggregate_repeated_experiments(frame)
    _validate_plan_dataset_scenarios(frame)

    features = [name for name in FEATURE_COLUMNS if name in frame.columns]
    unusable = [name for name in features if frame[name].isna().all()]
    if unusable:
        raise ValueError(f"Feature columns contain no usable values: {unusable}")

    folds = _workload_stratified_folds(frame, seed)
    all_trace_ids = set(frame["trace_id"])
    fold_reports: list[dict[str, Any]] = []
    fold_predictions: list[pd.DataFrame] = []
    tested_trace_ids: list[str] = []
    scenario_columns = ["trace_id", *CONFIGURATION_COLUMNS]

    for fold_number, test_trace_ids in enumerate(folds, start=1):
        test_trace_set = set(test_trace_ids)
        train_trace_set = all_trace_ids - test_trace_set
        if train_trace_set & test_trace_set:
            raise RuntimeError(f"Fold {fold_number} has overlapping trace groups.")
        if len(test_trace_set) != 3 or len(train_trace_set) != 12:
            raise RuntimeError(f"Fold {fold_number} has an invalid trace-group count.")

        test_mask = frame["trace_id"].isin(test_trace_set)
        train_data = frame.loc[~test_mask]
        test_data = frame.loc[test_mask]
        train_workloads = train_data.groupby("trace_id")["workload_type"].first()
        test_workloads = test_data.groupby("trace_id")["workload_type"].first()
        if test_workloads.value_counts().to_dict() != {
            "cnn": 1,
            "transformer": 1,
            "matmul": 1,
        }:
            raise RuntimeError(f"Fold {fold_number} is not balanced by workload.")
        if set(train_workloads.index) & set(test_workloads.index):
            raise RuntimeError(f"Fold {fold_number} has overlapping trace IDs.")

        model = _make_model(train_data, features)
        model.fit(train_data[features], train_data["hit_rate"])
        predicted = np.asarray(model.predict(test_data[features]), dtype=float).reshape(-1)
        if len(predicted) != len(test_data):
            raise RuntimeError(f"Fold {fold_number} returned an incomplete prediction set.")
        if not np.isfinite(predicted).all() or ((predicted < 0) | (predicted > 1)).any():
            raise ValueError(
                f"Fold {fold_number} produced non-finite or out-of-range hit_rate predictions."
            )

        output = test_data[["plan_id", *scenario_columns, "workload_type"]].copy()
        output.insert(0, "fold", fold_number)
        output["actual_hit_rate"] = test_data["hit_rate"].to_numpy(dtype=float)
        output["predicted_hit_rate"] = predicted
        fold_predictions.append(output)
        tested_trace_ids.extend(test_trace_ids)
        fold_reports.append(
            {
                "fold": fold_number,
                "training_row_count": int(len(train_data)),
                "testing_row_count": int(len(test_data)),
                "training_trace_ids": sorted(train_trace_set),
                "testing_trace_ids": sorted(test_trace_set),
                "train_test_trace_ids_disjoint": True,
            }
        )

    if len(tested_trace_ids) != len(all_trace_ids) or set(tested_trace_ids) != all_trace_ids:
        raise RuntimeError("Every trace must appear in exactly one test fold.")
    if len(tested_trace_ids) != len(set(tested_trace_ids)):
        raise RuntimeError("A trace appears in more than one test fold.")

    predictions = pd.concat(fold_predictions, ignore_index=True)
    predictions = predictions.sort_values(
        ["trace_id", *CONFIGURATION_COLUMNS], kind="stable"
    ).reset_index(drop=True)
    expected_scenarios = set(
        frame[scenario_columns].itertuples(index=False, name=None)
    )
    predicted_scenarios = set(
        predictions[scenario_columns].itertuples(index=False, name=None)
    )
    if (
        len(predictions) != len(frame)
        or predictions.duplicated(scenario_columns).any()
        or predicted_scenarios != expected_scenarios
    ):
        raise RuntimeError("Out-of-fold predictions do not cover each scenario exactly once.")

    pooled_metrics = _scores(
        predictions["actual_hit_rate"], predictions["predicted_hit_rate"].to_numpy()
    )
    per_workload: dict[str, dict[str, Any]] = {}
    for workload, rows in predictions.groupby("workload_type", sort=True):
        per_workload[str(workload)] = {
            "scenario_count": int(len(rows)),
            "trace_count": int(rows["trace_id"].nunique()),
            **_scores(rows["actual_hit_rate"], rows["predicted_hit_rate"].to_numpy()),
        }
    per_trace: dict[str, dict[str, Any]] = {}
    for trace_id, rows in predictions.groupby("trace_id", sort=True):
        per_trace[str(trace_id)] = {
            "workload_type": str(rows["workload_type"].iloc[0]),
            "scenario_count": int(len(rows)),
            **_scores(rows["actual_hit_rate"], rows["predicted_hit_rate"].to_numpy()),
        }

    report: dict[str, Any] = {
        "status": "evaluated",
        "plan_id": EXPECTED_PLAN_ID,
        "target": "hit_rate",
        "fold_count": 5,
        "fold_assignment_seed": seed,
        "random_forest_parameters": {
            "n_estimators": 300,
            "random_state": 42,
            "n_jobs": -1,
            "min_samples_leaf": 1,
        },
        "scenario_count": int(len(frame)),
        "distinct_trace_count": int(len(all_trace_ids)),
        "out_of_fold_prediction_count": int(len(predictions)),
        "out_of_fold_predictions_complete": True,
        "all_traces_tested_exactly_once": True,
        "all_folds_train_test_trace_ids_disjoint": True,
        "pooled_metrics": pooled_metrics,
        "per_workload_metrics": per_workload,
        "per_trace_metrics": per_trace,
        "folds": fold_reports,
        "report_path": str(report_file),
        "predictions_path": str(predictions_file),
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_file, index=False)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_model(
    dataset_path: str | Path,
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    test_size: float = 0.2,
    feature_importance_path: str | Path | None = None,
    allow_unplanned_dataset: bool = False,
) -> dict[str, Any]:
    """Train target models after validating provenance and grouped scenarios.

    Non-plan datasets require the explicit ``allow_unplanned_dataset`` opt-in;
    their Vivado provenance and measurement counters are still validated.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if type(allow_unplanned_dataset) is not bool:
        raise TypeError("allow_unplanned_dataset must be a bool.")
    frame = pd.read_csv(dataset_path)
    required = set(FEATURE_COLUMNS + TARGET_COLUMNS + ["trace_id"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    raw_record_count = len(frame)
    expected_plan_id = None if allow_unplanned_dataset else EXPECTED_PLAN_ID
    frame = _validate_training_dataset(frame, expected_plan_id).reset_index(drop=True)
    identified_record_count = len(frame)
    frame = _aggregate_repeated_experiments(frame)
    if expected_plan_id is not None:
        _validate_plan_dataset_scenarios(frame)
    groups = frame["trace_id"].astype(str)
    train_indices, test_indices = _split_by_trace(frame, test_size)
    features = [name for name in FEATURE_COLUMNS if name in frame.columns]
    unusable = [name for name in features if frame[name].isna().all()]
    if unusable:
        raise ValueError(f"Feature columns contain no usable values: {unusable}")
    models: dict[str, Pipeline] = {}
    scores: dict[str, dict[str, float | None]] = {}
    importances: list[dict[str, Any]] = []
    train_data, test_data = frame.iloc[train_indices], frame.iloc[test_indices]

    for target in TARGET_COLUMNS:
        valid_train = train_data[target].notna()
        if valid_train.sum() < 2:
            continue
        model = _make_model(frame, features)
        model.fit(train_data.loc[valid_train, features], train_data.loc[valid_train, target])
        models[target] = model
        valid_test = test_data[target].notna()
        if valid_test.any() and test_data.loc[valid_test, "trace_id"].nunique() >= MINIMUM_HOLDOUT_TRACE_GROUPS:
            predicted = model.predict(test_data.loc[valid_test, features])
            scores[target] = _scores(test_data.loc[valid_test, target], predicted)
        if target == "hit_rate":
            names = model.named_steps["preprocess"].get_feature_names_out()
            values = model.named_steps["regressor"].feature_importances_
            importances = sorted(
                [
                    {"feature": str(name), "importance": float(value)}
                    for name, value in zip(names, values)
                ],
                key=lambda item: item["importance"],
                reverse=True,
            )
    if not models:
        raise ValueError("No target has enough real, non-null training values.")

    unique_groups = int(groups.nunique())
    train_trace_ids = sorted(set(groups.iloc[train_indices]))
    test_trace_ids = sorted(set(groups.iloc[test_indices]))
    status = _evaluation_status(unique_groups, len(test_trace_ids))
    sufficient_data = status == "evaluated"
    report: dict[str, Any] = {
        "status": status,
        "distinct_workload_traces": unique_groups,
        "distinct_holdout_traces": len(test_trace_ids),
        "minimum_holdout_trace_groups": MINIMUM_HOLDOUT_TRACE_GROUPS,
        "minimum_traces_for_accuracy_claim": MINIMUM_EVALUATION_GROUPS,
        "raw_dataset_records": raw_record_count,
        "records_missing_trace_id": raw_record_count - identified_record_count,
        "unique_trace_configuration_scenarios": len(frame),
        "aggregated_repeated_records": max(0, identified_record_count - len(frame)),
        "training_record_count": len(train_data),
        "test_record_count": len(test_data),
        "training_sample_count": len(train_data),
        "testing_sample_count": len(test_data),
        "training_distinct_trace_count": len(train_trace_ids),
        "test_distinct_trace_count": len(test_trace_ids),
        "training_distinct_configuration_count": _distinct_configuration_count(train_data),
        "test_distinct_configuration_count": _distinct_configuration_count(test_data),
        "evaluation_note": (
            "Grouped holdout metrics; not evidence of general accuracy."
            if not sufficient_data
            else "Metrics are from traces held out by trace_id."
        ),
        "target_policy": {
            "primary": "hit_rate",
            "miss_rate": "derived as 1 - hit_rate; not trained independently",
            "estimated_amat": "excluded until documented timing assumptions are available",
        },
        "targets": list(models),
        "metrics": scores,
        "training_trace_ids": train_trace_ids,
        "test_trace_ids": test_trace_ids,
        "feature_importance": importances,
    }
    bundle = {
        "models": models,
        "feature_columns": features,
        "targets": list(models),
        "training_trace_ids": train_trace_ids,
        "test_trace_ids": report["test_trace_ids"],
        "training_report": report,
    }
    destination = Path(model_path) if model_path else MODELS_DIR / "cache_optimizer.joblib"
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    report["model_path"] = str(destination)
    metrics_file = Path(metrics_path) if metrics_path else TRAINING_DIR / "training_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    if feature_importance_path is not None:
        importance_file = Path(feature_importance_path)
        importance_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report["feature_importance"], columns=["feature", "importance"]).to_csv(
            importance_file, index=False
        )
        report["feature_importance_path"] = str(importance_file)
    metrics_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


