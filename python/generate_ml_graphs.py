"""Generate report-ready ML graphs from project results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "training"
RESULTS = ROOT / "results"
OUT = ROOT / "reports" / "graphs"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {target}")


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def feature_importance():
    df = pd.read_csv(TRAIN / "feature_importance.csv")
    df = df.sort_values("importance")

    labels = df["feature"].str.replace(
        "categorical__", "", regex=False
    ).str.replace("numeric__", "", regex=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, df["importance"] * 100)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Feature Importance (%)")
    ax.set_ylabel("Input Feature")
    ax.grid(axis="x", alpha=0.25)

    save(fig, "01_feature_importance.png")


def actual_vs_predicted():
    df = pd.read_csv(TRAIN / "grouped_cv_oof_predictions.csv")

    fig, ax = plt.subplots(figsize=(8, 8))

    for workload, group in df.groupby("workload_type"):
        ax.scatter(
            group["actual_hit_rate"] * 100,
            group["predicted_hit_rate"] * 100,
            label=str(workload).upper(),
            alpha=0.65
        )

    lo = min(
        df.actual_hit_rate.min(),
        df.predicted_hit_rate.min()
    ) * 100

    hi = max(
        df.actual_hit_rate.max(),
        df.predicted_hit_rate.max()
    ) * 100

    ax.plot([lo, hi], [lo, hi], "--", label="Ideal Prediction")

    ax.set_title("Actual vs Predicted Cache Hit Rate")
    ax.set_xlabel("Actual Hit Rate (%)")
    ax.set_ylabel("Predicted Hit Rate (%)")
    ax.legend()
    ax.grid(alpha=0.25)

    save(fig, "02_actual_vs_predicted.png")


def prediction_error():
    df = pd.read_csv(TRAIN / "grouped_cv_oof_predictions.csv")

    errors = (
        df["predicted_hit_rate"] - df["actual_hit_rate"]
    ) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(errors, bins=25)
    ax.axvline(0, linestyle="--", label="Zero Error")

    ax.set_title("Out-of-Fold Prediction Error Distribution")
    ax.set_xlabel("Prediction Error (Percentage Points)")
    ax.set_ylabel("Number of Scenarios")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save(fig, "03_prediction_error_distribution.png")


def cv_metrics():
    metrics = read_json(
        TRAIN / "grouped_cv_report.json"
    )["pooled_metrics"]

    names = ["MAE", "RMSE"]
    values = [
        metrics["mae"] * 100,
        metrics["rmse"] * 100
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, values)

    ax.set_title("Grouped 5-Fold Cross-Validation Error")
    ax.set_ylabel("Error (Percentage Points)")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    save(fig, "04_cross_validation_error_metrics.png")

    r2 = metrics["r2"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Grouped 5-Fold CV"], [r2])

    ax.set_ylim(0, 1.05)
    ax.set_title("Grouped 5-Fold Cross-Validation R²")
    ax.set_ylabel("R² Score")
    ax.grid(axis="y", alpha=0.25)

    ax.text(
        bars[0].get_x() + bars[0].get_width() / 2,
        r2,
        f"{r2:.6f}",
        ha="center",
        va="bottom"
    )

    save(fig, "05_cross_validation_r2.png")


def baseline_vs_ml():
    rows = []

    for path in sorted(
        RESULTS.glob("*_comparison_*.json")
    ):
        try:
            item = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue

        if (
            item.get("status") != "complete"
            or item.get("comparison_valid") is not True
            or item.get("measurement_source") != "vivado_xsim"
        ):
            continue

        name = path.name.lower()

        workload = (
            "CNN" if "cnn" in name
            else "Transformer" if "transformer" in name
            else "Matmul" if "matmul" in name
            else "Workload"
        )

        seed = (
            "Seed 100" if "seed100" in name
            else "Seed 42" if "seed42" in name
            else "Benchmark"
        )

        rows.append({
            "label": f"{workload} - {seed}",
            "baseline": item["baseline_actual_hit_rate"] * 100,
            "optimized": item["recommended_actual_hit_rate"] * 100
        })

    if not rows:
        raise FileNotFoundError(
            "No valid Vivado comparison JSON files found."
        )

    df = pd.DataFrame(rows).sort_values("label")

    x = list(range(len(df)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        [v - width / 2 for v in x],
        df["baseline"],
        width,
        label="Baseline"
    )

    ax.bar(
        [v + width / 2 for v in x],
        df["optimized"],
        width,
        label="ML Recommended"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=20, ha="right")
    ax.set_ylim(0, 105)

    ax.set_title("Baseline vs ML-Recommended Cache Hit Rate")
    ax.set_ylabel("Measured Cache Hit Rate (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save(fig, "06_baseline_vs_ml_hit_rate.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    feature_importance()
    actual_vs_predicted()
    prediction_error()
    cv_metrics()
    baseline_vs_ml()

    print("\nAll graphs generated successfully!")
    print(f"Output folder: {OUT}")


if __name__ == "__main__":
    main()