"""Create Matplotlib comparisons from actual benchmark metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import PLOTS_DIR

PLOT_COLUMNS = {
    "hit_rate": ("Hit rate", "Hit rate"),
    "miss_rate": ("Miss rate", "Miss rate"),
    "estimated_amat": ("Estimated AMAT", "Estimated AMAT (cycles)"),
}


def generate_visualizations(
    benchmark_path: str | Path,
    output_dir: str | Path = PLOTS_DIR,
    feature_importance_path: str | Path | None = None,
) -> list[Path]:
    """Save workload and cache-configuration charts from measured benchmark data."""
    frame = pd.read_csv(benchmark_path)
    if "synthetic" in frame and frame["synthetic"].astype(str).str.lower().isin(["true", "1"]).any():
        raise ValueError("Synthetic data cannot be presented as benchmark measurements.")
    if "status" in frame:
        frame = frame.loc[frame["status"] == "success"]
    if frame.empty:
        raise ValueError("No successful real benchmark measurements are available to plot.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = frame.apply(
        lambda row: f"{row.get('cache_size_bytes', '?')}B / {row.get('associativity', '?')}-way\n"
        f"{row.get('replacement_policy', '')}",
        axis=1,
    )
    generated: list[Path] = []
    for column, (title, ylabel) in PLOT_COLUMNS.items():
        if column not in frame or frame[column].notna().sum() == 0:
            continue
        path = output / f"{column}_comparison.png"
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(labels, frame[column])
        ax.set(title=title, ylabel=ylabel, xlabel="Cache configuration")
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)

    if "workload_type" in frame:
        for column, (title, ylabel) in PLOT_COLUMNS.items():
            if column not in frame or frame[column].notna().sum() == 0:
                continue
            grouped = frame.groupby("workload_type", dropna=False)[column].mean()
            path = output / f"{column}_by_workload.png"
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(grouped.index.astype(str), grouped.values)
            ax.set(title=f"{title} by workload", ylabel=ylabel, xlabel="Workload")
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            generated.append(path)

    if feature_importance_path:
        importance = pd.read_csv(feature_importance_path)
        if {"feature", "importance"}.issubset(importance.columns) and not importance.empty:
            top = importance.sort_values("importance").tail(20)
            path = output / "feature_importance.png"
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(top["feature"].astype(str), top["importance"])
            ax.set(title="Random forest feature importance", xlabel="Importance")
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            generated.append(path)
    return generated
