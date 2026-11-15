"""Command-line entry point for the cache-optimization workflow."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .benchmark import run_benchmark
from .cache_optimizer import recommend_cache_configuration
from .config import (
    DATASET_DIR,
    HIT_TIME_CYCLES,
    MISS_PENALTY_CYCLES,
    MODELS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    SUPPORTED_ASSOCIATIVITIES,
    SUPPORTED_BLOCK_SIZES_BYTES,
    SUPPORTED_CACHE_SIZES_BYTES,
    SUPPORTED_REPLACEMENT_POLICIES,
    RTL_DIR,
    SIMULATION_RESULTS_DIR,
    TRAINING_DIR,
    WORKLOAD_DIR,
    CacheConfig,
    ensure_project_directories,
    load_cache_config,
)
from .dataset_builder import (
    PLAN_DATASET_PATH,
    PLAN_QUALITY_REPORT_PATH,
    build_dataset,
    build_plan_dataset,
)
from .evaluate_model import evaluate_model
from .experiment_plan import (
    create_experiment_plan,
    execute_experiment_plan,
    write_experiment_plan,
)
from .feature_engineering import extract_workload_features, trace_identity
from .trace_exporter import export_trace
from .train_model import cross_validate_model, train_model
from .visualize_results import generate_visualizations
from .vivado_runner import count_trace_records, run_simulation
from .workload_generator import generate_workload, save_workload


def _read_features(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    trace = pd.read_csv(path)
    features = extract_workload_features(trace)
    features["trace_id"] = trace_identity(trace)
    return trace, features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Workload-aware cache optimization workflow."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate-workload", help="Generate a reproducible trace CSV.")
    generate.add_argument("--type", choices=("cnn", "transformer", "matmul"), required=True)
    generate.add_argument("--accesses", type=int, default=10_000)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", type=Path)

    export = commands.add_parser("export-trace", help="Export CSV and Verilog .mem files.")
    export.add_argument("csv", type=Path)
    export.add_argument("--output-csv", type=Path)
    export.add_argument("--mem", type=Path)
    export.add_argument("--address-width", type=int, default=32)
    export.add_argument("--length", type=int)

    simulate = commands.add_parser("run-vivado", help="Compile and run the RTL testbench.")
    simulate.add_argument("trace", type=Path, help="Two-column op/address .mem file.")
    simulate.add_argument("--config-file", type=Path)
    simulate.add_argument(
        "--cache-size", type=int, choices=SUPPORTED_CACHE_SIZES_BYTES, default=8192
    )
    simulate.add_argument(
        "--associativity", type=int, choices=SUPPORTED_ASSOCIATIVITIES, default=1
    )
    simulate.add_argument(
        "--policy", choices=SUPPORTED_REPLACEMENT_POLICIES, default="LRU"
    )
    simulate.add_argument(
        "--block-size", type=int, choices=SUPPORTED_BLOCK_SIZES_BYTES, default=64
    )
    simulate.add_argument("--rtl-dir", type=Path, default=RTL_DIR)
    simulate.add_argument("--testbench-dir", type=Path)
    simulate.add_argument("--top", default="cache_tb")
    simulate.add_argument("--results", type=Path, default=SIMULATION_RESULTS_DIR)
    simulate.add_argument("--workload-csv", type=Path)
    simulate.add_argument("--dry-run", action="store_true")

    experiment = commands.add_parser(
        "run-experiment",
        help="Generate a workload, simulate it, validate counters, and update the dataset.",
    )
    experiment.add_argument("--type", choices=("cnn", "transformer", "matmul"), required=True)
    experiment.add_argument("--accesses", type=int, default=10_000)
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument(
        "--cache-size", type=int, choices=SUPPORTED_CACHE_SIZES_BYTES, default=8192
    )
    experiment.add_argument(
        "--associativity", type=int, choices=SUPPORTED_ASSOCIATIVITIES, default=1
    )
    experiment.add_argument(
        "--block-size", type=int, choices=SUPPORTED_BLOCK_SIZES_BYTES, default=64
    )
    experiment.add_argument(
        "--policy", choices=SUPPORTED_REPLACEMENT_POLICIES, default="LRU"
    )
    experiment.add_argument("--rtl-dir", type=Path, default=RTL_DIR)
    experiment.add_argument("--testbench-dir", type=Path, default=None)
    experiment.add_argument("--top", default="cache_tb")
    experiment.add_argument("--results", type=Path, default=SIMULATION_RESULTS_DIR)
    experiment.add_argument("--dataset", type=Path, default=DATASET_DIR / "cache_dataset.csv")
    experiment.add_argument("--hit-time", type=float, default=HIT_TIME_CYCLES)
    experiment.add_argument("--miss-penalty", type=float, default=MISS_PENALTY_CYCLES)

    dataset = commands.add_parser("build-dataset", help="Build dataset from real simulation JSON.")
    dataset.add_argument("--results", type=Path, default=SIMULATION_RESULTS_DIR)
    dataset.add_argument("--output", type=Path)
    dataset.add_argument("--hit-time", type=float, default=HIT_TIME_CYCLES)
    dataset.add_argument("--miss-penalty", type=float, default=MISS_PENALTY_CYCLES)

    plan = commands.add_parser(
        "plan-experiments",
        help="Create a reproducible balanced plan without running simulations.",
    )
    plan.add_argument("--accesses", type=int, default=10_000)
    plan.add_argument(
        "--seeds",
        type=int,
        nargs=5,
        default=(42, 100, 200, 300, 400),
        metavar=("SEED1", "SEED2", "SEED3", "SEED4", "SEED5"),
    )
    plan.add_argument("--design-seed", type=int, default=42)
    plan.add_argument(
        "--output", type=Path, default=REPORTS_DIR / "balanced_training_plan.json"
    )

    run_plan = commands.add_parser(
        "run-experiment-plan",
        help="Execute a reviewed experiment plan using actual Vivado simulations.",
    )
    run_plan.add_argument("plan", type=Path)
    run_plan.add_argument("--results", type=Path, default=SIMULATION_RESULTS_DIR)
    run_plan.add_argument("--dataset", type=Path, default=DATASET_DIR / "cache_dataset.csv")
    run_plan.add_argument("--rtl-dir", type=Path, default=RTL_DIR)
    run_plan.add_argument("--testbench-dir", type=Path, default=None)
    run_plan.add_argument("--top", default="cache_tb")

    prepare_plan_data = commands.add_parser(
        "prepare-plan-dataset",
        help="Create a validated, plan-specific pooled training dataset.",
    )
    prepare_plan_data.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reports"
        / "balanced_training_plan.json",
    )
    prepare_plan_data.add_argument("--results", type=Path, default=SIMULATION_RESULTS_DIR)
    prepare_plan_data.add_argument("--output", type=Path, default=PLAN_DATASET_PATH)
    prepare_plan_data.add_argument(
        "--quality-report", type=Path, default=PLAN_QUALITY_REPORT_PATH
    )

    train = commands.add_parser("train-model", help="Train grouped random-forest models.")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--model", type=Path, default=MODELS_DIR / "cache_optimizer.joblib")
    train.add_argument("--metrics", type=Path)
    train.add_argument(
        "--feature-importance", type=Path, default=TRAINING_DIR / "feature_importance.csv"
    )
    train.add_argument(
        "--allow-unplanned-dataset",
        action="store_true",
        help=(
            "Allow a selected dataset without the required plan ID; real Vivado "
            "provenance and counter validation still apply."
        ),
    )

    cross_validate = commands.add_parser(
        "cross-validate-model",
        help="Evaluate the fixed random forest with workload-stratified grouped CV.",
    )
    cross_validate.add_argument("--dataset", type=Path, required=True)
    cross_validate.add_argument(
        "--report", type=Path, default=TRAINING_DIR / "grouped_cv_report.json"
    )
    cross_validate.add_argument(
        "--predictions",
        type=Path,
        default=TRAINING_DIR / "grouped_cv_oof_predictions.csv",
    )
    cross_validate.add_argument("--seed", type=int, default=42)

    evaluate = commands.add_parser("evaluate-model", help="Evaluate saved holdout traces.")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, default=MODELS_DIR / "cache_optimizer.joblib")
    evaluate.add_argument("--output", type=Path)

    recommend = commands.add_parser("recommend", help="Rank cache configurations for a trace.")
    recommend.add_argument("workload_csv", type=Path)
    recommend.add_argument("--model", type=Path, default=MODELS_DIR / "cache_optimizer.joblib")
    recommend.add_argument("--output", type=Path)
    recommend.add_argument("--max-cache-size", type=int)
    recommend.add_argument("--max-associativity", type=int)

    benchmark = commands.add_parser("benchmark", help="Simulate configurations and compare ML.")
    benchmark.add_argument("trace", type=Path)
    benchmark.add_argument("--workload-csv", type=Path, required=True)
    benchmark.add_argument("--model", type=Path, required=True)
    benchmark.add_argument("--rtl-dir", type=Path, default=RTL_DIR)
    benchmark.add_argument("--top", default="cache_tb")
    benchmark.add_argument("--max-cache-size", type=int)
    benchmark.add_argument("--max-associativity", type=int)
    benchmark.add_argument("--results", type=Path, default=SIMULATION_RESULTS_DIR)
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--comparison-output", type=Path)
    benchmark.add_argument("--hit-time", type=float, default=HIT_TIME_CYCLES)
    benchmark.add_argument("--miss-penalty", type=float, default=MISS_PENALTY_CYCLES)

    visualize = commands.add_parser("visualize", help="Plot actual benchmark results.")
    visualize.add_argument("--benchmark", type=Path, required=True)
    visualize.add_argument("--output-dir", type=Path)
    visualize.add_argument("--feature-importance", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_project_directories()

    if args.command == "generate-workload":
        trace = generate_workload(args.type, args.accesses, args.seed)
        path = save_workload(trace, args.output)
        print(f"Generated {len(trace)} {args.type} accesses: {path}")
    elif args.command == "export-trace":
        csv_path, mem_path = export_trace(
            args.csv, args.output_csv, args.mem, args.address_width, args.length
        )
        print(f"CSV trace: {csv_path}\nVerilog .mem trace: {mem_path}")
    elif args.command == "run-experiment":
        trace = generate_workload(args.type, args.accesses, args.seed)
        experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        workload_csv = WORKLOAD_DIR / f"experiment_{experiment_id}.csv"
        trace_file = workload_csv.with_suffix(".mem")
        csv_path, mem_path = export_trace(
            trace,
            csv_path=workload_csv,
            memory_path=trace_file,
        )
        expected_count = count_trace_records(mem_path)
        if expected_count != len(trace):
            raise RuntimeError(
                f"Exporter wrote {expected_count} records for a {len(trace)}-access workload."
            )
        _, features = _read_features(csv_path)
        features["seed"] = args.seed
        cache_config = CacheConfig(
            args.cache_size,
            args.associativity,
            args.policy,
            args.block_size,
        )
        result = run_simulation(
            trace_file=mem_path,
            cache_config=cache_config,
            rtl_dir=args.rtl_dir,
            top_module=args.top,
            result_dir=args.results,
            workload_features=features,
            trace_id=str(features["trace_id"]),
            testbench_dir=args.testbench_dir,
        )
        metrics = result.get("metrics", {})
        print(
            f"Experiment {experiment_id}: {args.type}, seed={args.seed}, "
            f"cache={args.cache_size}B/{args.associativity}-way/"
            f"{args.block_size}B/{args.policy}"
        )
        print(f"Workload CSV: {csv_path}")
        print(f"Vivado trace: {mem_path}")
        print(
            f"Simulation status: {result['status']}; expected accesses={expected_count}; "
            f"actual accesses={metrics.get('access_count')}; "
            f"hits={metrics.get('hit_count')}; misses={metrics.get('miss_count')}"
        )
        if result["status"] != "success" or result.get("validated") is not True:
            print("Experiment failed validation; dataset was not updated.")
            if result.get("errors"):
                print("Errors: " + "; ".join(result["errors"]))
            return 2
        dataset = build_dataset(
            results_dir=args.results,
            output_path=args.dataset,
            hit_time_cycles=args.hit_time,
            miss_penalty_cycles=args.miss_penalty,
        )
        print(
            f"Validated Vivado result: {result.get('result_file')}\n"
            f"Dataset: {args.dataset} ({len(dataset)} validated records)\n"
            f"Dataset quality report: {dataset.attrs['quality_report_path']}"
        )
    elif args.command == "run-vivado":
        if args.config_file:
            config = load_cache_config(args.config_file)
        else:
            config = CacheConfig(
                args.cache_size, args.associativity, args.policy, args.block_size
            )
        features: dict[str, Any] = {}
        trace_id = args.trace.stem
        if args.workload_csv:
            _, features = _read_features(args.workload_csv)
            trace_id = str(features["trace_id"])
        result = run_simulation(
            args.trace,
            config,
            args.rtl_dir,
            args.top,
            args.results,
            features,
            trace_id,
            args.dry_run,
            testbench_dir=args.testbench_dir or None,
        )
        print(json.dumps(result, indent=2))
        if result["status"] in {"unavailable", "failed", "incomplete"}:
            return 2
    elif args.command == "build-dataset":
        frame = build_dataset(
            args.results, args.output, args.hit_time, args.miss_penalty
        )
        quality = frame.attrs["quality_report"]
        print(
            f"Built {len(frame)} validated real-simulation records; "
            f"{quality['distinct_trace_count']} traces, "
            f"{quality['distinct_configuration_count']} configurations, "
            f"{quality['exact_duplicate_extra_rows']} exact duplicate rows."
        )
        print(f"Dataset quality report: {frame.attrs['quality_report_path']}")
    elif args.command == "plan-experiments":
        plan = create_experiment_plan(args.accesses, args.seeds, args.design_seed)
        path = write_experiment_plan(plan, args.output)
        coverage = plan["anchor_pairwise_coverage"]
        print(
            f"Created {plan['design']} plan {plan['plan_id']} at {path}\n"
            f"Traces: {plan['workload_trace_count']}; "
            f"legal configurations: {plan['candidate_configuration_count']}; "
            f"simulations: {plan['expected_simulation_count']}\n"
            f"Anchor pairwise coverage: {coverage['covered_pair_count']}/"
            f"{coverage['legal_pair_count']}; "
            f"uncovered pairs: {json.dumps(coverage['uncovered_pairs'])}\n"
            f"Per-workload runs: {json.dumps(plan['workload_distribution'])}"
        )
    elif args.command == "run-experiment-plan":
        summary = execute_experiment_plan(
            args.plan,
            args.results,
            args.dataset,
            args.rtl_dir,
            args.testbench_dir,
            args.top,
        )
        print(json.dumps(summary, indent=2))
        if summary["status"] != "success":
            return 2
    elif args.command == "prepare-plan-dataset":
        frame = build_plan_dataset(
            args.plan,
            args.results,
            args.output,
            args.quality_report,
        )
        quality = frame.attrs["quality_report"]
        print(
            f"Prepared {len(frame)} pooled scenarios for plan {quality['plan_id']}; "
            f"{quality['distinct_trace_count']} traces, "
            f"{quality['distinct_configuration_count']} configurations, "
            f"{quality['accepted_source_run_count']} validated Vivado measurements."
        )
        print(
            f"Dataset: {args.output}\n"
            f"Quality report: {frame.attrs['quality_report_path']}\n"
            f"Legacy exact-key matched runs: {quality['legacy_source_run_count']}"
        )
    elif args.command == "train-model":
        report = train_model(
            args.dataset,
            args.model,
            args.metrics,
            feature_importance_path=args.feature_importance,
            allow_unplanned_dataset=args.allow_unplanned_dataset,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "cross-validate-model":
        report = cross_validate_model(
            args.dataset,
            args.report,
            args.predictions,
            args.seed,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "evaluate-model":
        report = evaluate_model(args.dataset, args.model, args.output)
        print(json.dumps(report, indent=2))
    elif args.command == "recommend":
        _, features = _read_features(args.workload_csv)
        result = recommend_cache_configuration(
            features,
            args.model,
            args.output,
            max_cache_size_bytes=args.max_cache_size,
            max_associativity=args.max_associativity,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "benchmark":
        _, features = _read_features(args.workload_csv)
        frame = run_benchmark(
            args.trace,
            features,
            args.rtl_dir,
            args.top,
            args.model,
            args.max_cache_size,
            args.max_associativity,
            args.results,
            args.output,
            args.hit_time,
            args.miss_penalty,
            workload_csv_path=args.workload_csv,
            comparison_path=args.comparison_output,
        )
        print(
            f"Completed {len(frame)} configurations; "
            f"{int(frame['measurement_valid'].sum())} valid real simulations."
        )
        print(
            f"Records: {frame.attrs['csv_path']}\n"
            f"Records report: {frame.attrs['records_report_path']}\n"
            f"Comparison: {frame.attrs['comparison_report_path']}\n"
            f"Simulation results: {frame.attrs['simulation_results_dir']}"
        )
    elif args.command == "visualize":
        plots = generate_visualizations(
            args.benchmark, args.output_dir or PLOTS_DIR, args.feature_importance
        )
        print("Generated plots:\n" + "\n".join(str(path) for path in plots))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
