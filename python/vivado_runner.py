"""Vivado/xsim orchestration with an explicit Verilog trace-file interface."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import (
    REPLACEMENT_POLICY_CODES,
    RTL_DIR,
    SIMULATION_RESULTS_DIR,
    TESTBENCH_DIR,
    CacheConfig,
)
from .simulation_parser import parse_simulation_log


REQUIRED_COUNTERS = ("access_count", "hit_count", "miss_count")
TRACE_ADDRESS_MAX = (1 << 32) - 1


def _is_windows() -> bool:
    return os.name == "nt"


def _build_subprocess_command(
    command: list[str], *, windows: bool | None = None
) -> tuple[list[str] | str, str | None]:
    """Run Windows batch launchers through cmd.exe while preserving argv quoting."""
    is_windows = _is_windows() if windows is None else windows
    if not is_windows or Path(command[0]).suffix.lower() not in {".bat", ".cmd"}:
        return command, None
    comspec = os.environ.get("COMSPEC", "cmd.exe")
    quoted_arguments = []
    for argument in command:
        needs_quotes = (
            "=" in argument
            or any(character.isspace() for character in argument)
            or any(character in argument for character in "&|<>^()")
        )
        escaped = argument.replace('"', '""')
        quoted_arguments.append(f'"{escaped}"' if needs_quotes else escaped)
    command_line = " ".join(quoted_arguments)
    return f'"{comspec}" /d /c {command_line}', comspec


def count_trace_records(trace_file: str | Path) -> int:
    """Validate the exported op/hex-address format and count its transactions."""
    trace_path = Path(trace_file)
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace file does not exist: {trace_path}")

    record_count = 0
    with trace_path.open("r", encoding="ascii") as trace:
        for line_number, raw_line in enumerate(trace, start=1):
            fields = raw_line.split()
            if not fields:
                continue
            if len(fields) != 2 or fields[0] not in {"0", "1"}:
                raise ValueError(
                    f"Malformed trace record on line {line_number}; expected '<0|1> HEX_ADDRESS'."
                )
            if re.fullmatch(r"[0-9A-Fa-f]+", fields[1]) is None:
                raise ValueError(f"Invalid hexadecimal address on trace line {line_number}.")
            address = int(fields[1], 16)
            if address > TRACE_ADDRESS_MAX or address % 4:
                raise ValueError(
                    f"Trace address on line {line_number} must be a word-aligned 32-bit address."
                )
            record_count += 1
    if record_count == 0:
        raise ValueError("Trace file contains no valid access records.")
    return record_count


def validate_simulation_metrics(
    metrics: Mapping[str, Any], expected_access_count: int
) -> list[str]:
    """Return validation errors for required counters and their invariants."""
    errors: list[str] = []
    for name in REQUIRED_COUNTERS:
        value = metrics.get(name)
        if type(value) is not int or value < 0:
            errors.append(f"Required counter {name} is missing or is not a non-negative integer.")
    if errors:
        return errors

    access_count = metrics["access_count"]
    hit_count = metrics["hit_count"]
    miss_count = metrics["miss_count"]
    if access_count != expected_access_count:
        errors.append(
            f"ACCESS_COUNT={access_count} does not match exported trace count "
            f"{expected_access_count}."
        )
    if hit_count + miss_count != access_count:
        errors.append(
            f"HIT_COUNT + MISS_COUNT ({hit_count + miss_count}) does not match "
            f"ACCESS_COUNT ({access_count})."
        )
    return errors


def run_simulation(
    trace_file: str | Path,
    cache_config: CacheConfig,
    rtl_dir: str | Path = RTL_DIR,
    top_module: str = "cache_tb",
    result_dir: str | Path = SIMULATION_RESULTS_DIR,
    workload_features: dict[str, Any] | None = None,
    trace_id: str | None = None,
    dry_run: bool = False,
    testbench_dir: str | Path | None = None,
    plan_id: str | None = None,
) -> dict[str, Any]:
    """Compile and run RTL using xvlog/xelab/xsim, or explicitly skip in dry-run.

    RTL and testbench files are compiled together with geometry macros. The .mem
    trace is copied into the simulator run directory and passed via ``TRACE_FILE``;
    no CSV data is passed implicitly.
    """
    if plan_id is not None and (not isinstance(plan_id, str) or not plan_id):
        raise ValueError("plan_id must be a non-empty string when provided.")
    trace_path = Path(trace_file).resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace file does not exist: {trace_path}")
    expected_access_count = count_trace_records(trace_path)
    num_sets = cache_config.cache_size_bytes // (
        cache_config.associativity * cache_config.block_size_bytes
    )
    if (
        num_sets < 1
        or num_sets & (num_sets - 1)
        or cache_config.replacement_policy not in REPLACEMENT_POLICY_CODES
    ):
        raise ValueError("Invalid cache geometry or replacement policy for the RTL.")
    testbench_dir = Path(testbench_dir) if testbench_dir is not None else TESTBENCH_DIR
    output_dir = Path(result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = output_dir / f"{run_id}.log"
    result_path = output_dir / f"{run_id}.json"
    result: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if dry_run else "unavailable",
        "synthetic": bool(dry_run),
        "measurement_source": "synthetic_dry_run" if dry_run else None,
        "trace_file": str(trace_path),
        "expected_access_count": expected_access_count,
        "trace_id": trace_id or trace_path.stem,
        "workload_features": workload_features or {},
        "cache_config": cache_config.to_dict(),
        "metrics": {},
        "validated": False,
        "validation": {
            "passed": False,
            "expected_access_count": expected_access_count,
            "actual_access_count": None,
            "errors": [],
        },
        "log_file": str(log_path),
        "errors": [],
    }
    if plan_id is not None:
        result["plan_id"] = plan_id

    if dry_run:
        log_path.write_text(
            "DRY_RUN: no RTL simulation was run; no hardware metrics are available.\n",
            encoding="utf-8",
        )
    else:
        source_dir = Path(rtl_dir)
        testbench_source_dir = testbench_dir
        source_dirs = (source_dir, testbench_source_dir)
        sources = sorted(
            path
            for directory in source_dirs
            if directory.is_dir()
            for path in directory.rglob("*")
            if path.suffix.lower() in {".v", ".sv"}
        )
        tools: dict[str, str] = {}
        missing_tools: list[str] = []
        for name in ("xvlog", "xelab", "xsim"):
            tool_path = shutil.which(name)
            if tool_path is None:
                missing_tools.append(name)
            else:
                tools[name] = tool_path
        if not source_dir.is_dir():
            result["errors"].append(f"RTL directory is unavailable: {source_dir}")
        if not testbench_source_dir.is_dir():
            result["errors"].append(
                f"Testbench directory is unavailable: {testbench_source_dir}"
            )
        if not any(path.parent == source_dir or source_dir in path.parents for path in sources):
            result["errors"].append(f"No Verilog/SystemVerilog sources found in: {source_dir}")
        if not any(
            path.parent == testbench_source_dir or testbench_source_dir in path.parents
            for path in sources
        ):
            result["errors"].append(
                f"No Verilog/SystemVerilog testbench sources found in: {testbench_source_dir}"
            )
        if missing_tools:
            result["errors"].append(
                "Vivado simulation tools are unavailable on PATH: " + ", ".join(missing_tools)
            )

        if not result["errors"]:
            run_dir = output_dir / f"run_{run_id}"
            run_dir.mkdir()
            simulator_trace = run_dir / "trace.mem"
            shutil.copy2(trace_path, simulator_trace)
            expected_access_count = count_trace_records(simulator_trace)
            result["expected_access_count"] = expected_access_count
            result["validation"]["expected_access_count"] = expected_access_count
            commands = [
                [
                    tools["xvlog"],
                    "-d",
                    f"CACHE_SIZE_BYTES_CFG={cache_config.cache_size_bytes}",
                    "-d",
                    f"ASSOCIATIVITY_CFG={cache_config.associativity}",
                    "-d",
                    f"BLOCK_SIZE_BYTES_CFG={cache_config.block_size_bytes}",
                    "-d",
                    f"REPLACEMENT_POLICY_CFG="
                    f"{REPLACEMENT_POLICY_CODES[cache_config.replacement_policy]}",
                    *[str(source.resolve()) for source in sources],
                ],
                [tools["xelab"], top_module, "-s", "cache_snapshot"],
                [
                    tools["xsim"],
                    "cache_snapshot",
                    "-runall",
                    "-testplusarg",
                    "TRACE_FILE=trace.mem",
                    "-testplusarg",
                    f"CACHE_SIZE={cache_config.cache_size_bytes}",
                    "-testplusarg",
                    f"ASSOCIATIVITY={cache_config.associativity}",
                    "-testplusarg",
                    f"BLOCK_SIZE={cache_config.block_size_bytes}",
                    "-testplusarg",
                    f"REPLACEMENT_POLICY={cache_config.replacement_policy}",
                    "-testplusarg",
                    f"REPLACEMENT_POLICY_CODE="
                    f"{REPLACEMENT_POLICY_CODES[cache_config.replacement_policy]}",
                ],
            ]
            transcripts: list[str] = []
            try:
                for command in commands:
                    invocation, executable = _build_subprocess_command(command)
                    completed = subprocess.run(
                        invocation,
                        executable=executable,
                        cwd=run_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    logged_command = (
                        invocation
                        if isinstance(invocation, str)
                        else subprocess.list2cmdline(invocation)
                    )
                    transcripts.append(
                        f"$ {logged_command}\n{completed.stdout}\n{completed.stderr}"
                    )
                    if completed.returncode != 0:
                        result["errors"].append(
                            f"Command failed with exit code {completed.returncode}: {command[0]}"
                        )
                        break
                log_text = "\n".join(transcripts)
                parsed = parse_simulation_log(log_text)
                result["metrics"] = parsed["metrics"]
                result["errors"].extend(parsed["errors"])
                validation_errors = validate_simulation_metrics(
                    result["metrics"], expected_access_count
                )
                result["validation"]["actual_access_count"] = result["metrics"].get(
                    "access_count"
                )
                result["validation"]["errors"] = validation_errors
                if result["errors"]:
                    result["status"] = "failed"
                elif any(result["metrics"].get(name) is None for name in REQUIRED_COUNTERS):
                    result["status"] = "incomplete"
                    result["errors"].append(
                        "Simulation did not report all required integer counters."
                    )
                    result["errors"].extend(validation_errors)
                elif validation_errors:
                    result["status"] = "failed"
                    result["errors"].extend(validation_errors)
                else:
                    result["status"] = "success"
                    result["measurement_source"] = "vivado_xsim"
                    result["validated"] = True
                    result["validation"]["passed"] = True
            except OSError as exc:
                result["status"] = "failed"
                result["errors"].append(f"Could not execute simulator command: {exc}")
                log_text = "\n".join(transcripts)
            log_path.write_text(log_text, encoding="utf-8")

    if not log_path.exists():
        log_path.write_text("\n".join(result["errors"]), encoding="utf-8")
    result["result_file"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_log_file(path: str | Path) -> dict[str, Any]:
    """Parse a saved simulator log."""
    return parse_simulation_log(Path(path).read_text(encoding="utf-8", errors="replace"))
