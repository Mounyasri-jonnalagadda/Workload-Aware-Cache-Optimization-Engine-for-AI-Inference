# Workload-Aware Cache Optimization Engine for AI Inference

A Python workflow for generating CNN, Transformer, and matrix-multiplication memory
traces, passing them through an explicit text interface to an RTL testbench, and
training grouped random-forest predictors from **real successful simulations**.
The RTL implements a parameterized, write-through, write-allocate set-associative
cache with LRU, FIFO, or deterministic RANDOM replacement. The repository does
not claim FPGA synthesis or physical FPGA measurements. The target project is
AMD Xilinx Vivado 2025.2 / XA Artix-7 `xa7a12tcpg238-2I`.

## Requirements and installation

- Python 3.10 or later
- Packages in `requirements.txt`
- Vivado 2025.2 simulator tools `xvlog`, `xelab`, and `xsim` on `PATH`

From the project root:

```powershell
python -m pip install -r requirements.txt
```

The CLI can be invoked as `python -m python.main <command>`. Use the same Python
environment for installation, tests, and CLI commands.

## Trace and testbench interface

The CSV columns are `address,operation,workload_type,sequence`. The exporter
creates a separate `.mem` text file, one transaction per line:

```text
0 00000040
1 00000044
```

The first hexadecimal field is the operation (`0` read, `1` write); the second is
the address. This is deliberately not a CSV file. The testbench must open the path
provided in the `TRACE_FILE` simulator plusarg, parse the two fields, and use the
other plusargs: `CACHE_SIZE`, `ASSOCIATIVITY`, `BLOCK_SIZE`, and
`REPLACEMENT_POLICY`. `xvlog` also compiles the chosen geometry and replacement
code into the testbench as elaboration-time parameters. Supported cache sizes
are 4096, 8192, 16384, and 32768 bytes; associativities are 1, 2, or 4 ways;
block sizes are 32, 64, or 128 bytes; and policies are `LRU`, `FIFO`, and
`RANDOM`. The resulting set count must be a positive power of two. The controller
uses write-through, write-allocate behavior and a single outstanding request.

The testbench uses a simulation-only, fixed-array backing store with open
addressing for up to 65,536 distinct cache lines; overflow fails the simulation
instead of aliasing addresses. It checks read data and reports
`ACCESS_COUNT=<integer>`, `HIT_COUNT=<integer>`, and
`MISS_COUNT=<integer>`, which the Python parser recognizes. A write trace row
contains no write data, so the testbench supplies deterministic write payloads
for exercising the RTL; those payloads are not simulation measurements.

## RTL modules

- `rtl/address_decoder.v`: derives tag, set index, and block offset from a
  32-bit byte address.
- `rtl/cache_memory.v`: stores valid bits, tags, and block data for every set
  and way.
- `rtl/cache_controller.v`: searches all ways, selects invalid or replacement
  victims, refills, and performs word reads and write-through over a ready/valid
  line-memory interface. LRU uses per-set recency ranks, FIFO uses per-set
  insertion pointers, and RANDOM uses a reset-seeded deterministic LFSR.
- `rtl/performance_counter.v`: counts accepted accesses, hits, and misses.
- `testbench/cache_tb.v`: simulation-only trace reader, sparse backing-memory
  model, read-data checker, and metric printer.

The initial testbench cases are trace-driven. Include rows that exercise:

1. A cold read followed by a repeat read (one miss, then a hit).
2. A read to another word in the same block (a hit).
3. A different tag mapping to the same line, followed by a reread (eviction miss,
   then hit).
4. A write miss followed by a read, and a write hit followed by a read (write
   allocate/write-through and returned data are checked).
5. Same-set conflicts under 1-, 2-, and 4-way configurations, including LRU,
   FIFO, and deterministic RANDOM eviction.

For the default 8192-byte cache with 64-byte blocks, this example expects
`ACCESS_COUNT=10`, `HIT_COUNT=6`, and `MISS_COUNT=4`:

```text
0 00000000
0 00000000
0 00000004
0 00002000
0 00002000
0 00000000
1 00004000
0 00004000
1 00004000
0 00004000
```

## Workflow

Commands below run from the project root. Generated files go under `workloads/`,
`simulation_results/`, `datasets/`, `models/`, and `results/` by default.

1. Generate a repeatable workload:

   ```powershell
   python -m python.main generate-workload --type cnn --accesses 10000 --seed 42
   ```

   `--type` accepts `cnn`, `transformer`, or `matmul`. The CSV can be used to
   inspect and analyze the Python trace.

2. Export CSV and the explicit Verilog trace file:

   ```powershell
   python -m python.main export-trace workloads/workload.csv
   ```

   Optional `--length N` truncates both output files; `--address-width` controls
   address validation and the hexadecimal field width.

3. Run actual Vivado simulation after adding RTL:

   ```powershell
   python -m python.main run-vivado workloads/workload.mem `
     --workload-csv workloads/workload.csv --rtl-dir rtl --top cache_tb
   ```

   The runner compiles modules under `rtl/` and `testbench/cache_tb.v`, passing
   the selected geometry and replacement code at elaboration and configuration
   values via plusargs. Pass `--config-file models/recommendation.cfg` or specify
   `--cache-size`, `--associativity`, `--block-size`, and `--policy`.
   Associativity one is the direct-mapped special case.
   Missing simulator tools or RTL are reported as unavailable; no measurement is
   fabricated. `--dry-run` only writes an explicitly synthetic skip record without
   performance metrics; synthetic records are never added to the training set.

   For a single end-to-end validated experiment, generate the workload, export
   its trace, run Vivado, verify all three counters against the exact exported
   trace, and update the real-results dataset:

   ```powershell
   python -m python.main run-experiment --type cnn --accesses 100 --seed 42
   ```

   For example, run a 4-KiB, two-way cache with 32-byte blocks and FIFO:

   ```powershell
   python -m python.main run-experiment --type cnn --accesses 100 --seed 42 `
     --cache-size 4096 --associativity 2 --block-size 32 --policy FIFO
   ```

   This command exits unsuccessfully and does not update the dataset if simulation
   fails, a required counter is missing, `ACCESS_COUNT` differs from the exported
   row count, or hits plus misses do not equal accesses.

4. Build a dataset only from successful, validated, non-synthetic runs:

   ```powershell
   python -m python.main build-dataset
   ```

   Set valid `HIT_TIME_CYCLES` and `MISS_PENALTY_CYCLES` in
   `python/config.py`, or pass both `--hit-time` and `--miss-penalty`, to derive
   estimated AMAT as `hit_time + miss_rate * miss_penalty`. Without timing
   assumptions or a testbench AMAT counter, AMAT remains unavailable. Dataset
   rows include measured access, hit, and miss counts alongside the existing
   workload and cache configuration columns.

5. Train and evaluate with trace-grouped holdout data:

   ```powershell
   python -m python.main train-model --dataset datasets/cache_dataset.csv
   python -m python.main evaluate-model --dataset datasets/cache_dataset.csv
   ```

   All configurations from one `trace_id` stay in the same split. Models report
   MAE, RMSE, and R² when calculable. Fewer than five distinct traces is explicitly
   marked insufficient evidence for an accuracy claim. The training code omits an
   AMAT predictor when the dataset does not contain enough valid AMAT targets.

6. Recommend a cache configuration and export simulator settings:

   ```powershell
   python -m python.main recommend workloads/workload.csv
   ```

   The command saves `models/recommendation.json` and a simple `KEY=VALUE`
   `models/recommendation.cfg` containing the simulator plusargs. When a trained
   AMAT model exists, AMAT is the ranking objective; otherwise miss rate is used
   and that limitation is included with the recommendation. Optional
   `--max-cache-size` and `--max-associativity` restrict the search across
   supported cache sizes, ways, block sizes, and meaningful replacement policies.
   Direct-mapped candidates use LRU as their canonical policy because replacement
   choice cannot change a one-way cache. Resource utilization is reported as
   unavailable unless you add synthesis data.

7. Benchmark alternatives (real simulation required):

   ```powershell
   python -m python.main benchmark workloads/workload.mem `
     --workload-csv workloads/workload.csv --model models/cache_optimizer.joblib
   ```

   Before simulation, the benchmark verifies that the CSV and MEM files contain
   the same ordered accesses. It compares the fixed 8192-byte direct-mapped LRU
   baseline with the ML recommendation and avoids a duplicate run if they match.
   Timestamped CSV, run-report JSON, and comparison-report JSON files are written
   under `results/metrics/`; unavailable or invalid measurements are retained in
   the run report but are excluded from improvement claims. Use
   `--comparison-output <path.json>` to choose the comparison report's base path.
   Pass both `--hit-time` and
   `--miss-penalty` to calculate estimated AMAT and AMAT prediction error from
   miss-rate counters when the testbench does not directly print AMAT.

8. Plot benchmark results:

   ```powershell
   python -m python.main visualize --benchmark results/metrics/benchmark.csv
   ```

   Optional `--feature-importance` accepts a CSV with `feature,importance`
   columns. The plotting command refuses synthetic-only or empty measurements.

Run Python tests:

```powershell
python -m unittest discover -s tests -v
```

## Project layout

Python modules:

- `python/config.py`: shared directories, cache configuration validation, and config loading.
- `python/workload_generator.py`, `python/trace_exporter.py`, and
  `python/feature_engineering.py`: trace creation, Verilog trace encoding, and
  workload feature extraction.
- `python/vivado_runner.py` and `python/simulation_parser.py`: xsim orchestration,
  logging, plusargs, and counter parsing.
- `python/dataset_builder.py`, `python/train_model.py`, and
  `python/evaluate_model.py`: real-result filtering, grouped model training, and
  held-out evaluation.
- `python/cache_optimizer.py`, `python/benchmark.py`, and
  `python/visualize_results.py`: constrained recommendations, benchmarking, and
  plots.
- `python/main.py`: command-line workflow entry point.

```text
python/       Workflow modules and CLI
tests/        Unit tests
workloads/    Generated CSV and Verilog .mem traces
rtl/          Synthesizable address decoder, cache memory, controller, and counters
testbench/    Simulation-only cache testbench and backing-memory model
datasets/     Real-simulation training dataset
simulation_results/ Logs and JSON run records
training/     Training/evaluation reports
results/      Metrics, plots, and reports
models/       Joblib model and recommendations
```
