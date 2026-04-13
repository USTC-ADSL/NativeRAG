# Benchmark Runner

## Purpose / motivation

Phase 7 in `AGENTS.md` requires benchmark automation, ablation-friendly outputs, and replay support. The CLI already supports query-file execution, per-query JSONL / CSV traces, aggregate JSON batch reports, and chunk-state snapshots, but running the dense-only, static-tiered, and adaptive baselines by hand is still repetitive and error-prone.

This patch adds a small outer-loop benchmark runner that drives those existing exports without changing the native query hot path.

## What behavior changed

- A new tool exists at `tools/run_benchmark_matrix.py`.
- The tool runs a named preset matrix around `mobile_rag_cli --query --query-file ...`.
- By default it executes these presets:
  - `dense_only`
  - `static_tiered`
  - `adaptive_graph`
- Each preset produces its own artifact bundle under the selected output directory:
  - query trace JSONL
  - query summary CSV
  - batch report JSON
  - state snapshot output
  - stdout / stderr logs
- The runner also writes top-level artifacts:
  - `summary.json`
  - `summary.csv`
  - `manifest.json`
- `manifest.json` stores enough shared configuration to replay the matrix into a new output directory later.

## Key implementation points

- The runner lives outside the native runtime and shells out to the existing CLI binary.
- Preset selection is explicit and maps to existing query-time flags:
  - `dense_only`: no retrieval prefilter flags
  - `static_tiered`: `--lexical-prefilter --semantic-hash-prefilter`
  - `adaptive_graph`: `--lexical-prefilter --semantic-hash-prefilter --adaptive-graph`
- Each run reuses the same shared query/model/index inputs, which keeps baseline comparisons aligned.
- If `--state-snapshot-in` is provided, the same input snapshot is reused for every preset so replay and ablation runs start from the same saved state.
- Replay mode reads `manifest.json`, reconstructs the shared inputs plus preset list, and reruns the matrix into a fresh output directory.
- Focused smoke coverage lives in `tests/test_benchmark_runner.py`; it uses a fake CLI binary so the test stays fast and deterministic.

## Main files / modules touched

- `tools/run_benchmark_matrix.py`
- `tests/test_benchmark_runner.py`
- `tests/CMakeLists.txt`
- `docs/evaluation/benchmark-runner.md`

## Runtime path / execution flow

1. The user calls `tools/run_benchmark_matrix.py` with the query file, model paths, SQLite DB, Faiss index, and output directory.
2. The runner expands the requested preset list.
3. For each preset:
   - it creates `runs/<NN>_<preset>/`
   - it invokes `mobile_rag_cli --query --query-file ...` with the preset flags
   - it captures stdout / stderr into log files
   - it reads the preset's batch report JSON to extract top-level metrics
4. After all runs finish, the runner writes:
   - `summary.json` with one summary object per preset
   - `summary.csv` with one flat row per preset
   - `manifest.json` with the shared config, preset list, exact commands, and relative artifact paths
5. Replay mode loads `manifest.json` and reruns the same matrix into another output directory.

## Config flags / thresholds / defaults

- Fresh-run mode:
  - `--binary <path>`
  - `--query-file <path>`
  - `--llm-model <path>`
  - `--embedding-model <path>`
  - `--sqlite-db <path>`
  - `--index-path <path>`
  - `--output-dir <path>`
- Optional fresh-run flags:
  - `--state-snapshot-in <path>`
  - `--preset <name>` repeated; default is `dense_only`, `static_tiered`, `adaptive_graph`
  - `--top-k <num>`
  - `--threads <num>`
  - `--max-new-tokens <num>`
  - `--lexical-candidates <num>`
  - `--semantic-hash-candidates <num>`
  - `--semantic-hash-max-distance <num>`
- Replay mode:
  - `--replay-manifest <path>`
  - `--output-dir <path>`
  - optional `--binary <path>` override

This patch does not change retrieval/controller thresholds inside the native runtime; it only automates how existing flags are exercised.

## Fallback behavior

- If the runner is not used, normal CLI behavior is unchanged.
- If any required path is missing, the runner fails before launching the matrix.
- If one preset run exits non-zero, the runner fails closed and points to that preset's stderr log.
- Replay mode revalidates manifest paths before execution.

## Schema or storage changes

None.

The runner writes evaluation artifacts beside existing JSONL / CSV / snapshot outputs but does not change SQLite schema, Faiss layout, or runtime trace formats.

## Metrics / logs added

- `summary.json` and `summary.csv` add one row/object per preset with:
  - `query_count`
  - `escalation_count`
  - `average_total_ms`
  - `average_coverage_ratio`
  - `max_peak_rss_kb`
  - per-run artifact paths
- `manifest.json` adds:
  - shared query/model/index config
  - preset list
  - exact commands executed
  - relative artifact paths for each run
- Each preset bundle now also saves:
  - `stdout.log`
  - `stderr.log`

## How to test / reproduce

1. Reconfigure the host build if needed:
   - `cmake -S . -B build_progress_check -DUSE_PREBUILT=ON -DLLM_BACKEND=LlamaCpp -DBUILD_TESTS=ON`
2. Run the focused smoke test:
   - `ctest --test-dir build_progress_check -R 'BenchmarkRunnerTest' --output-on-failure`
3. Run a real matrix against the CLI:
   - `python3 tools/run_benchmark_matrix.py --binary ./build_progress_check/mobile_rag_cli --query-file ./queries.txt --llm-model <gguf> --embedding-model <emb-config> --sqlite-db <sqlite.db> --index-path <faiss.index> --output-dir /tmp/native_rag_bench --state-snapshot-in <snapshot.tsv>`
4. Inspect:
   - `/tmp/native_rag_bench/summary.json`
   - `/tmp/native_rag_bench/summary.csv`
   - `/tmp/native_rag_bench/manifest.json`
   - `/tmp/native_rag_bench/runs/01_dense_only/`
   - `/tmp/native_rag_bench/runs/02_static_tiered/`
   - `/tmp/native_rag_bench/runs/03_adaptive_graph/`
5. Replay the same matrix:
   - `python3 tools/run_benchmark_matrix.py --replay-manifest /tmp/native_rag_bench/manifest.json --output-dir /tmp/native_rag_bench_replay`

## Known limitations / TODOs

- The runner currently automates the preset matrix on the host side; it does not yet wrap `adb -s fd8657d6` device execution.
- Replay currently reuses the saved shared configuration and preset list, but it does not diff new artifacts against the original manifest.
- The summary only surfaces metrics already present in per-run batch reports; it does not compute percentiles or deeper citation metrics itself.
- Presets are fixed and heuristic for now; there is no external experiment spec file yet.
