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
- Additional optional presets now exist for Phase 5 ablations:
  - `dense_only_state_aware`
  - `state_aware_tiered`
  - `adaptive_state_aware`
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
- The top-level summary now carries both average end-to-end latency and batch-report `p50` / `p95` end-to-end latency for each preset.
- `manifest.json` stores enough shared configuration to replay the matrix into a new output directory later.

## Key implementation points

- The runner lives outside the native runtime and shells out to the existing CLI binary.
- The runner supports two execution modes:
  - local host execution against a local CLI binary
  - device execution through `adb`, with artifact pull-back into the local output directory
- In device mode the runner now executes the remote command from the CLI binary directory itself and prefixes the invocation with `env LD_LIBRARY_PATH=.` so Android can resolve colocated runtime libraries such as `libMNN.so`, `libllama.so`, and `libfaiss.so` from the same deployment directory as `mobile_rag_cli`.
- Preset selection is explicit and maps to existing query-time flags:
  - `dense_only`: no retrieval prefilter flags
  - `dense_only_state_aware`: `--state-aware-dense`
  - `static_tiered`: `--lexical-prefilter --semantic-hash-prefilter`
  - `state_aware_tiered`: `--lexical-prefilter --semantic-hash-prefilter --state-aware-dense`
  - `adaptive_graph`: `--lexical-prefilter --semantic-hash-prefilter --adaptive-graph`
  - `adaptive_state_aware`: `--lexical-prefilter --semantic-hash-prefilter --adaptive-graph --state-aware-dense`
- Each run reuses the same shared query/model/index inputs, which keeps baseline comparisons aligned.
- If `--state-snapshot-in` is provided, the same input snapshot is reused for every preset so replay and ablation runs start from the same saved state.
- In device mode the runner first checks `adb devices`, requires the selected serial to be in `device` state, runs the remote CLI once per preset, and pulls the generated artifacts back before summarizing them.
- The remote command derives the working directory from the CLI binary path itself instead of assuming the device shell already starts in a usable directory or already exports a usable `LD_LIBRARY_PATH`.
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
   - in device mode it also creates the remote run directory and then pulls the produced artifacts back locally
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
  - `--adb <path>`
    - optional adb executable override; defaults to `adb`
  - `--adb-serial <serial>`
    - enables device mode instead of local execution
  - `--remote-workdir <path>`
    - required in device mode; remote directory used for per-run outputs
  - `--preset <name>` repeated; default is `dense_only`, `static_tiered`, `adaptive_graph`
    - additional optional names: `dense_only_state_aware`, `state_aware_tiered`, `adaptive_state_aware`
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
  - `state_aware_dense_query_count`
  - `p50_total_ms`
  - `p95_total_ms`
  - `average_total_ms`
  - `average_coverage_ratio`
  - `average_state_filtered_candidate_count`
  - `max_peak_rss_kb`
  - per-run artifact paths
- `manifest.json` adds:
  - shared query/model/index config
  - execution mode and adb metadata when applicable
  - preset list
  - exact commands executed
  - relative artifact paths for each run
  - remote artifact paths for each run in device mode
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
6. Run the matrix on the default device serial when the inputs are already deployed remotely:
   - `python3 tools/run_benchmark_matrix.py --adb-serial fd8657d6 --remote-workdir /data/local/tmp/nativerag-bench --binary /data/local/tmp/nativerag-check/mobile_rag_cli --query-file /data/local/tmp/nativerag-check/runtime/query_batch.txt --llm-model /data/local/tmp/nativerag-check/models/Qwen3-4B-Q8_0.gguf --embedding-model /data/local/tmp/nativerag-check/models/Qwen3-0.6B-Embedding/config.json --sqlite-db /data/local/tmp/nativerag-check/runtime/state_demote.sqlite3 --index-path /data/local/tmp/nativerag-check/runtime/state_demote.faiss --state-snapshot-in /data/local/tmp/nativerag-check/runtime/state_demote_hot2.snapshot.tsv --output-dir /tmp/native_rag_bench_device`
7. Run a Phase 5 state-aware ablation bundle:
   - `python3 tools/run_benchmark_matrix.py --binary ./build_progress_check/mobile_rag_cli --query-file ./queries.txt --llm-model <gguf> --embedding-model <emb-config> --sqlite-db <sqlite.db> --index-path <faiss.index> --state-snapshot-in <snapshot.tsv> --output-dir /tmp/native_rag_bench_state --preset dense_only_state_aware --preset state_aware_tiered --preset adaptive_state_aware`

## Known limitations / TODOs

- Replay currently reuses the saved shared configuration and preset list, but it does not diff new artifacts against the original manifest.
- Device mode assumes the CLI binary, query file, models, and index inputs are already present on the target device; it does not push them automatically.
- Device mode assumes the required shared libraries are deployed alongside the CLI binary, because the runner `cd`s into the binary directory and launches the CLI with `LD_LIBRARY_PATH=.`
- The summary only surfaces metrics already present in per-run batch reports; it does not compute deeper citation metrics itself.
- Presets are fixed and heuristic for now; there is no external experiment spec file yet.
- The default preset list still prioritizes the original baseline trio; state-aware presets must be requested explicitly.
