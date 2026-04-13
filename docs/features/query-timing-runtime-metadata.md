# Query Timing And Runtime Metadata

## Purpose / motivation

`AGENTS.md` requires Phase 1 and Phase 7 to move beyond correctness-only traces. The repo already exported controller, retrieval, evidence, and chunk-state outcomes, but it still lacked structured timing and experiment-context fields that make runs comparable across devices, models, and index sizes.

This patch extends per-query and batch-level exports with lightweight timing metrics, a peak RSS proxy, and a stable runtime metadata snapshot.

## What behavior changed

- Per-query traces now include structured timing fields for:
  - query embedding
  - retrieval
  - evidence extraction
  - chunk-state updates
  - prompt construction
  - generation
  - end-to-end total
- Per-query traces now include a peak RSS proxy:
  - `peak_rss_kb`
- Per-query traces now include runtime metadata copied from the CLI invocation:
  - LLM backend
  - embedding backend
  - model paths
  - SQLite path
  - Faiss index path
  - query source (`inline` or `query_file`)
  - thread count
  - max new tokens
  - SQLite DB file size
  - Faiss index file size
- CSV summary export now appends these timing and runtime fields as flat columns.
- Batch report export now aggregates:
  - average timing values
  - average peak RSS proxy
  - maximum peak RSS proxy
  - one invocation-level runtime snapshot

## Key implementation points

- `include/RAGPipeline.hpp` now defines:
  - `QueryTiming`
  - `QuerySystemMetrics`
  - `TraceRuntimeMetadata`
- `src/RAGPipeline.cpp` now:
  - measures stage durations with `std::chrono::steady_clock`
  - samples `getrusage(RUSAGE_SELF).ru_maxrss` as a peak RSS proxy
  - serializes timing, system, and runtime sections into JSON / JSONL
  - appends the same fields into CSV summary output
- `src/main.cpp` and `src/main_with_dataset.cpp` now populate runtime metadata for query-mode pipelines from existing CLI config and local file sizes.
- `src/cli/BatchQueryReport.cpp` now accumulates averages for the new timing fields plus `peak_rss_kb` and exports the first query's runtime snapshot as the invocation metadata.

## Main files / modules touched

- `include/RAGPipeline.hpp`
- `src/RAGPipeline.cpp`
- `src/main.cpp`
- `src/main_with_dataset.cpp`
- `include/cli/BatchQueryReport.hpp`
- `src/cli/BatchQueryReport.cpp`
- `tests/test_rag_semantic_hash_prefilter.cpp`
- `tests/test_batch_query_report.cpp`

## Runtime path / execution flow

1. The CLI populates runtime metadata before query execution starts.
2. `RAGPipeline::answer_query(...)` records:
   - query embedding time
   - retrieval time, including any adaptive second pass
   - evidence extraction time
   - chunk-state update time
   - prompt-build time
   - generation time
   - total query time
3. After answer generation, the pipeline samples peak RSS and attaches the runtime metadata snapshot.
4. Query export paths write these fields into:
   - JSON
   - JSONL
   - CSV summary
5. Batch report aggregation computes invocation-level averages and maxima from those per-query values.

## Config flags / thresholds / defaults

No new flags are introduced.

The new timing and runtime fields are automatically populated whenever query traces or batch reports are exported.

## Fallback behavior

- If peak RSS sampling fails, `peak_rss_kb` falls back to `0`.
- If configured files do not exist or are not regular files, their size snapshot falls back to `0`.
- Timing collection is observational only and does not change controller or retrieval decisions.

## Schema or storage changes

None.

The new fields are emitted only in trace and report artifacts.

## Metrics / logs added

Structured exports now include:

- per-stage query timing
- peak RSS proxy
- runtime metadata snapshot

Batch reports now include:

- average per-stage timing
- average peak RSS proxy
- maximum peak RSS proxy

## How to test / reproduce

1. Configure a host build:
   - `cmake -S . -B build_progress_check -DUSE_PREBUILT=ON -DLLM_BACKEND=LlamaCpp -DBUILD_TESTS=ON`
2. Build focused targets:
   - `cmake --build build_progress_check --target mobile_rag_cli mobile_rag_dataset test_rag_semantic_hash_prefilter test_batch_query_report test_command_line_args -j4`
3. Run focused tests:
   - `ctest --test-dir build_progress_check -R 'RAGSemanticHashPrefilterTest|BatchQueryReportTest|CommandLineArgsTest' --output-on-failure`
4. Run a traced query or batch query and inspect:
   - `timings`
   - `system`
   - `runtime`
   - batch report `averages` and `maxima`

## Known limitations / TODOs

- This patch does not yet expose TTFT, prefill, or decode breakdown.
- `peak_rss_kb` is a process-level peak RSS proxy, not a precise per-operator memory profiler.
- There is still no thermal or energy instrumentation in the exported artifacts.
- Batch reports currently summarize averages and maxima only; they do not compute latency percentiles yet.
