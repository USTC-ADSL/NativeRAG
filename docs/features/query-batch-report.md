# Query Batch Report

## Purpose / motivation

Per-query JSONL and CSV traces are useful, but paper-style evaluation also needs an invocation-level summary that shows graph-selection distribution, fallback frequency, and average evidence quality without requiring an external post-processing step for every run.

This patch adds a lightweight aggregate JSON report for `--query` runs, especially useful together with `--query-file`.

## What behavior changed

- `--query` now accepts `--query-batch-report-out <path>`.
- After a successful query invocation, the CLI can export one aggregate JSON report covering all queries executed in that process.
- The batch report works for:
  - single-query mode
  - `--query-file` batch mode
- The aggregate report currently includes:
  - total query count
  - escalation count
  - initial graph distribution
  - final graph distribution
  - fallback-reason distribution
  - promotion / demotion totals
  - average top score, score margin, coverage ratio, and candidate counts

## Key implementation points

- `include/cli/BatchQueryReport.hpp` and `src/cli/BatchQueryReport.cpp` add a CLI-side accumulator that records the final `RAGPipeline::QueryTrace` for each completed query.
- `src/main.cpp` and `src/main_with_dataset.cpp` now:
  - create one report accumulator per invocation
  - record each completed query trace after `answer_query(...)`
  - export the aggregate JSON after all queries finish
- `src/cli/CommandLineArgs.cpp` now parses and validates:
  - `--query-batch-report-out <path>`

The aggregation lives in the CLI layer so the retrieval/controller hot path remains unchanged.

## Main files / modules touched

- `include/cli/CommandLineArgs.hpp`
- `src/cli/CommandLineArgs.cpp`
- `include/cli/BatchQueryReport.hpp`
- `src/cli/BatchQueryReport.cpp`
- `src/main.cpp`
- `src/main_with_dataset.cpp`
- `tests/test_command_line_args.cpp`
- `tests/test_batch_query_report.cpp`
- `tests/CMakeLists.txt`

## Runtime path / execution flow

1. Query mode runs normally and produces a final `QueryTrace` per query.
2. After each successful query, the CLI records that trace into the aggregate report accumulator.
3. When the invocation finishes:
   - JSONL and CSV outputs remain per-query artifacts
   - batch report export writes one aggregate JSON artifact for the whole invocation
4. Optional state snapshot export still runs separately.

## Config flags / thresholds / defaults

- `--query-batch-report-out <path>`
  - default: disabled
  - supported only for `--query`
  - valid for both inline query mode and `--query-file`
  - writes one JSON file per CLI invocation

This patch does not change retrieval/controller thresholds or graph-selection policy.

## Fallback behavior

- If the flag is omitted, query execution behaves exactly as before.
- If report export is requested but writing the file fails, the CLI exits with an error because the requested evaluation artifact was not produced.
- The report is written only after successful query completion, so aggregation does not alter retrieval behavior.

## Schema or storage changes

None.

The aggregate report is a standalone JSON artifact and does not change SQLite, Faiss, or snapshot storage.

## Metrics / logs added

This patch adds one aggregate JSON artifact with:

- query-count level totals
- graph-selection distributions
- fallback-reason distributions
- average evidence and shortlist metrics

No new hot-path stdout log family is added.

## How to test / reproduce

1. Configure a host build:
   - `cmake -S . -B build_progress_check -DUSE_PREBUILT=ON -DLLM_BACKEND=LlamaCpp -DBUILD_TESTS=ON`
2. Build the relevant targets:
   - `cmake --build build_progress_check --target mobile_rag_cli mobile_rag_dataset test_command_line_args test_batch_query_report -j4`
3. Run focused tests:
   - `ctest --test-dir build_progress_check -R 'CommandLineArgsTest|BatchQueryReportTest' --output-on-failure`
4. Run batch query mode with the new flag:
   - `mobile_rag --query --query-file queries.txt --llm-model <gguf> --embedding-model <emb-config> --query-trace-jsonl-out /tmp/query-trace.jsonl --query-summary-csv-out /tmp/query-summary.csv --query-batch-report-out /tmp/query-batch-report.json ...`
5. Confirm the report contains:
   - `query_count`
   - `initial_graph_counts`
   - `final_graph_counts`
   - `fallback_reason_counts`
   - `totals`
   - `averages`

## Known limitations / TODOs

- The aggregate report currently captures averages and distributions only; it does not compute latency percentiles yet.
- The output is JSON only; there is no companion CSV summary file for batch-level aggregates yet.
- The report summarizes final query traces, not intermediate candidate lists from every retrieval stage.
