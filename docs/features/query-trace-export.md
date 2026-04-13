# Query Trace Export

## Purpose / motivation

Phase 5 and Phase 7 in `AGENTS.md` both need more than console-only logs. The repo already emits useful `[CONTROLLER]`, `[RETRIEVAL]`, `[EVIDENCE]`, and `[INDEX_STATE]` lines, but replaying experiments from stdout alone is brittle.

This patch adds explicit per-query export paths so experiments can persist the final retrieval/controller trace as a structured artifact, plus append-friendly formats for batch accumulation.

## What behavior changed

- `--query` now accepts an optional `--query-trace-out <path>` flag.
- `--query` now accepts an optional `--query-trace-jsonl-out <path>` flag.
- `--query` now accepts an optional `--query-summary-csv-out <path>` flag.
- After a successful query run, the CLI can export the final query trace as JSON.
- After a successful query run, the CLI can append the final query trace as one JSONL row.
- After a successful query run, the CLI can also append a flat CSV summary row for batch analysis.
- `RAGPipeline` now keeps the most recent query trace in memory and exposes it through:
  - `last_query_trace()`
  - `export_last_query_trace(...)`
- The exported trace captures:
  - query-time retrieval config snapshot
  - controller decision fields
  - retrieval candidate counts and fallback reason
  - evidence features
  - index-state promotion/demotion counts
  - post-query state summary
  - final ranked result IDs, scores, and previews
  - the final answer text

## Key implementation points

- `include/RAGPipeline.hpp` now defines:
  - `QueryTraceResult`
  - `QueryTrace`
  - `last_query_trace()`
  - `export_last_query_trace(...)`
- `src/RAGPipeline.cpp` now:
  - clears the previous trace at the start of each query
  - records the active retrieval knobs used for the query:
    - `top_k`
    - lexical prefilter enablement and shortlist size
    - semantic-hash prefilter enablement, shortlist size, and max distance
  - records initial/final graph decisions and budget class
  - records escalation metadata when the controller upgrades the graph
  - records retrieval candidate counts, fallback reason, and final ranked results
  - records evidence and index-state summaries after query-time state transitions
  - writes a deterministic JSON object when export is requested
- `CommandLineArgs` now parses and validates:
  - `--query-trace-out <path>`
  - `--query-trace-jsonl-out <path>`
  - `--query-summary-csv-out <path>`
- `src/main.cpp` and `src/main_with_dataset.cpp` now export the trace after `--query` completes.
  - JSON export keeps the full structured artifact
  - JSONL export appends one compact structured row per query
  - CSV export appends a single flat row and writes the header automatically when the file is new

## Main files / modules touched

- `include/RAGPipeline.hpp`
- `src/RAGPipeline.cpp`
- `include/cli/CommandLineArgs.hpp`
- `src/cli/CommandLineArgs.cpp`
- `src/main.cpp`
- `src/main_with_dataset.cpp`
- `tests/test_command_line_args.cpp`
- `tests/test_rag_semantic_hash_prefilter.cpp`

## Runtime path / execution flow

1. The query executes through the normal adaptive or baseline retrieval path.
2. `RAGPipeline::answer_query(...)` collects the same controller/retrieval/evidence/index-state information already used for console logs.
3. After answer generation, the pipeline stores the final trace in memory as the most recent query trace.
4. If `--query-trace-out` is set:
   - the CLI calls `export_last_query_trace(...)`
   - the trace is serialized to a JSON file after the query finishes
5. If `--query-trace-jsonl-out` is set:
   - the CLI calls `append_last_query_trace_jsonl(...)`
   - the final query trace is serialized as one compact JSON object followed by `\n`
6. If `--query-summary-csv-out` is set:
   - the CLI calls `append_last_query_trace_summary_csv(...)`
   - the final query trace is flattened into one CSV row
   - the header is emitted automatically on first write
7. Snapshot export and trace export remain separate:
   - snapshot export persists SQLite chunk-state tables
   - query trace export persists the final query-level decision artifact
   - query trace JSONL persists one structured row per query
   - query summary CSV persists a batch-friendly row-oriented view

## Config flags / thresholds / defaults

- `--query-trace-out <path>`
  - default: disabled
  - supported only for `--query`
  - writes a JSON file for the most recent query
- `--query-trace-jsonl-out <path>`
  - default: disabled
  - supported only for `--query`
  - appends a compact JSON line for the most recent query
- `--query-summary-csv-out <path>`
  - default: disabled
  - supported only for `--query`
  - appends a CSV summary row for the most recent query

The trace uses existing runtime thresholds and flags. This patch does not add new controller thresholds.

## Fallback behavior

- If `--query-trace-out` is not provided, query execution behaves exactly as before.
- If `answer_query(...)` never completes successfully, `export_last_query_trace(...)` returns `false`.
- Trace export happens after query execution, so export failure does not affect retrieval/controller logic itself. The CLI treats export failure as a command failure because the requested artifact was not produced.

## Schema or storage changes

None.

This patch does not change SQLite schema, Faiss layout, or snapshot format. Query traces are written as separate JSON files.

## Metrics / logs added

No new stdout log family is added.

This patch adds a structured JSON artifact that mirrors and aggregates existing query-time logs:

- retrieval knob snapshot
- controller choice
- retrieval candidate counts
- evidence features
- index-state transition counts
- index-state summary
- ranked result IDs and scores

This patch also adds an append-friendly JSONL artifact:

- one JSON object per query
- same structured fields as the single-query JSON export
- suitable for incremental batch accumulation without flattening nested fields

This patch also adds a flat CSV artifact intended for aggregation across many queries:

- one row per query
- stable header
- top-level controller/retrieval/evidence/index-state fields
- top-1 result ID and score

## How to test / reproduce

1. Configure a host test build:
   - `cmake -S . -B build_progress_check -DUSE_PREBUILT=ON -DLLM_BACKEND=LlamaCpp -DBUILD_TESTS=ON`
2. Build the relevant targets:
   - `cmake --build build_progress_check --target mobile_rag_cli mobile_rag_dataset test_command_line_args test_rag_semantic_hash_prefilter -j4`
3. Run focused regressions:
   - `ctest --test-dir build_progress_check -R 'CommandLineArgsTest|RAGSemanticHashPrefilterTest' --output-on-failure`
4. Run a query with trace export:
   - `mobile_rag --query "..." --llm-model <gguf> --embedding-model <emb-config> --query-trace-out /tmp/query-trace.json ...`
5. Run a query with JSONL export:
   - `mobile_rag --query "..." --llm-model <gguf> --embedding-model <emb-config> --query-trace-jsonl-out /tmp/query-trace.jsonl ...`
6. Run a query with CSV summary export:
   - `mobile_rag --query "..." --llm-model <gguf> --embedding-model <emb-config> --query-summary-csv-out /tmp/query-summary.csv ...`
7. Confirm the JSON contains:
   - `initial_graph`
   - `final_graph`
   - `budget_class`
   - `fallback_reason`
   - `index_state`
   - `evidence`
   - `results`
8. Confirm the JSONL contains:
   - one JSON object per line
   - the same structured fields as the JSON export
9. Confirm the CSV contains:
   - a single header row when the file is first created
   - one appended row per query
   - flattened fields such as `budget_class`, `fallback_reason`, `coverage_ratio`, `top_result_id`

## Known limitations / TODOs

- `--query-trace-out` currently supports only single `--query` runs, not interactive multi-turn sessions.
- The JSON is written manually and intentionally stays small; it is not yet a full replay bundle.
- JSONL append still requires the caller to invoke the CLI once per query; there is no built-in multi-query driver yet.
- The trace captures final query results, not intermediate candidate lists before rerank.
- The CSV is intentionally flat and only includes the top-level summary plus top-1 result fields; it is not a full replay bundle.
