# Query File Batch Mode

## Purpose / motivation

Phase 7 needs a lightweight way to run many queries through the same loaded runtime without introducing a new benchmark harness yet. Re-launching the CLI once per query adds avoidable startup cost and makes append-style trace collection clumsy.

This patch adds a file-driven batch mode for `--query` so one process can answer multiple queries while reusing the same loaded SQLite state, Faiss index, embedding model, and LLM runtime.

## What behavior changed

- `--query` now accepts `--query-file <path>` as an alternative to an inline question.
- The query file loader:
  - trims leading and trailing whitespace
  - skips blank lines
  - skips comment lines whose trimmed form starts with `#`
- Query mode now requires exactly one source:
  - one inline query string
  - or one `--query-file`
- Batch query mode reuses the existing query-time pipeline in one process and runs queries sequentially.
- Batch query mode reuses the existing append-friendly exports:
  - `--query-trace-jsonl-out`
  - `--query-summary-csv-out`
- Batch query mode can also emit one invocation-level aggregate report through:
  - `--query-batch-report-out`
- Batch query mode rejects `--query-trace-out` because a single JSON file is ambiguous when multiple queries run in one invocation.

## Key implementation points

- `include/cli/QueryFileLoader.hpp` and `src/cli/QueryFileLoader.cpp` add a small loader dedicated to file-based query lists.
- `src/cli/CommandLineArgs.cpp` now:
  - parses `--query-file <path>`
  - joins positional query tokens for inline query mode
  - validates the inline-query vs query-file exclusivity
  - validates that the query file exists
  - rejects `--query-trace-out` together with `--query-file`
- `src/main.cpp` and `src/main_with_dataset.cpp` now:
  - load all queries before model/index initialization continues into query execution
  - execute each query through the same `RAGPipeline` / `RAGPipelineWithDataset` instance
  - append JSONL and CSV trace artifacts after each query
  - export the chunk-state snapshot once after the batch finishes

## Main files / modules touched

- `include/cli/CommandLineArgs.hpp`
- `src/cli/CommandLineArgs.cpp`
- `include/cli/QueryFileLoader.hpp`
- `src/cli/QueryFileLoader.cpp`
- `src/main.cpp`
- `src/main_with_dataset.cpp`
- `tests/test_command_line_args.cpp`
- `tests/test_query_file_loader.cpp`
- `tests/CMakeLists.txt`

## Runtime path / execution flow

1. The CLI parses `--query`.
2. It accepts either:
   - an inline query string
   - or `--query-file <path>`
3. If a query file is provided, the loader reads the file once and materializes the runnable query list.
4. The query runtime then loads the embedding model, LLM, SQLite DB, Faiss index, and optional state snapshot exactly once.
5. The CLI iterates through the loaded queries in order.
6. For each query:
   - `answer_query(...)` runs through the existing retrieval/controller path
   - the answer is written to stdout
   - JSONL / CSV exports append one row for that query when enabled
7. After the batch finishes, optional state-snapshot export runs once.

## Config flags / thresholds / defaults

- `--query-file <path>`
  - default: disabled
  - supported only for `--query`
  - mutually exclusive with inline query text
- `--query-trace-out <path>`
  - still supported for single-query mode
  - rejected with `--query-file`
- `--query-trace-jsonl-out <path>`
  - supported for both single-query and batch-query mode
  - appends one compact JSON object per query
- `--query-summary-csv-out <path>`
  - supported for both single-query and batch-query mode
  - appends one summary row per query
- `--query-batch-report-out <path>`
  - supported for both single-query and batch-query mode
  - writes one aggregate JSON report after the invocation finishes

This patch does not change retrieval thresholds, controller heuristics, or graph-selection defaults.

## Fallback behavior

- If the query file contains no runnable lines after trimming and comment filtering, the CLI exits with an error before query execution starts.
- If batch export is not requested, query-file mode still prints answers to stdout and otherwise uses the same retrieval/controller logic.
- If JSONL or CSV export fails for any query, the CLI fails immediately because the requested experiment artifact was not produced.

## Schema or storage changes

None.

This patch does not change SQLite schema, Faiss layout, snapshot format, or controller state storage.

## Metrics / logs added

No new query-time metrics are introduced.

This patch reuses existing query trace and summary metrics while making it easier to accumulate them across a query list in one invocation.

## How to test / reproduce

1. Configure a host build:
   - `cmake -S . -B build_progress_check -DUSE_PREBUILT=ON -DLLM_BACKEND=LlamaCpp -DBUILD_TESTS=ON`
2. Build the relevant targets:
   - `cmake --build build_progress_check --target mobile_rag_cli mobile_rag_dataset test_command_line_args test_query_file_loader -j4`
3. Run focused tests:
   - `ctest --test-dir build_progress_check -R 'CommandLineArgsTest|QueryFileLoaderTest' --output-on-failure`
4. Prepare a query file such as:
   - `queries.txt`
   - with one query per line
   - blank lines and `#` comment lines allowed
5. Run batch query mode:
   - `mobile_rag --query --query-file queries.txt --llm-model <gguf> --embedding-model <emb-config> --query-trace-jsonl-out /tmp/query-trace.jsonl --query-summary-csv-out /tmp/query-summary.csv --query-batch-report-out /tmp/query-batch-report.json ...`
6. Confirm:
   - one answer is emitted per runnable query
   - JSONL contains one JSON object per query
   - CSV contains one appended row per query
   - batch report contains one aggregate JSON summary for the invocation
   - `--query-trace-out` is rejected when `--query-file` is present

## Known limitations / TODOs

- Batch mode is sequential; it does not introduce concurrent query execution.
- Single-file JSON trace export remains single-query only.
- Answers are still printed as plain stdout lines; batch mode does not yet emit a structured answer bundle.
- Query-file mode is a small execution convenience, not a full benchmark harness or replay runner.
