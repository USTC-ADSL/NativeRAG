# State-Aware Dense Rerank

## Purpose / motivation

Phase 5 in `AGENTS.md` is about making the evolving hot / warm / cold index state affect runtime behavior, not only logging. The earlier chunk-state slice made SQLite the canonical source of residency metadata and replay state, but dense rerank still treated every chunk as equally eligible once it reached the shortlist.

This patch adds a minimal, opt-in runtime bridge from chunk state to retrieval behavior.

## What behavior changed

- Query mode now accepts `--state-aware-dense`.
- When the flag is enabled, SQLite dense rerank only considers shortlist candidates whose chunk state is `warm` or `hot`.
- `cold` chunks may still appear in lexical or semantic-hash shortlist generation, but they are filtered out before SQLite vector rerank.
- If every shortlisted candidate is filtered out by state, the pipeline falls back to the existing dense search path and records fallback reason `state_filtered_shortlist_empty`.
- If some candidates survive state filtering but SQLite rerank still returns no results, the pipeline records fallback reason `state_filtered_sqlite_rerank_empty` before using the existing dense fallback.
- Query traces, JSONL rows, CSV summaries, and stdout retrieval logs now expose:
  - whether state-aware dense mode was enabled
  - how many shortlist candidates were filtered by state

## Key implementation points

- `include/RAGPipeline.hpp` adds `StateAwareDenseConfig` plus new trace fields:
  - `state_aware_dense_enabled`
  - `state_filtered_candidate_count`
- `src/RAGPipeline.cpp` applies the filter only on the SQLite dense-rerank path used by lexical / hash shortlist graphs.
- `include/vector_db/SqliteVectorDB.hpp` and `src/vector_db/SqliteVectorDB.cpp` add `filter_ids_by_chunk_states(...)`, which preserves shortlist order while removing disallowed states.
- The allowed dense states are intentionally fixed to `warm` and `hot` for this first slice.
- Existing dense fallback behavior is preserved; this patch changes eligibility before SQLite rerank, not the fallback mechanism itself.

## Main files / modules touched

- `include/cli/CommandLineArgs.hpp`
- `src/cli/CommandLineArgs.cpp`
- `include/RAGPipeline.hpp`
- `src/RAGPipeline.cpp`
- `include/vector_db/SqliteVectorDB.hpp`
- `src/vector_db/SqliteVectorDB.cpp`
- `src/main.cpp`
- `src/main_with_dataset.cpp`
- `tests/test_command_line_args.cpp`
- `tests/test_rag_semantic_hash_prefilter.cpp`

## Runtime path / execution flow

1. The user enables `--state-aware-dense`.
2. Query-time retrieval still builds lexical and/or semantic-hash shortlists exactly as before.
3. Right before SQLite dense rerank, the pipeline asks SQLite for the canonical chunk state of each shortlisted ID.
4. Only `warm` and `hot` IDs are kept for SQLite dense rerank.
5. The pipeline records how many candidates were removed by state filtering.
6. If the filtered shortlist is empty, the pipeline falls back to the existing dense search path.
7. Query trace exports and stdout retrieval logs record the filtering decision and fallback reason.

## Config flags / thresholds / defaults

- `--state-aware-dense`
  - default: disabled
  - supported in query / interactive runtime paths
  - current allowed dense states when enabled: `warm`, `hot`

This patch does not add new thresholds. It is a boolean execution-mode switch.

## Fallback behavior

- If the flag is omitted, query behavior is unchanged.
- If SQLite is unavailable, retrieval still uses the existing dense fallback path.
- If all shortlisted candidates are filtered out by state, fallback reason is `state_filtered_shortlist_empty` and the pipeline reuses the existing dense search fallback.
- If filtered candidates remain but SQLite rerank yields no result, fallback reason is `state_filtered_sqlite_rerank_empty`.

## Schema or storage changes

None.

This patch consumes the existing `chunk_states` metadata already stored in SQLite. It does not add new tables, columns, or snapshot formats.

## Metrics / logs added

- `[RETRIEVAL]` logs now include:
  - `state_filtered_candidates`
- Query trace JSON / JSONL now include:
  - `state_aware_dense_enabled`
  - `state_filtered_candidate_count`
- Query summary CSV now includes:
  - `state_aware_dense_enabled`
  - `state_filtered_candidate_count`

## How to test / reproduce

1. Rebuild focused host targets:
   - `cmake --build build_progress_check --target mobile_rag_cli mobile_rag_dataset test_command_line_args test_rag_semantic_hash_prefilter -j4`
2. Run focused tests:
   - `ctest --test-dir build_progress_check -R 'CommandLineArgsTest|RAGSemanticHashPrefilterTest' --output-on-failure`
3. Run a query with the new flag and an imported state snapshot:
   - `mobile_rag --query --query-file queries.txt --llm-model <gguf> --embedding-model <emb-config> --sqlite-db <sqlite.db> --index-path <faiss.index> --lexical-prefilter --state-aware-dense --state-snapshot-in <snapshot.tsv> --query-trace-out /tmp/state-aware-trace.json`
4. Confirm the trace or stdout includes:
   - `state_aware_dense_enabled`
   - `state_filtered_candidate_count`
   - `fallback_reason` equal to either `none`, `state_filtered_shortlist_empty`, or `state_filtered_sqlite_rerank_empty`

## Known limitations / TODOs

- This slice only gates the SQLite dense-rerank path used after shortlist generation; it does not yet change the dense-only baseline path itself.
- Chunk states still do not evict vectors from Faiss or physically split dense storage.
- Allowed dense states are hard-coded to `warm` and `hot`; there is no external policy table yet.
- State filtering currently happens after shortlist generation, so cold chunks can still consume lexical/hash shortlist slots before being removed from dense rerank.
