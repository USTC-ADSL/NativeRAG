# Chunk State Lifecycle Metadata

## Purpose / motivation

This document records the first minimal Phase 5 slice for the evolving index story in `AGENTS.md`.

The immediate goal is not full dense promotion/demotion yet. The goal of this patch is to make SQLite explicitly track chunk residency state and transition history so later promotion, demotion, and reproducible snapshot work have a canonical state layer to build on.

## What behavior changed

- SQLite now persists per-chunk residency metadata under a canonical chunk-state table.
- SQLite now persists chunk-state transition history under a separate append-only transition log.
- SQLite now supports deterministic chunk-state snapshot export/import for experiment replay.
- Newly indexed chunks are initialized to `warm`.
- Query-time retrieved chunks are promoted to `hot`.
- Query traces now emit an `[INDEX_STATE]` line showing how many retrieved chunks were newly promoted to `hot`.
- Focused regressions validate:
  - chunk-state initialization
  - promotion and demotion history
  - persistence across reopen
  - query-time `warm -> hot` promotion inside `RAGPipeline`
  - snapshot export/import into a fresh SQLite state store

## Key implementation points

- `include/vector_db/SqliteVectorDB.hpp` now defines:
  - `ChunkState`
  - `initialize_chunk_states(...)`
  - `update_chunk_state(...)`
  - `get_chunk_state(...)`
  - `count_chunk_state_transitions(...)`
  - `export_chunk_state_snapshot(...)`
  - `import_chunk_state_snapshot(...)`
- `src/vector_db/SqliteVectorDB.cpp` now creates and maintains:
  - `chunk_states`
  - `chunk_state_transitions`
- Snapshot files are written as a deterministic TSV text format:
  - header: `STATE_SNAPSHOT_V1`
  - state rows sorted by chunk ID
  - transition rows sorted by chunk ID, timestamp, and event ID
  - importing a snapshot resets transition autoincrement tracking to the imported max event ID so replayed updates continue deterministically
- `RAGPipeline::add_text_embeddings(...)` initializes newly ingested chunks as `warm`.
- `RAGPipeline::answer_query(...)` promotes retrieved chunks to `hot` and emits an `[INDEX_STATE]` trace.
- CLI now exposes:
  - `--state-snapshot-in`
  - `--state-snapshot-out`

## Main files / modules touched

- `include/vector_db/SqliteVectorDB.hpp`
- `src/vector_db/SqliteVectorDB.cpp`
- `src/RAGPipeline.cpp`
- `tests/group_c_vectordb/test_sqlite_vectordb_backend.cpp`
- `tests/test_rag_semantic_hash_prefilter.cpp`

## Runtime path / execution flow

1. During indexing, `RAGPipeline` persists vectors, texts, and semantic hashes as before.
2. After successful SQLite persistence, the same chunk IDs are initialized to `warm`.
3. During query-time retrieval, the selected graph executes as before.
4. After final retrieval results are assembled, each retrieved chunk ID is promoted to `hot`.
5. The pipeline emits an `[INDEX_STATE]` log with the number of newly promoted chunks.
6. Retrieval and generation behavior otherwise remain unchanged.
7. When a snapshot flag is provided:
   - `--state-snapshot-in` restores chunk-state metadata before execution
   - `--state-snapshot-out` exports chunk-state metadata after execution

## Config flags / thresholds / defaults

- No new user-facing flags are added.
- Default initial state for indexed chunks: `warm`
- Default query-time promotion target: `hot`
- `--state-snapshot-in <path>`
  - restores a previously exported chunk-state snapshot
- `--state-snapshot-out <path>`
  - exports the current chunk-state snapshot after command completion
- Transition reason strings currently used by runtime:
  - `index_build`
  - `query_retrieval_hit`

## Fallback behavior

- If chunk-state initialization fails during indexing, the pipeline logs a warning and keeps the existing retrieval/indexing path alive.
- If query-time chunk-state promotion fails, retrieval and generation still complete.
- If a chunk is already in the requested state, no duplicate transition record is written.

## Schema or storage changes

Adds a canonical state table:

```sql
CREATE TABLE IF NOT EXISTS chunk_states (
  id INTEGER PRIMARY KEY,
  tier TEXT NOT NULL,
  last_transition_reason TEXT NOT NULL,
  last_transition_at_unix_ms INTEGER NOT NULL
);
```

Adds an append-only transition log:

```sql
CREATE TABLE IF NOT EXISTS chunk_state_transitions (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  id INTEGER NOT NULL,
  from_tier TEXT NOT NULL,
  to_tier TEXT NOT NULL,
  reason TEXT NOT NULL,
  created_at_unix_ms INTEGER NOT NULL
);
```

Snapshot export/import currently uses a deterministic TSV file with row prefixes:

```text
STATE_SNAPSHOT_V1
STATE    <id>    <tier>    <last_reason>    <last_transition_at_unix_ms>
TRANSITION    <event_id>    <id>    <from_tier>    <to_tier>    <reason>    <created_at_unix_ms>
```

## Metrics / logs added

- Adds `[INDEX_STATE]` query-time logs with:
  - `promoted_to_hot`
  - `retrieved_chunks`

## How to test / reproduce

1. Configure a host test build:
   - `cmake -S . -B build_progress_check -DUSE_PREBUILT=ON -DLLM_BACKEND=LlamaCpp -DBUILD_TESTS=ON`
2. Build focused targets:
   - `cmake --build build_progress_check --target test_command_line_args test_sqlite_vectordb_backend test_rag_semantic_hash_prefilter mobile_rag_cli -j4`
3. Run focused regressions:
   - `ctest --test-dir build_progress_check -R 'CommandLineArgsTest|SqliteVectorDBBackendTest|RAGSemanticHashPrefilterTest' --output-on-failure`
4. Export a snapshot during build or query:
   - `mobile_rag --query ... --state-snapshot-out /tmp/run.snapshot.tsv`
5. Restore a snapshot before a later run:
   - `mobile_rag --query ... --state-snapshot-in /tmp/run.snapshot.tsv`
6. On device, rebuild runtime assets after the schema change and run a normal query:
   - expect an `[INDEX_STATE]` line after retrieval result printing

## Known limitations / TODOs

- This patch tracks state metadata only; it does not yet evict dense vectors or physically split hot/cold storage.
- Promotion is query-hit based and heuristic.
- Automatic demotion policy is not yet wired into runtime decisions.
- Snapshot format is deterministic text, but it currently assumes runtime reason strings do not contain tabs or newlines.
- Replay currently restores only chunk-state metadata, not dense/text corpus contents.
