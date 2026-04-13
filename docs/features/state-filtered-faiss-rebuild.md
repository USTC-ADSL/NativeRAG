# State-Filtered Faiss Rebuild

## Purpose / motivation

Phase 5 in `AGENTS.md` is not complete if chunk state only changes query-time eligibility while the physical dense index always keeps every vector resident. This patch adds a reproducible offline rebuild path that materializes a warm/hot-only Faiss subindex from SQLite's canonical vectors plus the current chunk-state metadata.

The goal is not online eviction yet. The goal is to let experiments and offline packaging produce a physically smaller dense index that matches the current residency policy.

## What behavior changed

- The CLI now accepts `--rebuild-state-filtered-index` as a standalone offline command.
- The rebuild command reads vectors from SQLite instead of re-embedding source documents.
- The rebuild command optionally imports `--state-snapshot-in` before selecting vectors, so replayed chunk-state experiments can rebuild a matching dense subindex deterministically.
- The rebuilt Faiss index currently includes only chunks whose SQLite state is `warm` or `hot`.
- If the selected state set is empty, the command still writes a valid empty Faiss index using the stored vector dimension from SQLite.

## Key implementation points

- `include/vector_db/SqliteVectorDB.hpp` and `src/vector_db/SqliteVectorDB.cpp` now expose:
  - `get_vector_dimension()`
  - `load_vectors_by_chunk_states(...)`
- `include/vector_Index/FaissIndex.hpp` and `src/vector_db/FaissIndex.cpp` now expose `initialize_empty(int dimension)` so offline rebuilds can persist a valid empty index when no chunk currently qualifies for dense residency.
- `include/vector_db/IngestUtils.hpp` and `src/vector_db/IngestUtils.cpp` now expose:
  - `rebuild_faiss_index_from_sqlite_by_chunk_states(...)`
- `src/main.cpp` and `src/main_with_dataset.cpp` now wire the standalone rebuild command and reuse the existing snapshot import/export plumbing.
- The allowed rebuild states are intentionally fixed to `warm` and `hot` in this first slice so the physical-index policy matches the runtime `state-aware-dense` policy.

## Main files / modules touched

- `include/cli/CommandLineArgs.hpp`
- `src/cli/CommandLineArgs.cpp`
- `include/vector_db/SqliteVectorDB.hpp`
- `src/vector_db/SqliteVectorDB.cpp`
- `include/vector_Index/FaissIndex.hpp`
- `src/vector_db/FaissIndex.cpp`
- `include/vector_db/IngestUtils.hpp`
- `src/vector_db/IngestUtils.cpp`
- `src/main.cpp`
- `src/main_with_dataset.cpp`
- `tests/test_command_line_args.cpp`
- `tests/group_c_vectordb/test_sqlite_vectordb_backend.cpp`

## Runtime path / execution flow

1. The user runs `mobile_rag --rebuild-state-filtered-index ...`.
2. The CLI opens the SQLite database.
3. If `--state-snapshot-in` is provided, the CLI imports that snapshot first.
4. SQLite reports the stored vector dimension and exports only vectors whose chunk state is `warm` or `hot`.
5. The rebuild utility creates a Faiss index with the requested factory description.
6. If any `warm` / `hot` vectors exist, they are inserted with their original chunk IDs.
7. If no `warm` / `hot` vectors exist, the rebuild utility creates an empty Faiss index with the SQLite vector dimension.
8. The Faiss index is written to `--index-path`.
9. If `--state-snapshot-out` is provided, the CLI exports the final chunk-state snapshot after rebuild.

## Config flags / thresholds / defaults

- `--rebuild-state-filtered-index`
  - standalone offline command
- `--sqlite-db <path>`
  - required for the rebuild command
- `--index-path <path>`
  - required output Faiss index path
- `--faiss-type <desc>`
  - optional Faiss factory description, same as existing build/query commands
- `--state-snapshot-in <path>`
  - optional deterministic snapshot import before rebuild
- `--state-snapshot-out <path>`
  - optional snapshot export after rebuild
- Current allowed dense states for physical rebuild:
  - `warm`
  - `hot`

## Fallback behavior

- If SQLite contains no vectors at all, rebuild fails with an error.
- If SQLite contains vectors but no `warm` / `hot` chunk, rebuild still succeeds and writes an empty Faiss index.
- If snapshot import fails, the rebuild command stops before writing the output index.
- If the output index cannot be written, the command exits with an error.

## Schema or storage changes

None.

This patch reuses the existing `vectors` and `chunk_states` tables in SQLite and writes a standard Faiss index file.

## Metrics / logs added

- Rebuild logs now report:
  - input SQLite DB path
  - output Faiss path
  - fixed allowed state set (`warm,hot`)
  - whether a snapshot was imported/exported
- The rebuild utility also logs how many vectors were written into the state-filtered Faiss index.

## How to test / reproduce

1. Rebuild focused host targets:
   - `cmake --build build_progress_check --target mobile_rag_cli mobile_rag_dataset test_command_line_args test_sqlite_vectordb_backend -j4`
2. Run focused host tests:
   - `ctest --test-dir build_progress_check -R 'CommandLineArgsTest|SqliteVectorDBBackendTest' --output-on-failure`
3. Rebuild a warm/hot-only Faiss subindex from an existing SQLite DB:
   - `mobile_rag --rebuild-state-filtered-index --sqlite-db <sqlite.db> --index-path <filtered.faiss> --state-snapshot-in <snapshot.tsv>`
4. Verify the rebuilt index can be loaded and queried.
5. Verify an all-cold snapshot still produces a loadable empty Faiss index.

## Known limitations / TODOs

- The rebuild policy is still fixed to `warm` / `hot`; there is no user-configurable state set yet.
- This is an offline rebuild path, not an online Faiss eviction or background maintenance loop.
- The command rewrites a whole Faiss index; it does not perform incremental add/remove updates.
- Runtime query paths still decide dense eligibility independently; they do not yet auto-detect whether a physically state-filtered Faiss file is being used.
