# Semantic Hash Prefilter Baseline

## Purpose / motivation

This document records the first concrete Phase 2 slice for MorphRAG: adding a lightweight semantic hash representation that can be persisted in SQLite and queried through a native Hamming-distance prefilter.

The immediate goal is not to finish the full `FTS/hash -> Faiss` static pipeline in one patch. The goal is to establish the representation and prefilter substrate that later retrieval-path wiring can build on.

## What behavior changed

- The SQLite metadata store now creates and maintains a `semantic_hashes` table.
- `RAGPipeline` now persists dense vectors, text chunks, and semantic hashes into SQLite during indexing.
- Query-time retrieval can now optionally run a static semantic-hash shortlist before dense reranking.
- `SqliteVectorDB` now exposes:
  - a native Hamming-distance lookup API for semantic-hash prefiltering
  - a filtered dense rerank API that only scores a supplied candidate ID shortlist
- Focused regressions validate:
  - hash generation, persistence, reopening, and nearest-neighbor lookup by Hamming distance
  - query-time prefilter use
  - fallback to the original dense search path when the shortlist is empty
  - CLI parsing for the new retrieval flags

## Key implementation points

- `include/retrieval/SemanticHash.hpp` and `src/retrieval/SemanticHash.cpp` define a deterministic sign-based semantic hash baseline.
- The current hash generator is heuristic, not learned:
  - it mixes embedding dimensions deterministically
  - it writes one bit per mixed projection sign
  - it defaults to `128` bits
- `SqliteVectorDB` adds:
  - `add_semantic_hashes(...)`
  - `search_by_semantic_hash(...)`
  - `search_with_ids(...)`
- `RAGPipeline::add_text_embeddings(...)` now persists:
  - dense vectors into SQLite
  - text chunks into SQLite
  - semantic hashes into SQLite
- `RAGPipeline::answer_query(...)` now supports an optional static retrieval path:
  - build a query semantic hash from the query embedding
  - fetch a shortlist by Hamming distance from SQLite
  - rerank only that shortlist against dense vectors stored in SQLite
  - fall back to the existing dense index search if the shortlist or rerank result is empty
- `CommandLineArgs` and both CLI entrypoints now expose and wire the prefilter configuration.

## Main files / modules touched

- `include/retrieval/SemanticHash.hpp`
- `src/retrieval/SemanticHash.cpp`
- `include/vector_db/SqliteVectorDB.hpp`
- `src/vector_db/SqliteVectorDB.cpp`
- `src/RAGPipeline.cpp`
- `include/cli/CommandLineArgs.hpp`
- `src/cli/CommandLineArgs.cpp`
- `src/main.cpp`
- `src/main_with_dataset.cpp`
- `tests/group_c_vectordb/test_semantic_hashing.cpp`
- `tests/test_rag_semantic_hash_prefilter.cpp`
- `tests/test_command_line_args.cpp`
- `tests/CMakeLists.txt`

## Runtime path / execution flow

1. Offline indexing loads documents and computes embeddings as before.
2. `RAGPipeline` validates vectors and assigns stable IDs.
3. The pipeline persists:
   - vectors into Faiss
   - vectors into SQLite
   - texts into SQLite
   - semantic hash codes into SQLite
4. When `--semantic-hash-prefilter` is enabled at query time:
   - the query is embedded
   - a deterministic semantic hash is derived from that embedding
   - SQLite returns the closest `semantic_hash_candidate_limit` chunk IDs by Hamming distance
   - SQLite reranks only those IDs using stored dense vectors
5. If the semantic-hash shortlist is empty, or the SQLite dense rerank cannot return results, the query falls back to the original dense index search.
6. The retrieved chunks are then assembled into the LLM prompt as before.

This is intentionally still a static Phase 2 path. It is not yet the full `FTS/hash -> Faiss` baseline and it is not an adaptive controller.

## Config flags / thresholds / defaults

- Default semantic hash length: `128` bits
- Current generator type: deterministic sign-hash heuristic
- Current prefilter metric: Hamming distance
- Query-time prefilter is disabled by default
- `--semantic-hash-prefilter`
  - enables the static semantic-hash shortlist path
- `--semantic-hash-candidates`
  - default: `32`
  - sets the shortlist size before dense rerank
- `--semantic-hash-max-distance`
  - default: `-1`
  - `-1` disables the Hamming-distance cap
  - any non-negative value drops candidates farther than that threshold
- Hash bit length is still fixed in code and is not yet exposed as a CLI flag

## Fallback behavior

- If dense-vector persistence to SQLite fails, the pipeline logs a warning and preserves the existing Faiss-backed path.
- If text or semantic-hash persistence fails, the pipeline logs a warning and keeps the existing path alive.
- If semantic-hash prefiltering is disabled, the query uses the original dense search path.
- If semantic-hash prefiltering is enabled but SQLite is unavailable, the query falls back to the original dense search path.
- If the semantic-hash shortlist is empty, the query falls back to the original dense search path.
- If the SQLite dense rerank over the shortlist cannot return results, the query falls back to the original dense search path.
- If semantic-hash lookup receives invalid or mismatched byte lengths, candidates are skipped rather than crashing.

## Schema or storage changes

Adds a new SQLite table:

```sql
CREATE TABLE IF NOT EXISTS semantic_hashes (
  id INTEGER PRIMARY KEY,
  bit_count INTEGER NOT NULL,
  code BLOB NOT NULL
);
```

This table uses the same chunk/document IDs already used by the existing `vectors` and `texts` tables.

No new schema table is required for query-time reranking because the repo already had a `vectors` table in SQLite. This patch changes runtime behavior so that `RAGPipeline` now actually populates that canonical store during indexing.

## Metrics / logs added

- Offline indexing now logs how many dense vectors, text chunks, and semantic hashes were persisted into SQLite.
- Query-time retrieval now logs:
  - retrieval mode
  - semantic-hash candidate count
  - final dense result count
  - fallback reason
- No new latency or memory metrics are added in this first Phase 2 slice.

## How to test / reproduce

1. Configure a host test build:
   - `cmake -S . -B build_progress_check -DUSE_PREBUILT=ON -DLLM_BACKEND=LlamaCpp -DBUILD_TESTS=ON`
2. Build the focused tests:
   - `cmake --build build_progress_check --target test_command_line_args test_semantic_hashing test_rag_semantic_hash_prefilter -j4`
3. Run the tests:
   - `ctest --test-dir build_progress_check -R 'CommandLineArgsTest|SemanticHashingTest|RAGSemanticHashPrefilterTest' --output-on-failure`
4. Run the CLI path manually if needed:
   - build an index as usual
   - query with `--semantic-hash-prefilter`
   - optionally tune `--semantic-hash-candidates` and `--semantic-hash-max-distance`

## Known limitations / TODOs

- The current dense rerank over the semantic-hash shortlist runs from SQLite, not a constrained Faiss search.
- This patch still does not add SQLite FTS or the full static `FTS/hash -> Faiss` retrieval pipeline.
- The hash generator is a deterministic heuristic baseline, not a learned representation.
- Hash bit length is fixed in code.
- Query-time thresholds, graph selection, and adaptive escalation remain later Phase 2+ work.
