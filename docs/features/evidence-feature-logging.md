# Evidence Feature Logging

## Purpose / motivation

Phase 3 in `AGENTS.md` calls for evidence sufficiency features before introducing a full outer controller. This patch adds the first reusable evidence-feature extractor so query-time traces can expose controller inputs before any graph-selection policy is implemented.

## What behavior changed

- Query-time retrieval now computes and logs a small evidence feature vector after chunk retrieval and before prompt construction.
- The current feature set includes:
  - top retrieval score
  - second retrieval score
  - score margin
  - score sharpness
  - query term count
  - covered query term count
  - coverage ratio
  - retrieved chunk count
  - numeric constraint count / covered count / unresolved count
  - year-like constraint count / covered count / unresolved count
  - entity-like term count / covered count / unresolved count
  - total unresolved constraint count used as a controller-facing sufficiency signal
- A focused regression test validates the feature calculations on a deterministic example.

## Key implementation points

- `include/controller/EvidenceFeatures.hpp` defines the feature struct and extraction entry point.
- `src/controller/EvidenceFeatures.cpp` implements the current heuristic extractor.
- The extractor intentionally ignores a small set of common question words and stopwords so coverage focuses more on evidence-bearing content terms.
- Numeric constraints and year-like constraints are tracked separately from generic lexical coverage.
- Entity-like terms are currently identified heuristically from capitalized query tokens after stopword filtering.
- `src/RAGPipeline.cpp` now logs the computed feature vector under an `[EVIDENCE]` line.

## Main files / modules touched

- `include/controller/EvidenceFeatures.hpp`
- `src/controller/EvidenceFeatures.cpp`
- `src/RAGPipeline.cpp`
- `tests/test_evidence_features.cpp`
- `tests/CMakeLists.txt`

## Runtime path / execution flow

1. Retrieval produces ranked results and chunk texts.
2. `RAGPipeline` calls `compute_evidence_features(...)`.
3. The extractor derives score-based and coverage-based features.
4. `RAGPipeline` emits the `[EVIDENCE]` log line.
5. The existing prompt and generation path continue unchanged.

## Config flags / thresholds / defaults

- No new user-facing flags are added.
- The current feature extractor is always enabled for `RAGPipeline` query execution.

## Fallback behavior

- If there are no retrieval results, score-derived features remain zero.
- If there are no evidence-bearing query terms after filtering, coverage-derived features remain zero.
- Logging the feature vector does not change retrieval fallback behavior.

## Schema or storage changes

None.

## Metrics / logs added

- Adds a query-time `[EVIDENCE]` log line with:
  - `top_score`
  - `second_score`
  - `score_margin`
  - `score_sharpness`
  - `query_terms`
  - `covered_terms`
  - `coverage_ratio`
  - `unresolved_constraints`
  - `unresolved_numeric`
  - `unresolved_year`
  - `unresolved_entity_terms`
  - `retrieved_chunks`

## How to test / reproduce

1. Configure a host test build:
   - `cmake -S . -B build_progress_check -DUSE_PREBUILT=ON -DLLM_BACKEND=LlamaCpp -DBUILD_TESTS=ON`
2. Build the focused tests:
   - `cmake --build build_progress_check --target mobile_rag_cli test_evidence_features -j4`
3. Run the regression:
   - `ctest --test-dir build_progress_check -R 'EvidenceFeaturesTest' --output-on-failure`
4. Run a normal CLI query and confirm an `[EVIDENCE]` line appears in the logs.

## Known limitations / TODOs

- The current feature set is heuristic and intentionally small.
- The controller now consumes only a subset of these features; there is still no learned evidence policy.
- The controller now consumes unresolved-constraint signals, but the feature set still does not measure citation span availability or structured evidence slots.
- Stopword filtering is intentionally minimal and not a full linguistic normalization pipeline.
- Entity-like term detection is heuristic and currently based only on capitalization patterns in the raw query text.
