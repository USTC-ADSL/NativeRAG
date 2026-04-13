# Heuristic Adaptive Graph Selection

## Purpose / motivation

This document records the first minimal Phase 4 outer-controller slice for MorphRAG. The goal is not to introduce a full learned policy yet. The goal is to move the repo from static prefilter flags toward explicit retrieval-graph instantiation under a cheap, disable-able heuristic controller.

This keeps the paper story intact:

- the system now decides which retrieval graph to instantiate
- the controller remains separate from the inner dense/runtime execution path
- the original baselines still remain directly runnable

## What behavior changed

- Query-time execution now supports an opt-in adaptive controller through `--adaptive-graph`.
- The controller selects an initial retrieval graph from the already implemented Phase 2 baselines:
  - `dense_only`
  - `lexical_prefilter`
  - `semantic_hash_prefilter`
  - `lexical_hash_prefilter`
- The controller now also derives a cheap retrieval-budget proxy from existing shortlist knobs and `top-k`:
  - `tight`
  - `balanced`
  - `relaxed`
- When state-aware dense mode is active, the controller also checks whether SQLite currently exposes any `warm` / `hot` dense-eligible chunk before choosing `dense_only`.
- When evidence looks weak after the first retrieval pass, the controller can upgrade to a richer graph and rerun retrieval before prompt construction.
- When lexical coverage looks strong but constraint-like query tokens remain uncovered, the controller can still escalate instead of treating the evidence as sufficient.
- Query traces now emit `[CONTROLLER]` lines alongside the existing `[RETRIEVAL]` and `[EVIDENCE]` logs.
- The old manual prefilter flags still work exactly as before when `--adaptive-graph` is not enabled.

## Key implementation points

- `include/controller/GraphSelector.hpp` and `src/controller/GraphSelector.cpp` define the explicit outer-controller module.
- The current policy is heuristic, not learned:
  - numeric queries prefer `lexical_hash_prefilter` when both shortlists are available
  - under a tight shortlist budget, numeric queries stay on `lexical_prefilter` instead of paying for the merged graph
  - term-rich queries prefer `lexical_prefilter`
  - short content-bearing queries prefer `semantic_hash_prefilter`
  - if state-aware dense is enabled and SQLite reports no current `warm` / `hot` chunk, the controller avoids starting from `dense_only` when a SQLite-backed graph exists
  - if a query somehow still reaches `dense_only` while dense is state-unavailable, the controller upgrades to the richest available SQLite-backed graph instead of treating the dense path as a valid steady state
  - unresolved numeric or entity-like query constraints can force a richer graph even when score margin and lexical coverage look acceptable
  - low evidence can trigger an upgrade to a richer graph
  - under a tight shortlist budget, evidence-based upgrades are skipped
- `RAGPipeline::answer_query(...)` now:
  - asks `GraphSelector` for the initial graph
  - derives a budget class from currently enabled shortlist sizes and `top-k`
  - executes the selected graph
  - computes evidence features
  - optionally upgrades the graph and reruns retrieval
  - logs the final retrieval path before generation
- `CommandLineArgs` and both CLI entrypoints now expose and wire the adaptive controller flag.

## Main files / modules touched

- `include/controller/GraphSelector.hpp`
- `src/controller/GraphSelector.cpp`
- `include/RAGPipeline.hpp`
- `src/RAGPipeline.cpp`
- `include/cli/CommandLineArgs.hpp`
- `src/cli/CommandLineArgs.cpp`
- `src/main.cpp`
- `src/main_with_dataset.cpp`
- `tests/test_graph_selector.cpp`
- `tests/test_rag_semantic_hash_prefilter.cpp`
- `tests/test_command_line_args.cpp`
- `tests/CMakeLists.txt`

## Runtime path / execution flow

1. The query is embedded as before.
2. If `--adaptive-graph` is disabled:
   - the retrieval path remains the existing manual baseline chosen by prefilter flags
3. If `--adaptive-graph` is enabled:
   - `GraphSelector` derives simple query features from the query text
   - `GraphSelector` derives a retrieval-budget class from `top-k`, `lexical_candidate_limit`, and `semantic_hash_candidate_limit`
   - when state-aware dense is active, `RAGPipeline` also derives a binary dense-availability signal from the current SQLite chunk-state summary
   - the controller picks the cheapest currently available graph
4. `RAGPipeline` executes the selected graph:
   - dense only
   - lexical shortlist + dense rerank
   - semantic-hash shortlist + dense rerank
   - merged lexical/hash shortlist + dense rerank
5. `RAGPipeline` computes evidence features from the first retrieval pass.
6. If evidence is weak, or if uncovered constraints remain in the retrieved evidence, and a richer graph exists, the controller upgrades the graph and reruns retrieval.
7. The final retrieval result is logged and forwarded to prompt construction and generation.

## Config flags / thresholds / defaults

- `--adaptive-graph`
  - default: disabled
  - enables heuristic outer-controller selection
- Current built-in controller thresholds:
  - lexical-rich query threshold: `3` content terms
  - weak-evidence score margin threshold: `0.15`
  - weak-evidence coverage threshold: `0.50`
  - tight shortlist budget threshold: `2 * top_k`
  - balanced shortlist budget threshold: `4 * top_k`
- These thresholds are intentionally still code defaults and are not yet exposed as CLI knobs.

## Fallback behavior

- If `--adaptive-graph` is disabled, the repo stays on the previous static retrieval behavior.
- If SQLite is unavailable, the adaptive controller falls back to `dense_only`.
- If state-aware dense is enabled and SQLite reports no `warm` / `hot` chunk, the adaptive controller avoids `dense_only` when lexical or semantic SQLite graphs are available.
- If a requested shortlist is empty, retrieval falls back to dense search as before.
- If the SQLite shortlist rerank returns no results, retrieval falls back to dense search as before.
- If the controller already selected the richest currently available graph, no further upgrade is attempted.

## Schema or storage changes

None.

This patch only changes query-time graph selection and logging. It does not change the SQLite schema, Faiss layout, or persisted semantic-hash storage.

## Metrics / logs added

- Adds query-time `[CONTROLLER]` logs for:
  - initial graph
  - derived budget class
  - selection reason
  - escalation source and destination when triggered
  - final graph
- New selection / escalation reasons related to state-aware dense availability:
  - `dense_state_unavailable`
  - `dense_state_unavailable_upgrade`
- Existing `[RETRIEVAL]` and `[EVIDENCE]` logs remain in place and now describe the final graph execution.

## How to test / reproduce

1. Configure a host test build:
   - `cmake -S . -B build_progress_check -DUSE_PREBUILT=ON -DLLM_BACKEND=LlamaCpp -DBUILD_TESTS=ON`
2. Build focused targets:
   - `cmake --build build_progress_check --target mobile_rag_cli test_evidence_features test_graph_selector test_command_line_args test_rag_semantic_hash_prefilter -j4`
3. Run focused regressions:
   - `ctest --test-dir build_progress_check -R 'GraphSelectorTest|CommandLineArgsTest|RAGSemanticHashPrefilterTest|EvidenceFeaturesTest' --output-on-failure`
4. Run a query with both prefilters and adaptive mode enabled:
   - expect `[CONTROLLER]` logs before `[RETRIEVAL]`
   - expect a `budget=tight|balanced|relaxed` field in the controller logs
   - expect the final retrieval graph to remain one of the existing baseline graphs
   - with `--state-aware-dense` and an imported snapshot whose chunks are all `cold`, expect the controller to log `reason=dense_state_unavailable` instead of starting from `dense_only`

## Known limitations / TODOs

- The current controller is heuristic and uses only simple query/evidence features.
- Budget signals are still shortlist-size proxies; there is no thermal or energy-aware policy yet.
- Dense availability is currently summary-based, not query-specific; the controller only knows whether any `warm` / `hot` chunk exists at all.
- Evidence-based upgrading is single-step only; there is no multi-hop escalation chain yet.
- Thresholds are fixed in code.
- The final rerank still uses the existing dense rerank implementation; this patch does not change the inner xPU placement logic.
- Constraint detection is heuristic and currently limited to numeric, year-like, and entity-like lexical signals.
