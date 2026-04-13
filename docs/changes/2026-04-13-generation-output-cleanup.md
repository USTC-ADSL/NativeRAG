# Generation Output Cleanup Tightening

## Purpose / motivation

Real-device validation on `fd8657d6` showed that the current `llama.cpp` path could still emit an answer followed by an untagged reasoning-style tail such as "Okay, so the user is asking ...".

This patch tightens post-processing so benchmark-facing and user-facing output is closer to the intended "final answer only" contract without changing retrieval logic or model sampling behavior.

## What behavior changed

- `cleanup_generation_output(...)` now strips a trailing meta-reasoning block when the model emits a valid answer line first and then switches into deliberation-like narration.
- The cleanup path still preserves the existing `<think>` / `<analysis>` tag removal and `Answer:` prefix stripping.
- A focused regression test now covers the exact "answer + meta reasoning tail" failure mode seen during device testing.

## Key implementation points

- `include/llm/PromptUtils.hpp` adds a narrow line-based heuristic:
  - preserve the first answer-bearing lines
  - stop once a later line looks like model-side deliberation rather than answer content
- The heuristic only truncates after answer content has already been collected, which avoids turning a pure answer into an empty string.
- The new regression lives in `tests/test_prompt_utils.cpp`.

## Main files / modules touched

- `include/llm/PromptUtils.hpp`
- `tests/test_prompt_utils.cpp`
- `tests/CMakeLists.txt`

## Runtime path / execution flow

1. The model generates raw text.
2. `cleanup_generation_output(...)` removes `<think>` / `<analysis>` blocks.
3. It strips a leading `Answer:` prefix when present.
4. It keeps answer-bearing lines and truncates later meta-reasoning lines such as "Okay, ..." or "The user is ...".
5. The cleaned string is returned to the caller.

## Config flags / thresholds / defaults

- No new user-facing flags are added.
- The cleanup heuristic is always on for backends that already call `cleanup_generation_output(...)`.

## Fallback behavior

- If the generated text is already clean, it is returned unchanged.
- If cleanup removes tag blocks and still leaves no content, existing caller behavior remains unchanged.
- If the model emits only a direct answer line, no truncation happens.

## Schema or storage changes

None.

## Metrics / logs added

- No new metrics are added.
- No new logs are added.

## How to test / reproduce

1. Configure a host test build:
   - `cmake -S . -B build_progress_check -DUSE_PREBUILT=ON -DLLM_BACKEND=LlamaCpp -DBUILD_TESTS=ON`
2. Build the focused tests:
   - `cmake --build build_progress_check --target test_prompt_utils test_command_line_args test_semantic_hashing test_rag_semantic_hash_prefilter test_sqlite_vectordb_backend -j4`
3. Run the focused regressions:
   - `ctest --test-dir build_progress_check -R 'PromptUtilsTest|CommandLineArgsTest|SemanticHashingTest|RAGSemanticHashPrefilterTest|SqliteVectorDBBackendTest' --output-on-failure`
4. Re-run the device query on `fd8657d6` and confirm the answer no longer includes the reasoning tail.

## Known limitations / TODOs

- The current cleanup is still heuristic, not a formal structured-output contract.
- If a model emits meta reasoning without a recognizable line boundary, this patch may not catch it.
- Longer-term, structured answer output would be more robust than post-hoc trimming.
