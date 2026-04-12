# Llama Build And Runtime Plumbing Update

## Purpose / motivation

This change tightens the current `llama.cpp` integration so the repo's default backend is easier to build, quieter at runtime, and easier to verify through the standard test entry points.

## What behavior changed

- Android builds now vendor SQLite from `third_party/sqlite-vec/vendor/sqlite3.c` instead of assuming the NDK provides SQLite headers and a link target.
- `llama.cpp` builds now link `ggml-base` in addition to the existing `llama` and `ggml` libraries.
- `LlamaCppModel` installs a quiet log sink during backend initialization so only warnings, errors, and continuation chunks from those log lines reach stderr.
- `CommandLineArgs::parse()` now treats `--help` and `-h` as an early exit even when they appear after the primary command.
- The new command-line and logging regression tests are now intended to run through the normal `ctest` entry point instead of manual invocation only.

## Key implementation points

- `CMakeLists.txt` switches the project to `LANGUAGES C CXX` so the Android build can compile the vendored `sqlite3.c` translation unit.
- `CMakeLists.txt` centralizes SQLite include/library/source selection so both `mobile_rag_cli` and `mobile_rag_dataset` use the same Android and host logic.
- `src/llm/LlamaCppLogging.cpp` implements the log filter and sink wiring used by `src/llm/LlamaCppModel.cpp`.
- `tests/CMakeLists.txt` now registers the new regression binaries with CTest and gives the llama logging test the same prebuilt runtime search path as the rest of the test suite.

## Main files / modules touched

- `CMakeLists.txt`
- `README.md`
- `src/cli/CommandLineArgs.cpp`
- `include/llm/LlamaCppLogging.hpp`
- `src/llm/LlamaCppLogging.cpp`
- `src/llm/LlamaCppModel.cpp`
- `tests/CMakeLists.txt`
- `tests/test_command_line_args.cpp`
- `tests/test_llama_logging.cpp`

## Runtime path / execution flow

1. `LlamaCppModel` hits `initialize_llama_backend_once()`.
2. The backend installs the quiet log callback before calling `llama_backend_init()` and `ggml_backend_load_all()`.
3. Later llama/ggml log events flow through `forward_llama_log()`.
4. Info/debug chunks are dropped; warn/error chunks and their continuations are forwarded to the configured sink.
5. The CLI parser now exits immediately when help is requested anywhere in the argument tail, avoiding unrelated validation errors.

## Config flags / thresholds / defaults

- `LLM_BACKEND` still supports `MNN`, `MLLM`, and `LlamaCpp`; the default remains `LlamaCpp`.
- `BUILD_TESTS` still defaults to `ON`.
- The quiet logging filter keeps:
  - `GGML_LOG_LEVEL_WARN`
  - `GGML_LOG_LEVEL_ERROR`
  - `GGML_LOG_LEVEL_CONT` only when it continues a previously emitted warn/error chunk

## Fallback behavior

- On host builds, SQLite still comes from `find_package(SQLite3 REQUIRED)`.
- If the runtime log sink is set to `nullptr`, the logger falls back to `stderr`.
- If `ggml-base` is unavailable, the main build still configures, but the current CMake logic only links it when found.

## Schema or storage changes

None.

## Metrics / logs added

- No new metrics are added in this patch.
- Runtime stderr output from `llama.cpp` is reduced by filtering out info/debug logging.

## How to test / reproduce

1. Configure a clean build for the default llama path:
   - `cmake -S . -B build_progress_check -DUSE_PREBUILT=ON -DLLM_BACKEND=LlamaCpp -DBUILD_TESTS=ON`
2. Build the new regression tests:
   - `cmake --build build_progress_check --target test_command_line_args test_llama_logging -j4`
3. Confirm CTest discovers them:
   - `ctest --test-dir build_progress_check -N`
4. Run the focused regression tests:
   - `ctest --test-dir build_progress_check -R 'CommandLineArgsTest|LlamaLoggingTest' --output-on-failure`

## Known limitations / TODOs

- This patch improves build/test plumbing, not the MorphRAG controller or retrieval-graph logic described in `AGENTS.md`.
- The repo still needs broader documentation alignment between `README.md` and the newer research direction in `AGENTS.md`.
- Android device-side validation on `adb -s fd8657d6` has not been run as part of this host-side cleanup.
