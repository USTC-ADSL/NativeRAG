#include "llm/LlamaCppLogging.hpp"

#include <cstdio>

#include "llama.h"

namespace mobile_rag {

namespace {

thread_local bool g_last_chunk_emitted = false;
FILE* g_llama_log_sink = stderr;

}  // namespace

bool should_emit_llama_log(enum ggml_log_level level, bool previous_emitted) {
  switch (level) {
    case GGML_LOG_LEVEL_WARN:
    case GGML_LOG_LEVEL_ERROR:
      return true;
    case GGML_LOG_LEVEL_CONT:
      return previous_emitted;
    case GGML_LOG_LEVEL_NONE:
    case GGML_LOG_LEVEL_DEBUG:
    case GGML_LOG_LEVEL_INFO:
    default:
      return false;
  }
}

void set_llama_log_sink(FILE* sink) {
  g_llama_log_sink = sink ? sink : stderr;
  g_last_chunk_emitted = false;
}

void forward_llama_log(enum ggml_log_level level, const char* text, void* /*user_data*/) {
  const bool emit = should_emit_llama_log(level, g_last_chunk_emitted);
  g_last_chunk_emitted = emit;

  if (!emit || text == nullptr) {
    return;
  }

  std::fputs(text, g_llama_log_sink ? g_llama_log_sink : stderr);
  std::fflush(g_llama_log_sink ? g_llama_log_sink : stderr);
}

void install_quiet_llama_logging() {
  set_llama_log_sink(stderr);
  ggml_log_set(forward_llama_log, nullptr);
  llama_log_set(forward_llama_log, nullptr);
}

}  // namespace mobile_rag
