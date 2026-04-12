#pragma once

#include <cstdio>

#include "ggml.h"

namespace mobile_rag {

bool should_emit_llama_log(enum ggml_log_level level, bool previous_emitted);
void set_llama_log_sink(FILE* sink);
void forward_llama_log(enum ggml_log_level level, const char* text, void* user_data);
void install_quiet_llama_logging();

}  // namespace mobile_rag
