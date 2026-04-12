#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "llm/LlamaCppLogging.hpp"

using namespace mobile_rag;

namespace {

std::string read_file(FILE* file) {
  std::fflush(file);
  std::rewind(file);

  std::string out;
  char buffer[256];
  while (const size_t read = std::fread(buffer, 1, sizeof(buffer), file)) {
    out.append(buffer, read);
  }
  return out;
}

}  // namespace

int main() {
  assert(!should_emit_llama_log(GGML_LOG_LEVEL_INFO, false));
  assert(!should_emit_llama_log(GGML_LOG_LEVEL_DEBUG, false));
  assert(should_emit_llama_log(GGML_LOG_LEVEL_WARN, false));
  assert(should_emit_llama_log(GGML_LOG_LEVEL_ERROR, false));
  assert(!should_emit_llama_log(GGML_LOG_LEVEL_CONT, false));
  assert(should_emit_llama_log(GGML_LOG_LEVEL_CONT, true));

  FILE* file = std::tmpfile();
  assert(file != nullptr);

  set_llama_log_sink(file);
  forward_llama_log(GGML_LOG_LEVEL_INFO, "info\n", nullptr);
  forward_llama_log(GGML_LOG_LEVEL_WARN, "warn ", nullptr);
  forward_llama_log(GGML_LOG_LEVEL_CONT, "continued\n", nullptr);
  forward_llama_log(GGML_LOG_LEVEL_ERROR, "error\n", nullptr);

  const auto captured = read_file(file);
  std::fclose(file);

  assert(captured.find("info") == std::string::npos);
  assert(captured.find("warn continued") != std::string::npos);
  assert(captured.find("error") != std::string::npos);

  std::cout << "Llama logging test passed\n";
  return 0;
}
