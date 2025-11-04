#include "llm/LLMFactory.hpp"

#include <iostream>
#include <memory>

namespace mobile_rag {

std::shared_ptr<ILargeLanguageModel> create_llm() {
  std::cerr << "[LLMFactory] MLLM backend selected but not implemented." << '\n';
  return nullptr;
}

}  // namespace mobile_rag



