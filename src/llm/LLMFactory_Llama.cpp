#include "llm/LLMFactory.hpp"

#include <memory>

#include "llm/LlamaCppModel.hpp"

namespace mobile_rag {

std::shared_ptr<ILargeLanguageModel> create_llm(int num_threads,
                                                int max_tokens) {
  return std::make_shared<LlamaCppModel>(num_threads, max_tokens);
}

}  // namespace mobile_rag


