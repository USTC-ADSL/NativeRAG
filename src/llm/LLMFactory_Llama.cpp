#include "llm/LLMFactory.hpp"

#include <memory>

#include "llm/LlamaCppModel.hpp"

namespace mobile_rag {

std::shared_ptr<ILargeLanguageModel> create_llm() {
  return std::make_shared<LlamaCppModel>();
}

}  // namespace mobile_rag



