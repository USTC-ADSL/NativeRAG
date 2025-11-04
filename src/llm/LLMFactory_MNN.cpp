#include "llm/LLMFactory.hpp"

#include <memory>

#include "llm/MNNModel.hpp"

namespace mobile_rag {

std::shared_ptr<ILargeLanguageModel> create_llm() {
  return std::make_shared<MNNModel>();
}

}  // namespace mobile_rag



