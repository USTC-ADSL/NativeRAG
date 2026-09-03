#pragma once

#include <memory>

#include "llm/ILargeLanguageModel.hpp"

namespace mobile_rag {

std::shared_ptr<ILargeLanguageModel> create_llm(int num_threads = 4,
                                                int max_tokens = 256);

}  // namespace mobile_rag


