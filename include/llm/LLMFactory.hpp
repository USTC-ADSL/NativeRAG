#pragma once

#include <memory>

#include "llm/ILargeLanguageModel.hpp"

namespace mobile_rag {

std::shared_ptr<ILargeLanguageModel> create_llm();

}  // namespace mobile_rag



