#pragma once

#include <string>
#include <vector>

#include "llama.h"

#include "llm/ILargeLanguageModel.hpp"

namespace mobile_rag {

class LlamaCppModel : public ILargeLanguageModel {
 public:
  LlamaCppModel() = default;
  ~LlamaCppModel() override = default;

  bool load_model(const std::string& /*model_path*/) override { return false; }

  std::string build_prompt(const std::string& /*query*/,
                           const std::vector<std::string>& /*contexts*/) override { return ""; }

  std::string generate(const std::string& /*prompt*/) override { return ""; }

 private:
  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
};

}  // namespace mobile_rag


