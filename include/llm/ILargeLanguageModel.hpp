#pragma once

#include <string>
#include <vector>

namespace mobile_rag {

class ILargeLanguageModel {
 public:
  virtual ~ILargeLanguageModel() = default;

  virtual bool load_model(const std::string& model_path) = 0;

  // Helper to build a full prompt from query and retrieved contexts
  virtual std::string build_prompt(const std::string& query,
                                   const std::vector<std::string>& contexts) = 0;

  virtual std::string generate(const std::string& prompt) = 0;
};

}  // namespace mobile_rag


