#pragma once

#include <memory>
#include <string>
#include <vector>

#include "llm/llm.hpp"

#include "llm/ILargeLanguageModel.hpp"

namespace mobile_rag {

class MNNModel : public ILargeLanguageModel {
 public:
  MNNModel() = default;
  ~MNNModel() override = default;

  bool load_model(const std::string& model_path) override;

  std::string build_prompt(const std::string& query,
                           const std::vector<std::string>& contexts) override;

  std::string generate(const std::string& prompt) override;

  void set_num_threads(int num_threads) override;

  void set_max_new_tokens(int max_new_tokens) override;

 private:
  std::unique_ptr<MNN::Transformer::Llm> llm_;
  int num_threads_ = 4;
  int max_new_tokens_ = 256;
};

}  // namespace mobile_rag


