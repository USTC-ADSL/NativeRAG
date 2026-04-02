#pragma once

#include <string>
#include <vector>

#include "llama.h"

#include "llm/ILargeLanguageModel.hpp"

namespace mobile_rag {

class LlamaCppModel : public ILargeLanguageModel {
 public:
  LlamaCppModel() = default;
  ~LlamaCppModel() override;

  bool load_model(const std::string& model_path) override;

  std::string build_prompt(const std::string& query,
                           const std::vector<std::string>& contexts) override;

  std::string generate(const std::string& prompt) override;

  void set_num_threads(int num_threads) override;

  void set_max_new_tokens(int max_new_tokens) override;

 private:
  void unload();
  bool initialize_context();
  bool initialize_sampler();
  std::string resolve_model_path(const std::string& model_path) const;
  bool tokenize_prompt(const std::string& prompt,
                       std::vector<llama_token>& prompt_tokens) const;
  std::string token_to_piece(llama_token token) const;

  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  llama_sampler* sampler_ = nullptr;
  const llama_vocab* vocab_ = nullptr;
  std::string model_path_;
  uint32_t n_ctx_ = 2048;
  int num_threads_ = 4;
  int max_new_tokens_ = 256;
};

}  // namespace mobile_rag
