#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

#include "llm/ILargeLanguageModel.hpp"

namespace mobile_rag {

class LlamaCppModel : public ILargeLanguageModel {
 public:
  explicit LlamaCppModel(int num_threads = 4, int max_tokens = 256,
                         int context_size = 4096);
  ~LlamaCppModel() override;

  LlamaCppModel(const LlamaCppModel&) = delete;
  LlamaCppModel& operator=(const LlamaCppModel&) = delete;

  bool load_model(const std::string& model_path) override;

  std::string build_prompt(const std::string& query,
                           const std::vector<std::string>& contexts) override;

  std::string generate(const std::string& prompt) override;

 private:
  void unload_model();

  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  const llama_vocab* vocab_ = nullptr;
  llama_sampler* sampler_ = nullptr;
  int num_threads_ = 4;
  int max_tokens_ = 256;
  int context_size_ = 4096;
  bool runtime_acquired_ = false;
  std::mutex mutex_;
};

}  // namespace mobile_rag

