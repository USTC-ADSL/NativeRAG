#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

#include "reranker/IReranker.hpp"

namespace mobile_rag {

class LlamaCppReranker : public IReranker {
 public:
  explicit LlamaCppReranker(int num_threads = 4, int context_size = 4096);
  ~LlamaCppReranker() override;

  LlamaCppReranker(const LlamaCppReranker&) = delete;
  LlamaCppReranker& operator=(const LlamaCppReranker&) = delete;

  bool load_model(const std::string& model_path) override;

  std::vector<float> score(
      const std::string& query,
      const std::vector<std::string>& documents) override;

 private:
  std::string build_prompt(const std::string& query,
                           const std::string& document) const;
  void unload_model();

  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  const llama_vocab* vocab_ = nullptr;
  std::string rerank_template_;
  llama_token true_token_ = LLAMA_TOKEN_NULL;
  llama_token false_token_ = LLAMA_TOKEN_NULL;
  int num_threads_ = 4;
  int context_size_ = 4096;
  bool use_rank_pooling_ = false;
  bool runtime_acquired_ = false;
  std::mutex mutex_;
};

}  // namespace mobile_rag
