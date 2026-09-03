#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

#include "embedding/IEmbeddingModel.hpp"

namespace mobile_rag {

class LlamaCppEmbedding : public IEmbeddingModel {
 public:
  explicit LlamaCppEmbedding(int num_threads = 4, int context_size = 2048,
                             int batch_size = 2048);
  ~LlamaCppEmbedding() override;

  LlamaCppEmbedding(const LlamaCppEmbedding&) = delete;
  LlamaCppEmbedding& operator=(const LlamaCppEmbedding&) = delete;

  bool load_model(const std::string& model_path) override;

  std::vector<float> embed_query(const std::string& text) override;

  std::vector<std::vector<float>> embed_documents(
      const std::vector<std::string>& texts) override;

 private:
  std::vector<std::vector<float>> embed_texts(
      const std::vector<std::string>& texts);
  void unload_model();

  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  const llama_vocab* vocab_ = nullptr;
  int embed_dim_ = 0;
  int num_threads_ = 4;
  int context_size_ = 2048;
  int batch_size_ = 2048;
  bool runtime_acquired_ = false;
  std::mutex mutex_;
};

}  // namespace mobile_rag


