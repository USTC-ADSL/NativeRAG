#pragma once

#include <string>
#include <vector>

#include "embedding/IEmbeddingModel.hpp"

namespace mobile_rag {

class LlamaCppEmbedding : public IEmbeddingModel {
 public:
  LlamaCppEmbedding() = default;
  ~LlamaCppEmbedding() override = default;

  bool load_model(const std::string& model_path) override;

  std::vector<float> embed_query(const std::string& text) override;

  std::vector<std::vector<float>> embed_documents(
      const std::vector<std::string>& texts) override;

 private:
  int embed_dim_ = 384;
};

}  // namespace mobile_rag



