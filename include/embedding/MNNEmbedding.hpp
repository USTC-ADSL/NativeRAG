#pragma once

#include <memory>
#include <string>
#include <vector>

#include "llm/llm.hpp"

#include "embedding/IEmbeddingModel.hpp"

namespace mobile_rag {

class MNNEmbedding : public IEmbeddingModel {
 public:
  MNNEmbedding() = default;
  ~MNNEmbedding() override = default;

  bool load_model(const std::string& model_path) override;

  std::vector<float> embed_query(const std::string& text) override;

  std::vector<std::vector<float>> embed_documents(
      const std::vector<std::string>& texts) override;

 private:
  std::unique_ptr<MNN::Transformer::Embedding> embedding_;
  int embed_dim_ = 384;
  bool model_loaded_ = false;
};

}  // namespace mobile_rag


