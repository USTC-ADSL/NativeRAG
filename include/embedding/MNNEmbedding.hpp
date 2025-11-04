#pragma once

#include <memory>
#include <string>
#include <vector>

#include "MNN/Interpreter.hpp"
#include "MNN/Session.hpp"

#include "mobile_rag/embedding/IEmbeddingModel.hpp"

namespace mobile_rag {

class MNNEmbedding : public IEmbeddingModel {
 public:
  MNNEmbedding() = default;
  ~MNNEmbedding() override = default;

  bool load_model(const std::string& /*model_path*/) override { return false; }

  std::vector<float> embed_query(const std::string& /*text*/) override { return {}; }

  std::vector<std::vector<float>> embed_documents(
      const std::vector<std::string>& /*texts*/) override { return {}; }

 private:
  std::shared_ptr<MNN::Interpreter> interpreter_;
  MNN::Session* session_ = nullptr;
};

}  // namespace mobile_rag


