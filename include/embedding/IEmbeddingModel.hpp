#pragma once

#include <string>
#include <vector>

namespace mobile_rag {

class IEmbeddingModel {
 public:
  virtual ~IEmbeddingModel() = default;

  virtual bool load_model(const std::string& model_path) = 0;

  virtual std::vector<float> embed_query(const std::string& text) = 0;

  virtual std::vector<std::vector<float>> embed_documents(
      const std::vector<std::string>& texts) = 0;

  virtual void set_num_threads(int /*num_threads*/) {}
};

}  // namespace mobile_rag

