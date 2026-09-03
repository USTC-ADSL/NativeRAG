#pragma once

#include <string>
#include <vector>

namespace mobile_rag {

class IReranker {
 public:
  virtual ~IReranker() = default;

  virtual bool load_model(const std::string& model_path) = 0;

  virtual std::vector<float> score(
      const std::string& query,
      const std::vector<std::string>& documents) = 0;
};

}  // namespace mobile_rag
