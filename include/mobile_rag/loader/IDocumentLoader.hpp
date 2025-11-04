#pragma once

#include <string>
#include <vector>

namespace mobile_rag {

class IDocumentLoader {
 public:
  virtual ~IDocumentLoader() = default;

  virtual std::vector<std::string> load_and_split(const std::string& file_path) = 0;
};

}  // namespace mobile_rag


