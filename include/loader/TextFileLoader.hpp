#pragma once

#include <string>
#include <vector>

#include "mobile_rag/loader/IDocumentLoader.hpp"

namespace mobile_rag {

class TextFileLoader : public IDocumentLoader {
 public:
  TextFileLoader() = default;
  ~TextFileLoader() override = default;

  std::vector<std::string> load_and_split(const std::string& file_path) override;
};

}  // namespace mobile_rag



