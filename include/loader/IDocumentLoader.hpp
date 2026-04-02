#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace mobile_rag {

class IDocumentLoader {
 public:
  virtual ~IDocumentLoader() = default;

  virtual std::vector<std::string> load_and_split(const std::string& file_path) = 0;

  virtual void set_num_threads(unsigned int /*num_threads*/) {}

  virtual void set_chunking(size_t /*chunk_size*/, size_t /*chunk_overlap*/) {}
};

}  // namespace mobile_rag

