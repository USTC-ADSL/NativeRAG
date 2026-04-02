#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "loader/IDocumentLoader.hpp"

namespace mobile_rag {

class TextFileLoader : public IDocumentLoader {
 public:
  TextFileLoader(unsigned int num_threads = 1, size_t chunk_size = 1000,
                 size_t chunk_overlap = 200);
  ~TextFileLoader() override = default;

  std::vector<std::string> load_and_split(const std::string& file_path) override;

  void set_num_threads(unsigned int num_threads) override;

  void set_chunking(size_t chunk_size, size_t chunk_overlap) override;

 private:
  unsigned int num_threads_ = 1;
  size_t chunk_size_ = 1000;
  size_t chunk_overlap_ = 200;
};

}  // namespace mobile_rag


