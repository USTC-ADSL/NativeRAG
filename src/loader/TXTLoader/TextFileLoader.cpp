#include "loader/TextFileLoader.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "BaseLoader.hpp"
#include "TxtLoader.hpp"
#include "../Chunk/CharacterSplitter.hpp"

namespace mobile_rag {

TextFileLoader::TextFileLoader(unsigned int num_threads, size_t chunk_size,
                               size_t chunk_overlap)
    : num_threads_(std::max(1u, num_threads)),
      chunk_size_(chunk_size > 0 ? chunk_size : 1000),
      chunk_overlap_(std::min(chunk_overlap, chunk_size_ > 0 ? chunk_size_ - 1 : 0)) {}

void TextFileLoader::set_num_threads(unsigned int num_threads) {
  num_threads_ = std::max(1u, num_threads);
}

void TextFileLoader::set_chunking(size_t chunk_size, size_t chunk_overlap) {
  if (chunk_size == 0) {
    return;
  }

  chunk_size_ = chunk_size;
  chunk_overlap_ = std::min(chunk_overlap, chunk_size_ - 1);
}

std::vector<std::string> TextFileLoader::load_and_split(const std::string& file_path) {
  // Phase 1: only .txt
  TxtLoader loader(num_threads_);
  auto texts = loader.load_texts(file_path);
  if (texts.empty()) {
    std::cerr << "[TextFileLoader] No .txt content found at: " << file_path << '\n';
    return {};
  }

  CharacterSplitter splitter(chunk_size_, chunk_overlap_);

  std::vector<std::string> chunks;
  for (const auto& t : texts) {
    auto c = splitter.split(t);
    chunks.insert(chunks.end(), c.begin(), c.end());
  }
  return chunks;
}

}  // namespace mobile_rag


