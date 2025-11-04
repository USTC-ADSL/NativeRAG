#include "mobile_rag/loader/TextFileLoader.hpp"

#include <iostream>
#include <string>
#include <vector>

#include "BaseLoader.hpp"
#include "TxtLoader.hpp"
#include "../Chunk/CharacterSplitter.hpp"

namespace mobile_rag {

std::vector<std::string> TextFileLoader::load_and_split(const std::string& file_path) {
  // Phase 1: only .txt
  TxtLoader loader(/*num_threads=*/1);
  auto texts = loader.load_texts(file_path);
  if (texts.empty()) {
    std::cerr << "[TextFileLoader] No .txt content found at: " << file_path << '\n';
    return {};
  }

  constexpr size_t kChunkSize = 1000;
  constexpr size_t kOverlap = 200;
  CharacterSplitter splitter(kChunkSize, kOverlap);

  std::vector<std::string> chunks;
  for (const auto& t : texts) {
    auto c = splitter.split(t);
    chunks.insert(chunks.end(), c.begin(), c.end());
  }
  return chunks;
}

}  // namespace mobile_rag



