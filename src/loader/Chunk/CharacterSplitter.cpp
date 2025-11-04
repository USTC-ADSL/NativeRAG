#include "CharacterSplitter.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace mobile_rag {

std::vector<std::string> CharacterSplitter::split(const std::string& input) const {
  if (chunk_size_ == 0) return {};
  size_t step = chunk_size_ > overlap_ ? (chunk_size_ - overlap_) : 1;
  size_t chunk_count = static_cast<size_t>(std::ceil(static_cast<long double>(input.size()) / static_cast<long double>(step)));

  std::vector<std::string> chunks;
  chunks.reserve(chunk_count);
  for (size_t i = 0; i < chunk_count; ++i) {
    size_t start_idx = i * step;
    size_t end_idx = start_idx + chunk_size_;
    if (end_idx > input.size()) end_idx = input.size();
    chunks.emplace_back(input.substr(start_idx, end_idx - start_idx));
  }
  return chunks;
}

}  // namespace mobile_rag



