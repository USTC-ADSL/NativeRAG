#pragma once

#include <string>
#include <vector>

namespace mobile_rag {

class CharacterSplitter {
 public:
  CharacterSplitter(size_t chunk_size, size_t overlap)
      : chunk_size_(chunk_size), overlap_(overlap) {}

  std::vector<std::string> split(const std::string& input) const;

 private:
  size_t chunk_size_;
  size_t overlap_;
};

}  // namespace mobile_rag



