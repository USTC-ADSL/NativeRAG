#include "retrieval/SemanticHash.hpp"

#include <limits>

namespace mobile_rag {

std::vector<std::uint8_t> build_sign_semantic_hash(const std::vector<float>& embedding,
                                                   std::size_t bit_count) {
  if (embedding.empty() || bit_count == 0) {
    return {};
  }

  const std::size_t byte_count = (bit_count + 7) / 8;
  std::vector<std::uint8_t> code(byte_count, 0);
  const std::size_t dim = embedding.size();

  for (std::size_t bit = 0; bit < bit_count; ++bit) {
    const std::size_t primary_index = bit % dim;
    const std::size_t mixed_index = (bit * 31 + 7) % dim;
    float mixed_value = embedding[primary_index];
    if (dim > 1) {
      mixed_value += 0.5f * embedding[mixed_index];
    }

    if (mixed_value >= 0.0f) {
      code[bit / 8] |= static_cast<std::uint8_t>(1u << (bit % 8));
    }
  }

  return code;
}

int hamming_distance(const std::vector<std::uint8_t>& lhs,
                     const std::vector<std::uint8_t>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<int>::max();
  }

  int distance = 0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    distance += __builtin_popcount(
        static_cast<unsigned int>(lhs[i] ^ rhs[i]));
  }
  return distance;
}

}  // namespace mobile_rag
