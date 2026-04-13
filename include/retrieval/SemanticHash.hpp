#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mobile_rag {

// Phase 2 baseline: deterministic sign-based semantic hash.
// This is a cheap heuristic representation, not a learned controller.
inline constexpr std::size_t kDefaultSemanticHashBits = 128;

std::vector<std::uint8_t> build_sign_semantic_hash(
    const std::vector<float>& embedding,
    std::size_t bit_count = kDefaultSemanticHashBits);

int hamming_distance(const std::vector<std::uint8_t>& lhs,
                     const std::vector<std::uint8_t>& rhs);

}  // namespace mobile_rag
