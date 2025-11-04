#include "embedding/LlamaCppEmbedding.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>

namespace mobile_rag {

bool LlamaCppEmbedding::load_model(const std::string& /*model_path*/) {
  // Placeholder: accept path and keep a default embedding dimension.
  // A real implementation would initialize llama.cpp model/context here.
  embed_dim_ = std::max(1, embed_dim_);
  return true;
}

static std::vector<float> hash_embed(const std::string& text, int dim) {
  std::vector<float> vec(static_cast<size_t>(dim), 0.0f);
  if (dim <= 0) return vec;
  uint32_t h1 = 2166136261u;
  uint32_t h2 = 16777619u;
  for (char c : text) {
    unsigned char uc = static_cast<unsigned char>(c);
    h1 ^= uc;
    h1 *= 16777619u;
    h2 = (h2 ^ (uc + 0x9e3779b9u + (h2 << 6) + (h2 >> 2)));
  }
  for (int i = 0; i < dim; ++i) {
    uint32_t mix = h1 ^ (h2 + 0x9e3779b9u + (static_cast<uint32_t>(i) << 6) +
                        (static_cast<uint32_t>(i) >> 2));
    float v = static_cast<float>((mix % 2000) - 1000) / 1000.0f;
    vec[static_cast<size_t>(i)] = v;
  }
  float norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f));
  if (norm > 1e-6f) {
    for (float& x : vec) x /= norm;
  }
  return vec;
}

std::vector<float> LlamaCppEmbedding::embed_query(const std::string& text) {
  return hash_embed(text, embed_dim_);
}

std::vector<std::vector<float>> LlamaCppEmbedding::embed_documents(
    const std::vector<std::string>& texts) {
  std::vector<std::vector<float>> out;
  out.reserve(texts.size());
  for (const std::string& t : texts) {
    out.push_back(embed_query(t));
  }
  return out;
}

}  // namespace mobile_rag



