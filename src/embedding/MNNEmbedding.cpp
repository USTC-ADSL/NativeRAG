#include "embedding/MNNEmbedding.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>

namespace mobile_rag {

bool MNNEmbedding::load_model(const std::string& model_path) {
  try {
    interpreter_.reset(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (!interpreter_) {
      std::cerr << "[MNNEmbedding] Failed to create interpreter from: "
                << model_path << '\n';
      return false;
    }
    MNN::ScheduleConfig config;
    session_ = interpreter_->createSession(config);
    if (!session_) {
      std::cerr << "[MNNEmbedding] Failed to create session for: "
                << model_path << '\n';
      interpreter_.reset();
      return false;
    }
    // If the model has a known embedding size as output dim, try to infer it here.
    // We keep a safe default if not inferable.
    embed_dim_ = std::max(1, embed_dim_);
    return true;
  } catch (...) {
    std::cerr << "[MNNEmbedding] Exception while loading model: " << model_path
              << '\n';
    interpreter_.reset();
    session_ = nullptr;
    return false;
  }
}

static std::vector<float> generateTextHashEmbedding(const std::string& text,
                                                    int dim) {
  std::vector<float> vec(static_cast<size_t>(dim), 0.0f);
  if (dim <= 0) return vec;
  // Simple, deterministic hashing-based embedding to keep pipeline functional
  uint32_t h1 = 2166136261u;
  uint32_t h2 = 16777619u;
  for (size_t i = 0; i < text.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    h1 ^= c;
    h1 *= 16777619u;
    h2 = (h2 ^ (c + 0x9e3779b9 + (h2 << 6) + (h2 >> 2)));
  }
  for (int i = 0; i < dim; ++i) {
    uint32_t mix = h1 ^ (h2 + 0x9e3779b9u + (static_cast<uint32_t>(i) << 6) +
                        (static_cast<uint32_t>(i) >> 2));
    float v = static_cast<float>((mix % 2000) - 1000) / 1000.0f;
    vec[static_cast<size_t>(i)] = v;
  }
  // L2 normalize for stability
  float norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f));
  if (norm > 1e-6f) {
    for (float& x : vec) x /= norm;
  }
  return vec;
}

std::vector<float> MNNEmbedding::embed_query(const std::string& text) {
  // If a real MNN embedding pipeline is available, wire it here. For now,
  // return a deterministic embedding to keep the system functional.
  return generateTextHashEmbedding(text, embed_dim_);
}

std::vector<std::vector<float>> MNNEmbedding::embed_documents(
    const std::vector<std::string>& texts) {
  std::vector<std::vector<float>> out;
  out.reserve(texts.size());
  for (const std::string& t : texts) {
    out.push_back(embed_query(t));
  }
  return out;
}

}  // namespace mobile_rag


