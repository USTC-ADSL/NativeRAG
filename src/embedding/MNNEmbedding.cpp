#include "embedding/MNNEmbedding.hpp"

#include <algorithm>
#include <iostream>

using namespace MNN;
using namespace MNN::Transformer;
using namespace MNN::Express;

namespace mobile_rag {

bool MNNEmbedding::load_model(const std::string& model_path) {
  try {
    embedding_.reset(Embedding::createEmbedding(model_path, true));
    if (!embedding_) {
      std::cerr << "[MNNEmbedding] Failed to create MNN Embedding with config: "
                << model_path << '\n';
      return false;
    }
    embed_dim_ = embedding_->dim();
    return true;
  } catch (...) {
    std::cerr << "[MNNEmbedding] Exception while loading model: " << model_path
              << '\n';
    embedding_.reset();
    embed_dim_ = 0;
    return false;
  }
}

std::vector<float> MNNEmbedding::embed_query(const std::string& text) {
  if (!embedding_) {
    std::cerr << "[MNNEmbedding] Model not loaded" << '\n';
    return std::vector<float>();
  }
  
  auto var = embedding_->txt_embedding(text);
  embedding_->reset();
  const float* ptr = var->readMap<float>();
  if (ptr == nullptr) {
    std::cerr << "[MNNEmbedding] Embedding readMap returned nullptr" << '\n';
    return std::vector<float>();
  }
  
  int d = embed_dim_;
  std::vector<float> vec;
  vec.resize(d);
  std::copy(ptr, ptr + d, vec.begin());
  return vec;
}

std::vector<std::vector<float>> MNNEmbedding::embed_documents(
    const std::vector<std::string>& texts) {
  std::vector<std::vector<float>> out;
  out.reserve(texts.size());
  for (const auto& t : texts) {
    out.push_back(embed_query(t));
  }
  return out;
}

}  // namespace mobile_rag


