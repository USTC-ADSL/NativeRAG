#include "embedding/MNNEmbedding.hpp"

#include <algorithm>
#include <iostream>
#include <filesystem>

using namespace MNN;
using namespace MNN::Transformer;
using namespace MNN::Express;

namespace mobile_rag {

void MNNEmbedding::set_num_threads(int num_threads) {
  if (num_threads <= 0) {
    return;
  }

  num_threads_ = num_threads;
  if (embedding_) {
    const std::string config =
        std::string("{\"thread_num\":") + std::to_string(num_threads_) + "}";
    if (!embedding_->set_config(config)) {
      std::cerr << "[MNNEmbedding] Warning: failed to update thread_num to "
                << num_threads_ << '\n';
    }
  }
}

bool MNNEmbedding::load_model(const std::string& model_path) {
  model_loaded_ = false;
  try {
    // If model_path is empty, it means we're trying to load default model
    // which likely doesn't exist, so return false early
    if (model_path.empty()) {
      std::cerr << "[MNNEmbedding] Model path is empty, cannot load model\n";
      embedding_.reset();
      embed_dim_ = 0;
      return false;
    }

    // Check if the model file exists
    if (!std::filesystem::exists(model_path)) {
      std::cerr << "[MNNEmbedding] Model file does not exist: " << model_path << '\n';
      embedding_.reset();
      embed_dim_ = 0;
      return false;
    }

    auto embedding = Embedding::createEmbedding(model_path, true);
    if (!embedding) {
      std::cerr << "[MNNEmbedding] Failed to create MNN Embedding with config: "
                << model_path << '\n';
      embedding_.reset();
      embed_dim_ = 0;
      return false;
    }

    const std::string config =
        std::string("{\"thread_num\":") + std::to_string(num_threads_) + "}";
    if (!embedding->set_config(config)) {
      std::cerr << "[MNNEmbedding] Warning: failed to set thread_num to "
                << num_threads_ << '\n';
    }

    embedding_.reset(embedding);
    embed_dim_ = embedding_->dim();

    if (embed_dim_ <= 0) {
      std::cerr << "[MNNEmbedding] Invalid embedding dimension: " << embed_dim_ << '\n';
      embedding_.reset();
      embed_dim_ = 0;
      return false;
    }

    model_loaded_ = true;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[MNNEmbedding] Exception while loading model: " << model_path
              << " - " << e.what() << '\n';
    embedding_.reset();
    embed_dim_ = 0;
    return false;
  } catch (...) {
    std::cerr << "[MNNEmbedding] Unknown exception while loading model: " << model_path
              << '\n';
    embedding_.reset();
    embed_dim_ = 0;
    return false;
  }
}

std::vector<float> MNNEmbedding::embed_query(const std::string& text) {
  if (!model_loaded_ || !embedding_) {
    std::cerr << "[MNNEmbedding] Model not loaded" << '\n';
    return std::vector<float>();
  }

  try {
    auto var = embedding_->txt_embedding(text);
    if (var.get() == nullptr) {
      std::cerr << "[MNNEmbedding] txt_embedding returned nullptr for text: " << text << '\n';
      return std::vector<float>();
    }

    embedding_->reset();
    const float* ptr = var->readMap<float>();
    if (ptr == nullptr) {
      std::cerr << "[MNNEmbedding] Embedding readMap returned nullptr" << '\n';
      return std::vector<float>();
    }

    int d = embed_dim_;
    if (d <= 0) {
      std::cerr << "[MNNEmbedding] Invalid embedding dimension: " << d << '\n';
      return std::vector<float>();
    }

    std::vector<float> vec;
    vec.resize(d);
    std::copy(ptr, ptr + d, vec.begin());
    return vec;
  } catch (const std::exception& e) {
    std::cerr << "[MNNEmbedding] Exception in embed_query: " << e.what() << '\n';
    return std::vector<float>();
  } catch (...) {
    std::cerr << "[MNNEmbedding] Unknown exception in embed_query" << '\n';
    return std::vector<float>();
  }
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

