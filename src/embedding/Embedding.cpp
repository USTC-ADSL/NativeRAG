#include "embedding/Embedding.hpp"

#include <algorithm>
#include <iostream>

using namespace MNN;
using namespace MNN::Transformer;
using namespace MNN::Express;

EmbeddingRunner::EmbeddingRunner(const std::string& config_path) {
    mEmbedding.reset(Embedding::createEmbedding(config_path, true));
    if (!mEmbedding) {
        std::cout << "Failed to create MNN Embedding with config: " + config_path << std::endl;
    }
}

int EmbeddingRunner::dim() const {
    return mEmbedding ? mEmbedding->dim() : 0;
}

std::pair<std::vector<float>, std::string> EmbeddingRunner::encodeOne(const std::string& text) {
    auto var = mEmbedding->txt_embedding(text);
    mEmbedding->reset();
    const float* ptr = var->readMap<float>();
    if (ptr == nullptr) {
        std::cout << "Embedding readMap returned nullptr" << std::endl;
    }
    int d = dim();
    std::vector<float> vec;
    vec.resize(d);
    std::copy(ptr, ptr + d, vec.begin());
    return {std::move(vec), text};
}

std::vector<std::pair<std::vector<float>, std::string>>
EmbeddingRunner::encode(const std::vector<std::string>& texts) {
    std::vector<std::pair<std::vector<float>, std::string>> results;
    results.reserve(texts.size());
    for (const auto& t : texts) {
        results.emplace_back(encodeOne(t));
    }
    return results;
}


