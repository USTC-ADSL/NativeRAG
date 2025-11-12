#ifndef EMBEDDING_HPP_
#define EMBEDDING_HPP_

#include <string>
#include <vector>
#include <memory>

#include "MNN/llm/llm.hpp"

class EmbeddingRunner {
public:
    explicit EmbeddingRunner(const std::string& config_path);
    std::vector<std::pair<std::vector<float>, std::string>>
    encode(const std::vector<std::string>& texts);
    std::pair<std::vector<float>, std::string> encodeOne(const std::string& text);
    int dim() const;
private:
    std::unique_ptr<MNN::Transformer::Embedding> mEmbedding;
};

#endif // EMBEDDING_HPP_


