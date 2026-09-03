#include "embedding/EmbeddingFactory.hpp"

#include <memory>

#include "embedding/LlamaCppEmbedding.hpp"

namespace mobile_rag {

std::shared_ptr<IEmbeddingModel> create_embedding(int num_threads) {
  return std::make_shared<LlamaCppEmbedding>(num_threads);
}

}  // namespace mobile_rag
