#include "reranker/RerankerFactory.hpp"

#include <memory>

#include "reranker/LlamaCppReranker.hpp"

namespace mobile_rag {

std::shared_ptr<IReranker> create_reranker(int num_threads) {
  return std::make_shared<LlamaCppReranker>(num_threads);
}

}  // namespace mobile_rag
