#pragma once

#include <memory>

#include "embedding/IEmbeddingModel.hpp"

namespace mobile_rag {

std::shared_ptr<IEmbeddingModel> create_embedding(int num_threads = 4);

}  // namespace mobile_rag
