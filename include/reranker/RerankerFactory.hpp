#pragma once

#include <memory>

#include "reranker/IReranker.hpp"

namespace mobile_rag {

std::shared_ptr<IReranker> create_reranker(int num_threads = 4);

}  // namespace mobile_rag
