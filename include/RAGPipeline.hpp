#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "loader/IDocumentLoader.hpp"
#include "embedding/IEmbeddingModel.hpp"
#include "vector_Index/IVectorIndex.hpp"
#include "llm/ILargeLanguageModel.hpp"

namespace mobile_rag {

class RAGPipeline {
 public:
  RAGPipeline(std::shared_ptr<IDocumentLoader> loader,
              std::shared_ptr<IEmbeddingModel> embedder,
              std::shared_ptr<IVectorIndex> index,
              std::shared_ptr<ILargeLanguageModel> llm);

  void build_index_from_file(const std::string& file_path);

  std::string answer_query(const std::string& query);

 private:
  std::shared_ptr<IDocumentLoader> loader_;
  std::shared_ptr<IEmbeddingModel> embedder_;
  std::shared_ptr<IVectorIndex> index_;
  std::shared_ptr<ILargeLanguageModel> llm_;

  std::map<int64_t, std::string> id_to_chunk_;
  int64_t next_id_ = 0;
};

}  // namespace mobile_rag


