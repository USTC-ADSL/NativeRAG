#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mobile_rag/loader/IDocumentLoader.hpp"
#include "mobile_rag/embedding/IEmbeddingModel.hpp"
#include "mobile_rag/vector_db/IVectorDB.hpp"
#include "mobile_rag/llm/ILargeLanguageModel.hpp"

namespace mobile_rag {

class RAGPipeline {
 public:
  RAGPipeline(std::shared_ptr<IDocumentLoader> loader,
              std::shared_ptr<IEmbeddingModel> embedder,
              std::shared_ptr<IVectorDB> db,
              std::shared_ptr<ILargeLanguageModel> llm);

  void build_index_from_file(const std::string& file_path);

  std::string answer_query(const std::string& query);

 private:
  std::shared_ptr<IDocumentLoader> loader_;
  std::shared_ptr<IEmbeddingModel> embedder_;
  std::shared_ptr<IVectorDB> db_;
  std::shared_ptr<ILargeLanguageModel> llm_;

  std::map<int64_t, std::string> id_to_chunk_;
  int64_t next_id_ = 0;
};

}  // namespace mobile_rag


