#include "RAGPipeline.hpp"

#include <algorithm>
#include <iostream>

namespace mobile_rag {

RAGPipeline::RAGPipeline(std::shared_ptr<IDocumentLoader> loader,
                         std::shared_ptr<IEmbeddingModel> embedder,
                         std::shared_ptr<IVectorDB> db,
                         std::shared_ptr<ILargeLanguageModel> llm)
    : loader_(std::move(loader)),
      embedder_(std::move(embedder)),
      db_(std::move(db)),
      llm_(std::move(llm)) {}

void RAGPipeline::build_index_from_file(const std::string& file_path) {
  std::vector<std::string> chunks = loader_->load_and_split(file_path);
  if (chunks.empty()) {
    std::cerr << "[RAGPipeline] No chunks produced from file: " << file_path << '\n';
    return;
  }

  std::vector<std::vector<float>> vectors = embedder_->embed_documents(chunks);
  if (vectors.size() != chunks.size()) {
    std::cerr << "[RAGPipeline] Embeddings size mismatch: got " << vectors.size()
              << ", expected " << chunks.size() << '\n';
  }

  std::vector<int64_t> ids;
  ids.reserve(chunks.size());
  for (size_t i = 0; i < chunks.size(); ++i) {
    const int64_t id = next_id_++;
    ids.push_back(id);
    id_to_chunk_[id] = chunks[i];
  }

  if (!vectors.empty()) {
    db_->add_vectors(vectors, ids);
  }
}

std::string RAGPipeline::answer_query(const std::string& query) {
  std::vector<float> q = embedder_->embed_query(query);
  if (q.empty()) {
    std::cerr << "[RAGPipeline] Empty embedding for query." << '\n';
    return {};
  }

  // Choose a small default k for demo purposes
  constexpr int kTopK = 5;
  std::vector<std::pair<int64_t, float>> results = db_->search(q, kTopK);

  std::vector<std::string> retrieved_chunks;
  retrieved_chunks.reserve(results.size());
  for (const auto& [id, score] : results) {
    auto it = id_to_chunk_.find(id);
    if (it != id_to_chunk_.end()) {
      retrieved_chunks.push_back(it->second);
    }
  }

  std::string prompt = llm_->build_prompt(query, retrieved_chunks);
  return llm_->generate(prompt);
}

}  // namespace mobile_rag


