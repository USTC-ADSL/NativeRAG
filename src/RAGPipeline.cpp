#include "RAGPipeline.hpp"

#include <algorithm>
#include <iostream>

namespace mobile_rag {

RAGPipeline::RAGPipeline(std::shared_ptr<IDocumentLoader> loader,
                         std::shared_ptr<IEmbeddingModel> embedder,
                         std::shared_ptr<IVectorIndex> index,
                         std::shared_ptr<ILargeLanguageModel> llm,
                         std::shared_ptr<SqliteVectorDB> sqlite_db)
    : loader_(std::move(loader)),
      embedder_(std::move(embedder)),
      index_(std::move(index)),
      llm_(std::move(llm)),
      sqlite_db_(std::move(sqlite_db)) {}

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
    index_->add_vectors(vectors, ids);
  }

  // Persist id->text mapping to SQLite if available
  if (sqlite_db_) {
    if (!sqlite_db_->add_texts(chunks, ids)) {
      std::cerr << "[RAGPipeline] Warning: Failed to persist texts to SQLite\n";
    } else {
      std::cout << "[RAGPipeline] Persisted " << chunks.size()
                << " text chunks to SQLite\n";
    }
  }
}

bool RAGPipeline::save_index(const std::string& index_path) {
  if (!index_) {
    std::cerr << "[RAGPipeline] Vector index is not initialized." << '\n';
    return false;
  }

  bool success = index_->save_index(index_path);
  if (success) {
    std::cout << "[RAGPipeline] Index saved to: " << index_path << '\n';
  } else {
    std::cerr << "[RAGPipeline] Failed to save index to: " << index_path << '\n';
  }
  return success;
}

bool RAGPipeline::load_index(const std::string& index_path) {
  if (!index_) {
    std::cerr << "[RAGPipeline] Vector index is not initialized." << '\n';
    return false;
  }

  bool success = index_->load_index(index_path);
  if (success) {
    std::cout << "[RAGPipeline] Index loaded from: " << index_path << '\n';
  } else {
    std::cerr << "[RAGPipeline] Failed to load index from: " << index_path << '\n';
    return false;
  }

  // Load id->text mapping from SQLite if available
  if (sqlite_db_) {
    // Clear existing in-memory mapping
    id_to_chunk_.clear();
    next_id_ = 0;

    // Note: We need to query all texts from SQLite
    // For now, we rely on the SQLite DB being available during query phase
    std::cout << "[RAGPipeline] Text mapping will be loaded from SQLite on-demand\n";
  }

  return success;
}

std::string RAGPipeline::answer_query(const std::string& query) {
  std::vector<float> q = embedder_->embed_query(query);
  if (q.empty()) {
    std::cerr << "[RAGPipeline] Empty embedding for query." << '\n';
    return {};
  }

  // Choose a small default k for demo purposes
  constexpr int kTopK = 5;
  std::vector<std::pair<int64_t, float>> results = index_->search(q, kTopK);

  std::vector<std::string> retrieved_chunks;
  retrieved_chunks.reserve(results.size());
  for (const auto& [id, score] : results) {
    std::string chunk;

    // Try to get text from SQLite first (persistent storage)
    if (sqlite_db_) {
      chunk = sqlite_db_->get_text_for_id(id);
    }

    // Fall back to in-memory mapping if SQLite is not available
    if (chunk.empty()) {
      auto it = id_to_chunk_.find(id);
      if (it != id_to_chunk_.end()) {
        chunk = it->second;
      }
    }

    if (!chunk.empty()) {
      retrieved_chunks.push_back(chunk);
    }
  }

  std::string prompt = llm_->build_prompt(query, retrieved_chunks);
  return llm_->generate(prompt);
}

}  // namespace mobile_rag


