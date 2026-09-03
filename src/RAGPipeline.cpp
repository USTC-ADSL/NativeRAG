#include "RAGPipeline.hpp"

#include <algorithm>
#include <iostream>
#include <iomanip>

namespace mobile_rag {

namespace {

struct RetrievedCandidate {
  int64_t id = -1;
  float retrieval_score = 0.0f;
  float rerank_score = 0.0f;
  size_t retrieval_rank = 0;
  std::string text;
};

std::string make_preview(const std::string& text, size_t max_bytes) {
  size_t length = std::min(text.size(), max_bytes);
  while (length > 0 && length < text.size() &&
         (static_cast<unsigned char>(text[length]) & 0xc0) == 0x80) {
    --length;
  }
  std::string preview = text.substr(0, length);
  for (char& character : preview) {
    if (character == '\n' || character == '\r' || character == '\t') {
      character = ' ';
    }
  }
  return preview;
}

}  // namespace

RAGPipeline::RAGPipeline(std::shared_ptr<IDocumentLoader> loader,
                         std::shared_ptr<IEmbeddingModel> embedder,
                         std::shared_ptr<IVectorIndex> index,
                         std::shared_ptr<ILargeLanguageModel> llm,
                         std::shared_ptr<SqliteVectorDB> sqlite_db,
                         int top_k,
                         std::shared_ptr<IReranker> reranker,
                         int rerank_candidates)
    : loader_(std::move(loader)),
      embedder_(std::move(embedder)),
      index_(std::move(index)),
      llm_(std::move(llm)),
      reranker_(std::move(reranker)),
      sqlite_db_(std::move(sqlite_db)),
      top_k_(std::max(1, top_k)),
      rerank_candidates_(std::max(top_k_, rerank_candidates)) {}

bool RAGPipeline::build_index_from_file(const std::string& file_path) {
  if (!loader_ || !embedder_ || !index_) {
    std::cerr << "[RAGPipeline] Offline pipeline is not initialized\n";
    return false;
  }
  std::vector<std::string> chunks = loader_->load_and_split(file_path);
  if (chunks.empty()) {
    std::cerr << "[RAGPipeline] No chunks produced from file: " << file_path << '\n';
    return false;
  }

  std::vector<std::vector<float>> vectors = embedder_->embed_documents(chunks);
  if (vectors.size() != chunks.size()) {
    std::cerr << "[RAGPipeline] Embeddings size mismatch: got " << vectors.size()
              << ", expected " << chunks.size() << '\n';
    return false;
  }

  std::vector<int64_t> ids;
  ids.reserve(chunks.size());
  for (size_t i = 0; i < chunks.size(); ++i) {
    const int64_t id = next_id_++;
    ids.push_back(id);
    id_to_chunk_[id] = chunks[i];
  }

  if (vectors.empty() || !index_->add_vectors(vectors, ids)) {
    std::cerr << "[RAGPipeline] Failed to add embeddings to vector index\n";
    return false;
  }

  // Persist id->text mapping to SQLite if available
  if (sqlite_db_) {
    if (!sqlite_db_->add_texts(chunks, ids)) {
      std::cerr << "[RAGPipeline] Warning: Failed to persist texts to SQLite\n";
      return false;
    } else {
      std::cout << "[RAGPipeline] Persisted " << chunks.size()
                << " text chunks to SQLite\n";
    }
  }
  return true;
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

std::vector<std::string> RAGPipeline::retrieve_contexts(
    const std::string& query) {
  if (!embedder_ || !index_) {
    std::cerr << "[RAGPipeline] Retrieval pipeline is not initialized\n";
    return {};
  }
  std::vector<float> q = embedder_->embed_query(query);
  if (q.empty()) {
    std::cerr << "[RAGPipeline] Empty embedding for query." << '\n';
    return {};
  }

  const int candidate_count = reranker_ ? rerank_candidates_ : top_k_;
  std::vector<std::pair<int64_t, float>> results =
      index_->search(q, candidate_count);

  // Print query and retrieval results for debugging/inspection
  std::cout << "\n[QUERY] " << query << '\n';
  if (results.empty()) {
    std::cout << "[RETRIEVAL] No results\n";
    return {};
  }

  std::vector<RetrievedCandidate> candidates;
  candidates.reserve(results.size());
  for (size_t rank = 0; rank < results.size(); ++rank) {
    const auto& [id, score] = results[rank];
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
      candidates.push_back({id, score, 0.0f, rank + 1, std::move(chunk)});
    } else {
      std::cout << "[CANDIDATE-" << (rank + 1) << "] id=" << id
                << " retrieval_score=" << std::fixed << std::setprecision(4)
                << score
                << " | (text not found)" << '\n';
    }
  }

  if (candidates.empty()) {
    std::cout << "[RETRIEVAL] Candidate text not found\n";
    return {};
  }

  if (reranker_) {
    std::vector<std::string> documents;
    documents.reserve(candidates.size());
    for (const auto& candidate : candidates) {
      documents.push_back(candidate.text);
    }
    const std::vector<float> rerank_scores = reranker_->score(query, documents);
    if (rerank_scores.size() != candidates.size()) {
      std::cerr << "[RAGPipeline] Reranker failed to score all candidates\n";
      return {};
    }
    for (size_t i = 0; i < candidates.size(); ++i) {
      candidates[i].rerank_score = rerank_scores[i];
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const RetrievedCandidate& left,
                        const RetrievedCandidate& right) {
                       return left.rerank_score > right.rerank_score;
                     });
    std::cout << "[RERANK] Scored " << candidates.size()
              << " Faiss candidate(s) with llama.cpp\n";
  }

  const size_t output_count =
      std::min(candidates.size(), static_cast<size_t>(top_k_));
  std::vector<std::string> retrieved_chunks;
  retrieved_chunks.reserve(output_count);
  for (size_t rank = 0; rank < output_count; ++rank) {
    const auto& candidate = candidates[rank];
    retrieved_chunks.push_back(candidate.text);
    const std::string preview = make_preview(candidate.text, 120);
    std::cout << "[TOP-" << (rank + 1) << "] id=" << candidate.id;
    if (reranker_) {
      std::cout << " faiss_rank=" << candidate.retrieval_rank
                << " retrieval_score=" << std::fixed << std::setprecision(4)
                << candidate.retrieval_score << " rerank_score="
                << candidate.rerank_score;
    } else {
      std::cout << " score=" << std::fixed << std::setprecision(4)
                << candidate.retrieval_score;
    }
    std::cout << " | " << preview
              << (candidate.text.size() > 120 ? "..." : "") << '\n';
  }

  return retrieved_chunks;
}

std::string RAGPipeline::answer_query(const std::string& query) {
  if (!llm_) {
    std::cerr << "[RAGPipeline] LLM is not initialized\n";
    return {};
  }
  std::vector<std::string> retrieved_chunks = retrieve_contexts(query);
  if (retrieved_chunks.empty()) {
    return {};
  }
  std::string prompt = llm_->build_prompt(query, retrieved_chunks);
  return llm_->generate(prompt);
}

}  // namespace mobile_rag
