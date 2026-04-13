#include "RAGPipeline.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <iomanip>
#include <unordered_set>

#include "llm/PromptUtils.hpp"
#include "retrieval/SemanticHash.hpp"

namespace mobile_rag {

namespace {

std::vector<std::string> tokenize_terms(const std::string& text) {
  std::vector<std::string> terms;
  std::string current;
  for (char ch : text) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      current.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    } else if (!current.empty()) {
      if (current.size() > 2) {
        terms.push_back(current);
      }
      current.clear();
    }
  }
  if (!current.empty() && current.size() > 2) {
    terms.push_back(current);
  }
  return terms;
}

std::vector<std::string> split_sentences(const std::string& text) {
  std::vector<std::string> sentences;
  std::string current;
  for (char ch : text) {
    current.push_back(ch);
    if (ch == '.' || ch == '!' || ch == '?' || ch == '\n') {
      auto sentence = trim_copy(current);
      if (!sentence.empty()) {
        sentences.push_back(std::move(sentence));
      }
      current.clear();
    }
  }

  auto tail = trim_copy(current);
  if (!tail.empty()) {
    sentences.push_back(std::move(tail));
  }

  return sentences;
}

std::string fallback_answer_from_chunks(const std::string& query,
                                        const std::vector<std::string>& chunks) {
  if (chunks.empty()) {
    return {};
  }

  const auto query_terms_vec = tokenize_terms(query);
  const std::unordered_set<std::string> query_terms(query_terms_vec.begin(),
                                                    query_terms_vec.end());

  int best_score = -1;
  std::string best_sentence;
  for (const auto& chunk : chunks) {
    for (const auto& sentence : split_sentences(chunk)) {
      const auto sentence_terms = tokenize_terms(sentence);
      int score = 0;
      for (const auto& term : sentence_terms) {
        if (query_terms.count(term) > 0) {
          ++score;
        }
      }

      if (score > best_score ||
          (score == best_score && !best_sentence.empty() &&
           sentence.size() < best_sentence.size())) {
        best_score = score;
        best_sentence = sentence;
      }
    }
  }

  if (!best_sentence.empty()) {
    return best_sentence;
  }

  return trim_copy(chunks.front());
}

}  // namespace

RAGPipeline::RAGPipeline(std::shared_ptr<IDocumentLoader> loader,
                         std::shared_ptr<IEmbeddingModel> embedder,
                         std::shared_ptr<IVectorIndex> index,
                         std::shared_ptr<ILargeLanguageModel> llm,
                         std::shared_ptr<SqliteVectorDB> sqlite_db,
                         int top_k,
                         size_t chunk_size,
                         size_t chunk_overlap)
    : loader_(std::move(loader)),
      embedder_(std::move(embedder)),
      index_(std::move(index)),
      llm_(std::move(llm)),
      sqlite_db_(std::move(sqlite_db)),
      top_k_(std::max(1, top_k)),
      chunk_size_(chunk_size > 0 ? chunk_size : 1000),
      chunk_overlap_(std::min(chunk_overlap, chunk_size_ > 0 ? chunk_size_ - 1 : 0)) {}

void RAGPipeline::set_semantic_hash_prefilter(SemanticHashPrefilterConfig config) {
  semantic_hash_prefilter_.enabled = config.enabled;
  semantic_hash_prefilter_.candidate_limit =
      std::max(top_k_, std::max(1, config.candidate_limit));
  semantic_hash_prefilter_.max_hamming_distance = config.max_hamming_distance;
}

void RAGPipeline::set_lexical_prefilter(LexicalPrefilterConfig config) {
  lexical_prefilter_.enabled = config.enabled;
  lexical_prefilter_.candidate_limit =
      std::max(top_k_, std::max(1, config.candidate_limit));
}

bool RAGPipeline::add_text_embeddings(const std::vector<std::string>& texts,
                                      const std::vector<std::vector<float>>& vectors,
                                      const std::string& source_label) {
  if (!index_) {
    std::cerr << "[RAGPipeline] Vector index is not initialized for "
              << source_label << '\n';
    return false;
  }

  if (texts.empty() || vectors.empty()) {
    std::cerr << "[RAGPipeline] No texts or vectors available for "
              << source_label << '\n';
    return false;
  }

  const size_t pair_count = std::min(texts.size(), vectors.size());
  if (texts.size() != vectors.size()) {
    std::cerr << "[RAGPipeline] Embeddings size mismatch for " << source_label
              << ": got " << vectors.size() << ", expected " << texts.size()
              << ". Truncating to " << pair_count << " paired items.\n";
  }

  std::vector<std::string> valid_texts;
  std::vector<std::vector<float>> valid_vectors;
  valid_texts.reserve(pair_count);
  valid_vectors.reserve(pair_count);

  size_t dropped = 0;
  size_t expected_dim = 0;
  for (size_t i = 0; i < pair_count; ++i) {
    const auto& vector = vectors[i];
    if (vector.empty()) {
      ++dropped;
      continue;
    }

    if (expected_dim == 0) {
      expected_dim = vector.size();
    }

    if (vector.size() != expected_dim) {
      ++dropped;
      continue;
    }

    valid_texts.push_back(texts[i]);
    valid_vectors.push_back(vector);
  }

  if (valid_vectors.empty()) {
    std::cerr << "[RAGPipeline] No valid embeddings available for "
              << source_label << '\n';
    return false;
  }

  if (dropped > 0) {
    std::cerr << "[RAGPipeline] Dropped " << dropped << " invalid text/vector pairs for "
              << source_label << '\n';
  }

  std::vector<int64_t> ids;
  ids.reserve(valid_texts.size());
  for (size_t i = 0; i < valid_texts.size(); ++i) {
    ids.push_back(next_id_ + static_cast<int64_t>(i));
  }

  if (!index_->add_vectors(valid_vectors, ids)) {
    std::cerr << "[RAGPipeline] Failed to add vectors to index for "
              << source_label << '\n';
    return false;
  }

  next_id_ += static_cast<int64_t>(valid_texts.size());
  for (size_t i = 0; i < valid_texts.size(); ++i) {
    id_to_chunk_[ids[i]] = valid_texts[i];
  }

  if (sqlite_db_) {
    if (!sqlite_db_->add_vectors(valid_vectors, ids)) {
      std::cerr << "[RAGPipeline] Warning: Failed to persist dense vectors to SQLite for "
                << source_label << '\n';
    } else {
      std::cout << "[RAGPipeline] Persisted " << valid_vectors.size()
                << " dense vectors to SQLite\n";
    }

    if (!sqlite_db_->add_texts(valid_texts, ids)) {
      std::cerr << "[RAGPipeline] Warning: Failed to persist texts to SQLite for "
                << source_label << '\n';
    } else {
      std::cout << "[RAGPipeline] Persisted " << valid_texts.size()
                << " text chunks to SQLite\n";
    }

    std::vector<std::vector<std::uint8_t>> semantic_hashes;
    semantic_hashes.reserve(valid_vectors.size());
    for (const auto& vector : valid_vectors) {
      semantic_hashes.push_back(build_sign_semantic_hash(vector));
    }

    if (!sqlite_db_->add_semantic_hashes(
            semantic_hashes, ids, static_cast<int>(kDefaultSemanticHashBits))) {
      std::cerr << "[RAGPipeline] Warning: Failed to persist semantic hashes to SQLite for "
                << source_label << '\n';
    } else {
      std::cout << "[RAGPipeline] Persisted " << semantic_hashes.size()
                << " semantic hashes to SQLite\n";
    }
  }

  return true;
}

void RAGPipeline::build_index_from_file(const std::string& file_path) {
  if (!loader_) {
    std::cerr << "[RAGPipeline] Document loader is not initialized." << '\n';
    return;
  }

  if (!embedder_) {
    std::cerr << "[RAGPipeline] Embedder is not initialized." << '\n';
    return;
  }

  std::vector<std::string> chunks = loader_->load_and_split(file_path);
  if (chunks.empty()) {
    std::cerr << "[RAGPipeline] No chunks produced from file: " << file_path << '\n';
    return;
  }

  std::vector<std::vector<float>> vectors = embedder_->embed_documents(chunks);
  add_text_embeddings(chunks, vectors, "file indexing");
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
  if (!embedder_) {
    std::cerr << "[RAGPipeline] Embedder is not initialized." << '\n';
    return {};
  }

  if (!index_) {
    std::cerr << "[RAGPipeline] Vector index is not initialized." << '\n';
    return {};
  }

  if (!llm_) {
    std::cerr << "[RAGPipeline] LLM backend is not initialized." << '\n';
    return {};
  }

  std::vector<float> q = embedder_->embed_query(query);
  if (q.empty()) {
    std::cerr << "[RAGPipeline] Empty embedding for query." << '\n';
    return {};
  }

  std::vector<std::pair<int64_t, float>> results;
  std::string retrieval_mode = "dense_only";
  std::string fallback_reason = "prefilter_disabled";
  size_t lexical_candidate_count = 0;
  size_t hash_candidate_count = 0;

  if ((lexical_prefilter_.enabled || semantic_hash_prefilter_.enabled) && sqlite_db_) {
    if (lexical_prefilter_.enabled && semantic_hash_prefilter_.enabled) {
      retrieval_mode = "lexical_hash_prefilter";
    } else if (lexical_prefilter_.enabled) {
      retrieval_mode = "lexical_prefilter";
    } else {
      retrieval_mode = "semantic_hash_prefilter";
    }
    fallback_reason = "none";

    std::vector<int64_t> candidate_ids;
    std::unordered_set<int64_t> seen_ids;

    if (lexical_prefilter_.enabled) {
      const auto lexical_matches = sqlite_db_->search_text_lexical(
          query, lexical_prefilter_.candidate_limit);
      lexical_candidate_count = lexical_matches.size();
      candidate_ids.reserve(candidate_ids.size() + lexical_matches.size());
      for (const auto& [id, /*score*/ _] : lexical_matches) {
        if (seen_ids.insert(id).second) {
          candidate_ids.push_back(id);
        }
      }
    }

    if (semantic_hash_prefilter_.enabled) {
      const auto query_code = build_sign_semantic_hash(q);
      const auto hash_matches = sqlite_db_->search_by_semantic_hash(
          query_code,
          semantic_hash_prefilter_.candidate_limit,
          semantic_hash_prefilter_.max_hamming_distance);
      hash_candidate_count = hash_matches.size();
      candidate_ids.reserve(candidate_ids.size() + hash_matches.size());
      for (const auto& [id, /*distance*/ _] : hash_matches) {
        if (seen_ids.insert(id).second) {
          candidate_ids.push_back(id);
        }
      }
    }

    if (!candidate_ids.empty()) {
      results = sqlite_db_->search_with_ids(q, candidate_ids, top_k_);
      if (results.empty()) {
        fallback_reason = "sqlite_rerank_empty";
      }
    } else {
      fallback_reason = "empty_shortlist";
    }

    if (results.empty()) {
      results = index_->search(q, top_k_);
    }
  } else if (lexical_prefilter_.enabled || semantic_hash_prefilter_.enabled) {
    if (lexical_prefilter_.enabled && semantic_hash_prefilter_.enabled) {
      retrieval_mode = "lexical_hash_prefilter";
    } else if (lexical_prefilter_.enabled) {
      retrieval_mode = "lexical_prefilter";
    } else {
      retrieval_mode = "semantic_hash_prefilter";
    }
    fallback_reason = "sqlite_unavailable";
    results = index_->search(q, top_k_);
  } else {
    results = index_->search(q, top_k_);
  }

  std::cout << "[RETRIEVAL] mode=" << retrieval_mode
            << " lexical_candidates=" << lexical_candidate_count
            << " hash_candidates=" << hash_candidate_count
            << " dense_results=" << results.size()
            << " fallback=" << fallback_reason << '\n';

  // Print query and retrieval results for debugging/inspection
  std::cout << "\n[QUERY] " << query << '\n';
  if (results.empty()) {
    std::cout << "[RETRIEVAL] No results\n";
  }

  std::vector<std::string> retrieved_chunks;
  retrieved_chunks.reserve(results.size());
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
      retrieved_chunks.push_back(chunk);
      // Prepare a one-line preview
      std::string preview = chunk.substr(0, 120);
      for (char& c : preview) {
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
      }
      std::cout << "[TOP-" << (rank + 1) << "] id=" << id
                << " score=" << std::fixed << std::setprecision(4) << score
                << " | " << preview << (chunk.size() > 120 ? "..." : "") << '\n';
    } else {
      std::cout << "[TOP-" << (rank + 1) << "] id=" << id
                << " score=" << std::fixed << std::setprecision(4) << score
                << " | (text not found)" << '\n';
    }
  }

  std::string prompt = llm_->build_prompt(query, retrieved_chunks);
  std::string answer = llm_->generate(prompt);
  if (answer.empty()) {
    answer = fallback_answer_from_chunks(query, retrieved_chunks);
  }
  return answer;
}

}  // namespace mobile_rag
