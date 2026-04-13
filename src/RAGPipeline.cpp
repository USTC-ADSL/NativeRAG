#include "RAGPipeline.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_set>

#include "controller/EvidenceFeatures.hpp"
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

std::string make_chunk_preview(const std::string& chunk) {
  std::string preview = chunk.substr(0, 120);
  for (char& c : preview) {
    if (c == '\n' || c == '\r' || c == '\t') c = ' ';
  }
  return preview;
}

std::string json_escape(const std::string& input) {
  std::ostringstream escaped;
  for (const unsigned char ch : input) {
    switch (ch) {
      case '\\':
        escaped << "\\\\";
        break;
      case '"':
        escaped << "\\\"";
        break;
      case '\b':
        escaped << "\\b";
        break;
      case '\f':
        escaped << "\\f";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        if (ch < 0x20) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<int>(ch) << std::dec << std::setfill(' ');
        } else {
          escaped << static_cast<char>(ch);
        }
        break;
    }
  }
  return escaped.str();
}

struct RetrievalExecution {
  std::vector<std::pair<int64_t, float>> results;
  size_t lexical_candidate_count = 0;
  size_t hash_candidate_count = 0;
  std::string fallback_reason = "prefilter_disabled";
};

RetrievalGraph graph_from_prefilter_flags(bool lexical_prefilter_enabled,
                                          bool semantic_hash_prefilter_enabled) {
  if (lexical_prefilter_enabled && semantic_hash_prefilter_enabled) {
    return RetrievalGraph::LEXICAL_HASH_PREFILTER;
  }
  if (lexical_prefilter_enabled) {
    return RetrievalGraph::LEXICAL_PREFILTER;
  }
  if (semantic_hash_prefilter_enabled) {
    return RetrievalGraph::SEMANTIC_HASH_PREFILTER;
  }
  return RetrievalGraph::DENSE_ONLY;
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

void RAGPipeline::set_graph_selector_config(GraphSelector::Config config) {
  graph_selector_.set_config(config);
}

bool RAGPipeline::export_last_query_trace(const std::string& output_path) const {
  if (!has_last_query_trace_ || output_path.empty()) {
    return false;
  }

  std::ofstream out(output_path, std::ios::trunc);
  if (!out) {
    return false;
  }

  out << "{\n"
      << "  \"query\":\"" << json_escape(last_query_trace_.query) << "\",\n"
      << "  \"answer\":\"" << json_escape(last_query_trace_.answer) << "\",\n"
      << "  \"adaptive_graph_enabled\":"
      << (last_query_trace_.adaptive_graph_enabled ? "true" : "false") << ",\n"
      << "  \"budget_class\":\"" << json_escape(last_query_trace_.budget_class) << "\",\n"
      << "  \"initial_graph\":\"" << json_escape(last_query_trace_.initial_graph) << "\",\n"
      << "  \"final_graph\":\"" << json_escape(last_query_trace_.final_graph) << "\",\n"
      << "  \"initial_reason\":\"" << json_escape(last_query_trace_.initial_reason) << "\",\n"
      << "  \"final_reason\":\"" << json_escape(last_query_trace_.final_reason) << "\",\n"
      << "  \"top_k\":" << last_query_trace_.top_k << ",\n"
      << "  \"lexical_prefilter_enabled\":"
      << (last_query_trace_.lexical_prefilter_enabled ? "true" : "false") << ",\n"
      << "  \"lexical_candidate_limit\":" << last_query_trace_.lexical_candidate_limit << ",\n"
      << "  \"semantic_hash_prefilter_enabled\":"
      << (last_query_trace_.semantic_hash_prefilter_enabled ? "true" : "false") << ",\n"
      << "  \"semantic_hash_candidate_limit\":"
      << last_query_trace_.semantic_hash_candidate_limit << ",\n"
      << "  \"semantic_hash_max_distance\":"
      << last_query_trace_.semantic_hash_max_distance << ",\n"
      << "  \"escalated\":"
      << (last_query_trace_.escalated ? "true" : "false") << ",\n"
      << "  \"escalation_from\":\"" << json_escape(last_query_trace_.escalation_from) << "\",\n"
      << "  \"escalation_to\":\"" << json_escape(last_query_trace_.escalation_to) << "\",\n"
      << "  \"escalation_reason\":\"" << json_escape(last_query_trace_.escalation_reason)
      << "\",\n"
      << "  \"lexical_candidate_count\":" << last_query_trace_.lexical_candidate_count << ",\n"
      << "  \"hash_candidate_count\":" << last_query_trace_.hash_candidate_count << ",\n"
      << "  \"dense_result_count\":" << last_query_trace_.dense_result_count << ",\n"
      << "  \"fallback_reason\":\"" << json_escape(last_query_trace_.fallback_reason)
      << "\",\n"
      << "  \"promoted_to_hot\":" << last_query_trace_.promoted_to_hot << ",\n"
      << "  \"demoted_to_warm\":" << last_query_trace_.demoted_to_warm << ",\n"
      << "  \"index_state\":{\n"
      << "    \"hot\":" << last_query_trace_.index_state.hot_count << ",\n"
      << "    \"warm\":" << last_query_trace_.index_state.warm_count << ",\n"
      << "    \"cold\":" << last_query_trace_.index_state.cold_count << ",\n"
      << "    \"transition_count\":" << last_query_trace_.index_state.transition_count << "\n"
      << "  },\n"
      << "  \"evidence\":{\n"
      << "    \"top_score\":" << last_query_trace_.evidence.top_score << ",\n"
      << "    \"second_score\":" << last_query_trace_.evidence.second_score << ",\n"
      << "    \"score_margin\":" << last_query_trace_.evidence.score_margin << ",\n"
      << "    \"score_sharpness\":" << last_query_trace_.evidence.score_sharpness << ",\n"
      << "    \"retrieved_chunk_count\":"
      << last_query_trace_.evidence.retrieved_chunk_count << ",\n"
      << "    \"query_term_count\":" << last_query_trace_.evidence.query_term_count << ",\n"
      << "    \"covered_query_terms\":"
      << last_query_trace_.evidence.covered_query_terms << ",\n"
      << "    \"coverage_ratio\":" << last_query_trace_.evidence.coverage_ratio << "\n"
      << "  },\n"
      << "  \"results\":[\n";

  for (size_t i = 0; i < last_query_trace_.results.size(); ++i) {
    const auto& result = last_query_trace_.results[i];
    out << "    {\"id\":" << result.id
        << ",\"score\":" << result.score
        << ",\"preview\":\"" << json_escape(result.preview) << "\"}";
    if (i + 1 < last_query_trace_.results.size()) {
      out << ",";
    }
    out << "\n";
  }

  out << "  ]\n"
      << "}\n";
  return static_cast<bool>(out);
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

    if (!sqlite_db_->initialize_chunk_states(ids, ChunkState::WARM, "index_build")) {
      std::cerr << "[RAGPipeline] Warning: Failed to initialize chunk states in SQLite for "
                << source_label << '\n';
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
  last_query_trace_ = QueryTrace{};
  has_last_query_trace_ = false;

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

  GraphSelector::Availability availability;
  availability.sqlite_available = sqlite_db_ != nullptr;
  availability.lexical_graph_available = lexical_prefilter_.enabled;
  availability.semantic_hash_graph_available = semantic_hash_prefilter_.enabled;
  GraphSelector::BudgetContext budget_context;
  budget_context.top_k = top_k_;
  budget_context.lexical_candidate_limit = lexical_prefilter_.enabled
                                               ? lexical_prefilter_.candidate_limit
                                               : 0;
  budget_context.semantic_hash_candidate_limit = semantic_hash_prefilter_.enabled
                                                     ? semantic_hash_prefilter_.candidate_limit
                                                     : 0;
  const auto budget_class = graph_selector_.classify_budget(availability, budget_context);
  QueryTrace query_trace;
  query_trace.query = query;
  query_trace.adaptive_graph_enabled = graph_selector_.config().enabled;
  query_trace.budget_class = budget_class_name(budget_class);
  query_trace.top_k = top_k_;
  query_trace.lexical_prefilter_enabled = lexical_prefilter_.enabled;
  query_trace.lexical_candidate_limit = lexical_prefilter_.candidate_limit;
  query_trace.semantic_hash_prefilter_enabled = semantic_hash_prefilter_.enabled;
  query_trace.semantic_hash_candidate_limit = semantic_hash_prefilter_.candidate_limit;
  query_trace.semantic_hash_max_distance = semantic_hash_prefilter_.max_hamming_distance;

  RetrievalGraph active_graph =
      graph_from_prefilter_flags(lexical_prefilter_.enabled, semantic_hash_prefilter_.enabled);
  query_trace.initial_graph = retrieval_graph_name(active_graph);
  if (graph_selector_.config().enabled) {
    const auto initial_decision =
        graph_selector_.choose_initial_graph(query, availability, budget_context);
    active_graph = initial_decision.graph;
    query_trace.initial_graph = retrieval_graph_name(active_graph);
    query_trace.initial_reason = initial_decision.reason;
    std::cout << "[CONTROLLER] adaptive=on initial_graph="
              << retrieval_graph_name(active_graph)
              << " budget=" << budget_class_name(budget_class)
              << " reason=" << initial_decision.reason << '\n';
  }

  auto execute_retrieval = [&](RetrievalGraph graph) {
    RetrievalExecution execution;
    if (graph == RetrievalGraph::DENSE_ONLY) {
      execution.results = index_->search(q, top_k_);
      return execution;
    }

    if (!sqlite_db_) {
      execution.fallback_reason = "sqlite_unavailable";
      execution.results = index_->search(q, top_k_);
      return execution;
    }

    execution.fallback_reason = "none";

    std::vector<int64_t> candidate_ids;
    std::unordered_set<int64_t> seen_ids;

    if (graph == RetrievalGraph::LEXICAL_PREFILTER ||
        graph == RetrievalGraph::LEXICAL_HASH_PREFILTER) {
      const auto lexical_matches =
          sqlite_db_->search_text_lexical(query, lexical_prefilter_.candidate_limit);
      execution.lexical_candidate_count = lexical_matches.size();
      candidate_ids.reserve(candidate_ids.size() + lexical_matches.size());
      for (const auto& [id, /*score*/ _] : lexical_matches) {
        if (seen_ids.insert(id).second) {
          candidate_ids.push_back(id);
        }
      }
    }

    if (graph == RetrievalGraph::SEMANTIC_HASH_PREFILTER ||
        graph == RetrievalGraph::LEXICAL_HASH_PREFILTER) {
      const auto query_code = build_sign_semantic_hash(q);
      const auto hash_matches = sqlite_db_->search_by_semantic_hash(
          query_code,
          semantic_hash_prefilter_.candidate_limit,
          semantic_hash_prefilter_.max_hamming_distance);
      execution.hash_candidate_count = hash_matches.size();
      candidate_ids.reserve(candidate_ids.size() + hash_matches.size());
      for (const auto& [id, /*distance*/ _] : hash_matches) {
        if (seen_ids.insert(id).second) {
          candidate_ids.push_back(id);
        }
      }
    }

    if (!candidate_ids.empty()) {
      execution.results = sqlite_db_->search_with_ids(q, candidate_ids, top_k_);
      if (execution.results.empty()) {
        execution.fallback_reason = "sqlite_rerank_empty";
      }
    } else {
      execution.fallback_reason = "empty_shortlist";
    }

    if (execution.results.empty()) {
      execution.results = index_->search(q, top_k_);
    }

    return execution;
  };

  auto resolve_chunk_text = [&](int64_t id) {
    std::string chunk;
    if (sqlite_db_) {
      chunk = sqlite_db_->get_text_for_id(id);
    }
    if (chunk.empty()) {
      auto it = id_to_chunk_.find(id);
      if (it != id_to_chunk_.end()) {
        chunk = it->second;
      }
    }
    return chunk;
  };

  auto collect_chunks = [&](const std::vector<std::pair<int64_t, float>>& ranked_results) {
    std::vector<std::string> chunks;
    chunks.reserve(ranked_results.size());
    for (const auto& [id, /*score*/ _] : ranked_results) {
      chunks.push_back(resolve_chunk_text(id));
    }
    return chunks;
  };

  RetrievalExecution execution = execute_retrieval(active_graph);
  auto retrieved_chunks = collect_chunks(execution.results);
  auto evidence_features =
      compute_evidence_features(query, execution.results, retrieved_chunks);

  std::string final_controller_reason = "adaptive_disabled";
  if (graph_selector_.config().enabled) {
    const auto escalation =
        graph_selector_.maybe_escalate(
            active_graph, availability, budget_context, evidence_features);
    final_controller_reason = escalation.reason;
    if (escalation.escalated && escalation.graph != active_graph) {
      query_trace.escalated = true;
      query_trace.escalation_from = retrieval_graph_name(active_graph);
      query_trace.escalation_to = retrieval_graph_name(escalation.graph);
      query_trace.escalation_reason = escalation.reason;
      std::cout << "[CONTROLLER] escalate_from=" << retrieval_graph_name(active_graph)
                << " to=" << retrieval_graph_name(escalation.graph)
                << " budget=" << budget_class_name(budget_class)
                << " reason=" << escalation.reason
                << " coverage_ratio=" << std::fixed << std::setprecision(4)
                << evidence_features.coverage_ratio
                << " score_margin=" << evidence_features.score_margin << '\n';
      active_graph = escalation.graph;
      execution = execute_retrieval(active_graph);
      retrieved_chunks = collect_chunks(execution.results);
      evidence_features =
          compute_evidence_features(query, execution.results, retrieved_chunks);
    }
    std::cout << "[CONTROLLER] final_graph=" << retrieval_graph_name(active_graph)
              << " budget=" << budget_class_name(budget_class)
              << " reason=" << final_controller_reason << '\n';
  }
  query_trace.final_graph = retrieval_graph_name(active_graph);
  query_trace.final_reason = final_controller_reason;

  std::cout << "[RETRIEVAL] mode=" << retrieval_graph_name(active_graph)
            << " lexical_candidates=" << execution.lexical_candidate_count
            << " hash_candidates=" << execution.hash_candidate_count
            << " dense_results=" << execution.results.size()
            << " fallback=" << execution.fallback_reason << '\n';
  query_trace.lexical_candidate_count = execution.lexical_candidate_count;
  query_trace.hash_candidate_count = execution.hash_candidate_count;
  query_trace.dense_result_count = execution.results.size();
  query_trace.fallback_reason = execution.fallback_reason;

  // Print query and retrieval results for debugging/inspection
  std::cout << "\n[QUERY] " << query << '\n';
  if (execution.results.empty()) {
    std::cout << "[RETRIEVAL] No results\n";
  }

  for (size_t rank = 0; rank < execution.results.size(); ++rank) {
    const auto& [id, score] = execution.results[rank];
    const auto& chunk = retrieved_chunks[rank];
    const std::string preview = chunk.empty() ? std::string() : make_chunk_preview(chunk);
    query_trace.results.push_back({id, score, preview});

    if (!chunk.empty()) {
      std::cout << "[TOP-" << (rank + 1) << "] id=" << id
                << " score=" << std::fixed << std::setprecision(4) << score
                << " | " << preview << (chunk.size() > 120 ? "..." : "") << '\n';
    } else {
      std::cout << "[TOP-" << (rank + 1) << "] id=" << id
                << " score=" << std::fixed << std::setprecision(4) << score
                << " | (text not found)" << '\n';
    }
  }

  if (sqlite_db_) {
    std::vector<int64_t> retrieved_ids;
    retrieved_ids.reserve(execution.results.size());
    for (const auto& [id, /*score*/ _] : execution.results) {
      retrieved_ids.push_back(id);
    }

    const int demoted_to_warm = sqlite_db_->demote_non_retrieved_hot_chunks(
        retrieved_ids, ChunkState::WARM, "query_retrieval_miss");
    int promoted_to_hot = 0;
    for (const auto& [id, /*score*/ _] : execution.results) {
      const auto previous_state = sqlite_db_->get_chunk_state(id);
      if (sqlite_db_->update_chunk_state(id, ChunkState::HOT, "query_retrieval_hit") &&
          previous_state != chunk_state_name(ChunkState::HOT)) {
        ++promoted_to_hot;
      }
    }
    std::cout << "[INDEX_STATE] promoted_to_hot=" << promoted_to_hot
              << " demoted_to_warm=" << demoted_to_warm
              << " retrieved_chunks=" << execution.results.size() << '\n';
    query_trace.promoted_to_hot = promoted_to_hot;
    query_trace.demoted_to_warm = demoted_to_warm;
    const auto state_summary = sqlite_db_->get_chunk_state_summary();
    std::cout << "[INDEX_STATE_SUMMARY] hot=" << state_summary.hot_count
              << " warm=" << state_summary.warm_count
              << " cold=" << state_summary.cold_count
              << " transitions=" << state_summary.transition_count << '\n';
    query_trace.index_state = state_summary;
  }

  std::cout << "[EVIDENCE] top_score=" << std::fixed << std::setprecision(4)
            << evidence_features.top_score
            << " second_score=" << evidence_features.second_score
            << " score_margin=" << evidence_features.score_margin
            << " score_sharpness=" << evidence_features.score_sharpness
            << " query_terms=" << evidence_features.query_term_count
            << " covered_terms=" << evidence_features.covered_query_terms
            << " coverage_ratio=" << evidence_features.coverage_ratio
            << " retrieved_chunks=" << evidence_features.retrieved_chunk_count
            << '\n';

  std::string prompt = llm_->build_prompt(query, retrieved_chunks);
  std::string answer = llm_->generate(prompt);
  if (answer.empty()) {
    answer = fallback_answer_from_chunks(query, retrieved_chunks);
  }
  query_trace.answer = answer;
  query_trace.evidence = evidence_features;
  last_query_trace_ = std::move(query_trace);
  has_last_query_trace_ = true;
  return answer;
}

}  // namespace mobile_rag
