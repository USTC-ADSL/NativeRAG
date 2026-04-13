#include "RAGPipeline.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/resource.h>
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

std::string csv_escape(const std::string& input) {
  bool needs_quotes = false;
  for (const unsigned char ch : input) {
    if (ch == '"' || ch == ',' || ch == '\n' || ch == '\r') {
      needs_quotes = true;
      break;
    }
  }

  if (!needs_quotes) {
    return input;
  }

  std::string escaped = "\"";
  for (const unsigned char ch : input) {
    if (ch == '"') {
      escaped += "\"\"";
    } else {
      escaped.push_back(static_cast<char>(ch));
    }
  }
  escaped += "\"";
  return escaped;
}

template <typename ClockT>
double elapsed_ms_since(const std::chrono::time_point<ClockT>& start) {
  return std::chrono::duration<double, std::milli>(ClockT::now() - start).count();
}

uint64_t current_peak_rss_kb() {
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0;
  }

  if (usage.ru_maxrss < 0) {
    return 0;
  }

  return static_cast<uint64_t>(usage.ru_maxrss);
}

std::string serialize_query_trace_json(const RAGPipeline::QueryTrace& trace, bool pretty) {
  const std::string newline = pretty ? "\n" : "";
  const std::string colon = pretty ? ": " : ":";
  auto indent = [&](int level) {
    return pretty ? std::string(static_cast<size_t>(level) * 2, ' ') : std::string();
  };

  std::ostringstream out;
  out << "{" << newline
      << indent(1) << "\"query\"" << colon << "\"" << json_escape(trace.query) << "\"," << newline
      << indent(1) << "\"answer\"" << colon << "\"" << json_escape(trace.answer) << "\"," << newline
      << indent(1) << "\"adaptive_graph_enabled\"" << colon
      << (trace.adaptive_graph_enabled ? "true" : "false") << "," << newline
      << indent(1) << "\"budget_class\"" << colon << "\"" << json_escape(trace.budget_class) << "\"," << newline
      << indent(1) << "\"initial_graph\"" << colon << "\"" << json_escape(trace.initial_graph) << "\"," << newline
      << indent(1) << "\"final_graph\"" << colon << "\"" << json_escape(trace.final_graph) << "\"," << newline
      << indent(1) << "\"initial_reason\"" << colon << "\"" << json_escape(trace.initial_reason) << "\"," << newline
      << indent(1) << "\"final_reason\"" << colon << "\"" << json_escape(trace.final_reason) << "\"," << newline
      << indent(1) << "\"top_k\"" << colon << trace.top_k << "," << newline
      << indent(1) << "\"lexical_prefilter_enabled\"" << colon
      << (trace.lexical_prefilter_enabled ? "true" : "false") << "," << newline
      << indent(1) << "\"lexical_candidate_limit\"" << colon << trace.lexical_candidate_limit << "," << newline
      << indent(1) << "\"semantic_hash_prefilter_enabled\"" << colon
      << (trace.semantic_hash_prefilter_enabled ? "true" : "false") << "," << newline
      << indent(1) << "\"semantic_hash_candidate_limit\"" << colon
      << trace.semantic_hash_candidate_limit << "," << newline
      << indent(1) << "\"semantic_hash_max_distance\"" << colon
      << trace.semantic_hash_max_distance << "," << newline
      << indent(1) << "\"state_aware_dense_enabled\"" << colon
      << (trace.state_aware_dense_enabled ? "true" : "false") << "," << newline
      << indent(1) << "\"escalated\"" << colon
      << (trace.escalated ? "true" : "false") << "," << newline
      << indent(1) << "\"escalation_from\"" << colon << "\"" << json_escape(trace.escalation_from) << "\"," << newline
      << indent(1) << "\"escalation_to\"" << colon << "\"" << json_escape(trace.escalation_to) << "\"," << newline
      << indent(1) << "\"escalation_reason\"" << colon << "\"" << json_escape(trace.escalation_reason) << "\"," << newline
      << indent(1) << "\"lexical_candidate_count\"" << colon << trace.lexical_candidate_count << "," << newline
      << indent(1) << "\"hash_candidate_count\"" << colon << trace.hash_candidate_count << "," << newline
      << indent(1) << "\"state_filtered_candidate_count\"" << colon
      << trace.state_filtered_candidate_count << "," << newline
      << indent(1) << "\"dense_result_count\"" << colon << trace.dense_result_count << "," << newline
      << indent(1) << "\"fallback_reason\"" << colon << "\"" << json_escape(trace.fallback_reason) << "\"," << newline
      << indent(1) << "\"promoted_to_hot\"" << colon << trace.promoted_to_hot << "," << newline
      << indent(1) << "\"demoted_to_warm\"" << colon << trace.demoted_to_warm << "," << newline
      << indent(1) << "\"timings\"" << colon << "{" << newline
      << indent(2) << "\"query_embedding_ms\"" << colon << trace.timing.query_embedding_ms << "," << newline
      << indent(2) << "\"retrieval_ms\"" << colon << trace.timing.retrieval_ms << "," << newline
      << indent(2) << "\"evidence_ms\"" << colon << trace.timing.evidence_ms << "," << newline
      << indent(2) << "\"state_update_ms\"" << colon << trace.timing.state_update_ms << "," << newline
      << indent(2) << "\"prompt_build_ms\"" << colon << trace.timing.prompt_build_ms << "," << newline
      << indent(2) << "\"generation_ms\"" << colon << trace.timing.generation_ms << "," << newline
      << indent(2) << "\"total_ms\"" << colon << trace.timing.total_ms << newline
      << indent(1) << "}," << newline
      << indent(1) << "\"system\"" << colon << "{" << newline
      << indent(2) << "\"peak_rss_kb\"" << colon << trace.system.peak_rss_kb << newline
      << indent(1) << "}," << newline
      << indent(1) << "\"runtime\"" << colon << "{" << newline
      << indent(2) << "\"llm_backend\"" << colon << "\"" << json_escape(trace.runtime.llm_backend) << "\"," << newline
      << indent(2) << "\"embedding_backend\"" << colon << "\"" << json_escape(trace.runtime.embedding_backend) << "\"," << newline
      << indent(2) << "\"llm_model_path\"" << colon << "\"" << json_escape(trace.runtime.llm_model_path) << "\"," << newline
      << indent(2) << "\"embedding_model_path\"" << colon << "\"" << json_escape(trace.runtime.embedding_model_path) << "\"," << newline
      << indent(2) << "\"sqlite_db_path\"" << colon << "\"" << json_escape(trace.runtime.sqlite_db_path) << "\"," << newline
      << indent(2) << "\"index_path\"" << colon << "\"" << json_escape(trace.runtime.index_path) << "\"," << newline
      << indent(2) << "\"query_source\"" << colon << "\"" << json_escape(trace.runtime.query_source) << "\"," << newline
      << indent(2) << "\"num_threads\"" << colon << trace.runtime.num_threads << "," << newline
      << indent(2) << "\"max_new_tokens\"" << colon << trace.runtime.max_new_tokens << "," << newline
      << indent(2) << "\"sqlite_db_size_bytes\"" << colon << trace.runtime.sqlite_db_size_bytes << "," << newline
      << indent(2) << "\"index_size_bytes\"" << colon << trace.runtime.index_size_bytes << newline
      << indent(1) << "}," << newline
      << indent(1) << "\"index_state\"" << colon << "{" << newline
      << indent(2) << "\"hot\"" << colon << trace.index_state.hot_count << "," << newline
      << indent(2) << "\"warm\"" << colon << trace.index_state.warm_count << "," << newline
      << indent(2) << "\"cold\"" << colon << trace.index_state.cold_count << "," << newline
      << indent(2) << "\"transition_count\"" << colon << trace.index_state.transition_count << newline
      << indent(1) << "}," << newline
      << indent(1) << "\"evidence\"" << colon << "{" << newline
      << indent(2) << "\"top_score\"" << colon << trace.evidence.top_score << "," << newline
      << indent(2) << "\"second_score\"" << colon << trace.evidence.second_score << "," << newline
      << indent(2) << "\"score_margin\"" << colon << trace.evidence.score_margin << "," << newline
      << indent(2) << "\"score_sharpness\"" << colon << trace.evidence.score_sharpness << "," << newline
      << indent(2) << "\"retrieved_chunk_count\"" << colon << trace.evidence.retrieved_chunk_count << "," << newline
      << indent(2) << "\"query_term_count\"" << colon << trace.evidence.query_term_count << "," << newline
      << indent(2) << "\"covered_query_terms\"" << colon << trace.evidence.covered_query_terms << "," << newline
      << indent(2) << "\"coverage_ratio\"" << colon << trace.evidence.coverage_ratio << "," << newline
      << indent(2) << "\"numeric_constraint_count\"" << colon << trace.evidence.numeric_constraint_count << "," << newline
      << indent(2) << "\"covered_numeric_constraints\"" << colon << trace.evidence.covered_numeric_constraints << "," << newline
      << indent(2) << "\"unresolved_numeric_constraints\"" << colon << trace.evidence.unresolved_numeric_constraints << "," << newline
      << indent(2) << "\"year_constraint_count\"" << colon << trace.evidence.year_constraint_count << "," << newline
      << indent(2) << "\"covered_year_constraints\"" << colon << trace.evidence.covered_year_constraints << "," << newline
      << indent(2) << "\"unresolved_year_constraints\"" << colon << trace.evidence.unresolved_year_constraints << "," << newline
      << indent(2) << "\"entity_like_term_count\"" << colon << trace.evidence.entity_like_term_count << "," << newline
      << indent(2) << "\"covered_entity_like_terms\"" << colon << trace.evidence.covered_entity_like_terms << "," << newline
      << indent(2) << "\"unresolved_entity_like_terms\"" << colon << trace.evidence.unresolved_entity_like_terms << "," << newline
      << indent(2) << "\"unresolved_constraint_count\"" << colon << trace.evidence.unresolved_constraint_count << newline
      << indent(1) << "}," << newline
      << indent(1) << "\"results\"" << colon << "[";

  if (pretty && !trace.results.empty()) {
    out << newline;
  }

  for (size_t i = 0; i < trace.results.size(); ++i) {
    const auto& result = trace.results[i];
    out << indent(pretty ? 2 : 0)
        << "{\"id\"" << colon << result.id
        << ",\"score\"" << colon << result.score
        << ",\"preview\"" << colon << "\"" << json_escape(result.preview) << "\"}";
    if (i + 1 < trace.results.size()) {
      out << ",";
    }
    if (pretty) {
      out << newline;
    }
  }

  if (pretty && !trace.results.empty()) {
    out << indent(1);
  }

  out << "]" << newline
      << indent(0) << "}";
  return out.str();
}

struct RetrievalExecution {
  std::vector<std::pair<int64_t, float>> results;
  size_t lexical_candidate_count = 0;
  size_t hash_candidate_count = 0;
  size_t state_filtered_candidate_count = 0;
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

void RAGPipeline::set_state_aware_dense(StateAwareDenseConfig config) {
  state_aware_dense_.enabled = config.enabled;
}

void RAGPipeline::set_graph_selector_config(GraphSelector::Config config) {
  graph_selector_.set_config(config);
}

void RAGPipeline::set_trace_runtime_metadata(TraceRuntimeMetadata metadata) {
  trace_runtime_metadata_ = std::move(metadata);
}

bool RAGPipeline::export_last_query_trace(const std::string& output_path) const {
  if (!has_last_query_trace_ || output_path.empty()) {
    return false;
  }

  std::ofstream out(output_path, std::ios::trunc);
  if (!out) {
    return false;
  }

  out << serialize_query_trace_json(last_query_trace_, true) << '\n';
  return static_cast<bool>(out);
}

bool RAGPipeline::append_last_query_trace_jsonl(const std::string& output_path) const {
  if (!has_last_query_trace_ || output_path.empty()) {
    return false;
  }

  std::ofstream out(output_path, std::ios::app);
  if (!out) {
    return false;
  }

  out << serialize_query_trace_json(last_query_trace_, false) << '\n';
  return static_cast<bool>(out);
}

bool RAGPipeline::append_last_query_trace_summary_csv(const std::string& output_path) const {
  if (!has_last_query_trace_ || output_path.empty()) {
    return false;
  }

  const bool write_header =
      !std::filesystem::exists(output_path) || std::filesystem::is_empty(output_path);
  std::ofstream out(output_path, std::ios::app);
  if (!out) {
    return false;
  }

  if (write_header) {
    out << "query,answer,adaptive_graph_enabled,budget_class,initial_graph,final_graph,"
        << "initial_reason,final_reason,top_k,lexical_prefilter_enabled,"
        << "lexical_candidate_limit,semantic_hash_prefilter_enabled,"
        << "semantic_hash_candidate_limit,semantic_hash_max_distance,"
        << "state_aware_dense_enabled,lexical_candidate_count,hash_candidate_count,"
        << "state_filtered_candidate_count,dense_result_count,"
        << "fallback_reason,promoted_to_hot,demoted_to_warm,hot,warm,cold,"
        << "transition_count,top_score,second_score,score_margin,score_sharpness,"
        << "retrieved_chunk_count,query_term_count,covered_query_terms,coverage_ratio,"
        << "numeric_constraint_count,covered_numeric_constraints,"
        << "unresolved_numeric_constraints,year_constraint_count,"
        << "covered_year_constraints,unresolved_year_constraints,"
        << "entity_like_term_count,covered_entity_like_terms,"
        << "unresolved_entity_like_terms,unresolved_constraint_count,"
        << "query_embedding_ms,retrieval_ms,evidence_ms,state_update_ms,prompt_build_ms,"
        << "generation_ms,total_ms,peak_rss_kb,llm_backend,embedding_backend,query_source,"
        << "num_threads,max_new_tokens,sqlite_db_size_bytes,index_size_bytes,"
        << "top_result_id,top_result_score\n";
  }

  const auto top_result_id =
      last_query_trace_.results.empty() ? -1 : last_query_trace_.results.front().id;
  const auto top_result_score =
      last_query_trace_.results.empty() ? 0.0f : last_query_trace_.results.front().score;

  out << csv_escape(last_query_trace_.query) << ","
      << csv_escape(last_query_trace_.answer) << ","
      << (last_query_trace_.adaptive_graph_enabled ? "true" : "false") << ","
      << csv_escape(last_query_trace_.budget_class) << ","
      << csv_escape(last_query_trace_.initial_graph) << ","
      << csv_escape(last_query_trace_.final_graph) << ","
      << csv_escape(last_query_trace_.initial_reason) << ","
      << csv_escape(last_query_trace_.final_reason) << ","
      << last_query_trace_.top_k << ","
      << (last_query_trace_.lexical_prefilter_enabled ? "true" : "false") << ","
      << last_query_trace_.lexical_candidate_limit << ","
      << (last_query_trace_.semantic_hash_prefilter_enabled ? "true" : "false") << ","
      << last_query_trace_.semantic_hash_candidate_limit << ","
      << last_query_trace_.semantic_hash_max_distance << ","
      << (last_query_trace_.state_aware_dense_enabled ? "true" : "false") << ","
      << last_query_trace_.lexical_candidate_count << ","
      << last_query_trace_.hash_candidate_count << ","
      << last_query_trace_.state_filtered_candidate_count << ","
      << last_query_trace_.dense_result_count << ","
      << csv_escape(last_query_trace_.fallback_reason) << ","
      << last_query_trace_.promoted_to_hot << ","
      << last_query_trace_.demoted_to_warm << ","
      << last_query_trace_.index_state.hot_count << ","
      << last_query_trace_.index_state.warm_count << ","
      << last_query_trace_.index_state.cold_count << ","
      << last_query_trace_.index_state.transition_count << ","
      << last_query_trace_.evidence.top_score << ","
      << last_query_trace_.evidence.second_score << ","
      << last_query_trace_.evidence.score_margin << ","
      << last_query_trace_.evidence.score_sharpness << ","
      << last_query_trace_.evidence.retrieved_chunk_count << ","
      << last_query_trace_.evidence.query_term_count << ","
      << last_query_trace_.evidence.covered_query_terms << ","
      << last_query_trace_.evidence.coverage_ratio << ","
      << last_query_trace_.evidence.numeric_constraint_count << ","
      << last_query_trace_.evidence.covered_numeric_constraints << ","
      << last_query_trace_.evidence.unresolved_numeric_constraints << ","
      << last_query_trace_.evidence.year_constraint_count << ","
      << last_query_trace_.evidence.covered_year_constraints << ","
      << last_query_trace_.evidence.unresolved_year_constraints << ","
      << last_query_trace_.evidence.entity_like_term_count << ","
      << last_query_trace_.evidence.covered_entity_like_terms << ","
      << last_query_trace_.evidence.unresolved_entity_like_terms << ","
      << last_query_trace_.evidence.unresolved_constraint_count << ","
      << last_query_trace_.timing.query_embedding_ms << ","
      << last_query_trace_.timing.retrieval_ms << ","
      << last_query_trace_.timing.evidence_ms << ","
      << last_query_trace_.timing.state_update_ms << ","
      << last_query_trace_.timing.prompt_build_ms << ","
      << last_query_trace_.timing.generation_ms << ","
      << last_query_trace_.timing.total_ms << ","
      << last_query_trace_.system.peak_rss_kb << ","
      << csv_escape(last_query_trace_.runtime.llm_backend) << ","
      << csv_escape(last_query_trace_.runtime.embedding_backend) << ","
      << csv_escape(last_query_trace_.runtime.query_source) << ","
      << last_query_trace_.runtime.num_threads << ","
      << last_query_trace_.runtime.max_new_tokens << ","
      << last_query_trace_.runtime.sqlite_db_size_bytes << ","
      << last_query_trace_.runtime.index_size_bytes << ","
      << top_result_id << ","
      << top_result_score << "\n";
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
  const auto total_start = std::chrono::steady_clock::now();

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

  const auto query_embedding_start = std::chrono::steady_clock::now();
  std::vector<float> q = embedder_->embed_query(query);
  const double query_embedding_ms = elapsed_ms_since(query_embedding_start);
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
  query_trace.runtime = trace_runtime_metadata_;
  query_trace.timing.query_embedding_ms = query_embedding_ms;
  query_trace.lexical_prefilter_enabled = lexical_prefilter_.enabled;
  query_trace.lexical_candidate_limit = lexical_prefilter_.candidate_limit;
  query_trace.semantic_hash_prefilter_enabled = semantic_hash_prefilter_.enabled;
  query_trace.semantic_hash_candidate_limit = semantic_hash_prefilter_.candidate_limit;
  query_trace.semantic_hash_max_distance = semantic_hash_prefilter_.max_hamming_distance;
  query_trace.state_aware_dense_enabled = state_aware_dense_.enabled;

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

    if (state_aware_dense_.enabled && !candidate_ids.empty()) {
      const auto dense_candidate_ids = sqlite_db_->filter_ids_by_chunk_states(
          candidate_ids, {ChunkState::WARM, ChunkState::HOT});
      execution.state_filtered_candidate_count =
          candidate_ids.size() - dense_candidate_ids.size();
      candidate_ids = dense_candidate_ids;
    }

    if (!candidate_ids.empty()) {
      execution.results = sqlite_db_->search_with_ids(q, candidate_ids, top_k_);
      if (execution.results.empty()) {
        execution.fallback_reason =
            execution.state_filtered_candidate_count > 0
                ? "state_filtered_sqlite_rerank_empty"
                : "sqlite_rerank_empty";
      }
    } else {
      execution.fallback_reason =
          execution.state_filtered_candidate_count > 0
              ? "state_filtered_shortlist_empty"
              : "empty_shortlist";
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

  const auto first_retrieval_start = std::chrono::steady_clock::now();
  RetrievalExecution execution = execute_retrieval(active_graph);
  query_trace.timing.retrieval_ms += elapsed_ms_since(first_retrieval_start);
  auto retrieved_chunks = collect_chunks(execution.results);
  const auto first_evidence_start = std::chrono::steady_clock::now();
  auto evidence_features =
      compute_evidence_features(query, execution.results, retrieved_chunks);
  query_trace.timing.evidence_ms += elapsed_ms_since(first_evidence_start);

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
      const auto escalation_retrieval_start = std::chrono::steady_clock::now();
      execution = execute_retrieval(active_graph);
      query_trace.timing.retrieval_ms += elapsed_ms_since(escalation_retrieval_start);
      retrieved_chunks = collect_chunks(execution.results);
      const auto escalation_evidence_start = std::chrono::steady_clock::now();
      evidence_features =
          compute_evidence_features(query, execution.results, retrieved_chunks);
      query_trace.timing.evidence_ms += elapsed_ms_since(escalation_evidence_start);
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
            << " state_filtered_candidates=" << execution.state_filtered_candidate_count
            << " dense_results=" << execution.results.size()
            << " fallback=" << execution.fallback_reason << '\n';
  query_trace.lexical_candidate_count = execution.lexical_candidate_count;
  query_trace.hash_candidate_count = execution.hash_candidate_count;
  query_trace.state_filtered_candidate_count = execution.state_filtered_candidate_count;
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
    const auto state_update_start = std::chrono::steady_clock::now();
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
    query_trace.timing.state_update_ms += elapsed_ms_since(state_update_start);
  }

  std::cout << "[EVIDENCE] top_score=" << std::fixed << std::setprecision(4)
            << evidence_features.top_score
            << " second_score=" << evidence_features.second_score
            << " score_margin=" << evidence_features.score_margin
            << " score_sharpness=" << evidence_features.score_sharpness
            << " query_terms=" << evidence_features.query_term_count
            << " covered_terms=" << evidence_features.covered_query_terms
            << " coverage_ratio=" << evidence_features.coverage_ratio
            << " unresolved_constraints=" << evidence_features.unresolved_constraint_count
            << " unresolved_numeric=" << evidence_features.unresolved_numeric_constraints
            << " unresolved_year=" << evidence_features.unresolved_year_constraints
            << " unresolved_entity_terms=" << evidence_features.unresolved_entity_like_terms
            << " retrieved_chunks=" << evidence_features.retrieved_chunk_count
            << '\n';

  const auto prompt_build_start = std::chrono::steady_clock::now();
  std::string prompt = llm_->build_prompt(query, retrieved_chunks);
  query_trace.timing.prompt_build_ms = elapsed_ms_since(prompt_build_start);
  const auto generation_start = std::chrono::steady_clock::now();
  std::string answer = llm_->generate(prompt);
  query_trace.timing.generation_ms = elapsed_ms_since(generation_start);
  if (answer.empty()) {
    answer = fallback_answer_from_chunks(query, retrieved_chunks);
  }
  query_trace.answer = answer;
  query_trace.evidence = evidence_features;
  query_trace.timing.total_ms = elapsed_ms_since(total_start);
  query_trace.system.peak_rss_kb = current_peak_rss_kb();
  last_query_trace_ = std::move(query_trace);
  has_last_query_trace_ = true;
  return answer;
}

}  // namespace mobile_rag
