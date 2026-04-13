#include "controller/GraphSelector.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_set>
#include <vector>

namespace mobile_rag {

namespace {

bool should_ignore_term(const std::string& term) {
  static const std::unordered_set<std::string> kIgnoredTerms = {
      "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
      "does", "doing", "did", "done", "do", "is", "are", "was", "were", "be",
      "been", "being", "the", "this", "that", "these", "those", "and", "for",
      "with", "from", "into", "about", "your", "their", "there", "here",
  };
  return kIgnoredTerms.count(term) > 0;
}

std::vector<std::string> tokenize_terms(const std::string& text) {
  std::vector<std::string> terms;
  std::string current;
  for (char ch : text) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      current.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    } else if (!current.empty()) {
      if (current.size() > 2 && !should_ignore_term(current)) {
        terms.push_back(current);
      }
      current.clear();
    }
  }

  if (!current.empty() && current.size() > 2 && !should_ignore_term(current)) {
    terms.push_back(current);
  }

  return terms;
}

bool has_numeric_token(const std::string& text) {
  for (char ch : text) {
    if (std::isdigit(static_cast<unsigned char>(ch))) {
      return true;
    }
  }
  return false;
}

bool lexical_available(const GraphSelector::Availability& availability) {
  return availability.sqlite_available && availability.lexical_graph_available;
}

bool semantic_available(const GraphSelector::Availability& availability) {
  return availability.sqlite_available && availability.semantic_hash_graph_available;
}

}  // namespace

const char* retrieval_graph_name(RetrievalGraph graph) {
  switch (graph) {
    case RetrievalGraph::DENSE_ONLY:
      return "dense_only";
    case RetrievalGraph::LEXICAL_PREFILTER:
      return "lexical_prefilter";
    case RetrievalGraph::SEMANTIC_HASH_PREFILTER:
      return "semantic_hash_prefilter";
    case RetrievalGraph::LEXICAL_HASH_PREFILTER:
      return "lexical_hash_prefilter";
  }
  return "unknown";
}

GraphSelector::GraphSelector() {
  set_config(Config{});
}

GraphSelector::GraphSelector(Config config) {
  set_config(config);
}

void GraphSelector::set_config(Config config) {
  config.lexical_query_term_threshold = std::max(1, config.lexical_query_term_threshold);
  config.min_score_margin = std::max(0.0f, config.min_score_margin);
  config.min_coverage_ratio = std::max(0.0f, std::min(1.0f, config.min_coverage_ratio));
  config_ = config;
}

GraphSelector::Decision GraphSelector::choose_initial_graph(
    const std::string& query,
    const Availability& availability) const {
  if (!config_.enabled) {
    return {RetrievalGraph::DENSE_ONLY, "adaptive_disabled", false};
  }

  const bool lexical = lexical_available(availability);
  const bool semantic = semantic_available(availability);
  const int query_term_count = static_cast<int>(tokenize_terms(query).size());
  const bool numeric_query = has_numeric_token(query);

  if (!lexical && !semantic) {
    return {RetrievalGraph::DENSE_ONLY,
            availability.sqlite_available ? "no_prefilter_available" : "sqlite_unavailable",
            false};
  }

  if (lexical && semantic) {
    if (numeric_query) {
      return {RetrievalGraph::LEXICAL_HASH_PREFILTER, "numeric_query", false};
    }
    if (query_term_count >= config_.lexical_query_term_threshold) {
      return {RetrievalGraph::LEXICAL_PREFILTER, "term_rich_query", false};
    }
    return {RetrievalGraph::SEMANTIC_HASH_PREFILTER, "short_query", false};
  }

  if (lexical) {
    if (numeric_query) {
      return {RetrievalGraph::LEXICAL_PREFILTER, "numeric_query", false};
    }
    if (query_term_count >= config_.lexical_query_term_threshold) {
      return {RetrievalGraph::LEXICAL_PREFILTER, "lexical_only_available", false};
    }
    return {RetrievalGraph::DENSE_ONLY, "query_too_short_for_lexical", false};
  }

  if (query_term_count > 0) {
    return {RetrievalGraph::SEMANTIC_HASH_PREFILTER, "semantic_only_available", false};
  }

  return {RetrievalGraph::DENSE_ONLY, "no_content_terms", false};
}

GraphSelector::Decision GraphSelector::maybe_escalate(
    RetrievalGraph current_graph,
    const Availability& availability,
    const EvidenceFeatures& evidence) const {
  if (!config_.enabled) {
    return {current_graph, "adaptive_disabled", false};
  }

  if (current_graph == RetrievalGraph::LEXICAL_HASH_PREFILTER) {
    return {current_graph, "already_richest_graph", false};
  }

  const bool weak_evidence =
      evidence.retrieved_chunk_count == 0 ||
      evidence.coverage_ratio < config_.min_coverage_ratio ||
      evidence.score_margin < config_.min_score_margin;
  if (!weak_evidence) {
    return {current_graph, "evidence_sufficient", false};
  }

  const bool lexical = lexical_available(availability);
  const bool semantic = semantic_available(availability);

  if (lexical && semantic) {
    return {RetrievalGraph::LEXICAL_HASH_PREFILTER, "low_evidence_upgrade", true};
  }

  if (current_graph == RetrievalGraph::DENSE_ONLY) {
    if (lexical) {
      return {RetrievalGraph::LEXICAL_PREFILTER, "low_evidence_upgrade", true};
    }
    if (semantic) {
      return {RetrievalGraph::SEMANTIC_HASH_PREFILTER, "low_evidence_upgrade", true};
    }
  }

  return {current_graph, "no_richer_graph", false};
}

}  // namespace mobile_rag
