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

bool dense_available(const GraphSelector::Availability& availability) {
  return availability.dense_graph_available;
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

const char* budget_class_name(BudgetClass budget_class) {
  switch (budget_class) {
    case BudgetClass::TIGHT:
      return "tight";
    case BudgetClass::BALANCED:
      return "balanced";
    case BudgetClass::RELAXED:
      return "relaxed";
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
  config.tight_budget_shortlist_factor = std::max(1, config.tight_budget_shortlist_factor);
  config.balanced_budget_shortlist_factor = std::max(
      config.tight_budget_shortlist_factor + 1, config.balanced_budget_shortlist_factor);
  config_ = config;
}

BudgetClass GraphSelector::classify_budget(const Availability& availability,
                                           const BudgetContext& budget) const {
  const int safe_top_k = std::max(1, budget.top_k);

  int shortlist_budget = 0;
  if (lexical_available(availability)) {
    shortlist_budget += std::max(0, budget.lexical_candidate_limit);
  }
  if (semantic_available(availability)) {
    shortlist_budget += std::max(0, budget.semantic_hash_candidate_limit);
  }

  if (shortlist_budget <= safe_top_k * config_.tight_budget_shortlist_factor) {
    return BudgetClass::TIGHT;
  }
  if (shortlist_budget <= safe_top_k * config_.balanced_budget_shortlist_factor) {
    return BudgetClass::BALANCED;
  }
  return BudgetClass::RELAXED;
}

GraphSelector::Decision GraphSelector::choose_initial_graph(
    const std::string& query,
    const Availability& availability,
    const BudgetContext& budget) const {
  if (!config_.enabled) {
    return {RetrievalGraph::DENSE_ONLY, "adaptive_disabled", false};
  }

  const bool lexical = lexical_available(availability);
  const bool semantic = semantic_available(availability);
  const bool dense = dense_available(availability);
  const int query_term_count = static_cast<int>(tokenize_terms(query).size());
  const bool numeric_query = has_numeric_token(query);
  const auto budget_class = classify_budget(availability, budget);

  if (!lexical && !semantic) {
    return {RetrievalGraph::DENSE_ONLY,
            availability.sqlite_available ? "no_prefilter_available" : "sqlite_unavailable",
            false};
  }

  if (lexical && semantic) {
    if (numeric_query) {
      if (budget_class == BudgetClass::TIGHT) {
        return {RetrievalGraph::LEXICAL_PREFILTER, "numeric_query_budget_limited", false};
      }
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
    if (!dense) {
      return {RetrievalGraph::LEXICAL_PREFILTER, "dense_state_unavailable", false};
    }
    return {RetrievalGraph::DENSE_ONLY, "query_too_short_for_lexical", false};
  }

  if (query_term_count > 0) {
    return {RetrievalGraph::SEMANTIC_HASH_PREFILTER, "semantic_only_available", false};
  }

  if (!dense && semantic) {
    return {RetrievalGraph::SEMANTIC_HASH_PREFILTER, "dense_state_unavailable", false};
  }

  if (!dense) {
    return {RetrievalGraph::DENSE_ONLY,
            availability.sqlite_available ? "dense_state_unavailable" : "sqlite_unavailable",
            false};
  }

  return {RetrievalGraph::DENSE_ONLY, "no_content_terms", false};
}

GraphSelector::Decision GraphSelector::maybe_escalate(
    RetrievalGraph current_graph,
    const Availability& availability,
    const BudgetContext& budget,
    const EvidenceFeatures& evidence) const {
  if (!config_.enabled) {
    return {current_graph, "adaptive_disabled", false};
  }

  if (current_graph == RetrievalGraph::LEXICAL_HASH_PREFILTER) {
    return {current_graph, "already_richest_graph", false};
  }

  const bool lexical = lexical_available(availability);
  const bool semantic = semantic_available(availability);
  const bool dense = dense_available(availability);

  if (current_graph == RetrievalGraph::DENSE_ONLY && !dense) {
    if (lexical && semantic) {
      return {RetrievalGraph::LEXICAL_HASH_PREFILTER, "dense_state_unavailable_upgrade", true};
    }
    if (lexical) {
      return {RetrievalGraph::LEXICAL_PREFILTER, "dense_state_unavailable_upgrade", true};
    }
    if (semantic) {
      return {RetrievalGraph::SEMANTIC_HASH_PREFILTER, "dense_state_unavailable_upgrade", true};
    }
  }

  const bool unresolved_constraints =
      evidence.unresolved_constraint_count > 0;
  if (unresolved_constraints) {
    if (classify_budget(availability, budget) == BudgetClass::TIGHT) {
      return {current_graph, "budget_limited", false};
    }

    const bool lexical = lexical_available(availability);
    const bool semantic = semantic_available(availability);
    if (lexical && semantic) {
      return {RetrievalGraph::LEXICAL_HASH_PREFILTER,
              "constraint_uncovered_upgrade", true};
    }
    if (current_graph == RetrievalGraph::DENSE_ONLY) {
      if (lexical) {
        return {RetrievalGraph::LEXICAL_PREFILTER,
                "constraint_uncovered_upgrade", true};
      }
      if (semantic) {
        return {RetrievalGraph::SEMANTIC_HASH_PREFILTER,
                "constraint_uncovered_upgrade", true};
      }
    }
  }

  const bool weak_evidence =
      evidence.retrieved_chunk_count == 0 ||
      evidence.coverage_ratio < config_.min_coverage_ratio ||
      evidence.score_margin < config_.min_score_margin;
  if (!weak_evidence) {
    return {current_graph, "evidence_sufficient", false};
  }

  if (classify_budget(availability, budget) == BudgetClass::TIGHT) {
    return {current_graph, "budget_limited", false};
  }

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
