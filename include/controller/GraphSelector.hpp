#pragma once

#include <string>

#include "controller/EvidenceFeatures.hpp"

namespace mobile_rag {

enum class RetrievalGraph {
  DENSE_ONLY,
  LEXICAL_PREFILTER,
  SEMANTIC_HASH_PREFILTER,
  LEXICAL_HASH_PREFILTER,
};

const char* retrieval_graph_name(RetrievalGraph graph);

enum class BudgetClass {
  TIGHT,
  BALANCED,
  RELAXED,
};

const char* budget_class_name(BudgetClass budget_class);

class GraphSelector {
 public:
  struct Config {
    bool enabled = false;
    int lexical_query_term_threshold = 3;
    float min_score_margin = 0.15f;
    float min_coverage_ratio = 0.50f;
    int tight_budget_shortlist_factor = 2;
    int balanced_budget_shortlist_factor = 4;
  };

  struct Availability {
    bool sqlite_available = false;
    bool dense_graph_available = true;
    bool lexical_graph_available = false;
    bool semantic_hash_graph_available = false;
  };

  struct BudgetContext {
    int top_k = 1;
    int lexical_candidate_limit = 0;
    int semantic_hash_candidate_limit = 0;
  };

  struct Decision {
    RetrievalGraph graph = RetrievalGraph::DENSE_ONLY;
    std::string reason = "adaptive_disabled";
    bool escalated = false;
  };

  GraphSelector();
  explicit GraphSelector(Config config);

  void set_config(Config config);
  const Config& config() const { return config_; }

  BudgetClass classify_budget(const Availability& availability,
                              const BudgetContext& budget) const;
  Decision choose_initial_graph(const std::string& query,
                                const Availability& availability,
                                const BudgetContext& budget) const;
  Decision maybe_escalate(RetrievalGraph current_graph,
                          const Availability& availability,
                          const BudgetContext& budget,
                          const EvidenceFeatures& evidence) const;

 private:
  Config config_;
};

}  // namespace mobile_rag
