#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mobile_rag {

struct EvidenceFeatures {
  float top_score = 0.0f;
  float second_score = 0.0f;
  float score_margin = 0.0f;
  float score_sharpness = 0.0f;
  int retrieved_chunk_count = 0;
  int query_term_count = 0;
  int covered_query_terms = 0;
  float coverage_ratio = 0.0f;
  int numeric_constraint_count = 0;
  int covered_numeric_constraints = 0;
  int unresolved_numeric_constraints = 0;
  int year_constraint_count = 0;
  int covered_year_constraints = 0;
  int unresolved_year_constraints = 0;
  int entity_like_term_count = 0;
  int covered_entity_like_terms = 0;
  int unresolved_entity_like_terms = 0;
  int unresolved_constraint_count = 0;
};

EvidenceFeatures compute_evidence_features(
    const std::string& query,
    const std::vector<std::pair<int64_t, float>>& ranked_results,
    const std::vector<std::string>& retrieved_chunks);

}  // namespace mobile_rag
