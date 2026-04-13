#include "controller/EvidenceFeatures.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_set>

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

}  // namespace

EvidenceFeatures compute_evidence_features(
    const std::string& query,
    const std::vector<std::pair<int64_t, float>>& ranked_results,
    const std::vector<std::string>& retrieved_chunks) {
  EvidenceFeatures features;
  features.retrieved_chunk_count = static_cast<int>(retrieved_chunks.size());

  if (!ranked_results.empty()) {
    features.top_score = ranked_results.front().second;
  }
  if (ranked_results.size() > 1) {
    features.second_score = ranked_results[1].second;
  }
  features.score_margin = features.top_score - features.second_score;

  const float denom = std::max(std::fabs(features.top_score), 1e-6f);
  features.score_sharpness = features.score_margin / denom;

  const auto query_terms_vec = tokenize_terms(query);
  const std::unordered_set<std::string> query_terms(
      query_terms_vec.begin(), query_terms_vec.end());
  features.query_term_count = static_cast<int>(query_terms.size());

  if (query_terms.empty() || retrieved_chunks.empty()) {
    return features;
  }

  std::unordered_set<std::string> covered_terms;
  for (const auto& chunk : retrieved_chunks) {
    const auto chunk_terms_vec = tokenize_terms(chunk);
    for (const auto& term : chunk_terms_vec) {
      if (query_terms.count(term) > 0) {
        covered_terms.insert(term);
      }
    }
  }

  features.covered_query_terms = static_cast<int>(covered_terms.size());
  features.coverage_ratio =
      static_cast<float>(features.covered_query_terms) /
      static_cast<float>(features.query_term_count);

  return features;
}

}  // namespace mobile_rag
