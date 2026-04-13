#include "controller/EvidenceFeatures.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_set>

namespace mobile_rag {

namespace {

struct QueryToken {
  std::string raw;
  std::string normalized;
};

bool should_ignore_term(const std::string& term) {
  static const std::unordered_set<std::string> kIgnoredTerms = {
      "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
      "does", "doing", "did", "done", "do", "is", "are", "was", "were", "be",
      "been", "being", "the", "this", "that", "these", "those", "and", "for",
      "with", "from", "into", "about", "your", "their", "there", "here",
  };
  return kIgnoredTerms.count(term) > 0;
}

bool is_numeric_token(const std::string& token) {
  return !token.empty() &&
         std::all_of(token.begin(), token.end(), [](unsigned char ch) {
           return std::isdigit(ch) != 0;
         });
}

bool is_year_token(const std::string& token) {
  return token.size() == 4 && is_numeric_token(token);
}

bool is_entity_like_query_token(const QueryToken& token) {
  return !token.raw.empty() &&
         std::isupper(static_cast<unsigned char>(token.raw.front())) &&
         token.normalized.size() > 2 &&
         !is_numeric_token(token.normalized) &&
         !should_ignore_term(token.normalized);
}

std::vector<std::string> tokenize_terms(const std::string& text) {
  std::vector<std::string> terms;
  std::string current;
  for (char ch : text) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      current.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    } else if (!current.empty()) {
      if ((is_numeric_token(current) || current.size() > 2) &&
          !should_ignore_term(current)) {
        terms.push_back(current);
      }
      current.clear();
    }
  }

  if (!current.empty() && (is_numeric_token(current) || current.size() > 2) &&
      !should_ignore_term(current)) {
    terms.push_back(current);
  }

  return terms;
}

std::vector<QueryToken> tokenize_query_tokens(const std::string& text) {
  std::vector<QueryToken> tokens;
  std::string current_raw;
  std::string current_normalized;
  for (char ch : text) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      current_raw.push_back(ch);
      current_normalized.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    } else if (!current_raw.empty()) {
      if ((is_numeric_token(current_normalized) || current_normalized.size() > 2) &&
          !should_ignore_term(current_normalized)) {
        tokens.push_back({current_raw, current_normalized});
      }
      current_raw.clear();
      current_normalized.clear();
    }
  }

  if (!current_raw.empty() &&
      (is_numeric_token(current_normalized) || current_normalized.size() > 2) &&
      !should_ignore_term(current_normalized)) {
    tokens.push_back({current_raw, current_normalized});
  }

  return tokens;
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

  const auto query_tokens_vec = tokenize_query_tokens(query);
  std::unordered_set<std::string> query_terms;
  std::unordered_set<std::string> numeric_constraints;
  std::unordered_set<std::string> year_constraints;
  std::unordered_set<std::string> entity_like_terms;
  for (const auto& token : query_tokens_vec) {
    query_terms.insert(token.normalized);
    if (is_numeric_token(token.normalized)) {
      numeric_constraints.insert(token.normalized);
    }
    if (is_year_token(token.normalized)) {
      year_constraints.insert(token.normalized);
    }
    if (is_entity_like_query_token(token)) {
      entity_like_terms.insert(token.normalized);
    }
  }
  features.query_term_count = static_cast<int>(query_terms.size());
  features.numeric_constraint_count = static_cast<int>(numeric_constraints.size());
  features.year_constraint_count = static_cast<int>(year_constraints.size());
  features.entity_like_term_count = static_cast<int>(entity_like_terms.size());

  if (query_terms.empty() || retrieved_chunks.empty()) {
    features.unresolved_numeric_constraints = features.numeric_constraint_count;
    features.unresolved_year_constraints = features.year_constraint_count;
    features.unresolved_entity_like_terms = features.entity_like_term_count;
    features.unresolved_constraint_count =
        features.unresolved_numeric_constraints +
        features.unresolved_entity_like_terms;
    return features;
  }

  std::unordered_set<std::string> covered_terms;
  std::unordered_set<std::string> covered_numeric_constraints;
  std::unordered_set<std::string> covered_year_constraints;
  std::unordered_set<std::string> covered_entity_like_terms;
  for (const auto& chunk : retrieved_chunks) {
    const auto chunk_terms_vec = tokenize_terms(chunk);
    for (const auto& term : chunk_terms_vec) {
      if (query_terms.count(term) > 0) {
        covered_terms.insert(term);
      }
      if (numeric_constraints.count(term) > 0) {
        covered_numeric_constraints.insert(term);
      }
      if (year_constraints.count(term) > 0) {
        covered_year_constraints.insert(term);
      }
      if (entity_like_terms.count(term) > 0) {
        covered_entity_like_terms.insert(term);
      }
    }
  }

  features.covered_query_terms = static_cast<int>(covered_terms.size());
  features.coverage_ratio =
      static_cast<float>(features.covered_query_terms) /
      static_cast<float>(features.query_term_count);
  features.covered_numeric_constraints =
      static_cast<int>(covered_numeric_constraints.size());
  features.unresolved_numeric_constraints =
      features.numeric_constraint_count - features.covered_numeric_constraints;
  features.covered_year_constraints =
      static_cast<int>(covered_year_constraints.size());
  features.unresolved_year_constraints =
      features.year_constraint_count - features.covered_year_constraints;
  features.covered_entity_like_terms =
      static_cast<int>(covered_entity_like_terms.size());
  features.unresolved_entity_like_terms =
      features.entity_like_term_count - features.covered_entity_like_terms;
  features.unresolved_constraint_count =
      features.unresolved_numeric_constraints +
      features.unresolved_entity_like_terms;

  return features;
}

}  // namespace mobile_rag
