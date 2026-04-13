#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "controller/EvidenceFeatures.hpp"

namespace {

void expect_close(float actual, float expected, float tolerance,
                  const std::string& label) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << label << " mismatch\n";
    std::cerr << "expected: " << expected << "\n";
    std::cerr << "actual:   " << actual << "\n";
    std::exit(1);
  }
}

void expect_equal(int actual, int expected, const std::string& label) {
  if (actual != expected) {
    std::cerr << label << " mismatch\n";
    std::cerr << "expected: " << expected << "\n";
    std::cerr << "actual:   " << actual << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  const std::vector<std::pair<int64_t, float>> ranked_results = {
      {11, 0.90f},
      {22, 0.55f},
  };
  const std::vector<std::string> chunks = {
      "SQLite stores metadata and retrieval traces.",
      "Faiss is an accelerator for dense reranking.",
  };

  const auto features = mobile_rag::compute_evidence_features(
      "What metadata does SQLite store?", ranked_results, chunks);

  expect_close(features.top_score, 0.90f, 1e-5f, "top_score");
  expect_close(features.second_score, 0.55f, 1e-5f, "second_score");
  expect_close(features.score_margin, 0.35f, 1e-5f, "score_margin");
  expect_equal(features.query_term_count, 3, "query_term_count");
  expect_equal(features.covered_query_terms, 2, "covered_query_terms");
  expect_close(features.coverage_ratio, 2.0f / 3.0f, 1e-5f, "coverage_ratio");

  std::cout << "EvidenceFeatures test passed\n";
  return 0;
}
