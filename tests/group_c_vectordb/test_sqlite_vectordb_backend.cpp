#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "vector_db/SqliteVectorDB.hpp"

using namespace mobile_rag;

namespace {

std::string test_db_path() {
  return "/tmp/native_rag_sqlite_vectordb_backend.db";
}

void assert_top_result(const std::vector<std::pair<int64_t, float>>& results,
                       int64_t expected_id) {
  assert(!results.empty());
  assert(results.front().first == expected_id);
}

}  // namespace

int main() {
  const std::string db_path = test_db_path();
  std::filesystem::remove(db_path);

  {
    SqliteVectorDB db(db_path);

    const std::vector<std::vector<float>> vectors = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.8f, 0.2f, 0.0f},
    };
    const std::vector<int64_t> ids = {10, 20, 30};
    const std::vector<std::string> texts = {
        "alpha",
        "beta",
        "alpha-ish",
    };

    assert(db.add_vectors(vectors, ids));
    assert(db.add_texts(texts, ids));

    const auto alpha_results = db.search({1.0f, 0.0f, 0.0f}, 2);
    assert(alpha_results.size() == 2);
    assert_top_result(alpha_results, 10);
    assert(alpha_results[1].first == 30);
    assert(db.get_text_for_id(30) == "alpha-ish");

    const auto beta_results = db.search({0.0f, 1.0f, 0.0f}, 1);
    assert(beta_results.size() == 1);
    assert_top_result(beta_results, 20);
  }

  {
    SqliteVectorDB reopened(db_path);
    const auto reopened_results = reopened.search({1.0f, 0.0f, 0.0f}, 3);
    assert(reopened_results.size() == 3);
    assert_top_result(reopened_results, 10);
    assert(reopened_results[1].first == 30);
    assert(reopened_results[2].first == 20);
  }

  std::filesystem::remove(db_path);
  std::cout << "SqliteVectorDB backend test passed\n";
  return 0;
}
