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
    assert(db.initialize_chunk_states(ids, ChunkState::WARM, "index_build"));

    const auto alpha_results = db.search({1.0f, 0.0f, 0.0f}, 2);
    assert(alpha_results.size() == 2);
    assert_top_result(alpha_results, 10);
    assert(alpha_results[1].first == 30);
    assert(db.get_text_for_id(30) == "alpha-ish");
    assert(db.get_chunk_state(10) == "warm");
    assert(db.count_chunk_state_transitions(10) == 1);

    const auto beta_results = db.search({0.0f, 1.0f, 0.0f}, 1);
    assert(beta_results.size() == 1);
    assert_top_result(beta_results, 20);

    const auto lexical_results = db.search_text_lexical("alpha metadata", 2);
    assert(lexical_results.size() == 2);
    assert(lexical_results[0].first == 10);
    assert(lexical_results[1].first == 30);

    assert(db.update_chunk_state(10, ChunkState::HOT, "retrieval_hit"));
    assert(db.get_chunk_state(10) == "hot");
    assert(db.count_chunk_state_transitions(10) == 2);
  }

  {
    SqliteVectorDB reopened(db_path);
    const auto reopened_results = reopened.search({1.0f, 0.0f, 0.0f}, 3);
    assert(reopened_results.size() == 3);
    assert_top_result(reopened_results, 10);
    assert(reopened_results[1].first == 30);
    assert(reopened_results[2].first == 20);

    const auto reopened_lexical = reopened.search_text_lexical("beta", 1);
    assert(reopened_lexical.size() == 1);
    assert(reopened_lexical[0].first == 20);
    assert(reopened.get_chunk_state(10) == "hot");
    assert(reopened.count_chunk_state_transitions(10) == 2);
    assert(reopened.update_chunk_state(10, ChunkState::COLD, "manual_demotion"));
    assert(reopened.get_chunk_state(10) == "cold");
    assert(reopened.count_chunk_state_transitions(10) == 3);
  }

  {
    sqlite3* raw_db = nullptr;
    assert(sqlite3_open(db_path.c_str(), &raw_db) == SQLITE_OK);
    char* err = nullptr;
    assert(sqlite3_exec(raw_db, "DELETE FROM texts_fts;", nullptr, nullptr, &err) == SQLITE_OK);
    if (err) {
      sqlite3_free(err);
    }
    sqlite3_close(raw_db);
  }

  {
    SqliteVectorDB reopened_after_fts_clear(db_path);
    const auto repaired_lexical = reopened_after_fts_clear.search_text_lexical("alpha", 2);
    assert(repaired_lexical.size() == 2);
    assert(repaired_lexical[0].first == 10);
    assert(repaired_lexical[1].first == 30);
  }

  std::filesystem::remove(db_path);
  std::cout << "SqliteVectorDB backend test passed\n";
  return 0;
}
