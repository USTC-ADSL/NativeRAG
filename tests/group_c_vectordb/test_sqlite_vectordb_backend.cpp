#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "vector_db/SqliteVectorDB.hpp"

using namespace mobile_rag;

namespace {

std::string test_db_path() {
  return "/tmp/native_rag_sqlite_vectordb_backend.db";
}

std::string snapshot_path() {
  return "/tmp/native_rag_sqlite_vectordb_backend.snapshot.tsv";
}

std::string warm_snapshot_path() {
  return "/tmp/native_rag_sqlite_vectordb_backend.warm.snapshot.tsv";
}

std::string replay_snapshot_path() {
  return "/tmp/native_rag_sqlite_vectordb_backend.replay.snapshot.tsv";
}

void assert_top_result(const std::vector<std::pair<int64_t, float>>& results,
                       int64_t expected_id) {
  assert(!results.empty());
  assert(results.front().first == expected_id);
}

}  // namespace

int main() {
  const std::string db_path = test_db_path();
  const std::string snapshot = snapshot_path();
  const std::string warm_snapshot = warm_snapshot_path();
  const std::string replay_snapshot = replay_snapshot_path();
  std::filesystem::remove(db_path);
  std::filesystem::remove(snapshot);
  std::filesystem::remove(warm_snapshot);
  std::filesystem::remove(replay_snapshot);

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
    assert(db.export_chunk_state_snapshot(warm_snapshot));

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
    assert(reopened.export_chunk_state_snapshot(snapshot));
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

  {
    const std::string restored_db_path = "/tmp/native_rag_sqlite_vectordb_backend_restored.db";
    std::filesystem::remove(restored_db_path);
    SqliteVectorDB restored(restored_db_path);
    assert(restored.import_chunk_state_snapshot(snapshot));
    assert(restored.get_chunk_state(10) == "cold");
    assert(restored.get_chunk_state(20) == "warm");
    assert(restored.count_chunk_state_transitions(10) == 3);
    assert(restored.count_chunk_state_transitions(20) == 1);
    std::filesystem::remove(restored_db_path);
  }

  {
    SqliteVectorDB replayed(db_path);
    assert(replayed.import_chunk_state_snapshot(warm_snapshot));
    assert(replayed.update_chunk_state(10, ChunkState::HOT, "replayed_hit"));
    assert(replayed.export_chunk_state_snapshot(replay_snapshot));

    std::ifstream snapshot_stream(replay_snapshot);
    std::string contents((std::istreambuf_iterator<char>(snapshot_stream)),
                         std::istreambuf_iterator<char>());
    assert(contents.find("TRANSITION\t4\t10\twarm\thot\treplayed_hit\t") !=
           std::string::npos);
  }

  std::filesystem::remove(db_path);
  std::filesystem::remove(snapshot);
  std::filesystem::remove(warm_snapshot);
  std::filesystem::remove(replay_snapshot);
  std::cout << "SqliteVectorDB backend test passed\n";
  return 0;
}
