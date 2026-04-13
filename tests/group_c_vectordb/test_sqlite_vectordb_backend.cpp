#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "vector_Index/FaissIndex.hpp"
#include "vector_db/IngestUtils.hpp"
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

std::string filtered_index_path() {
  return "/tmp/native_rag_sqlite_vectordb_backend.filtered.faiss";
}

std::string empty_filtered_index_path() {
  return "/tmp/native_rag_sqlite_vectordb_backend.filtered.empty.faiss";
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
  const std::string filtered_index = filtered_index_path();
  const std::string empty_filtered_index = empty_filtered_index_path();
  std::filesystem::remove(db_path);
  std::filesystem::remove(snapshot);
  std::filesystem::remove(warm_snapshot);
  std::filesystem::remove(replay_snapshot);
  std::filesystem::remove(filtered_index);
  std::filesystem::remove(empty_filtered_index);

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
    const auto initial_summary = db.get_chunk_state_summary();
    assert(initial_summary.cold_count == 0);
    assert(initial_summary.warm_count == 3);
    assert(initial_summary.hot_count == 0);
    assert(initial_summary.transition_count == 3);
    assert(db.export_chunk_state_snapshot(warm_snapshot));

    const auto beta_results = db.search({0.0f, 1.0f, 0.0f}, 1);
    assert(beta_results.size() == 1);
    assert_top_result(beta_results, 20);

    const auto lexical_results = db.search_text_lexical("alpha metadata", 2);
    assert(lexical_results.size() == 2);
    assert(lexical_results[0].first == 10);
    assert(lexical_results[1].first == 30);

    assert(db.update_chunk_state(10, ChunkState::HOT, "retrieval_hit"));
    assert(db.update_chunk_state(20, ChunkState::HOT, "retrieval_hit"));
    assert(db.get_chunk_state(10) == "hot");
    assert(db.count_chunk_state_transitions(10) == 2);
    assert(db.get_chunk_state(20) == "hot");

    assert(db.demote_non_retrieved_hot_chunks({20}, ChunkState::WARM, "query_retrieval_miss") ==
           1);
    assert(db.get_chunk_state(10) == "warm");
    assert(db.get_chunk_state(20) == "hot");
    assert(db.count_chunk_state_transitions(10) == 3);
    const auto demoted_summary = db.get_chunk_state_summary();
    assert(demoted_summary.cold_count == 0);
    assert(demoted_summary.warm_count == 2);
    assert(demoted_summary.hot_count == 1);
    assert(demoted_summary.transition_count == 6);
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
    assert(reopened.get_chunk_state(10) == "warm");
    assert(reopened.get_chunk_state(20) == "hot");
    assert(reopened.count_chunk_state_transitions(10) == 3);
    assert(reopened.update_chunk_state(10, ChunkState::COLD, "manual_demotion"));
    assert(reopened.get_chunk_state(10) == "cold");
    assert(reopened.count_chunk_state_transitions(10) == 4);
    assert(reopened.get_vector_dimension() == 3);
    const auto warm_hot_vectors =
        reopened.load_vectors_by_chunk_states({ChunkState::WARM, ChunkState::HOT});
    assert(warm_hot_vectors.size() == 2);
    assert(warm_hot_vectors[0].first == 20);
    assert(warm_hot_vectors[0].second.size() == 3);
    assert(warm_hot_vectors[0].second[1] == 1.0f);
    assert(warm_hot_vectors[1].first == 30);
    assert(warm_hot_vectors[1].second.size() == 3);
    assert(warm_hot_vectors[1].second[0] == 0.8f);
    const auto reopened_summary = reopened.get_chunk_state_summary();
    assert(reopened_summary.cold_count == 1);
    assert(reopened_summary.warm_count == 1);
    assert(reopened_summary.hot_count == 1);
    assert(reopened_summary.transition_count == 7);
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
    assert(restored.get_chunk_state(20) == "hot");
    assert(restored.count_chunk_state_transitions(10) == 4);
    assert(restored.count_chunk_state_transitions(20) == 2);
    const auto restored_summary = restored.get_chunk_state_summary();
    assert(restored_summary.cold_count == 1);
    assert(restored_summary.warm_count == 1);
    assert(restored_summary.hot_count == 1);
    assert(restored_summary.transition_count == 7);
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

  {
    SqliteVectorDB filtered_source(db_path);
    assert(filtered_source.import_chunk_state_snapshot(snapshot));
    assert(rebuild_faiss_index_from_sqlite_by_chunk_states(
        db_path, filtered_index, {ChunkState::WARM, ChunkState::HOT}));
    FaissIndex filtered_index_reader;
    assert(filtered_index_reader.load_index(filtered_index));
    const auto filtered_results = filtered_index_reader.search({1.0f, 0.0f, 0.0f}, 2);
    assert(filtered_results.size() == 2);
    assert(filtered_results[0].first == 30);
    assert(filtered_results[1].first == 20);
  }

  {
    SqliteVectorDB all_cold(db_path);
    assert(all_cold.update_chunk_state(20, ChunkState::COLD, "manual_cold"));
    assert(all_cold.update_chunk_state(30, ChunkState::COLD, "manual_cold"));
    assert(rebuild_faiss_index_from_sqlite_by_chunk_states(
        db_path, empty_filtered_index, {ChunkState::WARM, ChunkState::HOT}));
    FaissIndex empty_index_reader;
    assert(empty_index_reader.load_index(empty_filtered_index));
    const auto empty_results = empty_index_reader.search({1.0f, 0.0f, 0.0f}, 1);
    assert(empty_results.empty());
  }

  std::filesystem::remove(db_path);
  std::filesystem::remove(snapshot);
  std::filesystem::remove(warm_snapshot);
  std::filesystem::remove(replay_snapshot);
  std::filesystem::remove(filtered_index);
  std::filesystem::remove(empty_filtered_index);
  std::cout << "SqliteVectorDB backend test passed\n";
  return 0;
}
