#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <sqlite3.h>

#include "vector_db/IVectorDB.hpp"

namespace mobile_rag {

enum class ChunkState {
  COLD,
  WARM,
  HOT,
};

const char* chunk_state_name(ChunkState state);

struct ChunkStateSummary {
  int cold_count = 0;
  int warm_count = 0;
  int hot_count = 0;
  int transition_count = 0;
};

class SqliteVectorDB : public IVectorDB {
 public:
  explicit SqliteVectorDB(const std::string& db_path = ":memory:");
  ~SqliteVectorDB() override;

  bool add_vectors(const std::vector<std::vector<float>>& vectors,
                   const std::vector<int64_t>& ids) override;

  std::vector<std::pair<int64_t, float>> search(
      const std::vector<float>& query_vector, int k) override;

  bool save_index(const std::string& index_path) override;

  bool load_index(const std::string& index_path) override;

  // Optional convenience APIs for storing/retrieving associated text metadata
  bool add_texts(const std::vector<std::string>& texts,
                 const std::vector<int64_t>& ids);
  bool set_text_for_id(int64_t id, const std::string& text);
  std::string get_text_for_id(int64_t id) const;
  std::vector<std::pair<int64_t, float>> search_text_lexical(
      const std::string& query,
      int k) const;
  bool add_semantic_hashes(const std::vector<std::vector<std::uint8_t>>& codes,
                           const std::vector<int64_t>& ids,
                           int bit_count);
  std::vector<std::pair<int64_t, int>> search_by_semantic_hash(
      const std::vector<std::uint8_t>& query_code,
      int k,
      int max_hamming_distance = -1) const;
  std::vector<std::pair<int64_t, float>> search_with_ids(
      const std::vector<float>& query_vector,
      const std::vector<int64_t>& candidate_ids,
      int k) const;
  bool initialize_chunk_states(const std::vector<int64_t>& ids,
                               ChunkState state,
                               const std::string& reason);
  bool update_chunk_state(int64_t id,
                          ChunkState new_state,
                          const std::string& reason);
  std::string get_chunk_state(int64_t id) const;
  ChunkStateSummary get_chunk_state_summary() const;
  int count_chunk_state_transitions(int64_t id) const;
  int demote_non_retrieved_hot_chunks(const std::vector<int64_t>& retained_ids,
                                      ChunkState target_state,
                                      const std::string& reason);
  bool export_chunk_state_snapshot(const std::string& snapshot_path) const;
  bool import_chunk_state_snapshot(const std::string& snapshot_path);

 private:
  bool initialize_schema();

 private:
  sqlite3* db_ = nullptr;
  std::string db_path_;
};

}  // namespace mobile_rag
