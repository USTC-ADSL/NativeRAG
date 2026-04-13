#include "vector_db/SqliteVectorDB.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_set>

#include "retrieval/SemanticHash.hpp"

namespace mobile_rag {

namespace {
int exec_noop_callback(void*, int, char**, char**) { return 0; }

std::vector<std::string> tokenize_lexical_terms(const std::string& text) {
  std::vector<std::string> terms;
  std::string current;
  for (char ch : text) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      current.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    } else if (!current.empty()) {
      if (current.size() > 1) {
        terms.push_back(current);
      }
      current.clear();
    }
  }

  if (!current.empty() && current.size() > 1) {
    terms.push_back(current);
  }

  return terms;
}

std::string build_fts_match_query(const std::vector<std::string>& terms) {
  std::string query;
  for (size_t i = 0; i < terms.size(); ++i) {
    if (i > 0) {
      query += " OR ";
    }
    query += terms[i];
  }
  return query;
}

int lexical_overlap_score(const std::vector<std::string>& query_terms,
                          const std::string& candidate_text) {
  if (query_terms.empty() || candidate_text.empty()) {
    return 0;
  }

  const auto candidate_terms_vec = tokenize_lexical_terms(candidate_text);
  if (candidate_terms_vec.empty()) {
    return 0;
  }

  const std::unordered_set<std::string> candidate_terms(
      candidate_terms_vec.begin(), candidate_terms_vec.end());

  int score = 0;
  for (const auto& term : query_terms) {
    if (candidate_terms.count(term) > 0) {
      ++score;
    }
  }
  return score;
}

bool try_enable_text_fts(sqlite3* db) {
  const char* sql =
      "CREATE VIRTUAL TABLE IF NOT EXISTS texts_fts "
      "USING fts5(text, tokenize='unicode61');";
  char* err = nullptr;
  const int rc = sqlite3_exec(db, sql, exec_noop_callback, nullptr, &err);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Warning: Failed to enable texts_fts: "
              << (err ? err : sqlite3_errmsg(db))
              << ". Falling back to non-FTS lexical search.\n";
    if (err) {
      sqlite3_free(err);
    }
    return false;
  }
  if (err) {
    sqlite3_free(err);
  }
  return true;
}

void try_refresh_text_fts(sqlite3* db) {
  const char* sql =
      "DELETE FROM texts_fts;"
      "INSERT INTO texts_fts(rowid, text) SELECT id, text FROM texts;";
  char* err = nullptr;
  const int rc = sqlite3_exec(db, sql, exec_noop_callback, nullptr, &err);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Warning: Failed to refresh texts_fts from texts: "
              << (err ? err : sqlite3_errmsg(db))
              << ". Falling back to non-FTS lexical search.\n";
  }
  if (err) {
    sqlite3_free(err);
  }
}

float cosine_similarity(const std::vector<float>& lhs,
                        const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size() || lhs.empty()) {
    return -std::numeric_limits<float>::infinity();
  }

  float dot = 0.0f;
  float lhs_norm = 0.0f;
  float rhs_norm = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    dot += lhs[i] * rhs[i];
    lhs_norm += lhs[i] * lhs[i];
    rhs_norm += rhs[i] * rhs[i];
  }

  if (lhs_norm <= 0.0f || rhs_norm <= 0.0f) {
    return -std::numeric_limits<float>::infinity();
  }

  return dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm));
}

std::vector<float> vector_from_blob(const void* blob, int bytes, int dim) {
  if (blob == nullptr || dim <= 0 ||
      bytes != static_cast<int>(sizeof(float) * static_cast<size_t>(dim))) {
    return {};
  }

  std::vector<float> values(static_cast<size_t>(dim));
  std::memcpy(values.data(), blob, static_cast<size_t>(bytes));
  return values;
}

sqlite3_int64 unix_time_ms_now() {
  using namespace std::chrono;
  return static_cast<sqlite3_int64>(
      duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}

std::string load_chunk_state(sqlite3* db, int64_t id) {
  if (!db) {
    return {};
  }

  const char* sql = "SELECT tier FROM chunk_states WHERE id = ?;";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return {};
  }

  rc = sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
  if (rc != SQLITE_OK) {
    sqlite3_finalize(stmt);
    return {};
  }

  std::string state;
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) {
    const unsigned char* text = sqlite3_column_text(stmt, 0);
    if (text) {
      state.assign(reinterpret_cast<const char*>(text));
    }
  }
  sqlite3_finalize(stmt);
  return state;
}

bool write_chunk_state_row(sqlite3* db,
                           int64_t id,
                           const std::string& state,
                           const std::string& reason,
                           sqlite3_int64 timestamp_ms) {
  const char* sql =
      "INSERT INTO chunk_states (id, tier, last_transition_reason, last_transition_at_unix_ms) "
      "VALUES (?, ?, ?, ?) "
      "ON CONFLICT(id) DO UPDATE SET tier=excluded.tier, "
      "last_transition_reason=excluded.last_transition_reason, "
      "last_transition_at_unix_ms=excluded.last_transition_at_unix_ms;";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return false;
  }

  rc = sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
  rc = (rc == SQLITE_OK) ? sqlite3_bind_text(stmt, 2, state.c_str(), -1, SQLITE_TRANSIENT) : rc;
  rc = (rc == SQLITE_OK) ? sqlite3_bind_text(stmt, 3, reason.c_str(), -1, SQLITE_TRANSIENT) : rc;
  rc = (rc == SQLITE_OK) ? sqlite3_bind_int64(stmt, 4, timestamp_ms) : rc;
  if (rc != SQLITE_OK) {
    sqlite3_finalize(stmt);
    return false;
  }

  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE;
}

bool write_chunk_transition_row(sqlite3* db,
                                int64_t id,
                                const std::string& from_state,
                                const std::string& to_state,
                                const std::string& reason,
                                sqlite3_int64 timestamp_ms) {
  const char* sql =
      "INSERT INTO chunk_state_transitions "
      "(id, from_tier, to_tier, reason, created_at_unix_ms) VALUES (?, ?, ?, ?, ?);";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return false;
  }

  rc = sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
  rc = (rc == SQLITE_OK)
           ? sqlite3_bind_text(stmt, 2, from_state.c_str(), -1, SQLITE_TRANSIENT)
           : rc;
  rc = (rc == SQLITE_OK)
           ? sqlite3_bind_text(stmt, 3, to_state.c_str(), -1, SQLITE_TRANSIENT)
           : rc;
  rc = (rc == SQLITE_OK)
           ? sqlite3_bind_text(stmt, 4, reason.c_str(), -1, SQLITE_TRANSIENT)
           : rc;
  rc = (rc == SQLITE_OK) ? sqlite3_bind_int64(stmt, 5, timestamp_ms) : rc;
  if (rc != SQLITE_OK) {
    sqlite3_finalize(stmt);
    return false;
  }

  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE;
}

std::vector<std::string> split_tab_fields(const std::string& line) {
  std::vector<std::string> fields;
  std::string current;
  std::istringstream stream(line);
  while (std::getline(stream, current, '\t')) {
    fields.push_back(current);
  }
  if (!line.empty() && line.back() == '\t') {
    fields.emplace_back();
  }
  return fields;
}

}  // namespace

const char* chunk_state_name(ChunkState state) {
  switch (state) {
    case ChunkState::COLD:
      return "cold";
    case ChunkState::WARM:
      return "warm";
    case ChunkState::HOT:
      return "hot";
  }
  return "unknown";
}

SqliteVectorDB::SqliteVectorDB(const std::string& db_path) : db_path_(db_path) {
  if (sqlite3_open(db_path_.c_str(), &db_) != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to open DB: "
              << sqlite3_errmsg(db_) << '\n';
    db_ = nullptr;
    return;
  }
  initialize_schema();
}

SqliteVectorDB::~SqliteVectorDB() {
  if (db_) {
    sqlite3_close(db_);
    db_ = nullptr;
  }
}

bool SqliteVectorDB::initialize_schema() {
  if (!db_) return false;
  const char* create_table_sql =
      "CREATE TABLE IF NOT EXISTS vectors ("
      "id INTEGER PRIMARY KEY,"
      "dim INTEGER NOT NULL,"
      "data BLOB NOT NULL"
      ");"
      "CREATE INDEX IF NOT EXISTS idx_vectors_id ON vectors(id);"
      "CREATE TABLE IF NOT EXISTS texts ("
      "id INTEGER PRIMARY KEY,"
      "text TEXT NOT NULL"
      ");"
      "CREATE INDEX IF NOT EXISTS idx_texts_id ON texts(id);"
      "CREATE TABLE IF NOT EXISTS semantic_hashes ("
      "id INTEGER PRIMARY KEY,"
      "bit_count INTEGER NOT NULL,"
      "code BLOB NOT NULL"
      ");"
      "CREATE INDEX IF NOT EXISTS idx_semantic_hashes_id ON semantic_hashes(id);"
      "CREATE TABLE IF NOT EXISTS chunk_states ("
      "id INTEGER PRIMARY KEY,"
      "tier TEXT NOT NULL,"
      "last_transition_reason TEXT NOT NULL,"
      "last_transition_at_unix_ms INTEGER NOT NULL"
      ");"
      "CREATE INDEX IF NOT EXISTS idx_chunk_states_tier ON chunk_states(tier);"
      "CREATE TABLE IF NOT EXISTS chunk_state_transitions ("
      "event_id INTEGER PRIMARY KEY AUTOINCREMENT,"
      "id INTEGER NOT NULL,"
      "from_tier TEXT NOT NULL,"
      "to_tier TEXT NOT NULL,"
      "reason TEXT NOT NULL,"
      "created_at_unix_ms INTEGER NOT NULL"
      ");"
      "CREATE INDEX IF NOT EXISTS idx_chunk_state_transitions_id "
      "ON chunk_state_transitions(id);";
  char* err = nullptr;
  if (sqlite3_exec(db_, create_table_sql, exec_noop_callback, nullptr, &err) !=
      SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Schema creation failed: "
              << (err ? err : "unknown") << '\n';
    if (err) sqlite3_free(err);
    return false;
  }
  if (err) sqlite3_free(err);
  if (try_enable_text_fts(db_)) {
    try_refresh_text_fts(db_);
  }
  return true;
}

bool SqliteVectorDB::add_vectors(
    const std::vector<std::vector<float>>& vectors,
    const std::vector<int64_t>& ids) {
  if (!db_) return false;
  if (vectors.size() != ids.size()) return false;
  if (vectors.empty()) return true;

  char* err = nullptr;
  if (sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, &err) != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to begin vector transaction: "
              << (err ? err : "unknown") << '\n';
    if (err) sqlite3_free(err);
    return false;
  }

  const char* sql =
      "INSERT INTO vectors (id, dim, data) VALUES (?, ?, ?) "
      "ON CONFLICT(id) DO UPDATE SET dim=excluded.dim, data=excluded.data;";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to prepare statement for add_vectors: "
              << sqlite3_errmsg(db_) << '\n';
    sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
    return false;
  }

  for (size_t i = 0; i < vectors.size(); ++i) {
    const auto& vector = vectors[i];
    if (vector.empty()) {
      std::cerr << "[SqliteVectorDB] Refusing to store empty vector for id "
                << ids[i] << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }

    rc = sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(ids[i]));
    if (rc != SQLITE_OK) {
      std::cerr << "[SqliteVectorDB] bind id failed in add_vectors: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }

    rc = sqlite3_bind_int(stmt, 2, static_cast<int>(vector.size()));
    if (rc != SQLITE_OK) {
      std::cerr << "[SqliteVectorDB] bind dim failed in add_vectors: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }

    rc = sqlite3_bind_blob(
        stmt, 3, vector.data(),
        static_cast<int>(sizeof(float) * vector.size()), SQLITE_TRANSIENT);
    if (rc != SQLITE_OK) {
      std::cerr << "[SqliteVectorDB] bind blob failed in add_vectors: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
      std::cerr << "[SqliteVectorDB] step failed in add_vectors: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }

    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);
  }

  sqlite3_finalize(stmt);

  if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &err) != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Commit failed for add_vectors: "
              << (err ? err : "unknown") << '\n';
    if (err) sqlite3_free(err);
    return false;
  }
  if (err) sqlite3_free(err);
  return true;
}

std::vector<std::pair<int64_t, float>> SqliteVectorDB::search(
    const std::vector<float>& query_vector, int k) {
  if (!db_ || query_vector.empty() || k <= 0) {
    return {};
  }

  const char* sql = "SELECT id, dim, data FROM vectors;";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to prepare search query: "
              << sqlite3_errmsg(db_) << '\n';
    return {};
  }

  std::vector<std::pair<int64_t, float>> scored_results;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    const auto id = static_cast<int64_t>(sqlite3_column_int64(stmt, 0));
    const int dim = sqlite3_column_int(stmt, 1);
    const void* blob = sqlite3_column_blob(stmt, 2);
    const int bytes = sqlite3_column_bytes(stmt, 2);

    auto candidate = vector_from_blob(blob, bytes, dim);
    if (candidate.empty() || candidate.size() != query_vector.size()) {
      continue;
    }

    const float score = cosine_similarity(query_vector, candidate);
    if (!std::isfinite(score)) {
      continue;
    }

    scored_results.emplace_back(id, score);
  }

  if (rc != SQLITE_DONE) {
    std::cerr << "[SqliteVectorDB] Search iteration failed: "
              << sqlite3_errmsg(db_) << '\n';
  }
  sqlite3_finalize(stmt);

  if (scored_results.empty()) {
    return {};
  }

  const size_t limit = std::min(static_cast<size_t>(k), scored_results.size());
  std::partial_sort(
      scored_results.begin(), scored_results.begin() + static_cast<std::ptrdiff_t>(limit),
      scored_results.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
  scored_results.resize(limit);
  return scored_results;
}

std::vector<std::pair<int64_t, float>> SqliteVectorDB::search_with_ids(
    const std::vector<float>& query_vector,
    const std::vector<int64_t>& candidate_ids,
    int k) const {
  if (!db_ || query_vector.empty() || candidate_ids.empty() || k <= 0) {
    return {};
  }

  const char* sql = "SELECT dim, data FROM vectors WHERE id = ?;";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to prepare filtered vector search: "
              << sqlite3_errmsg(db_) << '\n';
    return {};
  }

  std::vector<std::pair<int64_t, float>> scored_results;
  scored_results.reserve(candidate_ids.size());
  for (const auto id : candidate_ids) {
    rc = sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
    if (rc != SQLITE_OK) {
      std::cerr << "[SqliteVectorDB] Failed to bind filtered vector id: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      return {};
    }

    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
      const int dim = sqlite3_column_int(stmt, 0);
      const void* blob = sqlite3_column_blob(stmt, 1);
      const int bytes = sqlite3_column_bytes(stmt, 1);

      auto candidate = vector_from_blob(blob, bytes, dim);
      if (!candidate.empty() && candidate.size() == query_vector.size()) {
        const float score = cosine_similarity(query_vector, candidate);
        if (std::isfinite(score)) {
          scored_results.emplace_back(id, score);
        }
      }
    } else if (rc != SQLITE_DONE) {
      std::cerr << "[SqliteVectorDB] Filtered vector lookup failed: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      return {};
    }

    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);
  }

  sqlite3_finalize(stmt);

  std::sort(scored_results.begin(), scored_results.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
              }
              return lhs.first < rhs.first;
            });

  if (scored_results.size() > static_cast<size_t>(k)) {
    scored_results.resize(static_cast<size_t>(k));
  }
  return scored_results;
}

std::vector<std::pair<int64_t, float>> SqliteVectorDB::search_text_lexical(
    const std::string& query,
    int k) const {
  if (!db_ || query.empty() || k <= 0) {
    return {};
  }

  const auto query_terms = tokenize_lexical_terms(query);
  if (query_terms.empty()) {
    return {};
  }

  const std::string fts_query = build_fts_match_query(query_terms);
  const char* fts_sql =
      "SELECT rowid, bm25(texts_fts) "
      "FROM texts_fts WHERE texts_fts MATCH ? "
      "ORDER BY bm25(texts_fts) LIMIT ?;";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, fts_sql, -1, &stmt, nullptr);
  if (rc == SQLITE_OK) {
    rc = sqlite3_bind_text(stmt, 1, fts_query.c_str(), -1, SQLITE_TRANSIENT);
    if (rc != SQLITE_OK) {
      std::cerr << "[SqliteVectorDB] Failed to bind lexical FTS query: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      return {};
    }

    rc = sqlite3_bind_int(stmt, 2, k);
    if (rc != SQLITE_OK) {
      std::cerr << "[SqliteVectorDB] Failed to bind lexical FTS limit: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      return {};
    }

    std::vector<std::pair<int64_t, float>> results;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      const auto id = static_cast<int64_t>(sqlite3_column_int64(stmt, 0));
      const double bm25_score = sqlite3_column_double(stmt, 1);
      results.emplace_back(id, static_cast<float>(-bm25_score));
    }

    if (rc != SQLITE_DONE) {
      std::cerr << "[SqliteVectorDB] Lexical FTS search failed: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      return {};
    }

    sqlite3_finalize(stmt);
    return results;
  }

  if (stmt) {
    sqlite3_finalize(stmt);
  }

  const char* sql = "SELECT id, text FROM texts;";
  rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to prepare lexical fallback query: "
              << sqlite3_errmsg(db_) << '\n';
    return {};
  }

  std::vector<std::pair<int64_t, float>> scored_results;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    const auto id = static_cast<int64_t>(sqlite3_column_int64(stmt, 0));
    const unsigned char* text = sqlite3_column_text(stmt, 1);
    if (!text) {
      continue;
    }

    const int score = lexical_overlap_score(
        query_terms, reinterpret_cast<const char*>(text));
    if (score > 0) {
      scored_results.emplace_back(id, static_cast<float>(score));
    }
  }

  if (rc != SQLITE_DONE) {
    std::cerr << "[SqliteVectorDB] Lexical fallback iteration failed: "
              << sqlite3_errmsg(db_) << '\n';
    sqlite3_finalize(stmt);
    return {};
  }
  sqlite3_finalize(stmt);

  std::sort(scored_results.begin(), scored_results.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
              }
              return lhs.first < rhs.first;
            });

  if (scored_results.size() > static_cast<size_t>(k)) {
    scored_results.resize(static_cast<size_t>(k));
  }
  return scored_results;
}

bool SqliteVectorDB::add_semantic_hashes(
    const std::vector<std::vector<std::uint8_t>>& codes,
    const std::vector<int64_t>& ids,
    int bit_count) {
  if (!db_) return false;
  if (codes.size() != ids.size()) return false;
  if (codes.empty()) return true;
  if (bit_count <= 0) return false;

  char* err = nullptr;
  if (sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, &err) != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to begin semantic hash transaction: "
              << (err ? err : "unknown") << '\n';
    if (err) sqlite3_free(err);
    return false;
  }

  const char* sql =
      "INSERT INTO semantic_hashes (id, bit_count, code) VALUES (?, ?, ?) "
      "ON CONFLICT(id) DO UPDATE SET bit_count=excluded.bit_count, code=excluded.code;";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to prepare statement for add_semantic_hashes: "
              << sqlite3_errmsg(db_) << '\n';
    sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
    return false;
  }

  for (size_t i = 0; i < codes.size(); ++i) {
    const auto& code = codes[i];
    if (code.empty()) {
      std::cerr << "[SqliteVectorDB] Refusing to store empty semantic hash for id "
                << ids[i] << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }

    rc = sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(ids[i]));
    if (rc != SQLITE_OK) {
      std::cerr << "[SqliteVectorDB] bind id failed in add_semantic_hashes: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }

    rc = sqlite3_bind_int(stmt, 2, bit_count);
    if (rc != SQLITE_OK) {
      std::cerr << "[SqliteVectorDB] bind bit_count failed in add_semantic_hashes: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }

    rc = sqlite3_bind_blob(stmt, 3, code.data(), static_cast<int>(code.size()),
                           SQLITE_TRANSIENT);
    if (rc != SQLITE_OK) {
      std::cerr << "[SqliteVectorDB] bind code failed in add_semantic_hashes: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
      std::cerr << "[SqliteVectorDB] step failed in add_semantic_hashes: "
                << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }

    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);
  }

  sqlite3_finalize(stmt);

  if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &err) != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Commit failed for add_semantic_hashes: "
              << (err ? err : "unknown") << '\n';
    if (err) sqlite3_free(err);
    return false;
  }
  if (err) sqlite3_free(err);
  return true;
}

std::vector<std::pair<int64_t, int>> SqliteVectorDB::search_by_semantic_hash(
    const std::vector<std::uint8_t>& query_code,
    int k,
    int max_hamming_distance) const {
  if (!db_ || query_code.empty() || k <= 0) {
    return {};
  }

  const int query_bit_count = static_cast<int>(query_code.size() * 8);
  const char* sql =
      "SELECT id, code FROM semantic_hashes WHERE bit_count = ?;";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to prepare semantic hash query: "
              << sqlite3_errmsg(db_) << '\n';
    return {};
  }

  rc = sqlite3_bind_int(stmt, 1, query_bit_count);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to bind semantic hash bit_count: "
              << sqlite3_errmsg(db_) << '\n';
    sqlite3_finalize(stmt);
    return {};
  }

  std::vector<std::pair<int64_t, int>> scored_results;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    const auto id = static_cast<int64_t>(sqlite3_column_int64(stmt, 0));
    const void* blob = sqlite3_column_blob(stmt, 1);
    const int bytes = sqlite3_column_bytes(stmt, 1);
    if (blob == nullptr || bytes != static_cast<int>(query_code.size())) {
      continue;
    }

    std::vector<std::uint8_t> candidate_code(static_cast<size_t>(bytes));
    std::memcpy(candidate_code.data(), blob, static_cast<size_t>(bytes));

    const int distance = hamming_distance(query_code, candidate_code);
    if (distance == std::numeric_limits<int>::max()) {
      continue;
    }
    if (max_hamming_distance >= 0 && distance > max_hamming_distance) {
      continue;
    }
    scored_results.emplace_back(id, distance);
  }

  if (rc != SQLITE_DONE) {
    std::cerr << "[SqliteVectorDB] Semantic hash iteration failed: "
              << sqlite3_errmsg(db_) << '\n';
  }
  sqlite3_finalize(stmt);

  std::sort(scored_results.begin(), scored_results.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second < rhs.second;
              }
              return lhs.first < rhs.first;
            });

  if (scored_results.size() > static_cast<size_t>(k)) {
    scored_results.resize(static_cast<size_t>(k));
  }
  return scored_results;
}

bool SqliteVectorDB::save_index(const std::string& /*index_path*/) {
  // SQLite DB is already persisted at db_path_
  return true;
}

bool SqliteVectorDB::load_index(const std::string& /*index_path*/) {
  // Already opened in constructor; return true if DB handle is valid
  return db_ != nullptr;
}

bool SqliteVectorDB::add_texts(const std::vector<std::string>& texts,
                               const std::vector<int64_t>& ids) {
  if (!db_) return false;
  if (texts.size() != ids.size()) return false;
  if (texts.empty()) return true;

  char* err = nullptr;
  if (sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, &err) != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to begin transaction: "
              << (err ? err : "unknown") << '\n';
    if (err) sqlite3_free(err);
    return false;
  }

  const char* sql = "INSERT INTO texts (id, text) VALUES (?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET text=excluded.text;";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to prepare statement for add_texts: "
              << sqlite3_errmsg(db_) << '\n';
    sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
    return false;
  }

  sqlite3_stmt* fts_delete_stmt = nullptr;
  sqlite3_stmt* fts_insert_stmt = nullptr;
  bool use_fts = false;
  if (sqlite3_prepare_v2(db_, "DELETE FROM texts_fts WHERE rowid = ?;",
                         -1, &fts_delete_stmt, nullptr) == SQLITE_OK &&
      sqlite3_prepare_v2(db_, "INSERT INTO texts_fts(rowid, text) VALUES (?, ?);",
                         -1, &fts_insert_stmt, nullptr) == SQLITE_OK) {
    use_fts = true;
  } else {
    if (fts_delete_stmt) sqlite3_finalize(fts_delete_stmt);
    if (fts_insert_stmt) sqlite3_finalize(fts_insert_stmt);
    fts_delete_stmt = nullptr;
    fts_insert_stmt = nullptr;
  }

  for (size_t i = 0; i < texts.size(); ++i) {
    rc = sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(ids[i]));
    if (rc != SQLITE_OK) {
      std::cerr << "[SqliteVectorDB] bind id failed: " << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }
    rc = sqlite3_bind_text(stmt, 2, texts[i].c_str(), -1, SQLITE_TRANSIENT);
    if (rc != SQLITE_OK) {
      std::cerr << "[SqliteVectorDB] bind text failed: " << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
      std::cerr << "[SqliteVectorDB] step failed: " << sqlite3_errmsg(db_) << '\n';
      sqlite3_finalize(stmt);
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }
    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);

    if (use_fts) {
      rc = sqlite3_bind_int64(fts_delete_stmt, 1, static_cast<sqlite3_int64>(ids[i]));
      if (rc != SQLITE_OK) {
        std::cerr << "[SqliteVectorDB] bind rowid failed for texts_fts delete: "
                  << sqlite3_errmsg(db_) << '\n';
        sqlite3_finalize(stmt);
        sqlite3_finalize(fts_delete_stmt);
        sqlite3_finalize(fts_insert_stmt);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
      }
      rc = sqlite3_step(fts_delete_stmt);
      if (rc != SQLITE_DONE) {
        std::cerr << "[SqliteVectorDB] delete failed for texts_fts: "
                  << sqlite3_errmsg(db_) << '\n';
        sqlite3_finalize(stmt);
        sqlite3_finalize(fts_delete_stmt);
        sqlite3_finalize(fts_insert_stmt);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
      }
      sqlite3_reset(fts_delete_stmt);
      sqlite3_clear_bindings(fts_delete_stmt);

      rc = sqlite3_bind_int64(fts_insert_stmt, 1, static_cast<sqlite3_int64>(ids[i]));
      if (rc != SQLITE_OK) {
        std::cerr << "[SqliteVectorDB] bind rowid failed for texts_fts insert: "
                  << sqlite3_errmsg(db_) << '\n';
        sqlite3_finalize(stmt);
        sqlite3_finalize(fts_delete_stmt);
        sqlite3_finalize(fts_insert_stmt);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
      }
      rc = sqlite3_bind_text(fts_insert_stmt, 2, texts[i].c_str(), -1, SQLITE_TRANSIENT);
      if (rc != SQLITE_OK) {
        std::cerr << "[SqliteVectorDB] bind text failed for texts_fts insert: "
                  << sqlite3_errmsg(db_) << '\n';
        sqlite3_finalize(stmt);
        sqlite3_finalize(fts_delete_stmt);
        sqlite3_finalize(fts_insert_stmt);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
      }
      rc = sqlite3_step(fts_insert_stmt);
      if (rc != SQLITE_DONE) {
        std::cerr << "[SqliteVectorDB] insert failed for texts_fts: "
                  << sqlite3_errmsg(db_) << '\n';
        sqlite3_finalize(stmt);
        sqlite3_finalize(fts_delete_stmt);
        sqlite3_finalize(fts_insert_stmt);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
      }
      sqlite3_reset(fts_insert_stmt);
      sqlite3_clear_bindings(fts_insert_stmt);
    }
  }
  sqlite3_finalize(stmt);
  if (fts_delete_stmt) sqlite3_finalize(fts_delete_stmt);
  if (fts_insert_stmt) sqlite3_finalize(fts_insert_stmt);

  if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &err) != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Commit failed: " << (err ? err : "unknown") << '\n';
    if (err) sqlite3_free(err);
    return false;
  }
  if (err) sqlite3_free(err);
  return true;
}

bool SqliteVectorDB::set_text_for_id(int64_t id, const std::string& text) {
  if (!db_) return false;
  const char* sql = "INSERT INTO texts (id, text) VALUES (?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET text=excluded.text;";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Prepare failed in set_text_for_id: "
              << sqlite3_errmsg(db_) << '\n';
    return false;
  }
  rc = sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] bind id failed: " << sqlite3_errmsg(db_) << '\n';
    sqlite3_finalize(stmt);
    return false;
  }
  rc = sqlite3_bind_text(stmt, 2, text.c_str(), -1, SQLITE_TRANSIENT);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] bind text failed: " << sqlite3_errmsg(db_) << '\n';
    sqlite3_finalize(stmt);
    return false;
  }
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_DONE) {
    std::cerr << "[SqliteVectorDB] step failed: " << sqlite3_errmsg(db_) << '\n';
    sqlite3_finalize(stmt);
    return false;
  }
  sqlite3_finalize(stmt);

  sqlite3_stmt* fts_delete_stmt = nullptr;
  sqlite3_stmt* fts_insert_stmt = nullptr;
  if (sqlite3_prepare_v2(db_, "DELETE FROM texts_fts WHERE rowid = ?;",
                         -1, &fts_delete_stmt, nullptr) == SQLITE_OK &&
      sqlite3_prepare_v2(db_, "INSERT INTO texts_fts(rowid, text) VALUES (?, ?);",
                         -1, &fts_insert_stmt, nullptr) == SQLITE_OK) {
    rc = sqlite3_bind_int64(fts_delete_stmt, 1, static_cast<sqlite3_int64>(id));
    if (rc != SQLITE_OK) {
      sqlite3_finalize(fts_delete_stmt);
      sqlite3_finalize(fts_insert_stmt);
      return false;
    }
    rc = sqlite3_step(fts_delete_stmt);
    if (rc != SQLITE_DONE) {
      sqlite3_finalize(fts_delete_stmt);
      sqlite3_finalize(fts_insert_stmt);
      return false;
    }
    sqlite3_finalize(fts_delete_stmt);

    rc = sqlite3_bind_int64(fts_insert_stmt, 1, static_cast<sqlite3_int64>(id));
    if (rc != SQLITE_OK) {
      sqlite3_finalize(fts_insert_stmt);
      return false;
    }
    rc = sqlite3_bind_text(fts_insert_stmt, 2, text.c_str(), -1, SQLITE_TRANSIENT);
    if (rc != SQLITE_OK) {
      sqlite3_finalize(fts_insert_stmt);
      return false;
    }
    rc = sqlite3_step(fts_insert_stmt);
    if (rc != SQLITE_DONE) {
      sqlite3_finalize(fts_insert_stmt);
      return false;
    }
    sqlite3_finalize(fts_insert_stmt);
  } else {
    if (fts_delete_stmt) sqlite3_finalize(fts_delete_stmt);
    if (fts_insert_stmt) sqlite3_finalize(fts_insert_stmt);
  }

  return true;
}

std::string SqliteVectorDB::get_text_for_id(int64_t id) const {
  if (!db_) return {};
  const char* sql = "SELECT text FROM texts WHERE id = ?;";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Prepare failed in get_text_for_id: "
              << sqlite3_errmsg(db_) << '\n';
    return {};
  }
  rc = sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] bind id failed: " << sqlite3_errmsg(db_) << '\n';
    sqlite3_finalize(stmt);
    return {};
  }
  std::string out;
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) {
    const unsigned char* txt = sqlite3_column_text(stmt, 0);
    if (txt) out.assign(reinterpret_cast<const char*>(txt));
  }
  sqlite3_finalize(stmt);
  return out;
}

bool SqliteVectorDB::initialize_chunk_states(const std::vector<int64_t>& ids,
                                             ChunkState state,
                                             const std::string& reason) {
  if (!db_) return false;
  if (ids.empty()) return true;

  char* err = nullptr;
  if (sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, &err) != SQLITE_OK) {
    if (err) sqlite3_free(err);
    return false;
  }

  const std::string target_state = chunk_state_name(state);
  for (const auto id : ids) {
    const std::string current_state = load_chunk_state(db_, id);
    if (current_state == target_state) {
      continue;
    }

    const auto timestamp_ms = unix_time_ms_now();
    const std::string from_state = current_state.empty() ? "unknown" : current_state;
    if (!write_chunk_state_row(db_, id, target_state, reason, timestamp_ms) ||
        !write_chunk_transition_row(db_, id, from_state, target_state, reason, timestamp_ms)) {
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }
  }

  if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &err) != SQLITE_OK) {
    if (err) sqlite3_free(err);
    return false;
  }
  if (err) sqlite3_free(err);
  return true;
}

bool SqliteVectorDB::update_chunk_state(int64_t id,
                                        ChunkState new_state,
                                        const std::string& reason) {
  return initialize_chunk_states({id}, new_state, reason);
}

std::string SqliteVectorDB::get_chunk_state(int64_t id) const {
  return load_chunk_state(db_, id);
}

int SqliteVectorDB::count_chunk_state_transitions(int64_t id) const {
  if (!db_) return 0;

  const char* sql = "SELECT COUNT(*) FROM chunk_state_transitions WHERE id = ?;";
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return 0;
  }

  rc = sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(id));
  if (rc != SQLITE_OK) {
    sqlite3_finalize(stmt);
    return 0;
  }

  int count = 0;
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) {
    count = sqlite3_column_int(stmt, 0);
  }
  sqlite3_finalize(stmt);
  return count;
}

bool SqliteVectorDB::export_chunk_state_snapshot(const std::string& snapshot_path) const {
  if (!db_ || snapshot_path.empty()) return false;

  std::ofstream out(snapshot_path, std::ios::trunc);
  if (!out) {
    std::cerr << "[SqliteVectorDB] Failed to open snapshot for writing: "
              << snapshot_path << '\n';
    return false;
  }

  out << "STATE_SNAPSHOT_V1\n";

  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(
      db_,
      "SELECT id, tier, last_transition_reason, last_transition_at_unix_ms "
      "FROM chunk_states ORDER BY id;",
      -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to prepare chunk_states snapshot query: "
              << sqlite3_errmsg(db_) << '\n';
    return false;
  }

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    out << "STATE"
        << '\t' << static_cast<int64_t>(sqlite3_column_int64(stmt, 0))
        << '\t' << reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1))
        << '\t' << reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2))
        << '\t' << static_cast<long long>(sqlite3_column_int64(stmt, 3))
        << '\n';
  }
  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE) {
    std::cerr << "[SqliteVectorDB] Failed while exporting chunk_states snapshot: "
              << sqlite3_errmsg(db_) << '\n';
    return false;
  }

  rc = sqlite3_prepare_v2(
      db_,
      "SELECT event_id, id, from_tier, to_tier, reason, created_at_unix_ms "
      "FROM chunk_state_transitions ORDER BY id, created_at_unix_ms, event_id;",
      -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Failed to prepare chunk_state_transitions snapshot query: "
              << sqlite3_errmsg(db_) << '\n';
    return false;
  }

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    out << "TRANSITION"
        << '\t' << static_cast<long long>(sqlite3_column_int64(stmt, 0))
        << '\t' << static_cast<int64_t>(sqlite3_column_int64(stmt, 1))
        << '\t' << reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2))
        << '\t' << reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3))
        << '\t' << reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4))
        << '\t' << static_cast<long long>(sqlite3_column_int64(stmt, 5))
        << '\n';
  }
  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE) {
    std::cerr << "[SqliteVectorDB] Failed while exporting chunk_state_transitions snapshot: "
              << sqlite3_errmsg(db_) << '\n';
    return false;
  }

  return true;
}

bool SqliteVectorDB::import_chunk_state_snapshot(const std::string& snapshot_path) {
  if (!db_ || snapshot_path.empty()) return false;

  std::ifstream in(snapshot_path);
  if (!in) {
    std::cerr << "[SqliteVectorDB] Failed to open snapshot for reading: "
              << snapshot_path << '\n';
    return false;
  }

  std::string line;
  if (!std::getline(in, line) || line != "STATE_SNAPSHOT_V1") {
    std::cerr << "[SqliteVectorDB] Invalid snapshot header in: "
              << snapshot_path << '\n';
    return false;
  }

  char* err = nullptr;
  if (sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, &err) != SQLITE_OK) {
    if (err) sqlite3_free(err);
    return false;
  }

  if (sqlite3_exec(db_, "DELETE FROM chunk_state_transitions; DELETE FROM chunk_states;",
                   nullptr, nullptr, &err) != SQLITE_OK) {
    if (err) sqlite3_free(err);
    sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
    return false;
  }
  if (err) {
    sqlite3_free(err);
    err = nullptr;
  }

  sqlite3_stmt* state_stmt = nullptr;
  sqlite3_stmt* transition_stmt = nullptr;
  int rc = sqlite3_prepare_v2(
      db_,
      "INSERT INTO chunk_states "
      "(id, tier, last_transition_reason, last_transition_at_unix_ms) "
      "VALUES (?, ?, ?, ?);",
      -1, &state_stmt, nullptr);
  if (rc == SQLITE_OK) {
    rc = sqlite3_prepare_v2(
        db_,
        "INSERT INTO chunk_state_transitions "
        "(event_id, id, from_tier, to_tier, reason, created_at_unix_ms) "
        "VALUES (?, ?, ?, ?, ?, ?);",
        -1, &transition_stmt, nullptr);
  }
  if (rc != SQLITE_OK) {
    if (state_stmt) sqlite3_finalize(state_stmt);
    if (transition_stmt) sqlite3_finalize(transition_stmt);
    sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
    return false;
  }

  sqlite3_int64 max_transition_event_id = 0;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }

    const auto fields = split_tab_fields(line);
    if (fields.empty()) {
      continue;
    }

    if (fields[0] == "STATE") {
      if (fields.size() != 5) {
        sqlite3_finalize(state_stmt);
        sqlite3_finalize(transition_stmt);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
      }

      rc = sqlite3_bind_int64(state_stmt, 1, static_cast<sqlite3_int64>(std::stoll(fields[1])));
      rc = (rc == SQLITE_OK)
               ? sqlite3_bind_text(state_stmt, 2, fields[2].c_str(), -1, SQLITE_TRANSIENT)
               : rc;
      rc = (rc == SQLITE_OK)
               ? sqlite3_bind_text(state_stmt, 3, fields[3].c_str(), -1, SQLITE_TRANSIENT)
               : rc;
      rc = (rc == SQLITE_OK)
               ? sqlite3_bind_int64(state_stmt, 4,
                                    static_cast<sqlite3_int64>(std::stoll(fields[4])))
               : rc;
      if (rc != SQLITE_OK || sqlite3_step(state_stmt) != SQLITE_DONE) {
        sqlite3_finalize(state_stmt);
        sqlite3_finalize(transition_stmt);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
      }
      sqlite3_reset(state_stmt);
      sqlite3_clear_bindings(state_stmt);
      continue;
    }

    if (fields[0] == "TRANSITION") {
      if (fields.size() != 7) {
        sqlite3_finalize(state_stmt);
        sqlite3_finalize(transition_stmt);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
      }

      const sqlite3_int64 event_id = static_cast<sqlite3_int64>(std::stoll(fields[1]));
      max_transition_event_id = std::max(max_transition_event_id, event_id);
      rc = sqlite3_bind_int64(
          transition_stmt, 1, event_id);
      rc = (rc == SQLITE_OK)
               ? sqlite3_bind_int64(
                     transition_stmt, 2, static_cast<sqlite3_int64>(std::stoll(fields[2])))
               : rc;
      rc = (rc == SQLITE_OK)
               ? sqlite3_bind_text(
                     transition_stmt, 3, fields[3].c_str(), -1, SQLITE_TRANSIENT)
               : rc;
      rc = (rc == SQLITE_OK)
               ? sqlite3_bind_text(
                     transition_stmt, 4, fields[4].c_str(), -1, SQLITE_TRANSIENT)
               : rc;
      rc = (rc == SQLITE_OK)
               ? sqlite3_bind_text(
                     transition_stmt, 5, fields[5].c_str(), -1, SQLITE_TRANSIENT)
               : rc;
      rc = (rc == SQLITE_OK)
               ? sqlite3_bind_int64(
                     transition_stmt, 6, static_cast<sqlite3_int64>(std::stoll(fields[6])))
               : rc;
      if (rc != SQLITE_OK || sqlite3_step(transition_stmt) != SQLITE_DONE) {
        sqlite3_finalize(state_stmt);
        sqlite3_finalize(transition_stmt);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
      }
      sqlite3_reset(transition_stmt);
      sqlite3_clear_bindings(transition_stmt);
      continue;
    }

    sqlite3_finalize(state_stmt);
    sqlite3_finalize(transition_stmt);
    sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
    return false;
  }

  sqlite3_finalize(state_stmt);
  sqlite3_finalize(transition_stmt);

  if (sqlite3_exec(
          db_, "DELETE FROM sqlite_sequence WHERE name = 'chunk_state_transitions';",
          nullptr, nullptr, &err) != SQLITE_OK) {
    const std::string error_message = err ? err : "";
    if (err) sqlite3_free(err);
    err = nullptr;
    if (error_message.find("no such table: sqlite_sequence") == std::string::npos) {
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return false;
    }
  }
  if (err) {
    sqlite3_free(err);
    err = nullptr;
  }

  if (max_transition_event_id > 0) {
    sqlite3_stmt* sequence_stmt = nullptr;
    rc = sqlite3_prepare_v2(
        db_,
        "INSERT OR REPLACE INTO sqlite_sequence(name, seq) "
        "VALUES('chunk_state_transitions', ?);",
        -1, &sequence_stmt, nullptr);
    if (rc != SQLITE_OK) {
      const std::string error_message = sqlite3_errmsg(db_);
      sqlite3_finalize(sequence_stmt);
      if (error_message.find("no such table: sqlite_sequence") == std::string::npos) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
      }
    } else {
      rc = sqlite3_bind_int64(sequence_stmt, 1, max_transition_event_id);
      if (rc != SQLITE_OK || sqlite3_step(sequence_stmt) != SQLITE_DONE) {
        sqlite3_finalize(sequence_stmt);
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
      }
      sqlite3_finalize(sequence_stmt);
    }
  }

  if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &err) != SQLITE_OK) {
    if (err) sqlite3_free(err);
    return false;
  }
  if (err) sqlite3_free(err);
  return true;
}

}  // namespace mobile_rag
