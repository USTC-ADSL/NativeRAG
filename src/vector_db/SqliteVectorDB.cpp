#include "vector_db/SqliteVectorDB.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace mobile_rag {

namespace {
int exec_noop_callback(void*, int, char**, char**) { return 0; }

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
      "CREATE INDEX IF NOT EXISTS idx_texts_id ON texts(id);";
  char* err = nullptr;
  if (sqlite3_exec(db_, create_table_sql, exec_noop_callback, nullptr, &err) !=
      SQLITE_OK) {
    std::cerr << "[SqliteVectorDB] Schema creation failed: "
              << (err ? err : "unknown") << '\n';
    if (err) sqlite3_free(err);
    return false;
  }
  if (err) sqlite3_free(err);
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
  }
  sqlite3_finalize(stmt);

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

}  // namespace mobile_rag


