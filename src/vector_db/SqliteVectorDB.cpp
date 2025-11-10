#include "vector_db/SqliteVectorDB.hpp"

#include <iostream>

namespace mobile_rag {

namespace {
int exec_noop_callback(void*, int, char**, char**) { return 0; }
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
    const std::vector<std::vector<float>>& /*vectors*/,
    const std::vector<int64_t>& /*ids*/) {
  // Text-only schema: embeddings are not stored in SQLite.
  // Nothing to do here; return success.
  return true;
}

std::vector<std::pair<int64_t, float>> SqliteVectorDB::search(
    const std::vector<float>& /*query_vector*/, int /*k*/) {
  // TODO: Implement similarity search via sqlite-vec; placeholder empty result
  return {};
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



