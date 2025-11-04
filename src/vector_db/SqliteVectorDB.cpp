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
      "CREATE TABLE IF NOT EXISTS vectors ("
      "id INTEGER PRIMARY KEY,"
      "embedding BLOB"
      ");";
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
  // TODO: Implement storage using sqlite-vec; placeholder success for now
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

}  // namespace mobile_rag



