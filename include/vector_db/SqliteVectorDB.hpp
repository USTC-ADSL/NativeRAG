#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <sqlite3.h>

namespace mobile_rag {

// Persistent metadata store. Vector search is performed exclusively by Faiss.
class SqliteVectorDB {
 public:
  explicit SqliteVectorDB(const std::string& db_path = ":memory:");
  ~SqliteVectorDB();

  bool add_texts(const std::vector<std::string>& texts,
                 const std::vector<int64_t>& ids);
  std::string get_text_for_id(int64_t id) const;

 private:
  bool initialize_schema();

 private:
  sqlite3* db_ = nullptr;
  std::string db_path_;
};

}  // namespace mobile_rag


