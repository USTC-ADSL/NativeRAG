#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <sqlite3.h>

#include "vector_db/IVectorDB.hpp"

namespace mobile_rag {

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

 private:
  bool initialize_schema();

 private:
  sqlite3* db_ = nullptr;
  std::string db_path_;
};

}  // namespace mobile_rag



