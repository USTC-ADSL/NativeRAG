#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "faiss/Index.h"
#include "faiss/index_io.h"

#include "mobile_rag/vector_db/IVectorDB.hpp"

namespace mobile_rag {

class FaissDB : public IVectorDB {
 public:
  FaissDB() = default;
  ~FaissDB() override = default;

  bool add_vectors(const std::vector<std::vector<float>>& vectors,
                   const std::vector<int64_t>& ids) override;

  std::vector<std::pair<int64_t, float>> search(
      const std::vector<float>& query_vector, int k) override;

  bool save_index(const std::string& index_path) override;

  bool load_index(const std::string& index_path) override;

 private:
  std::unique_ptr<faiss::Index> index_;
};

}  // namespace mobile_rag


