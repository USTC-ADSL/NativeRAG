#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "faiss/Index.h"
#include "faiss/index_io.h"

#include "vector_Index/IVectorIndex.hpp"

namespace mobile_rag {

class FaissIndex : public IVectorIndex {
 public:
  FaissIndex() = default;
  ~FaissIndex() override = default;

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



