#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "faiss/Index.h"
#include "faiss/index_io.h"
#include "faiss/MetricType.h"

#include "vector_Index/IVectorIndex.hpp"

namespace mobile_rag {

class FaissIndex : public IVectorIndex {
 public:
  // Default: Flat with Inner Product
  FaissIndex();
  // Customizable via FAISS factory description and metric
  FaissIndex(std::string factory_desc, faiss::MetricType metric);
  ~FaissIndex() override = default;

  bool add_vectors(const std::vector<std::vector<float>>& vectors,
                   const std::vector<int64_t>& ids) override;

  std::vector<std::pair<int64_t, float>> search(
      const std::vector<float>& query_vector, int k) override;

  bool initialize_empty(int dimension);

  bool save_index(const std::string& index_path) override;

  bool load_index(const std::string& index_path) override;

 private:
  bool ensure_index_created(int dimension);
  bool train_if_needed(const std::vector<std::vector<float>>& vectors);

  std::unique_ptr<faiss::Index> index_;
  std::string factory_desc_;
  faiss::MetricType metric_ = faiss::METRIC_INNER_PRODUCT;
};

}  // namespace mobile_rag


