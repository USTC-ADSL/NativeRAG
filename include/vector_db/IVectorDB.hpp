#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mobile_rag {

class IVectorDB {
 public:
  virtual ~IVectorDB() = default;

  // Note: IDs should be associated with metadata externally to this interface
  virtual bool add_vectors(const std::vector<std::vector<float>>& vectors,
                           const std::vector<int64_t>& ids) = 0;

  // Returns pairs of (ID, score)
  virtual std::vector<std::pair<int64_t, float>> search(
      const std::vector<float>& query_vector, int k) = 0;

  virtual bool save_index(const std::string& index_path) = 0;

  virtual bool load_index(const std::string& index_path) = 0;
};

}  // namespace mobile_rag


