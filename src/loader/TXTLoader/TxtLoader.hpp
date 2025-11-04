#pragma once

#include <string>
#include <vector>

#include "BaseLoader.hpp"

namespace mobile_rag {

class TxtLoader : public BaseLoader {
 public:
  explicit TxtLoader(unsigned int num_threads = 1) : num_threads_(num_threads) {}
  ~TxtLoader() override = default;

  std::vector<std::string> load_texts(const std::string& path) override;

 private:
  unsigned int num_threads_;
};

}  // namespace mobile_rag



