#pragma once

#include <string>
#include <vector>

#include "BaseLoader.hpp"

namespace mobile_rag {

class TxtLoader : public BaseLoader {
 public:
  TxtLoader() = default;
  ~TxtLoader() override = default;

  std::vector<std::string> load_texts(const std::string& path) override;
};

}  // namespace mobile_rag


