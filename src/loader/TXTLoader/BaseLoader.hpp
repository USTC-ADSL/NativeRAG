#pragma once

#include <string>
#include <vector>

namespace mobile_rag {

class BaseLoader {
 public:
  virtual ~BaseLoader() = default;
  virtual std::vector<std::string> load_texts(const std::string& path) = 0;
};

}  // namespace mobile_rag



