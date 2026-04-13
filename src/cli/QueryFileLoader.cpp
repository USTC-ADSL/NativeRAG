#include "cli/QueryFileLoader.hpp"

#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string trim_copy(const std::string& input) {
  const size_t begin = input.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }

  const size_t end = input.find_last_not_of(" \t\r\n");
  return input.substr(begin, end - begin + 1);
}

}  // namespace

namespace mobile_rag {

std::vector<std::string> load_query_file(const std::string& path) {
  std::ifstream in(path);
  std::vector<std::string> queries;
  if (!in) {
    return queries;
  }

  std::string line;
  while (std::getline(in, line)) {
    std::string trimmed = trim_copy(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }

    queries.push_back(std::move(trimmed));
  }

  return queries;
}

}  // namespace mobile_rag
