#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cli/QueryFileLoader.hpp"

namespace {

void test_load_query_file_skips_blank_and_comment_lines() {
  const std::filesystem::path path = "/tmp/native_rag_query_file_loader.txt";
  std::ofstream out(path.string(), std::ios::trunc);
  out << "\n";
  out << "   # ignore this line\n";
  out << "  Which store keeps metadata?   \n";
  out << "\t\n";
  out << "What accelerates dense search?\n";
  out.close();

  const auto queries = mobile_rag::load_query_file(path.string());
  assert(queries.size() == 2);
  assert(queries[0] == "Which store keeps metadata?");
  assert(queries[1] == "What accelerates dense search?");

  std::filesystem::remove(path);
}

}  // namespace

int main() {
  test_load_query_file_skips_blank_and_comment_lines();
  std::cout << "Query file loader tests passed\n";
  return 0;
}
