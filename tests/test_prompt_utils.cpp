#include <iostream>
#include <string>

#include "llm/PromptUtils.hpp"

namespace {

void expect_equal(const std::string& actual, const std::string& expected,
                  const std::string& label) {
  if (actual != expected) {
    std::cerr << label << " mismatch\n";
    std::cerr << "expected: [" << expected << "]\n";
    std::cerr << "actual:   [" << actual << "]\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  expect_equal(
      mobile_rag::cleanup_generation_output(
          "Answer: SQLite is the source of truth."),
      "SQLite is the source of truth.",
      "answer_prefix");

  expect_equal(
      mobile_rag::cleanup_generation_output(
          "SQLite is the source of truth for metadata, evidence spans, and retrieval traces.\n"
          "Okay, so the user is asking what SQLite is in this project."),
      "SQLite is the source of truth for metadata, evidence spans, and retrieval traces.",
      "meta_reasoning_tail");

  std::cout << "PromptUtils cleanup test passed\n";
  return 0;
}
