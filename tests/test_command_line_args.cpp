#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cli/CommandLineArgs.hpp"

namespace {

struct ParseResult {
  bool parsed = false;
  std::string stdout_text;
  std::string stderr_text;
};

ParseResult run_parse(const std::vector<std::string>& args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (const auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  std::ostringstream captured_stdout;
  std::ostringstream captured_stderr;
  auto* old_stdout = std::cout.rdbuf(captured_stdout.rdbuf());
  auto* old_stderr = std::cerr.rdbuf(captured_stderr.rdbuf());

  mobile_rag::CommandLineArgs parser(static_cast<int>(argv.size()), argv.data());
  const bool parsed = parser.parse();

  std::cout.rdbuf(old_stdout);
  std::cerr.rdbuf(old_stderr);

  return {
      parsed,
      captured_stdout.str(),
      captured_stderr.str(),
  };
}

}  // namespace

int main() {
  const auto result = run_parse({"mobile_rag", "--interactive", "--help"});

  if (result.parsed) {
    std::cerr << "parse() unexpectedly returned true\n";
    return 1;
  }
  if (result.stdout_text.find("NativeRAG - Two-Stage Pipeline") == std::string::npos) {
    std::cerr << "stdout did not contain help banner\n";
    std::cerr << "--- stdout ---\n" << result.stdout_text;
    std::cerr << "--- stderr ---\n" << result.stderr_text;
    return 1;
  }
  if (result.stderr_text.find("--llm-model is required") != std::string::npos) {
    std::cerr << "stderr still contains validation failure\n";
    std::cerr << "--- stdout ---\n" << result.stdout_text;
    std::cerr << "--- stderr ---\n" << result.stderr_text;
    return 1;
  }

  std::cout << "CommandLineArgs help regression test passed\n";
  return 0;
}
