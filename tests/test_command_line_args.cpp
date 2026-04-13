#include <cassert>
#include <filesystem>
#include <fstream>
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

void test_help_short_circuits_validation() {
  const auto result = run_parse({"mobile_rag", "--interactive", "--help"});

  if (result.parsed) {
    std::cerr << "parse() unexpectedly returned true\n";
    std::exit(1);
  }
  if (result.stdout_text.find("NativeRAG - Two-Stage Pipeline") == std::string::npos) {
    std::cerr << "stdout did not contain help banner\n";
    std::cerr << "--- stdout ---\n" << result.stdout_text;
    std::cerr << "--- stderr ---\n" << result.stderr_text;
    std::exit(1);
  }
  if (result.stderr_text.find("--llm-model is required") != std::string::npos) {
    std::cerr << "stderr still contains validation failure\n";
    std::cerr << "--- stdout ---\n" << result.stdout_text;
    std::cerr << "--- stderr ---\n" << result.stderr_text;
    std::exit(1);
  }
}

void test_query_file_parses_batch_exports() {
  const std::filesystem::path scratch_dir = "/tmp/native_rag_cli_flags";
  const auto llm_path = scratch_dir / "dummy-model.gguf";
  const auto embedding_path = scratch_dir / "dummy-embedding.json";
  const auto snapshot_in_path = scratch_dir / "state-in.snapshot.tsv";
  const auto query_file_path = scratch_dir / "queries.txt";

  std::filesystem::create_directories(scratch_dir);
  std::ofstream(llm_path.string()).put('\n');
  std::ofstream(embedding_path.string()).put('\n');
  std::ofstream(snapshot_in_path.string()) << "STATE_SNAPSHOT_V1\n";
  std::ofstream(query_file_path.string()) << "what is sqlite\n";

  std::vector<std::string> args = {
      "mobile_rag",
      "--query",
      "--llm-model",
      llm_path.string(),
      "--embedding-model",
      embedding_path.string(),
      "--query-file",
      query_file_path.string(),
      "--lexical-prefilter",
      "--lexical-candidates",
      "12",
      "--semantic-hash-prefilter",
      "--semantic-hash-candidates",
      "24",
      "--semantic-hash-max-distance",
      "12",
      "--adaptive-graph",
      "--state-snapshot-in",
      snapshot_in_path.string(),
      "--state-snapshot-out",
      "/tmp/out.snapshot.tsv",
      "--query-trace-jsonl-out",
      "/tmp/out.trace.jsonl",
      "--query-summary-csv-out",
      "/tmp/out.summary.csv",
      "--query-batch-report-out",
      "/tmp/out.batch.json",
      "--state-aware-dense",
  };

  std::vector<char*> argv;
  argv.reserve(args.size());
  for (const auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  mobile_rag::CommandLineArgs parser(static_cast<int>(argv.size()), argv.data());
  const bool parsed = parser.parse();
  assert(parsed);

  const auto& config = parser.get_config();
  assert(config.lexical_prefilter);
  assert(config.lexical_candidate_limit == 12);
  assert(config.semantic_hash_prefilter);
  assert(config.semantic_hash_candidate_limit == 24);
  assert(config.semantic_hash_max_distance == 12);
  assert(config.adaptive_graph);
  assert(config.query_file_path == query_file_path.string());
  assert(config.state_snapshot_in_path == snapshot_in_path.string());
  assert(config.state_snapshot_out_path == "/tmp/out.snapshot.tsv");
  assert(config.query_trace_out_path.empty());
  assert(config.query_trace_jsonl_out_path == "/tmp/out.trace.jsonl");
  assert(config.query_summary_csv_out_path == "/tmp/out.summary.csv");
  assert(config.query_batch_report_out_path == "/tmp/out.batch.json");
  assert(config.state_aware_dense);

  std::filesystem::remove_all(scratch_dir);
}

void test_query_file_rejects_single_json_trace_export() {
  const std::filesystem::path scratch_dir = "/tmp/native_rag_cli_query_file_conflict";
  const auto llm_path = scratch_dir / "dummy-model.gguf";
  const auto embedding_path = scratch_dir / "dummy-embedding.json";
  const auto query_file_path = scratch_dir / "queries.txt";

  std::filesystem::create_directories(scratch_dir);
  std::ofstream(llm_path.string()).put('\n');
  std::ofstream(embedding_path.string()).put('\n');
  std::ofstream(query_file_path.string()) << "what is sqlite\n";

  const auto result = run_parse({
      "mobile_rag",
      "--query",
      "--llm-model",
      llm_path.string(),
      "--embedding-model",
      embedding_path.string(),
      "--query-file",
      query_file_path.string(),
      "--query-trace-out",
      "/tmp/out.trace.json",
  });

  if (result.parsed) {
    std::cerr << "parse() unexpectedly accepted --query-file with --query-trace-out\n";
    std::exit(1);
  }
  if (result.stderr_text.find("--query-trace-out is not supported with --query-file") ==
      std::string::npos) {
    std::cerr << "stderr did not report the query-file trace conflict\n";
    std::cerr << "--- stdout ---\n" << result.stdout_text;
    std::cerr << "--- stderr ---\n" << result.stderr_text;
    std::exit(1);
  }

  std::filesystem::remove_all(scratch_dir);
}

void test_query_rejects_inline_query_and_query_file() {
  const std::filesystem::path scratch_dir = "/tmp/native_rag_cli_query_sources";
  const auto llm_path = scratch_dir / "dummy-model.gguf";
  const auto embedding_path = scratch_dir / "dummy-embedding.json";
  const auto query_file_path = scratch_dir / "queries.txt";

  std::filesystem::create_directories(scratch_dir);
  std::ofstream(llm_path.string()).put('\n');
  std::ofstream(embedding_path.string()).put('\n');
  std::ofstream(query_file_path.string()) << "what is sqlite\n";

  const auto result = run_parse({
      "mobile_rag",
      "--query",
      "What",
      "is",
      "SQLite",
      "--llm-model",
      llm_path.string(),
      "--embedding-model",
      embedding_path.string(),
      "--query-file",
      query_file_path.string(),
  });

  if (result.parsed) {
    std::cerr << "parse() unexpectedly accepted inline query together with --query-file\n";
    std::exit(1);
  }
  if (result.stderr_text.find("requires exactly one inline query or --query-file") ==
      std::string::npos) {
    std::cerr << "stderr did not report the query source conflict\n";
    std::cerr << "--- stdout ---\n" << result.stdout_text;
    std::cerr << "--- stderr ---\n" << result.stderr_text;
    std::exit(1);
  }

  std::filesystem::remove_all(scratch_dir);
}

void test_rebuild_state_filtered_index_parses_without_models() {
  const std::filesystem::path scratch_dir = "/tmp/native_rag_cli_rebuild_state_index";
  const auto sqlite_db_path = scratch_dir / "rag.sqlite3";
  const auto snapshot_in_path = scratch_dir / "state.snapshot.tsv";

  std::filesystem::create_directories(scratch_dir);
  std::ofstream(sqlite_db_path.string()).put('\n');
  std::ofstream(snapshot_in_path.string()) << "STATE_SNAPSHOT_V1\n";

  std::vector<std::string> args = {
      "mobile_rag",
      "--rebuild-state-filtered-index",
      "--sqlite-db",
      sqlite_db_path.string(),
      "--index-path",
      "/tmp/rebuilt-state.faiss",
      "--state-snapshot-in",
      snapshot_in_path.string(),
      "--faiss-type",
      "Flat",
      "--verbose",
  };

  std::vector<char*> argv;
  argv.reserve(args.size());
  for (const auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  mobile_rag::CommandLineArgs parser(static_cast<int>(argv.size()), argv.data());
  const bool parsed = parser.parse();
  assert(parsed);

  const auto& config = parser.get_config();
  assert(config.command == mobile_rag::CommandLineArgs::Command::REBUILD_STATE_FILTERED_INDEX);
  assert(config.sqlite_db_path == sqlite_db_path.string());
  assert(config.index_path == "/tmp/rebuilt-state.faiss");
  assert(config.state_snapshot_in_path == snapshot_in_path.string());
  assert(config.faiss_index_type == "Flat");
  assert(config.llm_model_path.empty());
  assert(config.embedding_model_path.empty());

  std::filesystem::remove_all(scratch_dir);
}

}  // namespace

int main() {
  test_help_short_circuits_validation();
  test_query_file_parses_batch_exports();
  test_query_file_rejects_single_json_trace_export();
  test_query_rejects_inline_query_and_query_file();
  test_rebuild_state_filtered_index_parses_without_models();
  std::cout << "CommandLineArgs regression tests passed\n";
  return 0;
}
