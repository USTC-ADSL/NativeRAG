#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "RAGPipeline.hpp"
#include "cli/BatchQueryReport.hpp"

namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

void test_batch_query_report_exports_aggregate_json() {
  mobile_rag::BatchQueryReport report;

  mobile_rag::RAGPipeline::QueryTrace sqlite_trace;
  sqlite_trace.query = "Which store keeps metadata?";
  sqlite_trace.initial_graph = "lexical_prefilter";
  sqlite_trace.final_graph = "lexical_prefilter";
  sqlite_trace.fallback_reason = "none";
  sqlite_trace.escalated = false;
  sqlite_trace.lexical_candidate_count = 2;
  sqlite_trace.hash_candidate_count = 0;
  sqlite_trace.dense_result_count = 1;
  sqlite_trace.promoted_to_hot = 1;
  sqlite_trace.demoted_to_warm = 1;
  sqlite_trace.evidence.top_score = 0.75f;
  sqlite_trace.evidence.score_margin = 0.75f;
  sqlite_trace.evidence.coverage_ratio = 0.60f;
  report.record(sqlite_trace);

  mobile_rag::RAGPipeline::QueryTrace faiss_trace;
  faiss_trace.query = "What accelerates dense search?";
  faiss_trace.initial_graph = "lexical_prefilter";
  faiss_trace.final_graph = "dense_only";
  faiss_trace.fallback_reason = "dense_fallback";
  faiss_trace.escalated = true;
  faiss_trace.lexical_candidate_count = 0;
  faiss_trace.hash_candidate_count = 1;
  faiss_trace.dense_result_count = 1;
  faiss_trace.promoted_to_hot = 1;
  faiss_trace.demoted_to_warm = 0;
  faiss_trace.evidence.top_score = 0.50f;
  faiss_trace.evidence.score_margin = 0.25f;
  faiss_trace.evidence.coverage_ratio = 0.40f;
  report.record(faiss_trace);

  const std::filesystem::path output_path = "/tmp/native_rag_batch_query_report.json";
  std::filesystem::remove(output_path);

  assert(report.export_json(output_path.string()));

  const std::string json = read_file(output_path);
  assert(json.find("\"query_count\": 2") != std::string::npos);
  assert(json.find("\"escalation_count\": 1") != std::string::npos);
  assert(json.find("\"lexical_prefilter\": 2") != std::string::npos);
  assert(json.find("\"dense_only\": 1") != std::string::npos);
  assert(json.find("\"dense_fallback\": 1") != std::string::npos);
  assert(json.find("\"coverage_ratio\": 0.5") != std::string::npos);
  assert(json.find("\"promoted_to_hot\": 2") != std::string::npos);

  std::filesystem::remove(output_path);
}

}  // namespace

int main() {
  test_batch_query_report_exports_aggregate_json();
  std::cout << "Batch query report tests passed\n";
  return 0;
}
