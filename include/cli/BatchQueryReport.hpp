#pragma once

#include <cstddef>
#include <map>
#include <string>

#include "RAGPipeline.hpp"

namespace mobile_rag {

class BatchQueryReport {
 public:
  void record(const RAGPipeline::QueryTrace& trace);
  bool export_json(const std::string& output_path) const;

 private:
  size_t query_count_ = 0;
  size_t escalation_count_ = 0;
  bool has_runtime_metadata_ = false;
  int promoted_to_hot_total_ = 0;
  int demoted_to_warm_total_ = 0;
  double top_score_sum_ = 0.0;
  double score_margin_sum_ = 0.0;
  double coverage_ratio_sum_ = 0.0;
  double lexical_candidate_count_sum_ = 0.0;
  double hash_candidate_count_sum_ = 0.0;
  double dense_result_count_sum_ = 0.0;
  double query_embedding_ms_sum_ = 0.0;
  double retrieval_ms_sum_ = 0.0;
  double evidence_ms_sum_ = 0.0;
  double state_update_ms_sum_ = 0.0;
  double prompt_build_ms_sum_ = 0.0;
  double generation_ms_sum_ = 0.0;
  double total_ms_sum_ = 0.0;
  double peak_rss_kb_sum_ = 0.0;
  uint64_t max_peak_rss_kb_ = 0;
  std::map<std::string, size_t> initial_graph_counts_;
  std::map<std::string, size_t> final_graph_counts_;
  std::map<std::string, size_t> fallback_reason_counts_;
  RAGPipeline::TraceRuntimeMetadata runtime_metadata_;
};

}  // namespace mobile_rag
