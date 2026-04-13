#include "cli/BatchQueryReport.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string json_escape(const std::string& input) {
  std::ostringstream escaped;
  for (const unsigned char ch : input) {
    switch (ch) {
      case '\\':
        escaped << "\\\\";
        break;
      case '"':
        escaped << "\\\"";
        break;
      case '\b':
        escaped << "\\b";
        break;
      case '\f':
        escaped << "\\f";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        if (ch < 0x20) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<int>(ch) << std::dec << std::setfill(' ');
        } else {
          escaped << static_cast<char>(ch);
        }
        break;
    }
  }
  return escaped.str();
}

template <typename MapT>
void write_count_map(std::ostringstream& out,
                     const std::string& label,
                     const MapT& counts) {
  out << "  \"" << label << "\": {\n";
  size_t index = 0;
  for (const auto& [key, value] : counts) {
    out << "    \"" << json_escape(key) << "\": " << value;
    if (index + 1 < counts.size()) {
      out << ",";
    }
    out << "\n";
    ++index;
  }
  out << "  }";
}

double interpolate_percentile(const std::vector<double>& samples, double percentile_rank) {
  if (samples.empty()) {
    return 0.0;
  }
  if (samples.size() == 1) {
    return samples.front();
  }

  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const double position =
      (percentile_rank / 100.0) * static_cast<double>(sorted.size() - 1);
  const auto lower_index = static_cast<size_t>(std::floor(position));
  const auto upper_index = static_cast<size_t>(std::ceil(position));
  if (lower_index == upper_index) {
    return sorted[lower_index];
  }

  const double fraction = position - static_cast<double>(lower_index);
  return sorted[lower_index] +
         (sorted[upper_index] - sorted[lower_index]) * fraction;
}

}  // namespace

namespace mobile_rag {

void BatchQueryReport::record(const RAGPipeline::QueryTrace& trace) {
  ++query_count_;
  if (trace.escalated) {
    ++escalation_count_;
  }
  if (trace.state_aware_dense_enabled) {
    ++state_aware_dense_query_count_;
  }
  if (!has_runtime_metadata_) {
    runtime_metadata_ = trace.runtime;
    has_runtime_metadata_ = true;
  }

  promoted_to_hot_total_ += trace.promoted_to_hot;
  demoted_to_warm_total_ += trace.demoted_to_warm;
  top_score_sum_ += trace.evidence.top_score;
  score_margin_sum_ += trace.evidence.score_margin;
  coverage_ratio_sum_ += trace.evidence.coverage_ratio;
  lexical_candidate_count_sum_ += static_cast<double>(trace.lexical_candidate_count);
  hash_candidate_count_sum_ += static_cast<double>(trace.hash_candidate_count);
  state_filtered_candidate_count_sum_ +=
      static_cast<double>(trace.state_filtered_candidate_count);
  dense_result_count_sum_ += static_cast<double>(trace.dense_result_count);
  query_embedding_ms_sum_ += trace.timing.query_embedding_ms;
  retrieval_ms_sum_ += trace.timing.retrieval_ms;
  evidence_ms_sum_ += trace.timing.evidence_ms;
  state_update_ms_sum_ += trace.timing.state_update_ms;
  prompt_build_ms_sum_ += trace.timing.prompt_build_ms;
  generation_ms_sum_ += trace.timing.generation_ms;
  total_ms_sum_ += trace.timing.total_ms;
  peak_rss_kb_sum_ += static_cast<double>(trace.system.peak_rss_kb);
  query_embedding_ms_samples_.push_back(trace.timing.query_embedding_ms);
  retrieval_ms_samples_.push_back(trace.timing.retrieval_ms);
  evidence_ms_samples_.push_back(trace.timing.evidence_ms);
  state_update_ms_samples_.push_back(trace.timing.state_update_ms);
  prompt_build_ms_samples_.push_back(trace.timing.prompt_build_ms);
  generation_ms_samples_.push_back(trace.timing.generation_ms);
  total_ms_samples_.push_back(trace.timing.total_ms);
  peak_rss_kb_samples_.push_back(static_cast<double>(trace.system.peak_rss_kb));
  max_peak_rss_kb_ = std::max(max_peak_rss_kb_, trace.system.peak_rss_kb);
  ++initial_graph_counts_[trace.initial_graph];
  ++final_graph_counts_[trace.final_graph];
  ++fallback_reason_counts_[trace.fallback_reason];
}

bool BatchQueryReport::export_json(const std::string& output_path) const {
  if (output_path.empty() || query_count_ == 0) {
    return false;
  }

  std::ofstream out(output_path, std::ios::trunc);
  if (!out) {
    return false;
  }

  const double query_count = static_cast<double>(query_count_);

  std::ostringstream json;
  json << "{\n"
       << "  \"query_count\": " << query_count_ << ",\n"
       << "  \"escalation_count\": " << escalation_count_ << ",\n"
       << "  \"state_aware_dense_query_count\": " << state_aware_dense_query_count_ << ",\n"
       << "  \"runtime\": {\n"
       << "    \"llm_backend\": \"" << json_escape(runtime_metadata_.llm_backend) << "\",\n"
       << "    \"embedding_backend\": \"" << json_escape(runtime_metadata_.embedding_backend) << "\",\n"
       << "    \"llm_model_path\": \"" << json_escape(runtime_metadata_.llm_model_path) << "\",\n"
       << "    \"embedding_model_path\": \"" << json_escape(runtime_metadata_.embedding_model_path) << "\",\n"
       << "    \"sqlite_db_path\": \"" << json_escape(runtime_metadata_.sqlite_db_path) << "\",\n"
       << "    \"index_path\": \"" << json_escape(runtime_metadata_.index_path) << "\",\n"
       << "    \"query_source\": \"" << json_escape(runtime_metadata_.query_source) << "\",\n"
       << "    \"num_threads\": " << runtime_metadata_.num_threads << ",\n"
       << "    \"max_new_tokens\": " << runtime_metadata_.max_new_tokens << ",\n"
       << "    \"sqlite_db_size_bytes\": " << runtime_metadata_.sqlite_db_size_bytes << ",\n"
       << "    \"index_size_bytes\": " << runtime_metadata_.index_size_bytes << "\n"
       << "  },\n";
  write_count_map(json, "initial_graph_counts", initial_graph_counts_);
  json << ",\n";
  write_count_map(json, "final_graph_counts", final_graph_counts_);
  json << ",\n";
  write_count_map(json, "fallback_reason_counts", fallback_reason_counts_);
  json << ",\n"
       << "  \"totals\": {\n"
       << "    \"promoted_to_hot\": " << promoted_to_hot_total_ << ",\n"
        << "    \"demoted_to_warm\": " << demoted_to_warm_total_ << "\n"
       << "  },\n"
       << "  \"maxima\": {\n"
       << "    \"max_peak_rss_kb\": " << max_peak_rss_kb_ << "\n"
       << "  },\n"
       << "  \"percentiles\": {\n"
       << "    \"p50\": {\n"
       << "      \"query_embedding_ms\": "
       << interpolate_percentile(query_embedding_ms_samples_, 50.0) << ",\n"
       << "      \"retrieval_ms\": "
       << interpolate_percentile(retrieval_ms_samples_, 50.0) << ",\n"
       << "      \"evidence_ms\": "
       << interpolate_percentile(evidence_ms_samples_, 50.0) << ",\n"
       << "      \"state_update_ms\": "
       << interpolate_percentile(state_update_ms_samples_, 50.0) << ",\n"
       << "      \"prompt_build_ms\": "
       << interpolate_percentile(prompt_build_ms_samples_, 50.0) << ",\n"
       << "      \"generation_ms\": "
       << interpolate_percentile(generation_ms_samples_, 50.0) << ",\n"
       << "      \"total_ms\": "
       << interpolate_percentile(total_ms_samples_, 50.0) << ",\n"
       << "      \"peak_rss_kb\": "
       << interpolate_percentile(peak_rss_kb_samples_, 50.0) << "\n"
       << "    },\n"
       << "    \"p95\": {\n"
       << "      \"query_embedding_ms\": "
       << interpolate_percentile(query_embedding_ms_samples_, 95.0) << ",\n"
       << "      \"retrieval_ms\": "
       << interpolate_percentile(retrieval_ms_samples_, 95.0) << ",\n"
       << "      \"evidence_ms\": "
       << interpolate_percentile(evidence_ms_samples_, 95.0) << ",\n"
       << "      \"state_update_ms\": "
       << interpolate_percentile(state_update_ms_samples_, 95.0) << ",\n"
       << "      \"prompt_build_ms\": "
       << interpolate_percentile(prompt_build_ms_samples_, 95.0) << ",\n"
       << "      \"generation_ms\": "
       << interpolate_percentile(generation_ms_samples_, 95.0) << ",\n"
       << "      \"total_ms\": "
       << interpolate_percentile(total_ms_samples_, 95.0) << ",\n"
       << "      \"peak_rss_kb\": "
       << interpolate_percentile(peak_rss_kb_samples_, 95.0) << "\n"
       << "    }\n"
       << "  },\n"
       << "  \"averages\": {\n"
       << "    \"top_score\": " << (top_score_sum_ / query_count) << ",\n"
       << "    \"score_margin\": " << (score_margin_sum_ / query_count) << ",\n"
       << "    \"coverage_ratio\": " << (coverage_ratio_sum_ / query_count) << ",\n"
       << "    \"lexical_candidate_count\": " << (lexical_candidate_count_sum_ / query_count) << ",\n"
       << "    \"hash_candidate_count\": " << (hash_candidate_count_sum_ / query_count) << ",\n"
       << "    \"state_filtered_candidate_count\": "
       << (state_filtered_candidate_count_sum_ / query_count) << ",\n"
       << "    \"dense_result_count\": " << (dense_result_count_sum_ / query_count) << ",\n"
       << "    \"query_embedding_ms\": " << (query_embedding_ms_sum_ / query_count) << ",\n"
       << "    \"retrieval_ms\": " << (retrieval_ms_sum_ / query_count) << ",\n"
       << "    \"evidence_ms\": " << (evidence_ms_sum_ / query_count) << ",\n"
       << "    \"state_update_ms\": " << (state_update_ms_sum_ / query_count) << ",\n"
       << "    \"prompt_build_ms\": " << (prompt_build_ms_sum_ / query_count) << ",\n"
       << "    \"generation_ms\": " << (generation_ms_sum_ / query_count) << ",\n"
       << "    \"total_ms\": " << (total_ms_sum_ / query_count) << ",\n"
       << "    \"peak_rss_kb\": " << (peak_rss_kb_sum_ / query_count) << "\n"
       << "  }\n"
       << "}\n";

  out << json.str();
  return static_cast<bool>(out);
}

}  // namespace mobile_rag
