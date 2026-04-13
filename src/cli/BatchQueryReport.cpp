#include "cli/BatchQueryReport.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

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

}  // namespace

namespace mobile_rag {

void BatchQueryReport::record(const RAGPipeline::QueryTrace& trace) {
  ++query_count_;
  if (trace.escalated) {
    ++escalation_count_;
  }

  promoted_to_hot_total_ += trace.promoted_to_hot;
  demoted_to_warm_total_ += trace.demoted_to_warm;
  top_score_sum_ += trace.evidence.top_score;
  score_margin_sum_ += trace.evidence.score_margin;
  coverage_ratio_sum_ += trace.evidence.coverage_ratio;
  lexical_candidate_count_sum_ += static_cast<double>(trace.lexical_candidate_count);
  hash_candidate_count_sum_ += static_cast<double>(trace.hash_candidate_count);
  dense_result_count_sum_ += static_cast<double>(trace.dense_result_count);
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
       << "  \"escalation_count\": " << escalation_count_ << ",\n";
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
       << "  \"averages\": {\n"
       << "    \"top_score\": " << (top_score_sum_ / query_count) << ",\n"
       << "    \"score_margin\": " << (score_margin_sum_ / query_count) << ",\n"
       << "    \"coverage_ratio\": " << (coverage_ratio_sum_ / query_count) << ",\n"
       << "    \"lexical_candidate_count\": " << (lexical_candidate_count_sum_ / query_count) << ",\n"
       << "    \"hash_candidate_count\": " << (hash_candidate_count_sum_ / query_count) << ",\n"
       << "    \"dense_result_count\": " << (dense_result_count_sum_ / query_count) << "\n"
       << "  }\n"
       << "}\n";

  out << json.str();
  return static_cast<bool>(out);
}

}  // namespace mobile_rag
