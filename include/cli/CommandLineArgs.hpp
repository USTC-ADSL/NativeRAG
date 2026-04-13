#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <filesystem>

namespace mobile_rag {

/**
 * Command line arguments parser for NativeRAG CLI
 * Supports various configuration options for LLM, Embedding, and Vector DB
 */
class CommandLineArgs {
 public:
  enum class Command {
    HELP,
    BUILD,
    QUERY,
    INTERACTIVE,
    UNKNOWN
  };

  // Configuration structure
  struct Config {
    enum class DataSource {
      TEXT,
      DATASET,
    };

    // Command
    Command command = Command::HELP;

    // Models
    std::string llm_model_path;
    std::string embedding_model_path;

    // Vector DB / index settings
    std::string sqlite_db_path = "./vector_store.sqlite3";
    std::string faiss_index_type = "Flat";
    std::string index_path = "./faiss_index.bin";  // Path to save/load Faiss index

    // Input/Output
    DataSource data_source = DataSource::TEXT;
    std::string text_path;
    std::string dataset_path;
    std::string input_file;
    std::string query;
    std::string output_file;
    std::string state_snapshot_in_path;
    std::string state_snapshot_out_path;
    std::string query_trace_out_path;
    std::string query_summary_csv_out_path;

    // Performance
    int num_threads = 4;
    int top_k = 5;  // Number of documents to retrieve
    int max_new_tokens = 256;
    int lexical_candidate_limit = 16;
    int semantic_hash_candidate_limit = 32;
    int semantic_hash_max_distance = -1;

    // Chunking
    size_t chunk_size = 1000;
    size_t chunk_overlap = 200;

    // Flags
    bool verbose = false;
    bool use_gpu = false;
    bool save_index = true;
    bool load_index = true;
    bool lexical_prefilter = false;
    bool semantic_hash_prefilter = false;
    bool adaptive_graph = false;
  };

  CommandLineArgs(int argc, char** argv);
  
  bool parse();
  
  const Config& get_config() const { return config_; }
  
  void print_usage() const;
  void print_help() const;

 private:
  int argc_;
  char** argv_;
  Config config_;
  
  Command parse_command(const std::string& cmd);
  bool parse_options();
  bool validate_config();
};

}  // namespace mobile_rag
