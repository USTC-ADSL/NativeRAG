#pragma once

#include <string>

namespace mobile_rag {

/**
 * Command line arguments parser for NativeRAG CLI
 * Supports various configuration options for LLM, Embedding, and Vector DB
 */
class CommandLineArgs {
 public:
  enum class Command {
    HELP,
    BACKEND_INFO,
    BUILD,
    QUERY,
    INTERACTIVE,
    UNKNOWN
  };

  struct Config {
    Command command = Command::HELP;

    std::string llm_model_path;
    std::string embedding_model_path;
    std::string reranker_model_path;

    std::string sqlite_db_path = "./vector_store.sqlite3";
    std::string faiss_index_type = "Flat";
    std::string index_path = "./faiss_index.bin";

    std::string text_path;
    std::string query;

    int num_threads = 4;
    int top_k = 5;
    int rerank_candidates = 20;
    int max_tokens = 256;

    bool verbose = false;
    bool retrieve_only = false;
    bool save_index = true;
    bool load_index = true;
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
