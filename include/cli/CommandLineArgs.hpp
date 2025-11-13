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
    // Command
    Command command = Command::HELP;
    
    // Model paths
    std::string llm_model_path = "../models/Qwen3-0.6B/config.json";
    std::string embedding_model_path = "";  // Will use default if empty
    
    // Vector DB
    std::string vector_db_path = "./vector_db.index";
    
    // Input/Output
    std::string input_file = "";
    std::string query = "";
    std::string output_file = "";
    
    // Performance
    int num_threads = 4;
    int top_k = 5;  // Number of documents to retrieve
    
    // Flags
    bool verbose = false;
    bool use_gpu = false;
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

