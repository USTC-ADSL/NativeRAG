#include "cli/CommandLineArgs.hpp"

namespace mobile_rag {

CommandLineArgs::CommandLineArgs(int argc, char** argv)
    : argc_(argc), argv_(argv) {}

bool CommandLineArgs::parse() {
  if (argc_ < 2) {
    print_usage();
    return false;
  }

  config_.command = parse_command(argv_[1]);
  
  if (config_.command == Command::UNKNOWN) {
    std::cerr << "Unknown command: " << argv_[1] << '\n';
    print_usage();
    return false;
  }

  if (config_.command == Command::HELP) {
    print_help();
    return false;  // Help doesn't proceed with execution
  }

  // Parse options and positional arguments
  if (!parse_options()) {
    return false;
  }

  // Validate configuration
  if (!validate_config()) {
    return false;
  }

  return true;
}

CommandLineArgs::Command CommandLineArgs::parse_command(const std::string& cmd) {
  if (cmd == "--help" || cmd == "-h") return Command::HELP;
  if (cmd == "--build") return Command::BUILD;
  if (cmd == "--query") return Command::QUERY;
  if (cmd == "--interactive" || cmd == "-i") return Command::INTERACTIVE;
  return Command::UNKNOWN;
}

bool CommandLineArgs::parse_options() {
  for (int i = 2; i < argc_; ++i) {
    std::string arg = argv_[i];

    if (arg == "--llm-model" && i + 1 < argc_) {
      config_.llm_model_path = argv_[++i];
    } else if (arg == "--embedding-model" && i + 1 < argc_) {
      config_.embedding_model_path = argv_[++i];
    } else if (arg == "--vector-db" && i + 1 < argc_) {
      config_.vector_db_path = argv_[++i];
    } else if (arg == "--output" && i + 1 < argc_) {
      config_.output_file = argv_[++i];
    } else if (arg == "--top-k" && i + 1 < argc_) {
      config_.top_k = std::stoi(argv_[++i]);
    } else if (arg == "--threads" && i + 1 < argc_) {
      config_.num_threads = std::stoi(argv_[++i]);
    } else if (arg == "--verbose" || arg == "-v") {
      config_.verbose = true;
    } else if (arg == "--gpu") {
      config_.use_gpu = true;
    } else if (arg == "--no-save-index") {
      config_.save_index = false;
    } else if (arg == "--no-load-index") {
      config_.load_index = false;
    } else if (arg[0] != '-') {
      // Positional argument
      if (config_.command == Command::BUILD && config_.input_file.empty()) {
        config_.input_file = arg;
      } else if (config_.command == Command::QUERY && config_.query.empty()) {
        config_.query = arg;
      }
    }
  }

  // For QUERY command, join remaining args as query string
  if (config_.command == Command::QUERY && config_.query.empty()) {
    for (int i = 2; i < argc_; ++i) {
      if (argv_[i][0] != '-') {
        if (!config_.query.empty()) config_.query += ' ';
        config_.query += argv_[i];
      }
    }
  }

  return true;
}

bool CommandLineArgs::validate_config() {
  if (config_.command == Command::BUILD) {
    if (config_.input_file.empty()) {
      std::cerr << "Error: --build requires an input file path\n";
      return false;
    }
    if (!std::filesystem::exists(config_.input_file)) {
      std::cerr << "Error: Input file not found: " << config_.input_file << '\n';
      return false;
    }
  } else if (config_.command == Command::QUERY) {
    if (config_.query.empty()) {
      std::cerr << "Error: --query requires a query string\n";
      return false;
    }
  }

  if (config_.top_k <= 0) {
    std::cerr << "Error: --top-k must be positive\n";
    return false;
  }

  if (config_.num_threads <= 0) {
    std::cerr << "Error: --threads must be positive\n";
    return false;
  }

  return true;
}

void CommandLineArgs::print_usage() const {
  std::cout << "Usage: mobile_rag <command> [options] [arguments]\n"
            << "Commands:\n"
            << "  --build <file>        Build index from file\n"
            << "  --query <question>    Query the RAG system\n"
            << "  --interactive, -i     Interactive mode\n"
            << "  --help, -h            Show this help message\n";
}

void CommandLineArgs::print_help() const {
  std::cout << "\n╔════════════════════════════════════════════════════════════╗\n"
            << "║         NativeRAG - Command Line Interface                 ║\n"
            << "╚════════════════════════════════════════════════════════════╝\n\n"
            << "USAGE:\n"
            << "  mobile_rag <command> [options] [arguments]\n\n"
            << "COMMANDS:\n"
            << "  --build <file>        Build vector index from document file\n"
            << "  --query <question>    Query the RAG system with a question\n"
            << "  --interactive, -i     Start interactive mode\n"
            << "  --help, -h            Show this help message\n\n"
            << "OPTIONS:\n"
            << "  --llm-model <path>           Path to LLM model config (default: ../models/Qwen3-0.6B/config.json)\n"
            << "  --embedding-model <path>     Path to embedding model\n"
            << "  --vector-db <path>           Path to vector database (default: ./vector_db.index)\n"
            << "  --output <file>              Output file for results\n"
            << "  --top-k <num>                Number of documents to retrieve (default: 5)\n"
            << "  --threads <num>              Number of threads (default: 4)\n"
            << "  --verbose, -v                Enable verbose output\n"
            << "  --gpu                        Use GPU acceleration (if available)\n"
            << "  --no-save-index              Don't save index after building\n"
            << "  --no-load-index              Don't load existing index\n\n"
            << "EXAMPLES:\n"
            << "  # Build index from a text file\n"
            << "  mobile_rag --build documents.txt --vector-db my_index.db\n\n"
            << "  # Query with custom LLM model\n"
            << "  mobile_rag --query \"What is AI?\" --llm-model ./models/my_model/config.json\n\n"
            << "  # Interactive mode with verbose output\n"
            << "  mobile_rag --interactive --verbose --top-k 10\n\n";
}

}  // namespace mobile_rag

