#include "cli/CommandLineArgs.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

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

  if (!parse_options()) {
    return false;
  }

  if (!validate_config()) {
    return false;
  }

  return true;
}

CommandLineArgs::Command CommandLineArgs::parse_command(const std::string& cmd) {
  if (cmd == "--help" || cmd == "-h") return Command::HELP;
  if (cmd == "--backend-info") return Command::BACKEND_INFO;
  if (cmd == "--build") return Command::BUILD;
  if (cmd == "--query") return Command::QUERY;
  if (cmd == "--interactive" || cmd == "-i") return Command::INTERACTIVE;
  return Command::UNKNOWN;
}

bool CommandLineArgs::parse_options() {
  try {
    for (int i = 2; i < argc_; ++i) {
      std::string arg = argv_[i];

      if (arg == "--llm-model" && i + 1 < argc_) {
        config_.llm_model_path = argv_[++i];
      } else if (arg == "--embedding-model" && i + 1 < argc_) {
        config_.embedding_model_path = argv_[++i];
      } else if (arg == "--reranker-model" && i + 1 < argc_) {
        config_.reranker_model_path = argv_[++i];
      } else if ((arg == "--sqlite-db" || arg == "--vector-db") &&
                 i + 1 < argc_) {
        config_.sqlite_db_path = argv_[++i];
      } else if (arg == "--faiss-type" && i + 1 < argc_) {
        config_.faiss_index_type = argv_[++i];
      } else if (arg == "--index-path" && i + 1 < argc_) {
        config_.index_path = argv_[++i];
      } else if (arg == "--text-path" && i + 1 < argc_) {
        config_.text_path = argv_[++i];
      } else if (arg == "--top-k" && i + 1 < argc_) {
        config_.top_k = std::stoi(argv_[++i]);
      } else if (arg == "--rerank-candidates" && i + 1 < argc_) {
        config_.rerank_candidates = std::stoi(argv_[++i]);
      } else if (arg == "--threads" && i + 1 < argc_) {
        config_.num_threads = std::stoi(argv_[++i]);
      } else if (arg == "--max-tokens" && i + 1 < argc_) {
        config_.max_tokens = std::stoi(argv_[++i]);
      } else if (arg == "--verbose" || arg == "-v") {
        config_.verbose = true;
      } else if (arg == "--retrieve-only") {
        config_.retrieve_only = true;
      } else if (arg == "--no-save-index") {
        config_.save_index = false;
      } else if (arg == "--no-load-index") {
        config_.load_index = false;
      } else if (!arg.empty() && arg[0] != '-') {
        if (config_.command == Command::BUILD && config_.text_path.empty()) {
          config_.text_path = arg;
        } else if (config_.command == Command::QUERY) {
          if (!config_.query.empty()) config_.query += ' ';
          config_.query += arg;
        } else {
          std::cerr << "Error: unexpected positional argument: " << arg << '\n';
          return false;
        }
      } else {
        std::cerr << "Error: unknown or incomplete option: " << arg << '\n';
        return false;
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: invalid numeric option: " << error.what() << '\n';
    return false;
  }

  return true;
}

bool CommandLineArgs::validate_config() {
  if (config_.command == Command::BUILD) {
    if (config_.text_path.empty()) {
      std::cerr << "Error: --build requires --text-path or a positional path\n";
      return false;
    }
    if (!std::filesystem::exists(config_.text_path)) {
      std::cerr << "Error: Text path not found: " << config_.text_path << '\n';
      return false;
    }

    if (config_.embedding_model_path.empty()) {
      std::cerr << "Error: --embedding-model is required for --build\n";
      return false;
    }
    if (!std::filesystem::exists(config_.embedding_model_path)) {
      std::cerr << "Error: Embedding model path not found: "
                << config_.embedding_model_path << '\n';
      return false;
    }
  } else if (config_.command == Command::QUERY) {
    if (config_.query.empty()) {
      std::cerr << "Error: --query requires a query string\n";
      return false;
    }

    if (!config_.retrieve_only && config_.llm_model_path.empty()) {
      std::cerr << "Error: --llm-model is required for --query\n";
      return false;
    }
    if (!config_.retrieve_only &&
        !std::filesystem::exists(config_.llm_model_path)) {
      std::cerr << "Error: LLM model path not found: " << config_.llm_model_path << '\n';
      return false;
    }

    if (config_.embedding_model_path.empty()) {
      std::cerr << "Error: --embedding-model is required for --query\n";
      return false;
    }
    if (!std::filesystem::exists(config_.embedding_model_path)) {
      std::cerr << "Error: Embedding model path not found: "
                << config_.embedding_model_path << '\n';
      return false;
    }
  } else if (config_.command == Command::INTERACTIVE) {
    if (config_.llm_model_path.empty()) {
      std::cerr << "Error: --llm-model is required for --interactive\n";
      return false;
    }
    if (!std::filesystem::exists(config_.llm_model_path)) {
      std::cerr << "Error: LLM model path not found: " << config_.llm_model_path << '\n';
      return false;
    }

    if (config_.embedding_model_path.empty()) {
      std::cerr << "Error: --embedding-model is required for --interactive\n";
      return false;
    }
    if (!std::filesystem::exists(config_.embedding_model_path)) {
      std::cerr << "Error: Embedding model path not found: "
                << config_.embedding_model_path << '\n';
      return false;
    }
  } else if (config_.command == Command::BACKEND_INFO) {
    return true;
  }

  if (config_.faiss_index_type.empty()) {
    std::cerr << "Error: --faiss-type cannot be empty\n";
    return false;
  }

  if (config_.top_k <= 0) {
    std::cerr << "Error: --top-k must be positive\n";
    return false;
  }

  if (config_.rerank_candidates <= 0) {
    std::cerr << "Error: --rerank-candidates must be positive\n";
    return false;
  }
  if (!config_.reranker_model_path.empty()) {
    if (!std::filesystem::exists(config_.reranker_model_path)) {
      std::cerr << "Error: Reranker model path not found: "
                << config_.reranker_model_path << '\n';
      return false;
    }
    if (config_.rerank_candidates < config_.top_k) {
      std::cerr << "Error: --rerank-candidates must be greater than or equal "
                   "to --top-k\n";
      return false;
    }
  }

  if (config_.num_threads <= 0) {
    std::cerr << "Error: --threads must be positive\n";
    return false;
  }
  if (config_.max_tokens <= 0) {
    std::cerr << "Error: --max-tokens must be positive\n";
    return false;
  }

  return true;
}

void CommandLineArgs::print_usage() const {
  std::cout << "Usage: mobile_rag_cli <command> [options] [arguments]\n"
            << "Commands:\n"
            << "  --backend-info        Initialize and print the compiled llama.cpp device\n"
            << "  --build <path>        Build an index from a .txt file or directory\n"
            << "  --query <question>    Query the RAG system\n"
            << "  --interactive, -i     Interactive mode\n"
            << "  --help, -h            Show this help message\n";
}

void CommandLineArgs::print_help() const {
  std::cout << "\n╔════════════════════════════════════════════════════════════╗\n"
            << "║         NativeRAG - Two-Stage Pipeline                     ║\n"
            << "║    (Offline Indexing + Online Query)                       ║\n"
            << "╚════════════════════════════════════════════════════════════╝\n\n"
            << "USAGE:\n"
            << "  mobile_rag_cli <command> [options] [arguments]\n\n"
            << "COMMANDS:\n"
            << "  --backend-info        Print the compile-time accelerator and selected device\n"
            << "  --build <path>        Build an index from a .txt file or directory\n"
            << "  --query <question>    Query the RAG system with a question (Online Phase)\n"
            << "  --interactive, -i     Start interactive mode (Online Phase)\n"
            << "  --help, -h            Show this help message\n\n"
            << "REQUIRED OPTIONS BY COMMAND:\n"
            << "  BUILD (Offline Phase):\n"
            << "    --embedding-model <path>     Path to an embedding GGUF (REQUIRED)\n"
            << "    --text-path <path>           Text file/directory (REQUIRED)\n\n"
            << "  QUERY/INTERACTIVE (Online Phase):\n"
            << "    --llm-model <path>           Generation GGUF (except --retrieve-only)\n"
            << "    --embedding-model <path>     Embedding GGUF (REQUIRED)\n\n"
            << "OPTIONAL OPTIONS:\n"
            << "  --reranker-model <path>       Reranker GGUF; enables second-stage reranking\n"
            << "  --rerank-candidates <num>     Faiss candidates scored by reranker (default: 20)\n"
            << "  --sqlite-db <path>           Path to sqlite vector/text store (default: ./vector_store.sqlite3)\n"
            << "  --faiss-type <desc>          Faiss factory description (default: Flat)\n"
            << "  --index-path <path>          Path to save/load Faiss index (default: ./faiss_index.bin)\n"
            << "  --top-k <num>                Number of documents to retrieve (default: 5)\n"
            << "  --threads <num>              Number of threads (default: 4)\n"
            << "  --max-tokens <num>           Maximum generated tokens (default: 256)\n"
            << "  --verbose, -v                Enable verbose output\n"
            << "  --retrieve-only              Run embedding and retrieval without loading an LLM\n"
            << "  --no-save-index              Don't save index after building\n"
            << "  --no-load-index              Don't load existing index\n\n"
            << "EXAMPLES:\n"
            << "  # Stage 1: Build index from a text file (only needs embedding model)\n"
            << "  mobile_rag_cli --build --text-path documents \\\n"
            << "             --embedding-model ./models/embedding.gguf\n\n"
            << "  # Stage 2: Query (needs both embedding and LLM models)\n"
            << "  mobile_rag_cli --query \"What is AI?\" \\\n"
            << "             --llm-model ./models/llm.gguf \\\n"
            << "             --embedding-model ./models/embedding.gguf \\\n"
            << "             --reranker-model ./models/reranker.gguf \\\n"
            << "             --rerank-candidates 20\n\n"
            << "  # Stage 2: Interactive mode (needs both embedding and LLM models)\n"
            << "  mobile_rag_cli --interactive --verbose --top-k 10 \\\n"
            << "             --llm-model ./models/llm.gguf \\\n"
            << "             --embedding-model ./models/embedding.gguf\n\n";
}

}  // namespace mobile_rag
