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

  for (int i = 2; i < argc_; ++i) {
    const std::string arg = argv_[i];
    if (arg == "--help" || arg == "-h") {
      print_help();
      return false;
    }
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
  std::vector<std::string> query_tokens;

  for (int i = 2; i < argc_; ++i) {
    std::string arg = argv_[i];

    if (arg == "--llm-model" && i + 1 < argc_) {
      config_.llm_model_path = argv_[++i];
    } else if (arg == "--embedding-model" && i + 1 < argc_) {
      config_.embedding_model_path = argv_[++i];
    } else if ((arg == "--sqlite-db" || arg == "--vector-db") && i + 1 < argc_) {
      config_.sqlite_db_path = argv_[++i];
    } else if (arg == "--faiss-type" && i + 1 < argc_) {
      config_.faiss_index_type = argv_[++i];
    } else if (arg == "--index-path" && i + 1 < argc_) {
      config_.index_path = argv_[++i];
    } else if (arg == "--dataset-path" && i + 1 < argc_) {
      config_.dataset_path = argv_[++i];
      config_.data_source = Config::DataSource::DATASET;
      config_.input_file = config_.dataset_path;
    } else if (arg == "--text-path" && i + 1 < argc_) {
      config_.text_path = argv_[++i];
      config_.data_source = Config::DataSource::TEXT;
      config_.input_file = config_.text_path;
    } else if (arg == "--data-source" && i + 1 < argc_) {
      std::string mode = argv_[++i];
      if (mode == "dataset") {
        config_.data_source = Config::DataSource::DATASET;
      } else if (mode == "txt" || mode == "text") {
        config_.data_source = Config::DataSource::TEXT;
      } else {
        std::cerr << "Error: --data-source expects 'dataset' or 'txt'\n";
        return false;
      }
    } else if (arg == "--output" && i + 1 < argc_) {
      config_.output_file = argv_[++i];
    } else if (arg == "--query-file" && i + 1 < argc_) {
      config_.query_file_path = argv_[++i];
    } else if (arg == "--state-snapshot-in" && i + 1 < argc_) {
      config_.state_snapshot_in_path = argv_[++i];
    } else if (arg == "--state-snapshot-out" && i + 1 < argc_) {
      config_.state_snapshot_out_path = argv_[++i];
    } else if (arg == "--query-trace-out" && i + 1 < argc_) {
      config_.query_trace_out_path = argv_[++i];
    } else if (arg == "--query-trace-jsonl-out" && i + 1 < argc_) {
      config_.query_trace_jsonl_out_path = argv_[++i];
    } else if (arg == "--query-summary-csv-out" && i + 1 < argc_) {
      config_.query_summary_csv_out_path = argv_[++i];
    } else if (arg == "--top-k" && i + 1 < argc_) {
      config_.top_k = std::stoi(argv_[++i]);
    } else if (arg == "--threads" && i + 1 < argc_) {
      config_.num_threads = std::stoi(argv_[++i]);
    } else if (arg == "--max-new-tokens" && i + 1 < argc_) {
      config_.max_new_tokens = std::stoi(argv_[++i]);
    } else if (arg == "--lexical-prefilter") {
      config_.lexical_prefilter = true;
    } else if (arg == "--lexical-candidates" && i + 1 < argc_) {
      config_.lexical_candidate_limit = std::stoi(argv_[++i]);
    } else if (arg == "--semantic-hash-prefilter") {
      config_.semantic_hash_prefilter = true;
    } else if (arg == "--semantic-hash-candidates" && i + 1 < argc_) {
      config_.semantic_hash_candidate_limit = std::stoi(argv_[++i]);
    } else if (arg == "--semantic-hash-max-distance" && i + 1 < argc_) {
      config_.semantic_hash_max_distance = std::stoi(argv_[++i]);
    } else if (arg == "--adaptive-graph") {
      config_.adaptive_graph = true;
    } else if (arg == "--chunk-size" && i + 1 < argc_) {
      config_.chunk_size = static_cast<size_t>(std::stoull(argv_[++i]));
    } else if (arg == "--chunk-overlap" && i + 1 < argc_) {
      config_.chunk_overlap = static_cast<size_t>(std::stoull(argv_[++i]));
    } else if (arg == "--verbose" || arg == "-v") {
      config_.verbose = true;
    } else if (arg == "--gpu") {
      config_.use_gpu = true;
    } else if (arg == "--no-save-index") {
      config_.save_index = false;
    } else if (arg == "--no-load-index") {
      config_.load_index = false;
    } else if (!arg.empty() && arg[0] != '-') {
      // Positional argument
      if (config_.command == Command::BUILD && config_.input_file.empty()) {
        config_.input_file = arg;
        if (config_.data_source == Config::DataSource::DATASET) {
          config_.dataset_path = arg;
        } else {
          config_.text_path = arg;
        }
      } else if (config_.command == Command::QUERY) {
        query_tokens.push_back(arg);
      }
    }
  }

  if (config_.command == Command::QUERY && !query_tokens.empty()) {
    for (const auto& token : query_tokens) {
      if (!config_.query.empty()) {
        config_.query += ' ';
      }
      config_.query += token;
    }
  }

  if (config_.command == Command::BUILD) {
    if (config_.data_source == Config::DataSource::DATASET) {
      if (config_.dataset_path.empty()) {
        config_.dataset_path = config_.input_file;
      }
      if (config_.input_file.empty()) {
        config_.input_file = config_.dataset_path;
      }
    } else {
      if (config_.text_path.empty()) {
        config_.text_path = config_.input_file;
      }
      if (config_.input_file.empty()) {
        config_.input_file = config_.text_path;
      }
    }
  }

  return true;
}

bool CommandLineArgs::validate_config() {
  if (config_.command == Command::BUILD) {
    // BUILD phase: only needs embedding model, not LLM
    if (config_.data_source == Config::DataSource::DATASET) {
      if (config_.dataset_path.empty()) {
        std::cerr << "Error: dataset mode requires --dataset-path\n";
        return false;
      }
      if (!std::filesystem::exists(config_.dataset_path)) {
        std::cerr << "Error: Dataset path not found: " << config_.dataset_path << '\n';
        return false;
      }
      if (config_.input_file.empty()) {
        config_.input_file = config_.dataset_path;
      }
    } else {
      if (config_.text_path.empty()) {
        std::cerr << "Error: text mode requires --text-path or positional file\n";
        return false;
      }
      if (!std::filesystem::exists(config_.text_path)) {
        std::cerr << "Error: Text path not found: " << config_.text_path << '\n';
        return false;
      }
      if (config_.input_file.empty()) {
        config_.input_file = config_.text_path;
      }
    }

    // Validate embedding model for BUILD phase
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
    // QUERY phase: needs both embedding model and LLM
    const bool has_inline_query = !config_.query.empty();
    const bool has_query_file = !config_.query_file_path.empty();
    if (has_inline_query == has_query_file) {
      std::cerr << "Error: --query requires exactly one inline query or --query-file\n";
      return false;
    }
    if (has_query_file) {
      if (!std::filesystem::exists(config_.query_file_path)) {
        std::cerr << "Error: query file path not found: " << config_.query_file_path << '\n';
        return false;
      }
      if (!config_.query_trace_out_path.empty()) {
        std::cerr << "Error: --query-trace-out is not supported with --query-file\n";
        return false;
      }
    }

    // Validate LLM model for QUERY phase
    if (config_.llm_model_path.empty()) {
      std::cerr << "Error: --llm-model is required for --query\n";
      return false;
    }
    if (!std::filesystem::exists(config_.llm_model_path)) {
      std::cerr << "Error: LLM model path not found: " << config_.llm_model_path << '\n';
      return false;
    }

    // Validate embedding model for QUERY phase
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
    // INTERACTIVE phase: needs both embedding model and LLM
    // Validate LLM model for INTERACTIVE phase
    if (config_.llm_model_path.empty()) {
      std::cerr << "Error: --llm-model is required for --interactive\n";
      return false;
    }
    if (!std::filesystem::exists(config_.llm_model_path)) {
      std::cerr << "Error: LLM model path not found: " << config_.llm_model_path << '\n';
      return false;
    }

    // Validate embedding model for INTERACTIVE phase
    if (config_.embedding_model_path.empty()) {
      std::cerr << "Error: --embedding-model is required for --interactive\n";
      return false;
    }
    if (!std::filesystem::exists(config_.embedding_model_path)) {
      std::cerr << "Error: Embedding model path not found: "
                << config_.embedding_model_path << '\n';
      return false;
    }
  }

  if (config_.faiss_index_type.empty()) {
    std::cerr << "Error: --faiss-type cannot be empty\n";
    return false;
  }

  if (!config_.state_snapshot_in_path.empty() &&
      !std::filesystem::exists(config_.state_snapshot_in_path)) {
    std::cerr << "Error: state snapshot path not found: "
              << config_.state_snapshot_in_path << '\n';
    return false;
  }

  if (!config_.query_file_path.empty() &&
      config_.command != Command::QUERY) {
    std::cerr << "Error: --query-file is only supported for --query\n";
    return false;
  }

  if (!config_.query_trace_out_path.empty() &&
      config_.command != Command::QUERY) {
    std::cerr << "Error: --query-trace-out is only supported for --query\n";
    return false;
  }

  if (!config_.query_trace_jsonl_out_path.empty() &&
      config_.command != Command::QUERY) {
    std::cerr << "Error: --query-trace-jsonl-out is only supported for --query\n";
    return false;
  }

  if (!config_.query_summary_csv_out_path.empty() &&
      config_.command != Command::QUERY) {
    std::cerr << "Error: --query-summary-csv-out is only supported for --query\n";
    return false;
  }

  if (config_.top_k <= 0) {
    std::cerr << "Error: --top-k must be positive\n";
    return false;
  }

  if (config_.num_threads <= 0) {
    std::cerr << "Error: --threads must be positive\n";
    return false;
  }

  if (config_.max_new_tokens <= 0) {
    std::cerr << "Error: --max-new-tokens must be positive\n";
    return false;
  }

  if (config_.lexical_candidate_limit <= 0) {
    std::cerr << "Error: --lexical-candidates must be positive\n";
    return false;
  }

  if (config_.semantic_hash_candidate_limit <= 0) {
    std::cerr << "Error: --semantic-hash-candidates must be positive\n";
    return false;
  }

  if (config_.semantic_hash_max_distance < -1) {
    std::cerr << "Error: --semantic-hash-max-distance must be -1 or greater\n";
    return false;
  }

  if (config_.chunk_size == 0) {
    std::cerr << "Error: --chunk-size must be positive\n";
    return false;
  }

  if (config_.chunk_overlap >= config_.chunk_size) {
    std::cerr << "Error: --chunk-overlap must be smaller than --chunk-size\n";
    return false;
  }

  return true;
}

void CommandLineArgs::print_usage() const {
  std::cout << "Usage: mobile_rag <command> [options] [arguments]\n"
            << "Commands:\n"
            << "  --build <path>        Build index from txt file/dir or dataset file\n"
            << "  --query <question>    Query the RAG system\n"
            << "  --query --query-file <path>\n"
            << "                       Run query mode for each line in a file\n"
            << "  --interactive, -i     Interactive mode\n"
            << "  --help, -h            Show this help message\n";
}

void CommandLineArgs::print_help() const {
#if defined(LLM_BACKEND_LLAMA)
  constexpr const char* llm_backend_name = "LlamaCpp";
  constexpr const char* llm_model_hint =
      "GGUF file or a directory containing GGUF weights";
#elif defined(LLM_BACKEND_MNN)
  constexpr const char* llm_backend_name = "MNN";
  constexpr const char* llm_model_hint = "MNN config.json";
#elif defined(LLM_BACKEND_MLLM)
  constexpr const char* llm_backend_name = "MLLM";
  constexpr const char* llm_model_hint = "MLLM model path";
#else
  constexpr const char* llm_backend_name = "Unknown";
  constexpr const char* llm_model_hint = "model path accepted by the compiled backend";
#endif

  std::cout << "\n╔════════════════════════════════════════════════════════════╗\n"
            << "║         NativeRAG - Two-Stage Pipeline                     ║\n"
            << "║    (Offline Indexing + Online Query)                       ║\n"
            << "╚════════════════════════════════════════════════════════════╝\n\n"
            << "USAGE:\n"
            << "  mobile_rag <command> [options] [arguments]\n\n"
            << "BUILD NOTES:\n"
            << "  Current LLM backend: " << llm_backend_name << '\n'
            << "  --llm-model expects: " << llm_model_hint << '\n'
            << "  Embedding backend: MNN (embedding model should point to its config)\n\n"
            << "COMMANDS:\n"
            << "  --build <file>        Build vector index from document file (Offline Phase)\n"
            << "  --query <question>    Query the RAG system with a question (Online Phase)\n"
            << "  --query --query-file <path>\n"
            << "                        Run query mode once per non-empty line in a file\n"
            << "  --interactive, -i     Start interactive mode (Online Phase)\n"
            << "  --help, -h            Show this help message\n\n"
            << "REQUIRED OPTIONS BY COMMAND:\n"
            << "  BUILD (Offline Phase):\n"
            << "    --embedding-model <path>     Path to embedding model config (REQUIRED)\n"
            << "    --text-path <path>           Text file/directory (for txt mode)\n"
            << "    --dataset-path <path>        Dataset json path (for dataset mode)\n\n"
            << "  QUERY/INTERACTIVE (Online Phase):\n"
            << "    --llm-model <path>           Path to LLM model config (REQUIRED)\n"
            << "    --embedding-model <path>     Path to embedding model config (REQUIRED)\n\n"
            << "OPTIONAL OPTIONS:\n"
            << "  --sqlite-db <path>           Path to sqlite vector/text store (default: ./vector_store.sqlite3)\n"
            << "  --faiss-type <desc>          Faiss factory description (default: Flat)\n"
            << "  --index-path <path>          Path to save/load Faiss index (default: ./faiss_index.bin)\n"
            << "  --data-source <txt|dataset>  Choose loader mode (default: txt)\n"
            << "  --output <file>              Output file for results\n"
            << "  --state-snapshot-in <path>   Restore chunk-state snapshot before execution\n"
            << "  --state-snapshot-out <path>  Export chunk-state snapshot after execution\n"
            << "  --query-file <path>         Load one query per line for batch query mode\n"
            << "  --query-trace-out <path>     Export the final query trace as JSON (query mode only)\n"
            << "  --query-trace-jsonl-out <path>\n"
            << "                               Append the final query trace as one JSONL row (query mode only)\n"
            << "  --query-summary-csv-out <path>\n"
            << "                               Append a flat query summary row to CSV (query mode only)\n"
            << "  --top-k <num>                Number of documents to retrieve (default: 5)\n"
            << "  --threads <num>              Number of threads (default: 4)\n"
            << "  --max-new-tokens <num>       Maximum generated tokens (default: 256)\n"
            << "  --lexical-prefilter          Enable SQLite lexical shortlist before dense rerank\n"
            << "  --lexical-candidates <num>   Lexical shortlist size (default: 16)\n"
            << "  --semantic-hash-prefilter    Enable SQLite semantic-hash shortlist before dense rerank\n"
            << "  --semantic-hash-candidates <num>\n"
            << "                               Semantic-hash shortlist size (default: 32)\n"
            << "  --semantic-hash-max-distance <num>\n"
            << "                               Max Hamming distance for shortlist, -1 disables the cap (default: -1)\n"
            << "  --adaptive-graph            Enable heuristic graph selection and evidence-based upgrades\n"
            << "  --chunk-size <num>           Chunk size for offline indexing (default: 1000)\n"
            << "  --chunk-overlap <num>        Chunk overlap for offline indexing (default: 200)\n"
            << "  --verbose, -v                Enable verbose output\n"
            << "  --gpu                        Use GPU acceleration (if available)\n"
            << "  --no-save-index              Don't save index after building\n"
            << "  --no-load-index              Don't load existing index\n\n"
            << "EXAMPLES:\n"
            << "  # Stage 1: Build index from a text file (only needs embedding model)\n"
            << "  mobile_rag --build --data-source txt --text-path documents.txt \\\n"
            << "             --embedding-model ./models/emb/config.json\n\n"
            << "  # Stage 1: Build index from a dataset json (only needs embedding model)\n"
            << "  mobile_rag --build --data-source dataset --dataset-path dataset/data.json \\\n"
            << "             --embedding-model ./models/emb/config.json \\\n"
            << "             --faiss-type IVF512,PQ64\n\n"
            << "  # Stage 2: Query (needs both embedding and LLM models)\n"
            << "  mobile_rag --query \"What is AI?\" \\\n"
            << "             --llm-model ./models/llm/model.gguf \\\n"
            << "             --embedding-model ./models/emb/config.json\n\n"
            << "  # Stage 2: Batch query file with append-friendly exports\n"
            << "  mobile_rag --query --query-file queries.txt \\\n"
            << "             --llm-model ./models/llm/model.gguf \\\n"
            << "             --embedding-model ./models/emb/config.json \\\n"
            << "             --query-trace-jsonl-out /tmp/query-trace.jsonl \\\n"
            << "             --query-summary-csv-out /tmp/query-summary.csv\n\n"
            << "  # Stage 2: Interactive mode (needs both embedding and LLM models)\n"
            << "  mobile_rag --interactive --verbose --top-k 10 \\\n"
            << "             --llm-model ./models/llm/model.gguf \\\n"
            << "             --embedding-model ./models/emb/config.json\n\n";
}

}  // namespace mobile_rag
