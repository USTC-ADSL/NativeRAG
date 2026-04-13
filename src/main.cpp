#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>

#include "RAGPipeline.hpp"
#include "cli/BatchQueryReport.hpp"
#include "cli/CommandLineArgs.hpp"
#include "cli/QueryFileLoader.hpp"

#include "loader/TextFileLoader.hpp"
#include "embedding/MNNEmbedding.hpp"
#include "vector_Index/FaissIndex.hpp"
#include "vector_db/SqliteVectorDB.hpp"
#include "llm/LLMFactory.hpp"

namespace {

const char* current_llm_backend_name() {
#if defined(LLM_BACKEND_LLAMA)
  return "LlamaCpp";
#elif defined(LLM_BACKEND_MNN)
  return "MNN";
#elif defined(LLM_BACKEND_MLLM)
  return "MLLM";
#else
  return "Unknown";
#endif
}

const char* current_embedding_backend_name() {
  return "MNN";
}

uint64_t file_size_or_zero(const std::string& path) {
  if (path.empty()) {
    return 0;
  }

  std::error_code error_code;
  if (!std::filesystem::exists(path, error_code) ||
      !std::filesystem::is_regular_file(path, error_code)) {
    return 0;
  }

  return static_cast<uint64_t>(std::filesystem::file_size(path, error_code));
}

}  // namespace

// Helper function to suppress MNN startup messages
void suppress_mnn_startup_output() {
  // Redirect stdout to /dev/null temporarily during MNN initialization
  // This suppresses CPU detection messages like "CPU Group: [...]"
  fflush(stdout);
  int saved_stdout = dup(STDOUT_FILENO);
  int devnull = open("/dev/null", O_WRONLY);
  dup2(devnull, STDOUT_FILENO);
  close(devnull);

  // Store the saved stdout for later restoration
  // We'll restore it after the first model load
  static int g_saved_stdout = saved_stdout;
}

void restore_stdout() {
  // This will be called after MNN initialization
  static int g_saved_stdout = -1;
  if (g_saved_stdout != -1) {
    fflush(stdout);
    dup2(g_saved_stdout, STDOUT_FILENO);
    close(g_saved_stdout);
    g_saved_stdout = -1;
  }
}

int main(int argc, char** argv) {
  using namespace mobile_rag;

  // Parse command line arguments
  CommandLineArgs args(argc, argv);
  if (!args.parse()) {
    return 1;
  }

  const auto& config = args.get_config();

  if (config.verbose) {
    std::cout << "[INFO] Configuration:\n"
              << "  LLM Backend: " << current_llm_backend_name() << '\n'
              << "  Embedding Backend: " << current_embedding_backend_name() << '\n'
              << "  LLM Model: " << config.llm_model_path << '\n'
              << "  Embedding Model: " << config.embedding_model_path << '\n'
              << "  SQLite DB: " << config.sqlite_db_path << '\n'
              << "  Faiss Type: " << config.faiss_index_type << '\n'
              << "  Data Source: "
              << (config.data_source == CommandLineArgs::Config::DataSource::DATASET ? "dataset"
                                                                                     : "txt")
              << '\n'
              << "  Top-K: " << config.top_k << '\n'
              << "  Threads: " << config.num_threads << '\n'
              << "  Max New Tokens: " << config.max_new_tokens << '\n'
              << "  Lexical Prefilter: "
              << (config.lexical_prefilter ? "enabled" : "disabled") << '\n'
              << "  Lexical Candidates: "
              << config.lexical_candidate_limit << '\n'
              << "  Semantic Hash Prefilter: "
              << (config.semantic_hash_prefilter ? "enabled" : "disabled") << '\n'
              << "  Semantic Hash Candidates: "
              << config.semantic_hash_candidate_limit << '\n'
              << "  Semantic Hash Max Distance: "
              << config.semantic_hash_max_distance << '\n'
              << "  Adaptive Graph: "
              << (config.adaptive_graph ? "enabled" : "disabled") << '\n'
              << "  State-Aware Dense: "
              << (config.state_aware_dense ? "enabled" : "disabled") << '\n'
              << "  State Snapshot In: "
              << (config.state_snapshot_in_path.empty() ? "(none)"
                                                        : config.state_snapshot_in_path) << '\n'
              << "  State Snapshot Out: "
              << (config.state_snapshot_out_path.empty() ? "(none)"
                                                         : config.state_snapshot_out_path) << '\n'
              << "  Query Trace Out: "
              << (config.query_trace_out_path.empty() ? "(none)"
                                                      : config.query_trace_out_path) << '\n'
              << "  Query Trace JSONL Out: "
              << (config.query_trace_jsonl_out_path.empty() ? "(none)"
                                                            : config.query_trace_jsonl_out_path)
              << '\n'
              << "  Query Summary CSV Out: "
              << (config.query_summary_csv_out_path.empty() ? "(none)"
                                                            : config.query_summary_csv_out_path)
              << '\n'
              << "  Query Batch Report Out: "
              << (config.query_batch_report_out_path.empty() ? "(none)"
                                                            : config.query_batch_report_out_path)
              << '\n'
              << "  Query File: "
              << (config.query_file_path.empty() ? "(none)"
                                                : config.query_file_path)
              << '\n'
              << "  Chunk Size: " << config.chunk_size << '\n'
              << "  Chunk Overlap: " << config.chunk_overlap << '\n';
  }

  // Initialize SQLite DB for persisting id->text mappings
  auto sqlite_db = std::make_shared<SqliteVectorDB>(config.sqlite_db_path);

  // Execute command
  if (config.command == CommandLineArgs::Command::BUILD) {
    // ========== 离线阶段 (Offline/Indexing Phase) ==========
    // Only need: loader, embedder, index
    // No need for: LLM

    auto loader = std::make_shared<TextFileLoader>(config.num_threads, config.chunk_size,
                                                   config.chunk_overlap);
    auto embedder = std::make_shared<MNNEmbedding>();
    auto index = std::make_shared<FaissIndex>(config.faiss_index_type,
                                              faiss::METRIC_INNER_PRODUCT);
    embedder->set_num_threads(config.num_threads);

    // Load embedding model
    if (!embedder->load_model(config.embedding_model_path)) {
      std::cerr << "[ERROR] Failed to load embedding model from: "
                << config.embedding_model_path << '\n';
      return 1;
    }

    if (config.verbose) {
      std::cout << "[INFO] Embedding model loaded successfully\n";
    }

    // Create pipeline without LLM for offline phase
    RAGPipeline pipeline(loader, embedder, index, nullptr, sqlite_db, config.top_k,
                         config.chunk_size, config.chunk_overlap);
    GraphSelector::Config graph_selector_config;
    graph_selector_config.enabled = config.adaptive_graph;
    pipeline.set_graph_selector_config(graph_selector_config);
    pipeline.set_lexical_prefilter(
        {config.lexical_prefilter,
         config.lexical_candidate_limit});
    pipeline.set_semantic_hash_prefilter(
        {config.semantic_hash_prefilter,
         config.semantic_hash_candidate_limit,
         config.semantic_hash_max_distance});
    pipeline.set_state_aware_dense({config.state_aware_dense});
    if (config.data_source == CommandLineArgs::Config::DataSource::DATASET) {
      std::cerr << "[ERROR] Dataset mode is not supported in this binary. "
                   "Use main_with_dataset instead."
                << '\n';
      return 1;
    }

    // ========== 离线阶段 (Offline/Indexing Phase) ==========
    if (config.verbose) {
      std::cout << "[INFO] === OFFLINE PHASE: Building Index ===\n"
                << "[INFO] Input file: " << config.input_file << '\n'
                << "[INFO] Index path: " << config.index_path << '\n';
    }

    // Step 1-3: Load documents, embed, and build index
    pipeline.build_index_from_file(config.input_file);
    std::cout << "✓ Index built from: " << config.input_file << '\n';

    if (!config.state_snapshot_in_path.empty()) {
      if (!sqlite_db->import_chunk_state_snapshot(config.state_snapshot_in_path)) {
        std::cerr << "[ERROR] Failed to import state snapshot: "
                  << config.state_snapshot_in_path << '\n';
        return 1;
      }
      std::cout << "✓ State snapshot imported from: "
                << config.state_snapshot_in_path << '\n';
    }

    // Save index to disk
    if (config.save_index) {
      if (config.verbose) {
        std::cout << "[INFO] Saving index to: " << config.index_path << '\n';
      }
      if (!pipeline.save_index(config.index_path)) {
        std::cerr << "[ERROR] Failed to save index\n";
        return 1;
      }
      std::cout << "✓ Index saved to: " << config.index_path << '\n';
    }

    if (!config.state_snapshot_out_path.empty()) {
      if (!sqlite_db->export_chunk_state_snapshot(config.state_snapshot_out_path)) {
        std::cerr << "[ERROR] Failed to export state snapshot: "
                  << config.state_snapshot_out_path << '\n';
        return 1;
      }
      std::cout << "✓ State snapshot exported to: "
                << config.state_snapshot_out_path << '\n';
    }
    return 0;
  } else if (config.command == CommandLineArgs::Command::QUERY) {
    // ========== 查询阶段 (Online/Query Phase) ==========
    // Only need: embedder, index, llm
    // No need for: loader

    std::vector<std::string> queries;
    if (config.query_file_path.empty()) {
      queries.push_back(config.query);
    } else {
      queries = load_query_file(config.query_file_path);
      if (queries.empty()) {
        std::cerr << "[ERROR] Query file did not contain any runnable queries: "
                  << config.query_file_path << '\n';
        return 1;
      }
    }

    auto embedder = std::make_shared<MNNEmbedding>();
    auto index = std::make_shared<FaissIndex>(config.faiss_index_type,
                                              faiss::METRIC_INNER_PRODUCT);
    auto llm = create_llm();
    embedder->set_num_threads(config.num_threads);
    llm->set_num_threads(config.num_threads);
    llm->set_max_new_tokens(config.max_new_tokens);

    // Load embedding model
    if (!embedder->load_model(config.embedding_model_path)) {
      std::cerr << "[ERROR] Failed to load embedding model from: "
                << config.embedding_model_path << '\n';
      return 1;
    }

    if (config.verbose) {
      std::cout << "[INFO] Embedding model loaded successfully\n";
    }

    // Load LLM model
    if (!llm->load_model(config.llm_model_path)) {
      std::cerr << "[ERROR] Failed to load LLM model from: " << config.llm_model_path << '\n';
      return 1;
    }

    if (config.verbose) {
      std::cout << "[INFO] LLM model loaded successfully\n";
    }

    // Create pipeline without loader for query phase
    RAGPipeline pipeline(nullptr, embedder, index, llm, sqlite_db, config.top_k,
                         config.chunk_size, config.chunk_overlap);
    RAGPipeline::TraceRuntimeMetadata trace_runtime_metadata;
    trace_runtime_metadata.llm_backend = current_llm_backend_name();
    trace_runtime_metadata.embedding_backend = current_embedding_backend_name();
    trace_runtime_metadata.llm_model_path = config.llm_model_path;
    trace_runtime_metadata.embedding_model_path = config.embedding_model_path;
    trace_runtime_metadata.sqlite_db_path = config.sqlite_db_path;
    trace_runtime_metadata.index_path = config.index_path;
    trace_runtime_metadata.query_source =
        config.query_file_path.empty() ? "inline" : "query_file";
    trace_runtime_metadata.num_threads = config.num_threads;
    trace_runtime_metadata.max_new_tokens = config.max_new_tokens;
    trace_runtime_metadata.sqlite_db_size_bytes = file_size_or_zero(config.sqlite_db_path);
    trace_runtime_metadata.index_size_bytes = file_size_or_zero(config.index_path);
    pipeline.set_trace_runtime_metadata(trace_runtime_metadata);
    BatchQueryReport batch_report;
    GraphSelector::Config graph_selector_config;
    graph_selector_config.enabled = config.adaptive_graph;
    pipeline.set_graph_selector_config(graph_selector_config);
    pipeline.set_lexical_prefilter(
        {config.lexical_prefilter,
         config.lexical_candidate_limit});
    pipeline.set_semantic_hash_prefilter(
        {config.semantic_hash_prefilter,
         config.semantic_hash_candidate_limit,
         config.semantic_hash_max_distance});
    pipeline.set_state_aware_dense({config.state_aware_dense});

    if (config.verbose) {
      std::cout << "[INFO] === ONLINE PHASE: Query Processing ===\n";
      if (config.query_file_path.empty()) {
        std::cout << "[INFO] Query: " << config.query << '\n';
      } else {
        std::cout << "[INFO] Query file: " << config.query_file_path << '\n'
                  << "[INFO] Batch query count: " << queries.size() << '\n';
      }
      std::cout << "[INFO] Index path: " << config.index_path << '\n';
    }

    // Load index from disk
    if (config.load_index) {
      if (config.verbose) {
        std::cout << "[INFO] Loading index from: " << config.index_path << '\n';
      }
      if (!pipeline.load_index(config.index_path)) {
        std::cerr << "[ERROR] Failed to load index\n";
        return 1;
      }
      std::cout << "✓ Index loaded from: " << config.index_path << '\n';
    }

    if (!config.state_snapshot_in_path.empty()) {
      if (!sqlite_db->import_chunk_state_snapshot(config.state_snapshot_in_path)) {
        std::cerr << "[ERROR] Failed to import state snapshot: "
                  << config.state_snapshot_in_path << '\n';
        return 1;
      }
      std::cout << "✓ State snapshot imported from: "
                << config.state_snapshot_in_path << '\n';
    }

    auto export_query_artifacts = [&]() -> bool {
      if (!config.query_trace_out_path.empty()) {
        if (!pipeline.export_last_query_trace(config.query_trace_out_path)) {
          std::cerr << "[ERROR] Failed to export query trace: "
                    << config.query_trace_out_path << '\n';
          return false;
        }
        std::cout << "✓ Query trace exported to: "
                  << config.query_trace_out_path << '\n';
      }

      if (!config.query_trace_jsonl_out_path.empty()) {
        if (!pipeline.append_last_query_trace_jsonl(config.query_trace_jsonl_out_path)) {
          std::cerr << "[ERROR] Failed to append query trace JSONL: "
                    << config.query_trace_jsonl_out_path << '\n';
          return false;
        }
        std::cout << "✓ Query trace JSONL appended to: "
                  << config.query_trace_jsonl_out_path << '\n';
      }

      if (!config.query_summary_csv_out_path.empty()) {
        if (!pipeline.append_last_query_trace_summary_csv(config.query_summary_csv_out_path)) {
          std::cerr << "[ERROR] Failed to append query summary CSV: "
                    << config.query_summary_csv_out_path << '\n';
          return false;
        }
        std::cout << "✓ Query summary CSV appended to: "
                  << config.query_summary_csv_out_path << '\n';
      }

      return true;
    };

    // Step 4-7: Query embedding, search, context retrieval, and LLM inference
    for (size_t query_index = 0; query_index < queries.size(); ++query_index) {
      const auto& query = queries[query_index];
      if (config.verbose) {
        if (queries.size() == 1) {
          std::cout << "[INFO] Processing query: " << query << '\n';
        } else {
          std::cout << "[INFO] Processing batch query "
                    << (query_index + 1) << "/" << queries.size()
                    << ": " << query << '\n';
        }
      }

      std::string answer = pipeline.answer_query(query);
      std::cout << answer << '\n';
      batch_report.record(pipeline.last_query_trace());

      if (!export_query_artifacts()) {
        return 1;
      }
    }

    if (!config.query_batch_report_out_path.empty()) {
      if (!batch_report.export_json(config.query_batch_report_out_path)) {
        std::cerr << "[ERROR] Failed to export query batch report: "
                  << config.query_batch_report_out_path << '\n';
        return 1;
      }
      std::cout << "✓ Query batch report exported to: "
                << config.query_batch_report_out_path << '\n';
    }

    if (!config.state_snapshot_out_path.empty()) {
      if (!sqlite_db->export_chunk_state_snapshot(config.state_snapshot_out_path)) {
        std::cerr << "[ERROR] Failed to export state snapshot: "
                  << config.state_snapshot_out_path << '\n';
        return 1;
      }
      std::cout << "✓ State snapshot exported to: "
                << config.state_snapshot_out_path << '\n';
    }
    return 0;
  } else if (config.command == CommandLineArgs::Command::INTERACTIVE) {
    // ========== 查询阶段 (Online/Query Phase) ==========
    // Only need: embedder, index, llm
    // No need for: loader

    auto embedder = std::make_shared<MNNEmbedding>();
    auto index = std::make_shared<FaissIndex>(config.faiss_index_type,
                                              faiss::METRIC_INNER_PRODUCT);
    auto llm = create_llm();
    embedder->set_num_threads(config.num_threads);
    llm->set_num_threads(config.num_threads);
    llm->set_max_new_tokens(config.max_new_tokens);

    // Load embedding model
    if (!embedder->load_model(config.embedding_model_path)) {
      std::cerr << "[ERROR] Failed to load embedding model from: "
                << config.embedding_model_path << '\n';
      return 1;
    }

    if (config.verbose) {
      std::cout << "[INFO] Embedding model loaded successfully\n";
    }

    // Load LLM model
    if (!llm->load_model(config.llm_model_path)) {
      std::cerr << "[ERROR] Failed to load LLM model from: " << config.llm_model_path << '\n';
      return 1;
    }

    if (config.verbose) {
      std::cout << "[INFO] LLM model loaded successfully\n";
    }

    // Create pipeline without loader for interactive phase
    RAGPipeline pipeline(nullptr, embedder, index, llm, sqlite_db, config.top_k,
                         config.chunk_size, config.chunk_overlap);
    GraphSelector::Config graph_selector_config;
    graph_selector_config.enabled = config.adaptive_graph;
    pipeline.set_graph_selector_config(graph_selector_config);
    pipeline.set_lexical_prefilter(
        {config.lexical_prefilter,
         config.lexical_candidate_limit});
    pipeline.set_semantic_hash_prefilter(
        {config.semantic_hash_prefilter,
         config.semantic_hash_candidate_limit,
         config.semantic_hash_max_distance});
    pipeline.set_state_aware_dense({config.state_aware_dense});

    std::cout << "╔════════════════════════════════════════════════════════════╗\n"
              << "║         NativeRAG - Interactive Mode                       ║\n"
              << "╚════════════════════════════════════════════════════════════╝\n\n";

    // Load index from disk
    if (config.load_index) {
      if (config.verbose) {
        std::cout << "[INFO] Loading index from: " << config.index_path << '\n';
      }
      if (!pipeline.load_index(config.index_path)) {
        std::cerr << "[ERROR] Failed to load index from: " << config.index_path << '\n';
        std::cerr << "[ERROR] Please run --build first to create an index\n";
        return 1;
      }
      std::cout << "✓ Index loaded from: " << config.index_path << '\n';
    }

    if (!config.state_snapshot_in_path.empty()) {
      if (!sqlite_db->import_chunk_state_snapshot(config.state_snapshot_in_path)) {
        std::cerr << "[ERROR] Failed to import state snapshot: "
                  << config.state_snapshot_in_path << '\n';
        return 1;
      }
      std::cout << "✓ State snapshot imported from: "
                << config.state_snapshot_in_path << '\n';
    }

    std::cout << "Type 'help' for commands, 'exit' to quit\n\n";

    std::string input;
    while (true) {
      std::cout << "rag> ";
      if (!std::getline(std::cin, input)) break;

      if (input == "exit" || input == "quit") {
        std::cout << "Goodbye!\n";
        break;
      } else if (input == "help") {
        std::cout << "Commands:\n"
                  << "  help              Show this help message\n"
                  << "  exit/quit         Exit interactive mode\n"
                  << "  <question>        Ask a question\n";
      } else if (!input.empty()) {
        std::string answer = pipeline.answer_query(input);
        std::cout << answer << "\n\n";
      }
    }

    if (!config.state_snapshot_out_path.empty()) {
      if (!sqlite_db->export_chunk_state_snapshot(config.state_snapshot_out_path)) {
        std::cerr << "[ERROR] Failed to export state snapshot: "
                  << config.state_snapshot_out_path << '\n';
        return 1;
      }
      std::cout << "✓ State snapshot exported to: "
                << config.state_snapshot_out_path << '\n';
    }
    return 0;
  }

  return 1;
}
