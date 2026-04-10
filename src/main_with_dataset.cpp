#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <iomanip>

#include "RAGPipelineWithDataset.hpp"
#include "cli/CommandLineArgs.hpp"

#include "loader/TextFileLoader.hpp"
#include "embedding/MNNEmbedding.hpp"
#include "vector_Index/FaissIndex.hpp"
#include "vector_db/SqliteVectorDB.hpp"
#include "llm/LLMFactory.hpp"
#include "dataset/TrivialQADataset.hpp"

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

}  // namespace

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
    RAGPipelineWithDataset pipeline(loader, embedder, index, nullptr, sqlite_db,
                                    config.top_k, config.chunk_size,
                                    config.chunk_overlap);
    // ========== 离线阶段 (Offline/Indexing Phase) ==========
    if (config.verbose) {
      std::cout << "[INFO] === OFFLINE PHASE: Building Index ===\n"
                << "[INFO] Index path: " << config.index_path << '\n';
    }

    if (config.data_source == CommandLineArgs::Config::DataSource::DATASET) {
      // Load dataset
      auto dataset = std::make_shared<TrivialQADataset>();
      if (!dataset->load(config.input_file)) {
        std::cerr << "[ERROR] Failed to load dataset from: " << config.input_file << '\n';
        return 1;
      }

      if (config.verbose) {
        std::cout << "[INFO] Dataset loaded with " << dataset->size() << " samples\n";
      }

      // Step 1-3: Build index from dataset
      pipeline.build_index_from_dataset(dataset, true);
      std::cout << "✓ Index built from dataset: " << config.input_file << '\n';
    } else {
      if (config.verbose) {
        std::cout << "[INFO] Building index from text source: " << config.input_file << '\n';
      }
      // Step 1-3: Build index from text file
      pipeline.build_index_from_file(config.input_file);
      std::cout << "✓ Index built from: " << config.input_file << '\n';
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
    return 0;

  } else if (config.command == CommandLineArgs::Command::QUERY) {
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

    // Create pipeline without loader for query phase
    RAGPipelineWithDataset pipeline(nullptr, embedder, index, llm, sqlite_db,
                                    config.top_k, config.chunk_size,
                                    config.chunk_overlap);

    if (config.verbose) {
      std::cout << "[INFO] === ONLINE PHASE: Query Processing ===\n"
                << "[INFO] Query: " << config.query << '\n'
                << "[INFO] Index path: " << config.index_path << '\n';
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

    // Step 4-7: Query embedding, search, context retrieval, and LLM inference
    if (config.verbose) {
      std::cout << "[INFO] Processing query: " << config.query << '\n';
    }
    std::string answer = pipeline.answer_query(config.query);
    std::cout << answer << '\n';
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
    RAGPipelineWithDataset pipeline(nullptr, embedder, index, llm, sqlite_db,
                                    config.top_k, config.chunk_size,
                                    config.chunk_overlap);

    std::cout << "╔════════════════════════════════════════════════════════════╗\n"
              << "║    NativeRAG with Dataset - Interactive Mode              ║\n"
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
    return 0;
  }

  return 1;
}
