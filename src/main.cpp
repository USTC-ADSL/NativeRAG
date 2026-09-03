#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "RAGPipeline.hpp"
#include "cli/CommandLineArgs.hpp"

#include "loader/TextFileLoader.hpp"
#include "embedding/EmbeddingFactory.hpp"
#include "vector_Index/FaissIndex.hpp"
#include "vector_db/SqliteVectorDB.hpp"
#include "llm/LLMFactory.hpp"
#include "reranker/RerankerFactory.hpp"

#include "llama/LlamaRuntime.hpp"

int main(int argc, char** argv) {
  using namespace mobile_rag;

  // Parse command line arguments
  CommandLineArgs args(argc, argv);
  if (!args.parse()) {
    return 1;
  }

  const auto& config = args.get_config();

  if (config.command == CommandLineArgs::Command::BACKEND_INFO) {
    if (!LlamaRuntime::acquire()) {
      return 1;
    }
    LlamaRuntime::release();
    return 0;
  }

  if (config.verbose) {
    std::cout << "[INFO] Configuration:\n"
              << "  LLM Model: " << config.llm_model_path << '\n'
              << "  Embedding Model: " << config.embedding_model_path << '\n'
              << "  Reranker Model: " << config.reranker_model_path << '\n'
              << "  SQLite DB: " << config.sqlite_db_path << '\n'
              << "  Faiss Type: " << config.faiss_index_type << '\n'
              << "  Text Path: " << config.text_path << '\n'
              << "  Top-K: " << config.top_k << '\n'
              << "  Rerank Candidates: " << config.rerank_candidates << '\n'
              << "  Threads: " << config.num_threads << '\n'
              << "  Max Tokens: " << config.max_tokens << '\n';
  }

  // Initialize SQLite DB for persisting id->text mappings
  auto sqlite_db = std::make_shared<SqliteVectorDB>(config.sqlite_db_path);

  // Execute command
  if (config.command == CommandLineArgs::Command::BUILD) {
    // ========== 离线阶段 (Offline/Indexing Phase) ==========
    // Only need: loader, embedder, index
    // No need for: LLM

    auto loader = std::make_shared<TextFileLoader>();
    auto embedder = create_embedding(config.num_threads);
    auto index = std::make_shared<FaissIndex>(config.faiss_index_type,
                                              faiss::METRIC_INNER_PRODUCT);

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
    RAGPipeline pipeline(loader, embedder, index, nullptr, sqlite_db,
                         config.top_k);
    // ========== 离线阶段 (Offline/Indexing Phase) ==========
    if (config.verbose) {
      std::cout << "[INFO] === OFFLINE PHASE: Building Index ===\n"
                << "[INFO] Input path: " << config.text_path << '\n'
                << "[INFO] Index path: " << config.index_path << '\n';
    }

    // Step 1-3: Load documents, embed, and build index
    if (!pipeline.build_index_from_file(config.text_path)) {
      std::cerr << "[ERROR] Failed to build index from input\n";
      return 1;
    }
    std::cout << "✓ Index built from: " << config.text_path << '\n';

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

    auto embedder = create_embedding(config.num_threads);
    auto index = std::make_shared<FaissIndex>(config.faiss_index_type,
                                              faiss::METRIC_INNER_PRODUCT);
    std::shared_ptr<IReranker> reranker;

    // Load embedding model
    if (!embedder->load_model(config.embedding_model_path)) {
      std::cerr << "[ERROR] Failed to load embedding model from: "
                << config.embedding_model_path << '\n';
      return 1;
    }

    if (config.verbose) {
      std::cout << "[INFO] Embedding model loaded successfully\n";
    }

    if (!config.reranker_model_path.empty()) {
      reranker = create_reranker(config.num_threads);
      if (!reranker->load_model(config.reranker_model_path)) {
        std::cerr << "[ERROR] Failed to load reranker model from: "
                  << config.reranker_model_path << '\n';
        return 1;
      }
      if (config.verbose) {
        std::cout << "[INFO] Reranker model loaded successfully\n";
      }
    }

    if (config.verbose) {
      std::cout << "[INFO] === ONLINE PHASE: Query Processing ===\n"
                << "[INFO] Query: " << config.query << '\n'
                << "[INFO] Index path: " << config.index_path << '\n';
    }

    std::vector<std::string> contexts;
    {
      RAGPipeline retrieval_pipeline(nullptr, embedder, index, nullptr,
                                     sqlite_db, config.top_k, reranker,
                                     config.rerank_candidates);

      if (config.load_index) {
        if (config.verbose) {
          std::cout << "[INFO] Loading index from: " << config.index_path
                    << '\n';
        }
        if (!retrieval_pipeline.load_index(config.index_path)) {
          std::cerr << "[ERROR] Failed to load index\n";
          return 1;
        }
        std::cout << "✓ Index loaded from: " << config.index_path << '\n';
      }

      if (config.verbose) {
        std::cout << "[INFO] Processing query: " << config.query << '\n';
      }
      contexts = retrieval_pipeline.retrieve_contexts(config.query);
    }

    if (contexts.empty()) {
      std::cerr << "[ERROR] Query did not retrieve any context\n";
      return 1;
    }
    if (config.retrieve_only) {
      std::cout << "[RETRIEVE-ONLY] Retrieved " << contexts.size()
                << " context(s)\n";
      return 0;
    }

    reranker.reset();
    embedder.reset();

    auto llm = create_llm(config.num_threads, config.max_tokens);
    if (!llm || !llm->load_model(config.llm_model_path)) {
      std::cerr << "[ERROR] Failed to load LLM model from: "
                << config.llm_model_path << '\n';
      return 1;
    }
    if (config.verbose) {
      std::cout << "[INFO] LLM model loaded successfully\n";
    }

    const std::string prompt = llm->build_prompt(config.query, contexts);
    std::string answer = llm->generate(prompt);
    if (answer.empty()) {
      std::cerr << "[ERROR] Query did not produce an answer\n";
      return 1;
    }
    std::cout << "[ANSWER] " << answer << '\n';
    return 0;
  } else if (config.command == CommandLineArgs::Command::INTERACTIVE) {
    // ========== 查询阶段 (Online/Query Phase) ==========
    // Only need: embedder, index, llm
    // No need for: loader

    auto embedder = create_embedding(config.num_threads);
    auto index = std::make_shared<FaissIndex>(config.faiss_index_type,
                                              faiss::METRIC_INNER_PRODUCT);
    auto llm = create_llm(config.num_threads, config.max_tokens);
    std::shared_ptr<IReranker> reranker;

    // Load embedding model
    if (!embedder->load_model(config.embedding_model_path)) {
      std::cerr << "[ERROR] Failed to load embedding model from: "
                << config.embedding_model_path << '\n';
      return 1;
    }

    if (config.verbose) {
      std::cout << "[INFO] Embedding model loaded successfully\n";
    }

    if (!config.reranker_model_path.empty()) {
      reranker = create_reranker(config.num_threads);
      if (!reranker->load_model(config.reranker_model_path)) {
        std::cerr << "[ERROR] Failed to load reranker model from: "
                  << config.reranker_model_path << '\n';
        return 1;
      }
      if (config.verbose) {
        std::cout << "[INFO] Reranker model loaded successfully\n";
      }
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
    RAGPipeline pipeline(nullptr, embedder, index, llm, sqlite_db,
                         config.top_k, reranker,
                         config.rerank_candidates);

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
