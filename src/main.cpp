#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "RAGPipeline.hpp"
#include "cli/CommandLineArgs.hpp"

#include "loader/TextFileLoader.hpp"
#include "embedding/MNNEmbedding.hpp"
#include "vector_Index/FaissIndex.hpp"
#include "llm/LLMFactory.hpp"

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
              << "  LLM Model: " << config.llm_model_path << '\n'
              << "  Vector DB: " << config.vector_db_path << '\n'
              << "  Top-K: " << config.top_k << '\n'
              << "  Threads: " << config.num_threads << '\n';
  }

  // Initialize components
  auto loader = std::make_shared<TextFileLoader>();
  auto embedder = std::make_shared<MNNEmbedding>();
  auto index = std::make_shared<FaissIndex>();
  auto llm = create_llm();

  // Load LLM model with custom path
  if (!llm->load_model(config.llm_model_path)) {
    std::cerr << "[ERROR] Failed to load LLM model from: " << config.llm_model_path << '\n';
    return 1;
  }

  if (config.verbose) {
    std::cout << "[INFO] LLM model loaded successfully\n";
  }

  RAGPipeline pipeline(loader, embedder, index, llm);

  // Execute command
  if (config.command == CommandLineArgs::Command::BUILD) {
    if (config.verbose) {
      std::cout << "[INFO] Building index from: " << config.input_file << '\n';
    }
    pipeline.build_index_from_file(config.input_file);
    std::cout << "✓ Index built from: " << config.input_file << '\n';
    return 0;
  } else if (config.command == CommandLineArgs::Command::QUERY) {
    if (config.verbose) {
      std::cout << "[INFO] Processing query: " << config.query << '\n';
    }
    std::string answer = pipeline.answer_query(config.query);
    std::cout << answer << '\n';
    return 0;
  } else if (config.command == CommandLineArgs::Command::INTERACTIVE) {
    std::cout << "╔════════════════════════════════════════════════════════════╗\n"
              << "║         NativeRAG - Interactive Mode                       ║\n"
              << "╚════════════════════════════════════════════════════════════╝\n\n"
              << "Type 'help' for commands, 'exit' to quit\n\n";

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


