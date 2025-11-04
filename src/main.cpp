#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "RAGPipeline.hpp"

#include "loader/TextFileLoader.hpp"
#include "embedding/MNNEmbedding.hpp"
#include "vector_db/FaissDB.hpp"
#include "llm/LlamaCppModel.hpp"

int main(int argc, char** argv) {
  using namespace mobile_rag;

  auto loader = std::make_shared<TextFileLoader>();
  auto embedder = std::make_shared<MNNEmbedding>();
  auto db = std::make_shared<FaissDB>();
  auto llm = std::make_shared<LlamaCppModel>();

  RAGPipeline pipeline(loader, embedder, db, llm);

  if (argc < 2) {
    std::cerr << "Usage:\n"
              << "  mobile_rag --build <file_path>\n"
              << "  mobile_rag --query \"your question\"\n";
    return 1;
  }

  std::string cmd = argv[1];
  if (cmd == "--build") {
    if (argc < 3) {
      std::cerr << "Missing <file_path> for --build" << '\n';
      return 1;
    }
    std::string file_path = argv[2];
    pipeline.build_index_from_file(file_path);
    std::cout << "Index built from: " << file_path << '\n';
    return 0;
  } else if (cmd == "--query") {
    if (argc < 3) {
      std::cerr << "Missing query string for --query" << '\n';
      return 1;
    }
    // Join remaining args as the query string to allow spaces without quotes
    std::string query;
    for (int i = 2; i < argc; ++i) {
      if (i > 2) query += ' ';
      query += argv[i];
    }
    std::string answer = pipeline.answer_query(query);
    std::cout << answer << '\n';
    return 0;
  } else {
    std::cerr << "Unknown command: " << cmd << '\n';
    return 1;
  }
}


