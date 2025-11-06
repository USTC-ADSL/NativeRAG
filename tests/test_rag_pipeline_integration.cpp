#include <cassert>
#include <iostream>
#include <memory>

#include "RAGPipelineWithDataset.hpp"
#include "dataset/TrivialQADataset.hpp"
#include "dataset/VectorDataGenerator.hpp"
#include "embedding/MNNEmbedding.hpp"
#include "llm/LLMFactory.hpp"
#include "loader/TextFileLoader.hpp"
#include "vector_Index/FaissIndex.hpp"
#include "TestBase.hpp"

using namespace mobile_rag;
using namespace mobile_rag::testing;

/**
 * RAG Pipeline 集成测试
 * 
 * 展示如何使用 RAGPipelineWithDataset 进行端到端的 RAG 测试
 */
class RAGPipelineIntegrationTest : public TestBase {
 public:
  bool test_pipeline_with_dataset() {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n"
              << "║    RAG Pipeline Integration Test                           ║\n"
              << "╚════════════════════════════════════════════════════════════╝\n";

    // 加载数据集
    if (!load_dataset("dataset/data/val00-100.json")) {
      std::cerr << "Failed to load dataset\n";
      return false;
    }

    // 创建 RAG Pipeline
    auto loader = std::make_shared<TextFileLoader>();
    auto embedder = std::make_shared<MNNEmbedding>();
    auto index = std::make_shared<FaissIndex>();
    auto llm = create_llm();

    auto pipeline = std::make_shared<RAGPipelineWithDataset>(
        loader, embedder, index, llm);

    // 从数据集构建索引
    std::cout << "\n[Step 1] Building index from dataset...\n";
    pipeline->build_index_from_dataset(dataset_, true);

    // 获取查询
    std::cout << "\n[Step 2] Getting queries from dataset...\n";
    auto queries = pipeline->get_queries_from_dataset(dataset_);
    std::cout << "Retrieved " << queries.size() << " queries\n";

    if (queries.size() > 0) {
      std::cout << "First query: " << queries[0].substr(0, 50) << "...\n";
    }

    // 获取答案
    std::cout << "\n[Step 3] Getting answers from dataset...\n";
    auto answers = pipeline->get_answers_from_dataset(dataset_);
    std::cout << "Retrieved " << answers.size() << " answer sets\n";

    if (answers.size() > 0) {
      std::cout << "First answer set has " << answers[0].size() << " answers\n";
      if (!answers[0].empty()) {
        std::cout << "First answer: " << answers[0][0] << '\n';
      }
    }

    // 测试查询
    std::cout << "\n[Step 4] Testing query answering...\n";
    if (!queries.empty()) {
      std::string answer = pipeline->answer_query(queries[0]);
      std::cout << "Query: " << queries[0].substr(0, 50) << "...\n";
      std::cout << "Answer: " << answer.substr(0, 100) << "...\n";
    }

    return true;
  }

  bool test_group_a_workflow() {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n"
              << "║    Group A (Embedding) Workflow Test                       ║\n"
              << "╚════════════════════════════════════════════════════════════╝\n";

    if (!load_dataset("dataset/data/val00-100.json")) {
      return false;
    }

    auto embedder = std::make_shared<MNNEmbedding>();
    embedder->load_model("");

    // 获取前 3 个样本
    auto samples = get_samples(3);

    std::cout << "\n[Embedding Workflow]\n";
    for (size_t i = 0; i < samples.size(); ++i) {
      const auto& sample = samples[i];
      
      std::cout << "\nSample " << (i + 1) << ":\n";
      std::cout << "  Query: " << sample.query.substr(0, 40) << "...\n";
      
      // 嵌入查询
      auto query_embedding = embedder->embed_query(sample.query);
      std::cout << "  Query embedding dimension: " << query_embedding.size()
                << '\n';

      // 嵌入文档
      if (!sample.documents.empty()) {
        auto doc_embeddings = embedder->embed_documents(sample.documents);
        std::cout << "  Document embeddings: " << doc_embeddings.size() << '\n';
      }
    }

    return true;
  }

  bool test_group_b_workflow() {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n"
              << "║    Group B (LLM) Workflow Test                             ║\n"
              << "╚════════════════════════════════════════════════════════════╝\n";

    if (!load_dataset("dataset/data/val00-100.json")) {
      return false;
    }

    auto llm = create_llm();

    // 获取前 2 个样本
    auto samples = get_samples(2);

    std::cout << "\n[LLM Workflow]\n";
    for (size_t i = 0; i < samples.size(); ++i) {
      const auto& sample = samples[i];
      
      std::cout << "\nSample " << (i + 1) << ":\n";
      std::cout << "  Query: " << sample.query.substr(0, 40) << "...\n";
      std::cout << "  Expected answers: " << sample.answers.size() << '\n';

      // 构建 prompt
      std::string prompt = llm->build_prompt(sample.query, sample.documents);
      std::cout << "  Prompt length: " << prompt.length() << '\n';

      // 生成答案
      std::string answer = llm->generate(prompt);
      std::cout << "  Generated answer: " << answer.substr(0, 50) << "...\n";
    }

    return true;
  }

  bool test_group_c_workflow() {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n"
              << "║    Group C (VectorDB) Workflow Test                        ║\n"
              << "╚════════════════════════════════════════════════════════════╝\n";

    if (!load_dataset("dataset/data/val00-100.json")) {
      return false;
    }

    auto index = std::make_shared<FaissIndex>();

    // 生成向量
    std::cout << "\n[Generating vectors from dataset...]\n";
    bool success = VectorDataGenerator::generate_from_dataset(
        dataset_, "/tmp/test_vectors_workflow", 384);

    if (!success) {
      std::cerr << "Failed to generate vectors\n";
      return false;
    }

    // 加载向量
    auto vectors = VectorDataGenerator::load_vectors(
        "/tmp/test_vectors_workflow/vectors.bin");
    std::cout << "Loaded " << vectors.size() << " vectors\n";

    // 添加到索引
    std::vector<int64_t> ids;
    for (int64_t i = 0; i < static_cast<int64_t>(vectors.size()); ++i) {
      ids.push_back(i);
    }

    index->add_vectors(vectors, ids);
    std::cout << "Added vectors to index\n";

    // 搜索
    if (!vectors.empty()) {
      auto results = index->search(vectors[0], 10);
      std::cout << "Search returned " << results.size() << " results\n";
    }

    return true;
  }

  void run_all_tests() {
    int passed = 0;
    int total = 4;

    if (test_pipeline_with_dataset()) passed++;
    if (test_group_a_workflow()) passed++;
    if (test_group_b_workflow()) passed++;
    if (test_group_c_workflow()) passed++;

    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n"
              << "║    Integration Test Results                                ║\n"
              << "║    " << passed << "/" << total << " tests passed"
              << "                                    ║\n"
              << "╚════════════════════════════════════════════════════════════╝\n";

    assert(passed == total && "Some tests failed!");
  }
};

int main() {
  try {
    RAGPipelineIntegrationTest test;
    test.run_all_tests();
    std::cout << "\n✓ All integration tests passed!\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << '\n';
    return 1;
  }
}

