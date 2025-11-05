#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "dataset/TrivialQADataset.hpp"
#include "dataset/VectorDataGenerator.hpp"
#include "vector_Index/FaissIndex.hpp"
#include "tests/TestBase.hpp"

using namespace mobile_rag;
using namespace mobile_rag::testing;

/**
 * Group C (VectorDB) 测试范式
 * 
 * 测试内容：
 * 1. 加载数据集
 * 2. 生成向量
 * 3. 添加向量到索引
 * 4. 向量搜索
 * 5. 索引持久化
 * 6. 索引加载
 */
class VectorDBTest : public TestBase {
 private:
  std::shared_ptr<IVectorIndex> index_;
  std::vector<std::vector<float>> vectors_;
  std::vector<int64_t> vector_ids_;

 public:
  bool test_load_dataset() {
    print_test_info("Test 1: Load TrivialQA Dataset");
    
    bool success = load_dataset("dataset/data/val00-100.json");
    
    if (success) {
      auto samples = get_samples();
      std::cout << "Loaded " << samples.size() << " samples\n";
      if (!samples.empty()) {
        std::cout << "First sample query: " << samples[0].query.substr(0, 50)
                  << "...\n";
        std::cout << "First sample documents: " << samples[0].documents.size()
                  << '\n';
      }
    }
    
    print_result("Load dataset", success);
    return success;
  }

  bool test_generate_vectors() {
    print_test_info("Test 2: Generate Vectors from Dataset");
    
    if (!dataset_) {
      print_result("Dataset exists", false);
      return false;
    }

    // 生成向量
    bool success = VectorDataGenerator::generate_from_dataset(
        dataset_, "/tmp/test_vectors_integrated", 384);

    if (success) {
      // 加载生成的向量
      vectors_ = VectorDataGenerator::load_vectors("/tmp/test_vectors_integrated/vectors.bin");
      std::cout << "Generated " << vectors_.size() << " vectors\n";
      if (!vectors_.empty()) {
        std::cout << "Vector dimension: " << vectors_[0].size() << '\n';
      }
    }

    print_result("Generate vectors", success);
    return success;
  }

  bool test_add_vectors_to_index() {
    print_test_info("Test 3: Add Vectors to Index");
    
    if (vectors_.empty()) {
      print_result("Vectors exist", false);
      return false;
    }

    index_ = std::make_shared<FaissIndex>();
    
    // 生成 ID
    vector_ids_.clear();
    for (int64_t i = 0; i < static_cast<int64_t>(vectors_.size()); ++i) {
      vector_ids_.push_back(i);
    }

    bool success = index_->add_vectors(vectors_, vector_ids_);

    if (success) {
      std::cout << "Added " << vectors_.size() << " vectors to index\n";
    }

    print_result("Add vectors to index", success);
    return success;
  }

  bool test_vector_search() {
    print_test_info("Test 4: Vector Search");
    
    if (!index_ || vectors_.empty()) {
      print_result("Index and vectors exist", false);
      return false;
    }

    // 使用第一个向量作为查询
    const auto& query_vector = vectors_[0];
    int k = 10;

    auto results = index_->search(query_vector, k);

    bool success = !results.empty();

    if (success) {
      std::cout << "Top-" << k << " search results:\n";
      for (size_t i = 0; i < results.size() && i < 5; ++i) {
        std::cout << "  " << (i + 1) << ". ID=" << results[i].first
                  << ", Similarity=" << results[i].second << '\n';
      }
    }

    print_result("Vector search", success);
    return success;
  }

  bool test_save_index() {
    print_test_info("Test 5: Save Index");
    
    if (!index_) {
      print_result("Index exists", false);
      return false;
    }

    bool success = index_->save_index("/tmp/test_index.faiss");

    if (success) {
      std::cout << "Index saved to /tmp/test_index.faiss\n";
    }

    print_result("Save index", success);
    return success;
  }

  bool test_load_index() {
    print_test_info("Test 6: Load Index");
    
    auto new_index = std::make_shared<FaissIndex>();
    bool success = new_index->load_index("/tmp/test_index.faiss");

    if (success) {
      std::cout << "Index loaded from /tmp/test_index.faiss\n";
      
      // 验证加载的索引是否可用
      if (!vectors_.empty()) {
        auto results = new_index->search(vectors_[0], 5);
        std::cout << "Search on loaded index returned " << results.size()
                  << " results\n";
      }
    }

    print_result("Load index", success);
    return success;
  }

  bool test_batch_search() {
    print_test_info("Test 7: Batch Search");
    
    if (!index_ || vectors_.size() < 5) {
      print_result("Index and vectors exist", false);
      return false;
    }

    int success_count = 0;
    int test_count = 5;

    for (int i = 0; i < test_count; ++i) {
      auto results = index_->search(vectors_[i], 10);
      if (!results.empty()) {
        success_count++;
      }
    }

    bool success = success_count == test_count;

    std::cout << "Batch search: " << success_count << "/" << test_count
              << " successful\n";

    print_result("Batch search", success);
    return success;
  }

  void run_all_tests() {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n"
              << "║         Group C (VectorDB) Unit Tests                      ║\n"
              << "╚════════════════════════════════════════════════════════════╝\n";

    int passed = 0;
    int total = 7;

    if (test_load_dataset()) passed++;
    if (test_generate_vectors()) passed++;
    if (test_add_vectors_to_index()) passed++;
    if (test_vector_search()) passed++;
    if (test_save_index()) passed++;
    if (test_load_index()) passed++;
    if (test_batch_search()) passed++;

    std::cout << "\n========================================\n"
              << "Results: " << passed << "/" << total << " tests passed\n"
              << "========================================\n";

    assert(passed == total && "Some tests failed!");
  }
};

int main() {
  try {
    VectorDBTest test;
    test.run_all_tests();
    std::cout << "\n✓ All VectorDB tests passed!\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << '\n';
    return 1;
  }
}

