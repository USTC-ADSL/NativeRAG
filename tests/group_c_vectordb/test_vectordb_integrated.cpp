#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <string>

#include "dataset/TrivialQADataset.hpp"
#include "dataset/VectorDataGenerator.hpp"
#include "vector_Index/FaissIndex.hpp"
#include "vector_db/SqliteVectorDB.hpp"
#include "TestBase.hpp"
#include "faiss/MetricType.h"

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
  std::vector<std::string> metadata_;
  std::shared_ptr<SqliteVectorDB> sqlite_;
  std::string faiss_factory_ = "Flat";
  faiss::MetricType faiss_metric_ = faiss::METRIC_INNER_PRODUCT;

 public:
  bool test_load_dataset() {
    print_test_info("Test 1: Check Pre-generated Embeddings");
    const std::string vectors_file = std::string(PROJECT_ROOT_DIR) + "/dataset/data/qwen3_embeddings/vectors.bin";
    const std::string metadata_file = std::string(PROJECT_ROOT_DIR) + "/dataset/data/qwen3_embeddings/metadata.txt";
    std::ifstream fv(vectors_file, std::ios::binary);
    std::ifstream fm(metadata_file);
    bool success = fv.good() && fm.good();
    print_result("Embeddings exist", success);
    return success;
  }

  bool test_generate_vectors() {
    print_test_info("Test 2: Load Pre-generated Vectors");
    const std::string vectors_file = std::string(PROJECT_ROOT_DIR) + "/dataset/data/qwen3_embeddings/vectors.bin";
    const std::string metadata_file = std::string(PROJECT_ROOT_DIR) + "/dataset/data/qwen3_embeddings/metadata.txt";
    vectors_ = VectorDataGenerator::load_vectors(vectors_file);
    metadata_ = VectorDataGenerator::load_metadata(metadata_file);
    bool success = !vectors_.empty() && vectors_.size() == metadata_.size();
    if (success) {
      std::cout << "Loaded " << vectors_.size() << " vectors, dim=" << vectors_[0].size() << '\n';
      std::cout << "Loaded " << metadata_.size() << " metadata entries\n";
    } else {
      std::cout << "Vectors size: " << vectors_.size() << ", metadata size: " << metadata_.size() << '\n';
    }
    print_result("Load vectors", success);
    return success;
  }

  bool test_add_vectors_to_index() {
    print_test_info("Test 3: Add Vectors to Index");
    
    if (vectors_.empty()) {
      print_result("Vectors exist", false);
      return false;
    }

    // Read FAISS config from environment variables (optional)
    if (const char* f = std::getenv("FAISS_FACTORY")) {
      faiss_factory_ = f;
    }
    if (const char* m = std::getenv("FAISS_METRIC")) {
      std::string ms = m;
      if (ms == "IP" || ms == "COSINE" || ms == "COS") faiss_metric_ = faiss::METRIC_INNER_PRODUCT;
      else if (ms == "L2" || ms == "EUCLIDEAN") faiss_metric_ = faiss::METRIC_L2;
    }
    std::cout << "[Config] FAISS factory = " << faiss_factory_
              << ", metric = " << (faiss_metric_ == faiss::METRIC_L2 ? "L2" : "IP") << '\n';

    index_ = std::make_shared<FaissIndex>(faiss_factory_, faiss_metric_);
    sqlite_ = std::make_shared<SqliteVectorDB>("/tmp/test_texts.db");
    
    // 生成 ID
    vector_ids_.clear();
    for (int64_t i = 0; i < static_cast<int64_t>(vectors_.size()); ++i) {
      vector_ids_.push_back(i);
    }

    // 1) 添加向量到 FAISS
    bool success = index_->add_vectors(vectors_, vector_ids_);

    if (success) {
      std::cout << "Added " << vectors_.size() << " vectors to index\n";
    }

    // 2) 写入 id->text 映射到 SQLite
    if (success) {
      if (!metadata_.empty() && metadata_.size() == vectors_.size()) {
        bool kv_ok = sqlite_->add_texts(metadata_, vector_ids_);
        success = success && kv_ok;
        std::cout << "Inserted " << metadata_.size() << " id->text into SQLite (/tmp/test_texts.db)\n";
      } else {
        std::cout << "Skip SQLite insert: metadata missing or size mismatch\n";
      }
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
      // 使用 SQLite 通过 ID 取回文本，验证并行使用
      if (sqlite_) {
        int show = std::min<int>(3, static_cast<int>(results.size()));
        for (int i = 0; i < show; ++i) {
          auto id = results[static_cast<size_t>(i)].first;
          auto text = sqlite_->get_text_for_id(id);
          std::cout << "  -> Text[" << id << "]: "
                    << (text.size() > 60 ? text.substr(0, 60) + "..." : text) << '\n';
          success = success && !text.empty();
        }
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

