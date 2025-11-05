#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <memory>
#include "dataset/IDataset.hpp"
#include "dataset/TrivialQADataset.hpp"
#include "dataset/VectorDataGenerator.hpp"

using namespace mobile_rag;

// 向量相似度计算（余弦相似度）
float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 0.0f;
    
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom > 0 ? dot / denom : 0.0f;
}

// 测试1：加载真实数据集
void testLoadDataset() {
    std::cout << "\n=== Test 1: Load TrivialQA Dataset ===" << std::endl;
    
    auto dataset = std::make_shared<TrivialQADataset>();
    bool success = dataset->load("dataset/data/val00-100.json");
    
    assert(success);
    assert(dataset->size() > 0);
    
    std::cout << "✓ Loaded " << dataset->size() << " samples from TrivialQA" << std::endl;
    
    // 打印第一个样本信息
    auto sample = dataset->get(0);
    std::cout << "  Sample ID: " << sample.id << std::endl;
    std::cout << "  Query: " << sample.query.substr(0, 50) << "..." << std::endl;
    std::cout << "  Documents: " << sample.documents.size() << std::endl;
}

// 测试2：从数据集生成向量
void testGenerateVectorsFromDataset() {
    std::cout << "\n=== Test 2: Generate Vectors from Dataset ===" << std::endl;
    
    auto dataset = std::make_shared<TrivialQADataset>();
    dataset->load("dataset/data/val00-100.json");
    
    std::string output_dir = "/tmp/test_vectors";
    bool success = VectorDataGenerator::generate_from_dataset(dataset, output_dir, 384);
    
    assert(success);
    std::cout << "✓ Generated vectors to " << output_dir << std::endl;
}

// 测试3：加载和验证向量
void testLoadVectors() {
    std::cout << "\n=== Test 3: Load and Verify Vectors ===" << std::endl;
    
    std::string vectors_file = "/tmp/test_vectors/vectors.bin";
    auto vectors = VectorDataGenerator::load_vectors(vectors_file);
    
    assert(!vectors.empty());
    assert(vectors[0].size() == 384);
    
    std::cout << "✓ Loaded " << vectors.size() << " vectors with dimension " 
              << vectors[0].size() << std::endl;
    
    // 验证向量是否归一化
    float norm = 0.0f;
    for (float v : vectors[0]) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    std::cout << "  First vector norm: " << norm << " (should be ~1.0)" << std::endl;
}

// 测试4：向量相似度计算
void testVectorSimilarity() {
    std::cout << "\n=== Test 4: Vector Similarity ===" << std::endl;
    
    std::string vectors_file = "/tmp/test_vectors/vectors.bin";
    auto vectors = VectorDataGenerator::load_vectors(vectors_file);
    
    // 测试相同向量的相似度
    float sim_same = cosineSimilarity(vectors[0], vectors[0]);
    assert(std::abs(sim_same - 1.0f) < 0.01f);
    std::cout << "✓ Similarity(same vector) = " << sim_same << " (expected ~1.0)" << std::endl;
    
    // 测试不同向量的相似度
    if (vectors.size() > 1) {
        float sim_diff = cosineSimilarity(vectors[0], vectors[1]);
        std::cout << "✓ Similarity(different vectors) = " << sim_diff << std::endl;
    }
}

// 测试5：向量搜索模拟
void testVectorSearch() {
    std::cout << "\n=== Test 5: Vector Search Simulation ===" << std::endl;
    
    std::string vectors_file = "/tmp/test_vectors/vectors.bin";
    auto vectors = VectorDataGenerator::load_vectors(vectors_file);
    
    // 使用第一个向量作为查询
    auto query = vectors[0];
    
    // 计算与所有向量的相似度
    std::vector<std::pair<int, float>> similarities;
    for (size_t i = 0; i < vectors.size(); ++i) {
        float sim = cosineSimilarity(query, vectors[i]);
        similarities.push_back({i, sim});
    }
    
    // 排序并获取 Top-10
    std::sort(similarities.begin(), similarities.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    int top_k = std::min(10, static_cast<int>(similarities.size()));
    std::cout << "✓ Top-" << top_k << " search results:" << std::endl;
    for (int i = 0; i < top_k; ++i) {
        std::cout << "  " << (i+1) << ". ID=" << similarities[i].first 
                  << ", Similarity=" << similarities[i].second << std::endl;
    }
}

// 测试6：元数据加载
void testMetadata() {
    std::cout << "\n=== Test 6: Metadata ===" << std::endl;
    
    std::string metadata_file = "/tmp/test_vectors/metadata.txt";
    auto metadata = VectorDataGenerator::load_metadata(metadata_file);
    
    assert(!metadata.empty());
    std::cout << "✓ Loaded " << metadata.size() << " metadata entries" << std::endl;
    std::cout << "  First entry: " << metadata[0] << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Group C (VectorDB) Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        testLoadDataset();
        testGenerateVectorsFromDataset();
        testLoadVectors();
        testVectorSimilarity();
        testVectorSearch();
        testMetadata();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "✓ All tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}

