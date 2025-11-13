#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "dataset/IDataset.hpp"
#include "dataset/TrivialQADataset.hpp"
#include "llm/ILargeLanguageModel.hpp"
#include "llm/LLMFactory.hpp"

namespace mobile_rag {
namespace testing {

/**
 * 测试基类，提供通用的测试工具和数据集加载
 */
class TestBase {
 public:
  TestBase() = default;
  virtual ~TestBase() = default;

  /**
   * 加载 TrivialQA 数据集
   * @param dataset_path 数据集文件路径
   * @return 加载成功返回 true
   */
  bool load_dataset(const std::string& dataset_path) {
    dataset_ = std::make_shared<TrivialQADataset>();

    // Try the provided path first
    if (dataset_->load(dataset_path)) {
      std::cout << "[TestBase] Loaded dataset with " << dataset_->size()
                << " samples" << '\n';
      return true;
    }

    // Try alternative paths
    std::vector<std::string> alternative_paths = {
        "../" + dataset_path,
        "../../" + dataset_path,
        // Handle the case where dataset_path is "dataset/data/val00-100.json"
        // but the actual file is in "dataset/data/trival_QA/val00-100.json"
        "dataset/data/trival_QA/val00-100.json",
        "../dataset/data/trival_QA/val00-100.json",
        "../../dataset/data/trival_QA/val00-100.json",
    };

    for (const auto& alt_path : alternative_paths) {
      if (dataset_->load(alt_path)) {
        std::cout << "[TestBase] Loaded dataset with " << dataset_->size()
                  << " samples from: " << alt_path << '\n';
        return true;
      }
    }

    std::cerr << "[TestBase] Failed to load dataset from any path" << '\n';
    return false;
  }

  /**
   * 获取加载的数据集
   */
  std::shared_ptr<IDataset> get_dataset() const { return dataset_; }

  /**
   * 获取数据集中的所有样本
   */
  std::vector<DataSample> get_samples() const {
    if (!dataset_) return {};
    return dataset_->get_all();
  }

  /**
   * 获取数据集中的前 N 个样本
   */
  std::vector<DataSample> get_samples(size_t count) const {
    auto all_samples = get_samples();
    if (count >= all_samples.size()) return all_samples;
    return std::vector<DataSample>(all_samples.begin(),
                                   all_samples.begin() + count);
  }

  /**
   * 打印测试信息
   */
  void print_test_info(const std::string& test_name) {
    std::cout << "\n========================================\n"
              << test_name << "\n"
              << "========================================\n";
  }

  /**
   * 打印测试结果
   */
  void print_result(const std::string& test_name, bool passed) {
    if (passed) {
      std::cout << "✓ " << test_name << " PASSED\n";
    } else {
      std::cout << "✗ " << test_name << " FAILED\n";
    }
  }

  /**
   * 创建并加载LLM模型
   * @return 加载成功返回LLM实例，失败返回nullptr
   */
  std::shared_ptr<ILargeLanguageModel> create_llm() {
    auto llm = mobile_rag::create_llm();
    if (!llm) {
      std::cerr << "[TestBase] Failed to create LLM instance\n";
      return nullptr;
    }

    // Try multiple possible paths for the model config
    // Order matters: try the most likely paths first to avoid unnecessary attempts
    std::vector<std::string> possible_paths = {
        "../models/Qwen3-0.6B/config.json",      // When run from build directory (most common)
        "models/Qwen3-0.6B/config.json",         // When run from project root
        "config.json",                           // When run from model directory
    };

    for (const auto& model_path : possible_paths) {
      if (llm->load_model(model_path)) {
        std::cout << "[TestBase] Model loaded successfully from: " << model_path << '\n';
        return llm;
      }
    }

    std::cerr << "[TestBase] Failed to load model from any of the possible paths\n";
    return nullptr;
  }

 protected:
  std::shared_ptr<IDataset> dataset_;
};

}  // namespace testing
}  // namespace mobile_rag

