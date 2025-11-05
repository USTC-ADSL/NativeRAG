#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "dataset/IDataset.hpp"
#include "dataset/TrivialQADataset.hpp"

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
    if (!dataset_->load(dataset_path)) {
      std::cerr << "[TestBase] Failed to load dataset: " << dataset_path
                << '\n';
      return false;
    }
    std::cout << "[TestBase] Loaded dataset with " << dataset_->size()
              << " samples" << '\n';
    return true;
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

 protected:
  std::shared_ptr<IDataset> dataset_;
};

}  // namespace testing
}  // namespace mobile_rag

