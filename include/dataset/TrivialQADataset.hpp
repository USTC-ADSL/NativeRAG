#pragma once

#include "IDataset.hpp"
#include <vector>
#include <map>

namespace mobile_rag {

/**
 * TrivialQA 数据集实现
 * 支持 JSON 格式的 TrivialQA 数据集
 * 
 * 数据格式:
 * {
 *   "docs": ["doc1", "doc2", ...],
 *   "qas": [
 *     {
 *       "id": "q1",
 *       "question": "What is...",
 *       "answers": ["answer1", "answer2"],
 *       "supporting_facts": [[0, 1], [1, 2]]  // 文档索引
 *     }
 *   ]
 * }
 */
class TrivialQADataset : public IDataset {
public:
    TrivialQADataset() = default;
    ~TrivialQADataset() override = default;

    bool load(const std::string& file_path) override;
    size_t size() const override;
    DataSample get(size_t index) const override;
    std::vector<DataSample> get_all() const override;
    std::string get_name() const override { return "TrivialQA"; }

private:
    std::vector<DataSample> samples_;
    std::vector<std::string> documents_;  // 所有文档的缓存
};

}  // namespace mobile_rag

