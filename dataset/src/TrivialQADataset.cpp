#include "dataset/TrivialQADataset.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace mobile_rag {

bool TrivialQADataset::load(const std::string& file_path) {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << file_path << std::endl;
            return false;
        }

        json data;
        file >> data;
        file.close();

        // 加载所有文档
        if (data.contains("docs") && data["docs"].is_array()) {
            documents_ = data["docs"].get<std::vector<std::string>>();
        }

        // 支持两种格式：
        // 1. 新格式：questions 和 answers 列表
        // 2. 旧格式：qas 数组

        if (data.contains("questions") && data.contains("answers")) {
            // 新格式：questions 和 answers 是平行的列表
            auto questions = data["questions"].get<std::vector<std::string>>();
            auto answers_data = data["answers"];

            for (size_t i = 0; i < questions.size(); ++i) {
                DataSample sample;
                sample.id = std::to_string(i);
                sample.query = questions[i];

                // 获取答案
                if (i < answers_data.size()) {
                    if (answers_data[i].is_array()) {
                        sample.answers = answers_data[i].get<std::vector<std::string>>();
                    } else if (answers_data[i].is_string()) {
                        sample.answers.push_back(answers_data[i].get<std::string>());
                    }
                }

                // 使用所有文档作为相关文档
                sample.documents = documents_;

                samples_.push_back(sample);
            }
        } else if (data.contains("qas") && data["qas"].is_array()) {
            // 旧格式：qas 数组
            for (const auto& qa : data["qas"]) {
                DataSample sample;

                // 获取问题ID
                if (qa.contains("id")) {
                    sample.id = qa["id"].get<std::string>();
                }

                // 获取问题
                if (qa.contains("question")) {
                    sample.query = qa["question"].get<std::string>();
                }

                // 获取答案
                if (qa.contains("answers") && qa["answers"].is_array()) {
                    sample.answers = qa["answers"].get<std::vector<std::string>>();
                }

                // 获取相关文档（通过 supporting_facts）
                if (qa.contains("supporting_facts") && qa["supporting_facts"].is_array()) {
                    for (const auto& fact : qa["supporting_facts"]) {
                        if (fact.is_array() && fact.size() > 0) {
                            int doc_idx = fact[0].get<int>();
                            if (doc_idx >= 0 && doc_idx < static_cast<int>(documents_.size())) {
                                sample.documents.push_back(documents_[doc_idx]);
                            }
                        }
                    }
                }

                // 如果没有通过 supporting_facts 获取文档，则使用所有文档
                if (sample.documents.empty()) {
                    sample.documents = documents_;
                }

                samples_.push_back(sample);
            }
        }

        std::cout << "Loaded " << samples_.size() << " samples from " << file_path << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        return false;
    }
}

size_t TrivialQADataset::size() const {
    return samples_.size();
}

DataSample TrivialQADataset::get(size_t index) const {
    if (index < samples_.size()) {
        return samples_[index];
    }
    return DataSample();
}

std::vector<DataSample> TrivialQADataset::get_all() const {
    return samples_;
}

}  // namespace mobile_rag

