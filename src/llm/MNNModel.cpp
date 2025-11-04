#include "llm/MNNModel.hpp"

#include <iostream>
#include <sstream>

namespace mobile_rag {

bool MNNModel::load_model(const std::string& model_path) {
  try {
    llm_.reset(MNN::Transformer::Llm::createLLM(model_path));
    if (!llm_) {
      std::cerr << "[MNNModel] Failed to create Llm from: " << model_path << '\n';
      return false;
    }
    llm_->set_config("{\"tmp_path\":\"tmp\"}");
    llm_->load();
    return true;
  } catch (...) {
    std::cerr << "[MNNModel] Exception while loading Llm config: " << model_path
              << '\n';
    llm_.reset();
    return false;
  }
}

std::string MNNModel::build_prompt(const std::string& query,
                                   const std::vector<std::string>& contexts) {
  std::string prompt = query;
  if (!contexts.empty()) {
    prompt += "\nRelated documents are:";
    for (const auto& doc : contexts) {
      prompt += doc;
      prompt += '\n';
    }
  }
  return prompt;
}

std::string MNNModel::generate(const std::string& prompt) {
  if (!llm_) {
    std::cerr << "[MNNModel] Llm is not loaded." << '\n';
    return {};
  }
  std::ostringstream oss;
  llm_->response(prompt, &oss, "\n");
  llm_->reset();
  return oss.str();
}

}  // namespace mobile_rag



