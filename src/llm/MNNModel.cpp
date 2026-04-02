#include "llm/MNNModel.hpp"

#include <iostream>
#include <sstream>
#include <filesystem>

#include "llm/PromptUtils.hpp"

namespace mobile_rag {

// Helper function to convert relative path to absolute path
static std::string resolve_path(const std::string& path) {
  try {
    auto fs_path = std::filesystem::path(path);
    if (fs_path.is_absolute()) {
      return path;
    }
    // Try to make it absolute relative to current working directory
    auto abs_path = std::filesystem::absolute(fs_path);
    return abs_path.string();
  } catch (const std::exception& e) {
    std::cerr << "[MNNModel] Error resolving path: " << e.what() << '\n';
    return path;
  }
}

void MNNModel::set_num_threads(int num_threads) {
  if (num_threads <= 0) {
    return;
  }

  num_threads_ = num_threads;
  if (llm_) {
    const std::string config =
        std::string("{\"thread_num\":") + std::to_string(num_threads_) + "}";
    if (!llm_->set_config(config)) {
      std::cerr << "[MNNModel] Warning: failed to update thread_num to "
                << num_threads_ << '\n';
    }
  }
}

void MNNModel::set_max_new_tokens(int max_new_tokens) {
  if (max_new_tokens <= 0) {
    return;
  }

  max_new_tokens_ = max_new_tokens;
  if (llm_) {
    const std::string config =
        std::string("{\"max_new_tokens\":") + std::to_string(max_new_tokens_) + "}";
    if (!llm_->set_config(config)) {
      std::cerr << "[MNNModel] Warning: failed to update max_new_tokens to "
                << max_new_tokens_ << '\n';
    }
  }
}

bool MNNModel::load_model(const std::string& model_path) {
  try {
    // Resolve the path to absolute path to avoid issues with relative paths
    std::string resolved_path = resolve_path(model_path);
    std::cout << "[MNNModel] Resolved model path: " << resolved_path << '\n';

    // Create LLM instance from config path
    llm_.reset(MNN::Transformer::Llm::createLLM(resolved_path));
    if (!llm_) {
      std::cerr << "[MNNModel] Failed to create Llm from: " << resolved_path << '\n';
      return false;
    }

    // Configure the LLM with temporary path, thread count, and a plain
    // assistant template that does not inject reasoning tags.
    std::ostringstream config_stream;
    config_stream << "{\"tmp_path\":\"tmp\",\"use_template\":true,"
                  << "\"thread_num\":" << num_threads_ << ','
                  << "\"max_new_tokens\":" << max_new_tokens_ << ','
                  << "\"assistant_prompt_template\":\"<|im_start|>assistant\\n<think>\\n</think>%s<|im_end|>\\n\"}";
    const std::string config = config_stream.str();
    if (!llm_->set_config(config)) {
      std::cerr << "[MNNModel] Failed to set config" << '\n';
      llm_.reset();
      return false;
    }

    // Load the model weights
    if (!llm_->load()) {
      std::cerr << "[MNNModel] Failed to load model weights" << '\n';
      llm_.reset();
      return false;
    }

    std::cout << "[MNNModel] Model loaded successfully from: " << model_path << '\n';
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[MNNModel] Exception while loading model: " << e.what() << '\n';
    llm_.reset();
    return false;
  } catch (...) {
    std::cerr << "[MNNModel] Unknown exception while loading model from: " << model_path << '\n';
    llm_.reset();
    return false;
  }
}

std::string MNNModel::build_prompt(const std::string& query,
                                   const std::vector<std::string>& contexts) {
  return build_rag_prompt(query, contexts) + "\n<think>\n</think>";
}

std::string MNNModel::generate(const std::string& prompt) {
  if (!llm_) {
    std::cerr << "[MNNModel] LLM is not loaded. Call load_model() first." << '\n';
    return {};
  }

  try {
    // Use stringstream to capture the model's output
    std::ostringstream oss;
    MNN::Transformer::ChatMessages messages = {
        {"system",
         "You are a retrieval-augmented assistant. "
         "Answer using only the provided reference documents. "
         "If the answer is missing, say \"I don't know based on the provided documents.\" "
         "Do not output analysis or <think> tags. Return only the final answer."},
        {"user", prompt},
    };

    // Use a non-newline stop string to avoid prematurely truncating the answer
    // after the first line break.
    llm_->response(messages, &oss, "<eop>", max_new_tokens_);

    // Reset the model state for the next inference
    llm_->reset();

    return cleanup_generation_output(oss.str());
  } catch (const std::exception& e) {
    std::cerr << "[MNNModel] Exception during generation: " << e.what() << '\n';
    return {};
  } catch (...) {
    std::cerr << "[MNNModel] Unknown exception during generation" << '\n';
    return {};
  }
}

}  // namespace mobile_rag
