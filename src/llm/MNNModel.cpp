#include "llm/MNNModel.hpp"

#include <iostream>
#include <sstream>
#include <filesystem>

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

    // Configure the LLM with temporary path for KV cache
    // Set use_template to true to use the model's built-in chat template
    std::string config = R"({"tmp_path":"tmp","use_template":true})";
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
  // Build a RAG-style prompt with query and retrieved documents
  std::string prompt = query;

  if (!contexts.empty()) {
    prompt += "\n\nRelated documents:\n";
    for (size_t i = 0; i < contexts.size(); ++i) {
      prompt += "[Document " + std::to_string(i + 1) + "]: ";
      prompt += contexts[i];
      prompt += "\n";
    }
  }

  return prompt;
}

std::string MNNModel::generate(const std::string& prompt) {
  if (!llm_) {
    std::cerr << "[MNNModel] LLM is not loaded. Call load_model() first." << '\n';
    return {};
  }

  try {
    // Use stringstream to capture the model's output
    std::ostringstream oss;

    // Call response with the prompt, output stream, and end token
    // The model will generate tokens until it reaches the end token or max tokens
    llm_->response(prompt, &oss, "\n");

    // Reset the model state for the next inference
    llm_->reset();

    std::string result = oss.str();

    // Trim trailing whitespace and newlines
    while (!result.empty() && (result.back() == '\n' || result.back() == ' ')) {
      result.pop_back();
    }

    return result;
  } catch (const std::exception& e) {
    std::cerr << "[MNNModel] Exception during generation: " << e.what() << '\n';
    return {};
  } catch (...) {
    std::cerr << "[MNNModel] Unknown exception during generation" << '\n';
    return {};
  }
}

}  // namespace mobile_rag



