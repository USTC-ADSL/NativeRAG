#include "llm/LlamaCppModel.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "llm/LlamaCppLogging.hpp"
#include "llm/PromptUtils.hpp"

namespace mobile_rag {

namespace {

void initialize_llama_backend_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    install_quiet_llama_logging();
    llama_backend_init();
    ggml_backend_load_all();
  });
}

}  // namespace

LlamaCppModel::~LlamaCppModel() {
  unload();
}

void LlamaCppModel::unload() {
  if (sampler_) {
    llama_sampler_free(sampler_);
    sampler_ = nullptr;
  }
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
  }
  vocab_ = nullptr;
  model_path_.clear();
}

void LlamaCppModel::set_num_threads(int num_threads) {
  if (num_threads <= 0) {
    return;
  }

  num_threads_ = num_threads;
  if (model_) {
    initialize_context();
  }
}

void LlamaCppModel::set_max_new_tokens(int max_new_tokens) {
  if (max_new_tokens > 0) {
    max_new_tokens_ = max_new_tokens;
  }
}

std::string LlamaCppModel::resolve_model_path(const std::string& model_path) const {
  namespace fs = std::filesystem;

  fs::path path(model_path);
  if (fs::is_regular_file(path)) {
    return model_path;
  }

  if (!fs::is_directory(path)) {
    return {};
  }

  std::vector<fs::path> candidates;
  for (const auto& entry : fs::directory_iterator(path)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    if (entry.path().extension() == ".gguf") {
      candidates.push_back(entry.path());
    }
  }

  if (candidates.empty()) {
    return {};
  }

  std::sort(candidates.begin(), candidates.end());
  return candidates.front().string();
}

bool LlamaCppModel::initialize_context() {
  if (!model_) {
    return false;
  }

  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }

  llama_context_params ctx_params = llama_context_default_params();
  const int32_t model_ctx = llama_model_n_ctx_train(model_);
  const uint32_t resolved_ctx =
      model_ctx > 0 ? static_cast<uint32_t>(std::clamp(model_ctx, 512, 2048))
                    : 2048u;

  n_ctx_ = resolved_ctx;
  ctx_params.n_ctx = n_ctx_;
  ctx_params.n_batch = n_ctx_;
  ctx_params.n_ubatch = std::min<uint32_t>(n_ctx_, 512u);
  ctx_params.n_seq_max = 1;
  ctx_params.n_threads = num_threads_;
  ctx_params.n_threads_batch = num_threads_;
  ctx_params.no_perf = true;

  ctx_ = llama_init_from_model(model_, ctx_params);
  if (!ctx_) {
    std::cerr << "[LlamaCppModel] Failed to create context" << '\n';
    return false;
  }

  return true;
}

bool LlamaCppModel::initialize_sampler() {
  if (sampler_) {
    llama_sampler_free(sampler_);
    sampler_ = nullptr;
  }

  auto sampler_params = llama_sampler_chain_default_params();
  sampler_params.no_perf = true;
  sampler_ = llama_sampler_chain_init(sampler_params);
  if (!sampler_) {
    std::cerr << "[LlamaCppModel] Failed to initialize sampler chain" << '\n';
    return false;
  }

  llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());
  return true;
}

bool LlamaCppModel::load_model(const std::string& model_path) {
  unload();
  initialize_llama_backend_once();

  model_path_ = resolve_model_path(model_path);
  if (model_path_.empty()) {
    std::cerr << "[LlamaCppModel] Could not resolve a GGUF model from: "
              << model_path << '\n';
    return false;
  }

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;

  model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
  if (!model_) {
    std::cerr << "[LlamaCppModel] Failed to load model from: "
              << model_path_ << '\n';
    unload();
    return false;
  }

  if (!llama_model_has_decoder(model_)) {
    std::cerr << "[LlamaCppModel] Unsupported model type: decoder is required" << '\n';
    unload();
    return false;
  }

  vocab_ = llama_model_get_vocab(model_);
  if (!vocab_) {
    std::cerr << "[LlamaCppModel] Failed to access model vocabulary" << '\n';
    unload();
    return false;
  }

  if (!initialize_context() || !initialize_sampler()) {
    unload();
    return false;
  }

  return true;
}

std::string LlamaCppModel::build_prompt(const std::string& query,
                                        const std::vector<std::string>& contexts) {
  return build_rag_prompt(query, contexts);
}

bool LlamaCppModel::tokenize_prompt(const std::string& prompt,
                                    std::vector<llama_token>& prompt_tokens) const {
  if (!vocab_) {
    return false;
  }

  const int32_t token_count =
      -llama_tokenize(vocab_, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                      nullptr, 0, true, true);
  if (token_count <= 0) {
    std::cerr << "[LlamaCppModel] Failed to tokenize prompt" << '\n';
    return false;
  }

  prompt_tokens.resize(static_cast<size_t>(token_count));
  const int32_t written =
      llama_tokenize(vocab_, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                     prompt_tokens.data(), token_count, true, true);
  if (written < 0) {
    std::cerr << "[LlamaCppModel] Prompt tokenization failed" << '\n';
    return false;
  }

  return true;
}

std::string LlamaCppModel::token_to_piece(llama_token token) const {
  if (!vocab_) {
    return {};
  }

  char stack_buffer[256];
  int32_t piece_len = llama_token_to_piece(vocab_, token, stack_buffer,
                                           static_cast<int32_t>(sizeof(stack_buffer)),
                                           0, true);
  if (piece_len >= 0) {
    return std::string(stack_buffer, static_cast<size_t>(piece_len));
  }

  std::string dynamic_buffer(static_cast<size_t>(-piece_len), '\0');
  piece_len = llama_token_to_piece(vocab_, token, dynamic_buffer.data(),
                                   static_cast<int32_t>(dynamic_buffer.size()),
                                   0, true);
  if (piece_len < 0) {
    return {};
  }

  dynamic_buffer.resize(static_cast<size_t>(piece_len));
  return dynamic_buffer;
}

std::string LlamaCppModel::generate(const std::string& prompt) {
  if (!model_ || !ctx_ || !sampler_ || !vocab_) {
    std::cerr << "[LlamaCppModel] Model is not loaded. Call load_model() first." << '\n';
    return {};
  }

  std::vector<llama_token> prompt_tokens;
  if (!tokenize_prompt(prompt, prompt_tokens)) {
    return {};
  }

  if (prompt_tokens.empty()) {
    return {};
  }

  const size_t reserved_generation = static_cast<size_t>(std::max(1, max_new_tokens_));
  const size_t max_prompt_tokens =
      n_ctx_ > reserved_generation + 1
          ? static_cast<size_t>(n_ctx_ - reserved_generation - 1)
          : static_cast<size_t>(n_ctx_ - 1);

  if (prompt_tokens.size() > max_prompt_tokens && max_prompt_tokens > 0) {
    prompt_tokens.erase(
        prompt_tokens.begin(),
        prompt_tokens.begin() +
            static_cast<std::vector<llama_token>::difference_type>(
                prompt_tokens.size() - max_prompt_tokens));
  }

  llama_memory_clear(llama_get_memory(ctx_), true);
  llama_sampler_reset(sampler_);

  if (llama_decode(ctx_, llama_batch_get_one(prompt_tokens.data(),
                                             static_cast<int32_t>(prompt_tokens.size()))) != 0) {
    std::cerr << "[LlamaCppModel] llama_decode failed during prompt evaluation" << '\n';
    return {};
  }

  std::string output;
  output.reserve(static_cast<size_t>(max_new_tokens_) * 4);

  for (int i = 0; i < max_new_tokens_; ++i) {
    llama_token token = llama_sampler_sample(sampler_, ctx_, -1);
    if (llama_vocab_is_eog(vocab_, token)) {
      break;
    }

    output += token_to_piece(token);

    if (llama_decode(ctx_, llama_batch_get_one(&token, 1)) != 0) {
      std::cerr << "[LlamaCppModel] llama_decode failed during generation" << '\n';
      break;
    }
  }

  return cleanup_generation_output(output);
}

}  // namespace mobile_rag
