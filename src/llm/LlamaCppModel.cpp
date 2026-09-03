#include "llm/LlamaCppModel.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <utility>

#include "llama/LlamaRuntime.hpp"

namespace mobile_rag {

namespace {

std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                  const std::string& text) {
  int32_t count = llama_tokenize(vocab, text.c_str(),
                                 static_cast<int32_t>(text.size()), nullptr, 0,
                                 true, true);
  if (count == INT32_MIN) {
    return {};
  }
  if (count < 0) {
    count = -count;
  }
  std::vector<llama_token> tokens(static_cast<size_t>(count));
  const int32_t actual = llama_tokenize(
      vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(),
      static_cast<int32_t>(tokens.size()), true, true);
  if (actual < 0) {
    return {};
  }
  tokens.resize(static_cast<size_t>(actual));
  return tokens;
}

std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
  char local_buffer[256];
  int32_t size = llama_token_to_piece(vocab, token, local_buffer,
                                      sizeof(local_buffer), 0, true);
  if (size >= 0) {
    return std::string(local_buffer, static_cast<size_t>(size));
  }
  std::vector<char> buffer(static_cast<size_t>(-size));
  size = llama_token_to_piece(vocab, token, buffer.data(),
                              static_cast<int32_t>(buffer.size()), 0, true);
  return size < 0 ? std::string{}
                  : std::string(buffer.data(), static_cast<size_t>(size));
}

std::string trim_response(std::string response) {
  const std::string closing_think = "</think>";
  const size_t think_end = response.find(closing_think);
  if (think_end != std::string::npos) {
    response.erase(0, think_end + closing_think.size());
  }
  const size_t first = response.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const size_t last = response.find_last_not_of(" \t\r\n");
  return response.substr(first, last - first + 1);
}

}  // namespace

LlamaCppModel::LlamaCppModel(int num_threads, int max_tokens, int context_size)
    : num_threads_(std::max(1, num_threads)),
      max_tokens_(std::max(1, max_tokens)),
      context_size_(std::max(256, context_size)) {}

LlamaCppModel::~LlamaCppModel() { unload_model(); }

void LlamaCppModel::unload_model() {
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
  if (runtime_acquired_) {
    LlamaRuntime::release();
    runtime_acquired_ = false;
  }
}

bool LlamaCppModel::load_model(const std::string& model_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  unload_model();
  if (model_path.empty() || !std::filesystem::is_regular_file(model_path)) {
    std::cerr << "[LlamaCppModel] Model file not found: " << model_path << '\n';
    return false;
  }

  if (!LlamaRuntime::acquire()) {
    return false;
  }
  runtime_acquired_ = true;

  llama_model_params model_params = llama_model_default_params();
  if (!LlamaRuntime::apply_model_profile(model_params)) {
    unload_model();
    return false;
  }
  model_ = llama_model_load_from_file(model_path.c_str(), model_params);
  if (!model_) {
    std::cerr << "[LlamaCppModel] Failed to load GGUF model: " << model_path
              << '\n';
    unload_model();
    return false;
  }
  if (!llama_model_has_decoder(model_)) {
    std::cerr << "[LlamaCppModel] Model has no decoder\n";
    unload_model();
    return false;
  }

  vocab_ = llama_model_get_vocab(model_);
  const int model_context = llama_model_n_ctx_train(model_);
  const int requested_context =
      model_context > 0 ? std::min(context_size_, model_context) : context_size_;

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = static_cast<uint32_t>(requested_context);
  context_params.n_batch =
      static_cast<uint32_t>(std::min(requested_context, 512));
  context_params.n_ubatch = context_params.n_batch;
  context_params.n_threads = num_threads_;
  context_params.n_threads_batch = num_threads_;
  context_params.embeddings = false;
  LlamaRuntime::apply_context_profile(context_params);

  ctx_ = llama_init_from_model(model_, context_params);
  if (!ctx_ || !vocab_) {
    std::cerr << "[LlamaCppModel] Failed to create generation context\n";
    unload_model();
    return false;
  }

  sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
  if (!sampler_) {
    std::cerr << "[LlamaCppModel] Failed to create sampler chain\n";
    unload_model();
    return false;
  }
  llama_sampler* greedy_sampler = llama_sampler_init_greedy();
  if (!greedy_sampler) {
    std::cerr << "[LlamaCppModel] Failed to create greedy sampler\n";
    unload_model();
    return false;
  }
  llama_sampler_chain_add(sampler_, greedy_sampler);
  std::cout << "[LlamaCppModel] Loaded " << LlamaRuntime::accelerator_name()
            << " model on " << LlamaRuntime::target_device_name()
            << ", context="
            << llama_n_ctx(ctx_) << ", max_tokens=" << max_tokens_ << '\n';
  return true;
}

std::string LlamaCppModel::build_prompt(
    const std::string& query, const std::vector<std::string>& contexts) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::string system =
      "你是一个设备端检索增强问答助手。只能依据给定的检索上下文回答。"
      "如果上下文没有答案，请明确说不知道。回答要简洁，不要展示推理过程。";

  std::ostringstream user_stream;
  user_stream << "检索上下文：\n";
  for (size_t i = 0; i < contexts.size(); ++i) {
    user_stream << '[' << (i + 1) << "] " << contexts[i] << '\n';
  }
  user_stream << "\n问题：" << query
              << "\n请直接给出答案，不要输出思考过程。/no_think";
  const std::string user = user_stream.str();

  if (!model_) {
    return system + "\n\n" + user + "\n\n回答：";
  }

  const char* chat_template = llama_model_chat_template(model_, nullptr);
  if (!chat_template) {
    return system + "\n\n" + user + "\n\n回答：";
  }

  const llama_chat_message messages[] = {
      {"system", system.c_str()},
      {"user", user.c_str()},
  };
  std::vector<char> formatted(system.size() + user.size() + 1024);
  int32_t length = llama_chat_apply_template(
      chat_template, messages, 2, true, formatted.data(),
      static_cast<int32_t>(formatted.size()));
  if (length > static_cast<int32_t>(formatted.size())) {
    formatted.resize(static_cast<size_t>(length));
    length = llama_chat_apply_template(
        chat_template, messages, 2, true, formatted.data(),
        static_cast<int32_t>(formatted.size()));
  }
  if (length < 0) {
    std::cerr << "[LlamaCppModel] Chat template failed; using plain prompt\n";
    return system + "\n\n" + user + "\n\n回答：";
  }
  return std::string(formatted.data(), static_cast<size_t>(length));
}

std::string LlamaCppModel::generate(const std::string& prompt) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!model_ || !ctx_ || !vocab_ || !sampler_) {
    std::cerr << "[LlamaCppModel] Model is not loaded\n";
    return {};
  }

  std::vector<llama_token> prompt_tokens = tokenize(vocab_, prompt);
  if (prompt_tokens.empty()) {
    std::cerr << "[LlamaCppModel] Failed to tokenize prompt\n";
    return {};
  }
  const int context_limit = static_cast<int>(llama_n_ctx(ctx_));
  if (static_cast<int>(prompt_tokens.size()) + max_tokens_ > context_limit) {
    std::cerr << "[LlamaCppModel] Prompt and output budget exceed context: "
              << prompt_tokens.size() << " + " << max_tokens_ << " > "
              << context_limit << '\n';
    return {};
  }

  llama_memory_clear(llama_get_memory(ctx_), true);
  llama_sampler_reset(sampler_);

  const int batch_limit = static_cast<int>(llama_n_batch(ctx_));
  for (size_t offset = 0; offset < prompt_tokens.size();) {
    const int count = std::min<int>(
        batch_limit, static_cast<int>(prompt_tokens.size() - offset));
    llama_batch batch =
        llama_batch_get_one(prompt_tokens.data() + offset, count);
    const int decode_result = llama_decode(ctx_, batch);
    if (decode_result != 0) {
      std::cerr << "[LlamaCppModel] Prompt decode failed: " << decode_result
                << '\n';
      return {};
    }
    offset += static_cast<size_t>(count);
  }

  std::string response;
  for (int generated = 0; generated < max_tokens_; ++generated) {
    const llama_token token = llama_sampler_sample(sampler_, ctx_, -1);
    if (llama_vocab_is_eog(vocab_, token)) {
      break;
    }

    std::string piece = token_to_piece(vocab_, token);
    if (piece.empty()) {
      std::cerr << "[LlamaCppModel] Failed to convert generated token\n";
      return {};
    }
    response += piece;

    if (generated + 1 < max_tokens_) {
      llama_token next = token;
      llama_batch batch = llama_batch_get_one(&next, 1);
      const int decode_result = llama_decode(ctx_, batch);
      if (decode_result != 0) {
        std::cerr << "[LlamaCppModel] Generation decode failed: "
                  << decode_result << '\n';
        return {};
      }
    }
  }
  return trim_response(std::move(response));
}

}  // namespace mobile_rag
