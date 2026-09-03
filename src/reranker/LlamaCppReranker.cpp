#include "reranker/LlamaCppReranker.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <utility>

#include "llama/LlamaRuntime.hpp"

namespace mobile_rag {
namespace {

constexpr const char* kDefaultInstruction =
    "Given a web search query, retrieve relevant passages that answer the "
    "query";

std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                  const std::string& text, bool add_special,
                                  bool parse_special) {
  int32_t count = llama_tokenize(vocab, text.c_str(),
                                 static_cast<int32_t>(text.size()), nullptr, 0,
                                 add_special, parse_special);
  if (count == INT32_MIN) {
    return {};
  }
  if (count < 0) {
    count = -count;
  }
  std::vector<llama_token> tokens(static_cast<size_t>(count));
  const int32_t actual = llama_tokenize(
      vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(),
      static_cast<int32_t>(tokens.size()), add_special, parse_special);
  if (actual < 0) {
    return {};
  }
  tokens.resize(static_cast<size_t>(actual));
  return tokens;
}

void replace_all(std::string* text, const std::string& needle,
                 const std::string& replacement) {
  size_t position = 0;
  while ((position = text->find(needle, position)) != std::string::npos) {
    text->replace(position, needle.size(), replacement);
    position += replacement.size();
  }
}

std::string model_name(const llama_model* model) {
  char buffer[512] = {};
  const int32_t length =
      llama_model_meta_val_str(model, "general.name", buffer, sizeof(buffer));
  if (length <= 0) {
    return {};
  }
  return std::string(buffer,
                     std::min<size_t>(static_cast<size_t>(length),
                                      sizeof(buffer) - 1));
}

bool contains_reranker(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return text.find("rerank") != std::string::npos;
}

float binary_probability(float true_logit, float false_logit) {
  const double difference =
      static_cast<double>(false_logit) - static_cast<double>(true_logit);
  if (difference >= 0.0) {
    const double exponential = std::exp(-difference);
    return static_cast<float>(exponential / (1.0 + exponential));
  }
  return static_cast<float>(1.0 / (1.0 + std::exp(difference)));
}

}  // namespace

LlamaCppReranker::LlamaCppReranker(int num_threads, int context_size)
    : num_threads_(std::max(1, num_threads)),
      context_size_(std::max(256, context_size)) {}

LlamaCppReranker::~LlamaCppReranker() { unload_model(); }

void LlamaCppReranker::unload_model() {
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
  }
  vocab_ = nullptr;
  rerank_template_.clear();
  true_token_ = LLAMA_TOKEN_NULL;
  false_token_ = LLAMA_TOKEN_NULL;
  use_rank_pooling_ = false;
  if (runtime_acquired_) {
    LlamaRuntime::release();
    runtime_acquired_ = false;
  }
}

bool LlamaCppReranker::load_model(const std::string& model_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  unload_model();
  if (model_path.empty() || !std::filesystem::is_regular_file(model_path)) {
    std::cerr << "[LlamaCppReranker] Model file not found: " << model_path
              << '\n';
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
    std::cerr << "[LlamaCppReranker] Failed to load GGUF model: " << model_path
              << '\n';
    unload_model();
    return false;
  }
  if (!llama_model_has_decoder(model_)) {
    std::cerr << "[LlamaCppReranker] Model has no decoder\n";
    unload_model();
    return false;
  }

  vocab_ = llama_model_get_vocab(model_);
  if (!vocab_) {
    std::cerr << "[LlamaCppReranker] Model has no vocabulary\n";
    unload_model();
    return false;
  }

  if (const char* prompt = llama_model_chat_template(model_, "rerank")) {
    rerank_template_ = prompt;
    use_rank_pooling_ = true;
  } else {
    const std::string name = model_name(model_);
    if (!contains_reranker(name)) {
      std::cerr << "[LlamaCppReranker] GGUF has neither a named rerank "
                   "template nor a reranker model name\n";
      unload_model();
      return false;
    }

    const auto true_tokens = tokenize(vocab_, "yes", false, false);
    const auto false_tokens = tokenize(vocab_, "no", false, false);
    if (true_tokens.size() != 1 || false_tokens.size() != 1) {
      std::cerr << "[LlamaCppReranker] Expected yes/no to each map to one "
                   "token\n";
      unload_model();
      return false;
    }
    true_token_ = true_tokens.front();
    false_token_ = false_tokens.front();
  }

  const int model_context = llama_model_n_ctx_train(model_);
  const int requested_context =
      model_context > 0 ? std::min(context_size_, model_context) : context_size_;
  const int requested_batch =
      std::min(requested_context, LlamaRuntime::embedding_batch_limit());

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = static_cast<uint32_t>(requested_context);
  context_params.n_batch = static_cast<uint32_t>(requested_batch);
  context_params.n_ubatch = static_cast<uint32_t>(requested_batch);
  context_params.n_seq_max = 1;
  context_params.n_threads = num_threads_;
  context_params.n_threads_batch = num_threads_;
  context_params.embeddings = use_rank_pooling_;
  if (use_rank_pooling_) {
    context_params.pooling_type = LLAMA_POOLING_TYPE_RANK;
  }
  LlamaRuntime::apply_context_profile(context_params);

  ctx_ = llama_init_from_model(model_, context_params);
  if (!ctx_) {
    std::cerr << "[LlamaCppReranker] Failed to create reranker context\n";
    unload_model();
    return false;
  }
  if (use_rank_pooling_ &&
      llama_pooling_type(ctx_) != LLAMA_POOLING_TYPE_RANK) {
    std::cerr << "[LlamaCppReranker] Model does not expose rank pooling\n";
    unload_model();
    return false;
  }

  std::cout << "[LlamaCppReranker] Loaded "
            << LlamaRuntime::accelerator_name() << " model on "
            << LlamaRuntime::target_device_name() << ", mode="
            << (use_rank_pooling_ ? "rank-pooling" : "yes-no-logits")
            << ", context=" << llama_n_ctx(ctx_)
            << ", batch=" << llama_n_batch(ctx_) << '\n';
  return true;
}

std::string LlamaCppReranker::build_prompt(
    const std::string& query, const std::string& document) const {
  if (use_rank_pooling_) {
    std::string prompt = rerank_template_;
    replace_all(&prompt, "{query}", query);
    replace_all(&prompt, "{document}", document);
    return prompt;
  }

  return std::string(
             "<|im_start|>system\nJudge whether the Document meets the "
             "requirements based on the Query and the Instruct provided. Note "
             "that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
             "<|im_start|>user\n<Instruct>: ") +
         kDefaultInstruction + "\n<Query>: " + query + "\n<Document>: " +
         document +
         "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

std::vector<float> LlamaCppReranker::score(
    const std::string& query,
    const std::vector<std::string>& documents) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!model_ || !ctx_ || !vocab_) {
    std::cerr << "[LlamaCppReranker] Model is not loaded\n";
    return {};
  }

  std::vector<float> scores;
  scores.reserve(documents.size());
  const int context_limit = static_cast<int>(llama_n_ctx(ctx_));
  const int batch_limit = static_cast<int>(llama_n_batch(ctx_));

  for (size_t document_index = 0; document_index < documents.size();
       ++document_index) {
    const std::string prompt = build_prompt(query, documents[document_index]);
    const auto tokens = tokenize(vocab_, prompt, false, true);
    if (tokens.empty()) {
      std::cerr << "[LlamaCppReranker] Failed to tokenize candidate "
                << document_index << '\n';
      return {};
    }
    if (static_cast<int>(tokens.size()) > context_limit) {
      std::cerr << "[LlamaCppReranker] Candidate " << document_index
                << " has " << tokens.size() << " tokens, exceeding context "
                << context_limit << '\n';
      return {};
    }
    if (use_rank_pooling_ && static_cast<int>(tokens.size()) > batch_limit) {
      std::cerr << "[LlamaCppReranker] Rank-pooling candidate "
                << document_index << " has " << tokens.size()
                << " tokens, exceeding batch size " << batch_limit << '\n';
      return {};
    }

    llama_memory_clear(llama_get_memory(ctx_), true);
    if (use_rank_pooling_) {
      llama_batch batch = llama_batch_get_one(
          const_cast<llama_token*>(tokens.data()),
          static_cast<int32_t>(tokens.size()));
      const int decode_result = llama_decode(ctx_, batch);
      if (decode_result != 0) {
        std::cerr << "[LlamaCppReranker] Rank decode failed for candidate "
                  << document_index << ": " << decode_result << '\n';
        return {};
      }
      const float* rank = llama_get_embeddings_seq(ctx_, 0);
      if (!rank || !std::isfinite(rank[0])) {
        std::cerr << "[LlamaCppReranker] Invalid rank output for candidate "
                  << document_index << '\n';
        return {};
      }
      scores.push_back(rank[0]);
      continue;
    }

    for (size_t offset = 0; offset < tokens.size();) {
      const int count = std::min<int>(
          batch_limit, static_cast<int>(tokens.size() - offset));
      llama_batch batch = llama_batch_get_one(
          const_cast<llama_token*>(tokens.data() + offset), count);
      const int decode_result = llama_decode(ctx_, batch);
      if (decode_result != 0) {
        std::cerr << "[LlamaCppReranker] Logits decode failed for candidate "
                  << document_index << ": " << decode_result << '\n';
        return {};
      }
      offset += static_cast<size_t>(count);
    }

    const float* logits = llama_get_logits_ith(ctx_, -1);
    if (!logits || !std::isfinite(logits[true_token_]) ||
        !std::isfinite(logits[false_token_])) {
      std::cerr << "[LlamaCppReranker] Invalid yes/no logits for candidate "
                << document_index << '\n';
      return {};
    }
    scores.push_back(
        binary_probability(logits[true_token_], logits[false_token_]));
  }

  return scores;
}

}  // namespace mobile_rag
