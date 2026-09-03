#include "embedding/LlamaCppEmbedding.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <iostream>
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

bool normalize_embedding(const float* source, int dimension,
                         std::vector<float>* destination) {
  destination->assign(source, source + dimension);
  double squared_norm = 0.0;
  for (float value : *destination) {
    if (!std::isfinite(value)) {
      return false;
    }
    squared_norm += static_cast<double>(value) * value;
  }
  const double norm = std::sqrt(squared_norm);
  if (!std::isfinite(norm) || norm <= 1e-12) {
    return false;
  }
  for (float& value : *destination) {
    value = static_cast<float>(value / norm);
  }
  return true;
}

}  // namespace

LlamaCppEmbedding::LlamaCppEmbedding(int num_threads, int context_size,
                                     int batch_size)
    : num_threads_(std::max(1, num_threads)),
      context_size_(std::max(32, context_size)),
      batch_size_(std::max(32, batch_size)) {}

LlamaCppEmbedding::~LlamaCppEmbedding() { unload_model(); }

void LlamaCppEmbedding::unload_model() {
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
  }
  vocab_ = nullptr;
  embed_dim_ = 0;
  if (runtime_acquired_) {
    LlamaRuntime::release();
    runtime_acquired_ = false;
  }
}

bool LlamaCppEmbedding::load_model(const std::string& model_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  unload_model();
  if (model_path.empty() || !std::filesystem::is_regular_file(model_path)) {
    std::cerr << "[LlamaCppEmbedding] Model file not found: " << model_path
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
    std::cerr << "[LlamaCppEmbedding] Failed to load GGUF model: "
              << model_path << '\n';
    unload_model();
    return false;
  }

  vocab_ = llama_model_get_vocab(model_);
  embed_dim_ = llama_model_n_embd_out(model_);
  if (!vocab_ || embed_dim_ <= 0) {
    std::cerr << "[LlamaCppEmbedding] Invalid vocabulary or output dimension\n";
    unload_model();
    return false;
  }

  const int model_context = llama_model_n_ctx_train(model_);
  const int requested_context =
      model_context > 0 ? std::min(context_size_, model_context) : context_size_;
  const int requested_batch =
      std::min({batch_size_, requested_context,
                LlamaRuntime::embedding_batch_limit()});

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = static_cast<uint32_t>(requested_context);
  context_params.n_batch = static_cast<uint32_t>(requested_batch);
  context_params.n_ubatch = static_cast<uint32_t>(requested_batch);
  context_params.n_seq_max = 16;
  context_params.n_threads = num_threads_;
  context_params.n_threads_batch = num_threads_;
  context_params.embeddings = true;
  LlamaRuntime::apply_context_profile(context_params);
  context_params.kv_unified = true;

  ctx_ = llama_init_from_model(model_, context_params);
  if (!ctx_) {
    std::cerr << "[LlamaCppEmbedding] Failed to create embedding context\n";
    unload_model();
    return false;
  }

  std::cout << "[LlamaCppEmbedding] Loaded "
            << LlamaRuntime::accelerator_name()
            << " model on " << LlamaRuntime::target_device_name()
            << ", dimension=" << embed_dim_
            << ", context=" << llama_n_ctx(ctx_)
            << ", pooling=" << static_cast<int>(llama_pooling_type(ctx_))
            << '\n';
  return true;
}

std::vector<std::vector<float>> LlamaCppEmbedding::embed_texts(
    const std::vector<std::string>& texts) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!model_ || !ctx_ || !vocab_ || embed_dim_ <= 0) {
    std::cerr << "[LlamaCppEmbedding] Model is not loaded\n";
    return {};
  }
  if (texts.empty()) {
    return {};
  }

  std::vector<std::vector<llama_token>> tokenized;
  tokenized.reserve(texts.size());
  const int n_batch = static_cast<int>(llama_n_batch(ctx_));
  for (const std::string& text : texts) {
    auto tokens = tokenize(vocab_, text);
    if (tokens.empty()) {
      std::cerr << "[LlamaCppEmbedding] Failed to tokenize input\n";
      return {};
    }
    if (static_cast<int>(tokens.size()) > n_batch) {
      std::cerr << "[LlamaCppEmbedding] Input has " << tokens.size()
                << " tokens, exceeding batch size " << n_batch << '\n';
      return {};
    }
    tokenized.push_back(std::move(tokens));
  }

  std::vector<std::vector<float>> output(texts.size());
  llama_batch batch = llama_batch_init(n_batch, 0, 1);
  const int max_sequences = static_cast<int>(llama_n_seq_max(ctx_));
  size_t begin = 0;

  while (begin < tokenized.size()) {
    size_t end = begin;
    int token_count = 0;
    while (end < tokenized.size() &&
           static_cast<int>(end - begin) < max_sequences &&
           token_count + static_cast<int>(tokenized[end].size()) <= n_batch) {
      token_count += static_cast<int>(tokenized[end].size());
      ++end;
    }
    if (end == begin) {
      llama_batch_free(batch);
      return {};
    }

    batch.n_tokens = 0;
    std::vector<int> last_token_indices;
    last_token_indices.reserve(end - begin);
    for (size_t input_index = begin; input_index < end; ++input_index) {
      const llama_seq_id sequence_id =
          static_cast<llama_seq_id>(input_index - begin);
      const auto& tokens = tokenized[input_index];
      for (size_t position = 0; position < tokens.size(); ++position) {
        const int batch_index = batch.n_tokens++;
        batch.token[batch_index] = tokens[position];
        batch.pos[batch_index] = static_cast<llama_pos>(position);
        batch.n_seq_id[batch_index] = 1;
        batch.seq_id[batch_index][0] = sequence_id;
        batch.logits[batch_index] = 1;
      }
      last_token_indices.push_back(batch.n_tokens - 1);
    }

    llama_memory_clear(llama_get_memory(ctx_), false);
    const int decode_result = llama_decode(ctx_, batch);
    if (decode_result != 0) {
      std::cerr << "[LlamaCppEmbedding] llama_decode failed: "
                << decode_result << '\n';
      llama_batch_free(batch);
      return {};
    }

    const enum llama_pooling_type pooling = llama_pooling_type(ctx_);
    for (size_t local_index = 0; local_index < end - begin; ++local_index) {
      const float* embedding = nullptr;
      if (pooling == LLAMA_POOLING_TYPE_NONE) {
        embedding = llama_get_embeddings_ith(ctx_,
                                             last_token_indices[local_index]);
      } else {
        embedding = llama_get_embeddings_seq(
            ctx_, static_cast<llama_seq_id>(local_index));
      }
      if (!embedding || !normalize_embedding(
                            embedding, embed_dim_, &output[begin + local_index])) {
        std::cerr << "[LlamaCppEmbedding] Invalid embedding output for input "
                  << (begin + local_index) << '\n';
        llama_batch_free(batch);
        return {};
      }
    }
    begin = end;
  }

  llama_batch_free(batch);
  return output;
}

std::vector<float> LlamaCppEmbedding::embed_query(const std::string& text) {
  const std::string instructed =
      "Instruct: Given a web search query, retrieve relevant passages that "
      "answer the query\nQuery: " +
      text;
  auto vectors = embed_texts({instructed});
  return vectors.empty() ? std::vector<float>{} : std::move(vectors.front());
}

std::vector<std::vector<float>> LlamaCppEmbedding::embed_documents(
    const std::vector<std::string>& texts) {
  return embed_texts(texts);
}

}  // namespace mobile_rag
