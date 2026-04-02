#include "RAGPipelineWithDataset.hpp"

#include <iostream>

#include "loader/Chunk/CharacterSplitter.hpp"

namespace mobile_rag {

RAGPipelineWithDataset::RAGPipelineWithDataset(
    std::shared_ptr<IDocumentLoader> loader,
    std::shared_ptr<IEmbeddingModel> embedder,
    std::shared_ptr<IVectorIndex> index,
    std::shared_ptr<ILargeLanguageModel> llm,
    std::shared_ptr<SqliteVectorDB> sqlite_db,
    int top_k,
    size_t chunk_size,
    size_t chunk_overlap)
    : RAGPipeline(std::move(loader), std::move(embedder), std::move(index),
                  std::move(llm), std::move(sqlite_db), top_k,
                  chunk_size, chunk_overlap) {}

void RAGPipelineWithDataset::build_index_from_dataset(
    const std::shared_ptr<IDataset>& dataset, bool use_documents) {
  if (!dataset || dataset->size() == 0) {
    std::cerr << "[RAGPipelineWithDataset] Dataset is empty or null" << '\n';
    return;
  }

  if (!embedder_) {
    std::cerr << "[RAGPipelineWithDataset] Embedder is not initialized" << '\n';
    return;
  }

  if (!index_) {
    std::cerr << "[RAGPipelineWithDataset] Vector index is not initialized" << '\n';
    return;
  }

  std::vector<std::string> texts_to_embed;
  // 收集要嵌入的文本
  auto samples = dataset->get_all();
  // 与 TextFileLoader 中保持一致的默认切分策略
  CharacterSplitter splitter(chunk_size_, chunk_overlap_);

  for (size_t i = 0; i < samples.size(); ++i) {
    const auto& sample = samples[i];
    
    if (use_documents) {
      // 使用文档：对文档进行切分后再嵌入
      for (size_t j = 0; j < sample.documents.size(); ++j) {
        const auto& doc = sample.documents[j];
        auto chunks = splitter.split(doc);
        if (chunks.empty()) continue;
        for (size_t k = 0; k < chunks.size(); ++k) {
          texts_to_embed.push_back(chunks[k]);
        }
      }
    } else {
      // 使用问题
      texts_to_embed.push_back(sample.query);
    }
  }

  if (texts_to_embed.empty()) {
    std::cerr << "[RAGPipelineWithDataset] No texts to embed from dataset"
              << '\n';
    return;
  }

  std::cout << "[RAGPipelineWithDataset] Embedding " << texts_to_embed.size()
            << " chunks from dataset..." << '\n';

  // 以批处理方式进行嵌入，并输出进度
  constexpr size_t kBatchSize = 64;
  const size_t total = texts_to_embed.size();
  size_t processed = 0;
  size_t total_added = 0;

  for (size_t start = 0; start < total; start += kBatchSize) {
    const size_t end = std::min(start + kBatchSize, total);
    std::vector<std::string> batch(texts_to_embed.begin() + start,
                                   texts_to_embed.begin() + end);

    auto vectors = embedder_->embed_documents(batch);
    const int64_t next_id_before = next_id_;
    if (add_text_embeddings(batch, vectors, "dataset indexing batch")) {
      total_added += static_cast<size_t>(next_id_ - next_id_before);
    } else {
      std::cerr << "[RAGPipelineWithDataset] Warning: Failed to add batch starting at "
                << start << '\n';
    }

    processed += batch.size();
    const size_t percent = static_cast<size_t>((processed * 100) / total);
    std::cout << "[RAGPipelineWithDataset] Embedded " << processed << "/" << total
              << " (" << percent << "%)" << '\n';
  }

  std::cout << "[RAGPipelineWithDataset] Index built with " << total_added
            << " vectors" << '\n';

}

std::vector<std::string> RAGPipelineWithDataset::get_queries_from_dataset(
    const std::shared_ptr<IDataset>& dataset) {
  std::vector<std::string> queries;
  if (!dataset) return queries;

  auto samples = dataset->get_all();
  for (const auto& sample : samples) {
    queries.push_back(sample.query);
  }
  return queries;
}

std::vector<std::vector<std::string>>
RAGPipelineWithDataset::get_answers_from_dataset(
    const std::shared_ptr<IDataset>& dataset) {
  std::vector<std::vector<std::string>> answers;
  if (!dataset) return answers;

  auto samples = dataset->get_all();
  for (const auto& sample : samples) {
    answers.push_back(sample.answers);
  }
  return answers;
}

}  // namespace mobile_rag
