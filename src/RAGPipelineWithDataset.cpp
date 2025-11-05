#include "RAGPipelineWithDataset.hpp"

#include <iostream>

namespace mobile_rag {

RAGPipelineWithDataset::RAGPipelineWithDataset(
    std::shared_ptr<IDocumentLoader> loader,
    std::shared_ptr<IEmbeddingModel> embedder,
    std::shared_ptr<IVectorIndex> index,
    std::shared_ptr<ILargeLanguageModel> llm)
    : RAGPipeline(std::move(loader), std::move(embedder), std::move(index),
                  std::move(llm)) {}

void RAGPipelineWithDataset::build_index_from_dataset(
    const std::shared_ptr<IDataset>& dataset, bool use_documents) {
  if (!dataset || dataset->size() == 0) {
    std::cerr << "[RAGPipelineWithDataset] Dataset is empty or null" << '\n';
    return;
  }

  std::vector<std::string> texts_to_embed;
  std::vector<std::string> metadata;

  // 收集要嵌入的文本
  auto samples = dataset->get_all();
  for (size_t i = 0; i < samples.size(); ++i) {
    const auto& sample = samples[i];
    
    if (use_documents) {
      // 使用文档
      for (size_t j = 0; j < sample.documents.size(); ++j) {
        texts_to_embed.push_back(sample.documents[j]);
        metadata.push_back(sample.id + "_doc_" + std::to_string(j));
      }
    } else {
      // 使用问题
      texts_to_embed.push_back(sample.query);
      metadata.push_back(sample.id + "_query");
    }
  }

  if (texts_to_embed.empty()) {
    std::cerr << "[RAGPipelineWithDataset] No texts to embed from dataset"
              << '\n';
    return;
  }

  std::cout << "[RAGPipelineWithDataset] Embedding " << texts_to_embed.size()
            << " texts from dataset..." << '\n';

  // 调用 RAGPipeline 的 build_index_from_file 的逻辑
  // 但这里我们直接嵌入文本而不是从文件加载
  std::vector<std::vector<float>> vectors =
      embedder_->embed_documents(texts_to_embed);

  if (vectors.size() != texts_to_embed.size()) {
    std::cerr << "[RAGPipelineWithDataset] Embeddings size mismatch: got "
              << vectors.size() << ", expected " << texts_to_embed.size()
              << '\n';
    return;
  }

  std::vector<int64_t> ids;
  ids.reserve(vectors.size());
  for (size_t i = 0; i < vectors.size(); ++i) {
    const int64_t id = next_id_++;
    ids.push_back(id);
    id_to_chunk_[id] = texts_to_embed[i];
  }

  if (!vectors.empty()) {
    index_->add_vectors(vectors, ids);
  }

  std::cout << "[RAGPipelineWithDataset] Index built with " << vectors.size()
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

