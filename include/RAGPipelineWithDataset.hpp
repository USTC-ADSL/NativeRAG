#pragma once

#include <memory>
#include <string>
#include <vector>

#include "RAGPipeline.hpp"
#include "dataset/IDataset.hpp"

namespace mobile_rag {

/**
 * RAGPipelineWithDataset 扩展了 RAGPipeline，支持从数据集加载数据
 * 用于测试和评估各个 Group 的功能
 */
class RAGPipelineWithDataset : public RAGPipeline {
 public:
  RAGPipelineWithDataset(std::shared_ptr<IDocumentLoader> loader,
                         std::shared_ptr<IEmbeddingModel> embedder,
                         std::shared_ptr<IVectorIndex> index,
                         std::shared_ptr<ILargeLanguageModel> llm);

  /**
   * 从数据集加载文档并构建索引
   * @param dataset 数据集实例
   * @param use_documents 是否使用数据集中的文档（true）或问题（false）
   */
  void build_index_from_dataset(const std::shared_ptr<IDataset>& dataset,
                                bool use_documents = true);

  /**
   * 从数据集中获取所有样本的问题
   */
  std::vector<std::string> get_queries_from_dataset(
      const std::shared_ptr<IDataset>& dataset);

  /**
   * 从数据集中获取所有样本的答案
   */
  std::vector<std::vector<std::string>> get_answers_from_dataset(
      const std::shared_ptr<IDataset>& dataset);
};

}  // namespace mobile_rag

