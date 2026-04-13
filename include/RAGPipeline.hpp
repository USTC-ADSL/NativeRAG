#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "loader/IDocumentLoader.hpp"
#include "controller/GraphSelector.hpp"
#include "embedding/IEmbeddingModel.hpp"
#include "vector_Index/IVectorIndex.hpp"
#include "llm/ILargeLanguageModel.hpp"
#include "vector_db/SqliteVectorDB.hpp"

namespace mobile_rag {

class RAGPipeline {
 public:
  struct SemanticHashPrefilterConfig {
    bool enabled = false;
    int candidate_limit = 32;
    int max_hamming_distance = -1;
  };

  struct LexicalPrefilterConfig {
    bool enabled = false;
    int candidate_limit = 16;
  };

  RAGPipeline(std::shared_ptr<IDocumentLoader> loader,
              std::shared_ptr<IEmbeddingModel> embedder,
              std::shared_ptr<IVectorIndex> index,
              std::shared_ptr<ILargeLanguageModel> llm,
              std::shared_ptr<SqliteVectorDB> sqlite_db = nullptr,
              int top_k = 5,
              size_t chunk_size = 1000,
              size_t chunk_overlap = 200);

  // ========== 离线阶段 (Offline/Indexing Phase) ==========
  /**
   * 从文件构建索引（离线阶段）
   * Step 1-3: 文档加载、向量化、索引构建
   */
  void build_index_from_file(const std::string& file_path);

  /**
   * 保存索引到磁盘（离线阶段）
   * 持久化向量索引和元数据
   */
  bool save_index(const std::string& index_path);

  // ========== 查询阶段 (Online/Query Phase) ==========
  /**
   * 从磁盘加载索引（查询阶段初始化）
   * 加载向量索引和元数据
   */
  bool load_index(const std::string& index_path);

  /**
   * 回答查询（查询阶段）
   * Step 4-7: 查询向量化、向量检索、上下文拼接、LLM推理
   */
  std::string answer_query(const std::string& query);

  void set_semantic_hash_prefilter(SemanticHashPrefilterConfig config);
  void set_lexical_prefilter(LexicalPrefilterConfig config);
  void set_graph_selector_config(GraphSelector::Config config);

 protected:
  bool add_text_embeddings(const std::vector<std::string>& texts,
                           const std::vector<std::vector<float>>& vectors,
                           const std::string& source_label);

  std::shared_ptr<IDocumentLoader> loader_;
  std::shared_ptr<IEmbeddingModel> embedder_;
  std::shared_ptr<IVectorIndex> index_;
  std::shared_ptr<ILargeLanguageModel> llm_;
  std::shared_ptr<SqliteVectorDB> sqlite_db_;

  std::map<int64_t, std::string> id_to_chunk_;
  int64_t next_id_ = 0;
  int top_k_ = 5;
  size_t chunk_size_ = 1000;
  size_t chunk_overlap_ = 200;
  SemanticHashPrefilterConfig semantic_hash_prefilter_;
  LexicalPrefilterConfig lexical_prefilter_;
  GraphSelector graph_selector_;
};

}  // namespace mobile_rag
