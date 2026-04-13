#include <cassert>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "RAGPipeline.hpp"
#include "embedding/IEmbeddingModel.hpp"
#include "llm/ILargeLanguageModel.hpp"
#include "llm/PromptUtils.hpp"
#include "vector_Index/IVectorIndex.hpp"
#include "vector_db/SqliteVectorDB.hpp"

using namespace mobile_rag;

namespace {

class FakeEmbeddingModel : public IEmbeddingModel {
 public:
  bool load_model(const std::string& /*model_path*/) override { return true; }

  std::vector<float> embed_query(const std::string& text) override {
    auto it = query_embeddings.find(text);
    if (it == query_embeddings.end()) {
      return {};
    }
    return it->second;
  }

  std::vector<std::vector<float>> embed_documents(
      const std::vector<std::string>& texts) override {
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(texts.size());
    for (const auto& text : texts) {
      auto it = document_embeddings.find(text);
      if (it == document_embeddings.end()) {
        embeddings.emplace_back();
      } else {
        embeddings.push_back(it->second);
      }
    }
    return embeddings;
  }

  std::unordered_map<std::string, std::vector<float>> query_embeddings;
  std::unordered_map<std::string, std::vector<float>> document_embeddings;
};

class FakeVectorIndex : public IVectorIndex {
 public:
  bool add_vectors(const std::vector<std::vector<float>>& /*vectors*/,
                   const std::vector<int64_t>& /*ids*/) override {
    return true;
  }

  std::vector<std::pair<int64_t, float>> search(
      const std::vector<float>& query_vector, int k) override {
    ++search_calls;
    last_query_vector = query_vector;
    last_k = k;
    return search_results;
  }

  bool save_index(const std::string& /*index_path*/) override { return true; }
  bool load_index(const std::string& /*index_path*/) override { return true; }

  int search_calls = 0;
  int last_k = 0;
  std::vector<float> last_query_vector;
  std::vector<std::pair<int64_t, float>> search_results;
};

class FakeLLM : public ILargeLanguageModel {
 public:
  bool load_model(const std::string& /*model_path*/) override { return true; }

  std::string build_prompt(const std::string& query,
                           const std::vector<std::string>& contexts) override {
    return build_rag_prompt(query, contexts);
  }

  std::string generate(const std::string& /*prompt*/) override {
    return {};
  }
};

class TestableRAGPipeline : public RAGPipeline {
 public:
  using RAGPipeline::RAGPipeline;

  bool ingest(const std::vector<std::string>& texts,
              const std::vector<std::vector<float>>& vectors,
              const std::string& source_label) {
    return add_text_embeddings(texts, vectors, source_label);
  }
};

std::string make_temp_db_path(const std::string& suffix) {
  return "/tmp/native_rag_semantic_hash_prefilter_" + suffix + ".sqlite3";
}

void test_prefilter_uses_sqlite_rerank_before_dense_fallback() {
  const std::string db_path = make_temp_db_path("shortlist");
  std::filesystem::remove(db_path);

  auto sqlite_db = std::make_shared<SqliteVectorDB>(db_path);
  auto embedder = std::make_shared<FakeEmbeddingModel>();
  auto index = std::make_shared<FakeVectorIndex>();
  auto llm = std::make_shared<FakeLLM>();

  TestableRAGPipeline pipeline(nullptr, embedder, index, llm, sqlite_db, 1, 128, 16);
  pipeline.set_semantic_hash_prefilter({true, 2, 0});

  const std::vector<std::string> texts = {
      "SQLite is the canonical store.",
      "Faiss is an accelerator.",
      "This chunk should never win.",
  };
  const std::vector<std::vector<float>> vectors = {
      {1.0f, 1.0f},
      {0.8f, 0.9f},
      {-1.0f, -1.0f},
  };
  assert(pipeline.ingest(texts, vectors, "semantic-hash-prefilter-test"));

  const std::string query = "What is the canonical store?";
  embedder->query_embeddings[query] = {1.0f, 1.0f};
  index->search_results = {{2, 0.99f}};

  const std::string answer = pipeline.answer_query(query);
  assert(index->search_calls == 0);
  assert(answer.find("SQLite is the canonical store.") != std::string::npos);

  std::filesystem::remove(db_path);
}

void test_prefilter_falls_back_to_dense_search_when_shortlist_is_empty() {
  const std::string db_path = make_temp_db_path("fallback");
  std::filesystem::remove(db_path);

  auto sqlite_db = std::make_shared<SqliteVectorDB>(db_path);
  auto embedder = std::make_shared<FakeEmbeddingModel>();
  auto index = std::make_shared<FakeVectorIndex>();
  auto llm = std::make_shared<FakeLLM>();

  TestableRAGPipeline pipeline(nullptr, embedder, index, llm, sqlite_db, 1, 128, 16);
  pipeline.set_semantic_hash_prefilter({true, 2, 0});

  const std::vector<std::string> texts = {
      "Dense fallback path.",
      "Another dense-only chunk.",
  };
  const std::vector<std::vector<float>> vectors = {
      {1.0f, 1.0f},
      {0.6f, 0.7f},
  };
  assert(pipeline.ingest(texts, vectors, "semantic-hash-prefilter-fallback-test"));

  const std::string query = "dense fallback path";
  embedder->query_embeddings[query] = {-1.0f, -1.0f};
  index->search_results = {{0, 0.77f}};

  const std::string answer = pipeline.answer_query(query);
  assert(index->search_calls == 1);
  assert(answer.find("Dense fallback path.") != std::string::npos);

  std::filesystem::remove(db_path);
}

void test_lexical_and_hash_shortlists_are_merged_before_dense_rerank() {
  const std::string db_path = make_temp_db_path("merged-shortlists");
  std::filesystem::remove(db_path);

  auto sqlite_db = std::make_shared<SqliteVectorDB>(db_path);
  auto embedder = std::make_shared<FakeEmbeddingModel>();
  auto index = std::make_shared<FakeVectorIndex>();
  auto llm = std::make_shared<FakeLLM>();

  TestableRAGPipeline pipeline(nullptr, embedder, index, llm, sqlite_db, 1, 128, 16);
  pipeline.set_semantic_hash_prefilter({true, 1, 0});
  pipeline.set_lexical_prefilter({true, 1});

  const std::vector<std::string> texts = {
      "Unrelated dense chunk without sqlite keyword.",
      "SQLite stores metadata and traces for this project.",
  };
  const std::vector<std::vector<float>> vectors = {
      {1.0f, 1.0f},
      {1.0f, -1.0f},
  };
  assert(pipeline.ingest(texts, vectors, "merged-prefilter-test"));

  const std::string query = "sqlite metadata";
  embedder->query_embeddings[query] = {1.0f, -1.0f};
  index->search_results = {{0, 0.99f}};

  const std::string answer = pipeline.answer_query(query);
  assert(index->search_calls == 0);
  assert(answer.find("SQLite stores metadata and traces for this project.") != std::string::npos);

  std::filesystem::remove(db_path);
}

}  // namespace

int main() {
  test_prefilter_uses_sqlite_rerank_before_dense_fallback();
  test_prefilter_falls_back_to_dense_search_when_shortlist_is_empty();
  test_lexical_and_hash_shortlists_are_merged_before_dense_rerank();
  return 0;
}
