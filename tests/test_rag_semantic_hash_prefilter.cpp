#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "RAGPipeline.hpp"
#include "controller/GraphSelector.hpp"
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

struct QueryRunResult {
  std::string answer;
  std::string stdout_text;
};

std::string make_temp_db_path(const std::string& suffix) {
  return "/tmp/native_rag_semantic_hash_prefilter_" + suffix + ".sqlite3";
}

QueryRunResult run_query_and_capture_stdout(TestableRAGPipeline& pipeline,
                                            const std::string& query) {
  std::ostringstream captured_stdout;
  auto* old_stdout = std::cout.rdbuf(captured_stdout.rdbuf());

  const std::string answer = pipeline.answer_query(query);

  std::cout.rdbuf(old_stdout);
  return {answer, captured_stdout.str()};
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

void test_adaptive_graph_logs_controller_selection() {
  const std::string db_path = make_temp_db_path("adaptive-controller");
  const std::string trace_path = "/tmp/native_rag_query_trace.json";
  const std::string trace_jsonl_path = "/tmp/native_rag_query_trace.jsonl";
  const std::string summary_csv_path = "/tmp/native_rag_query_trace_summary.csv";
  std::filesystem::remove(db_path);
  std::filesystem::remove(trace_path);
  std::filesystem::remove(trace_jsonl_path);
  std::filesystem::remove(summary_csv_path);

  auto sqlite_db = std::make_shared<SqliteVectorDB>(db_path);
  auto embedder = std::make_shared<FakeEmbeddingModel>();
  auto index = std::make_shared<FakeVectorIndex>();
  auto llm = std::make_shared<FakeLLM>();

  TestableRAGPipeline pipeline(nullptr, embedder, index, llm, sqlite_db, 1, 128, 16);
  pipeline.set_lexical_prefilter({true, 1});
  pipeline.set_semantic_hash_prefilter({true, 1, 0});
  pipeline.set_graph_selector_config({true, 3, 0.15f, 0.50f});

  const std::vector<std::string> texts = {
      "Unrelated dense chunk without sqlite keyword.",
      "SQLite stores metadata and traces for this project.",
  };
  const std::vector<std::vector<float>> vectors = {
      {1.0f, 1.0f},
      {1.0f, -1.0f},
  };
  assert(pipeline.ingest(texts, vectors, "adaptive-controller-test"));
  assert(sqlite_db->get_chunk_state(1) == "warm");

  const std::string query = "sqlite metadata retrieval traces";
  embedder->query_embeddings[query] = {1.0f, -1.0f};
  index->search_results = {{0, 0.99f}};

  const auto run = run_query_and_capture_stdout(pipeline, query);
  assert(run.answer.find("SQLite stores metadata and traces for this project.") !=
         std::string::npos);
  assert(run.stdout_text.find("[CONTROLLER]") != std::string::npos);
  assert(run.stdout_text.find("budget=tight") != std::string::npos);
  assert(run.stdout_text.find("initial_graph=lexical_prefilter") != std::string::npos);
  assert(run.stdout_text.find("reason=term_rich_query") != std::string::npos);
  assert(sqlite_db->get_chunk_state(1) == "hot");
  assert(sqlite_db->count_chunk_state_transitions(1) == 2);
  const auto& trace = pipeline.last_query_trace();
  assert(trace.query == query);
  assert(trace.initial_graph == "lexical_prefilter");
  assert(trace.final_graph == "lexical_prefilter");
  assert(trace.budget_class == "tight");
  assert(trace.initial_reason == "term_rich_query");
  assert(trace.final_reason == "evidence_sufficient");
  assert(trace.top_k == 1);
  assert(trace.lexical_prefilter_enabled);
  assert(trace.lexical_candidate_limit == 1);
  assert(trace.semantic_hash_prefilter_enabled);
  assert(trace.semantic_hash_candidate_limit == 1);
  assert(trace.semantic_hash_max_distance == 0);
  assert(trace.lexical_candidate_count == 1);
  assert(trace.hash_candidate_count == 0);
  assert(trace.dense_result_count == 1);
  assert(trace.fallback_reason == "none");
  assert(trace.promoted_to_hot == 1);
  assert(trace.demoted_to_warm == 0);
  assert(trace.index_state.hot_count == 1);
  assert(trace.index_state.warm_count == 1);
  assert(trace.index_state.cold_count == 0);
  assert(trace.index_state.transition_count == 3);
  assert(trace.evidence.retrieved_chunk_count == 1);
  assert(trace.results.size() == 1);
  assert(trace.results[0].id == 1);
  assert(trace.results[0].score > 0.0f);
  assert(pipeline.export_last_query_trace(trace_path));

  std::ifstream trace_stream(trace_path);
  std::string trace_json((std::istreambuf_iterator<char>(trace_stream)),
                         std::istreambuf_iterator<char>());
  assert(trace_json.find("\"query\": \"sqlite metadata retrieval traces\"") !=
         std::string::npos);
  assert(trace_json.find("\"initial_graph\": \"lexical_prefilter\"") !=
         std::string::npos);
  assert(trace_json.find("\"final_graph\": \"lexical_prefilter\"") !=
         std::string::npos);
  assert(trace_json.find("\"budget_class\": \"tight\"") != std::string::npos);
  assert(trace_json.find("\"top_k\": 1") != std::string::npos);
  assert(trace_json.find("\"lexical_prefilter_enabled\": true") != std::string::npos);
  assert(trace_json.find("\"semantic_hash_candidate_limit\": 1") != std::string::npos);
  assert(trace_json.find("\"promoted_to_hot\": 1") != std::string::npos);
  assert(trace_json.find("\"hot\": 1") != std::string::npos);
  assert(trace_json.find("\"transition_count\": 3") != std::string::npos);
  assert(pipeline.append_last_query_trace_jsonl(trace_jsonl_path));

  std::ifstream trace_jsonl_stream(trace_jsonl_path);
  std::string trace_jsonl((std::istreambuf_iterator<char>(trace_jsonl_stream)),
                          std::istreambuf_iterator<char>());
  assert(trace_jsonl.find("{\"query\":\"sqlite metadata retrieval traces\"") !=
         std::string::npos);
  assert(trace_jsonl.find("\"budget_class\":\"tight\"") != std::string::npos);
  assert(trace_jsonl.find("\"semantic_hash_candidate_limit\":1") != std::string::npos);
  assert(trace_jsonl.find("\"promoted_to_hot\":1") != std::string::npos);
  assert(trace_jsonl.find("\"transition_count\":3") != std::string::npos);
  assert(trace_jsonl.find('\n') != std::string::npos);
  assert(pipeline.append_last_query_trace_summary_csv(summary_csv_path));

  std::ifstream summary_stream(summary_csv_path);
  std::string summary_csv((std::istreambuf_iterator<char>(summary_stream)),
                          std::istreambuf_iterator<char>());
  assert(summary_csv.find("query,answer,adaptive_graph_enabled,budget_class") !=
         std::string::npos);
  assert(summary_csv.find("sqlite metadata retrieval traces,SQLite stores metadata and traces for this project.,true,tight") !=
         std::string::npos);
  assert(summary_csv.find(",lexical_prefilter,lexical_prefilter,term_rich_query,evidence_sufficient,1,true,1,true,1,0,1,0,1,none,1,0,1,1,0,3,") !=
         std::string::npos);

  std::filesystem::remove(db_path);
  std::filesystem::remove(trace_path);
  std::filesystem::remove(trace_jsonl_path);
  std::filesystem::remove(summary_csv_path);
}

void test_query_promotes_new_hit_and_demotes_previous_hot_chunk() {
  const std::string db_path = make_temp_db_path("state-demotion");
  std::filesystem::remove(db_path);

  auto sqlite_db = std::make_shared<SqliteVectorDB>(db_path);
  auto embedder = std::make_shared<FakeEmbeddingModel>();
  auto index = std::make_shared<FakeVectorIndex>();
  auto llm = std::make_shared<FakeLLM>();

  TestableRAGPipeline pipeline(nullptr, embedder, index, llm, sqlite_db, 1, 128, 16);

  const std::vector<std::string> texts = {
      "SQLite stores metadata and traces for this project.",
      "Faiss accelerates dense search for this project.",
  };
  const std::vector<std::vector<float>> vectors = {
      {1.0f, 0.0f},
      {0.0f, 1.0f},
  };
  assert(pipeline.ingest(texts, vectors, "state-demotion-test"));
  assert(sqlite_db->get_chunk_state(0) == "warm");
  assert(sqlite_db->get_chunk_state(1) == "warm");

  const std::string first_query = "sqlite metadata traces";
  embedder->query_embeddings[first_query] = {1.0f, 0.0f};
  index->search_results = {{0, 0.91f}};
  const auto first_run = run_query_and_capture_stdout(pipeline, first_query);
  assert(first_run.stdout_text.find("[INDEX_STATE] promoted_to_hot=1 demoted_to_warm=0") !=
         std::string::npos);
  assert(first_run.stdout_text.find("[INDEX_STATE_SUMMARY] hot=1 warm=1 cold=0 transitions=3") !=
         std::string::npos);
  assert(sqlite_db->get_chunk_state(0) == "hot");
  assert(sqlite_db->get_chunk_state(1) == "warm");

  const std::string second_query = "faiss dense search";
  embedder->query_embeddings[second_query] = {0.0f, 1.0f};
  index->search_results = {{1, 0.88f}};
  const auto second_run = run_query_and_capture_stdout(pipeline, second_query);
  assert(second_run.stdout_text.find("[INDEX_STATE] promoted_to_hot=1 demoted_to_warm=1") !=
         std::string::npos);
  assert(second_run.stdout_text.find("[INDEX_STATE_SUMMARY] hot=1 warm=1 cold=0 transitions=5") !=
         std::string::npos);
  assert(sqlite_db->get_chunk_state(0) == "warm");
  assert(sqlite_db->get_chunk_state(1) == "hot");

  std::filesystem::remove(db_path);
}

}  // namespace

int main() {
  test_prefilter_uses_sqlite_rerank_before_dense_fallback();
  test_prefilter_falls_back_to_dense_search_when_shortlist_is_empty();
  test_lexical_and_hash_shortlists_are_merged_before_dense_rerank();
  test_adaptive_graph_logs_controller_selection();
  test_query_promotes_new_hit_and_demotes_previous_hot_chunk();
  return 0;
}
