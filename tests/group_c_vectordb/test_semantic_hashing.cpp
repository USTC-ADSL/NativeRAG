#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "retrieval/SemanticHash.hpp"
#include "vector_db/SqliteVectorDB.hpp"

using namespace mobile_rag;

namespace {

std::string test_db_path() {
  return "/tmp/native_rag_semantic_hashing.db";
}

}  // namespace

int main() {
  const std::string db_path = test_db_path();
  std::filesystem::remove(db_path);

  const std::vector<std::vector<float>> embeddings = {
      {1.0f, 0.8f, -0.9f, -1.1f},
      {0.9f, 0.7f, -0.7f, -1.0f},
      {-1.0f, -0.8f, 0.9f, 1.1f},
  };
  const std::vector<int64_t> ids = {10, 20, 30};

  std::vector<std::vector<std::uint8_t>> codes;
  for (const auto& embedding : embeddings) {
    codes.push_back(build_sign_semantic_hash(embedding, 16));
  }

  assert(codes.size() == embeddings.size());
  assert(hamming_distance(codes[0], codes[1]) < hamming_distance(codes[0], codes[2]));

  {
    SqliteVectorDB db(db_path);
    assert(db.add_semantic_hashes(codes, ids, 16));

    const auto nearest = db.search_by_semantic_hash(codes[0], 2);
    assert(nearest.size() == 2);
    assert(nearest[0].first == 10);
    assert(nearest[0].second == 0);
    assert(nearest[1].first == 20);

    const auto filtered = db.search_by_semantic_hash(codes[0], 5, 2);
    assert(filtered.size() == 2);
    assert(filtered[0].first == 10);
    assert(filtered[1].first == 20);
  }

  {
    SqliteVectorDB reopened(db_path);
    const auto reopened_results = reopened.search_by_semantic_hash(codes[0], 3);
    assert(reopened_results.size() == 3);
    assert(reopened_results[0].first == 10);
    assert(reopened_results[1].first == 20);
    assert(reopened_results[2].first == 30);
  }

  std::filesystem::remove(db_path);
  std::cout << "Semantic hashing test passed\n";
  return 0;
}
