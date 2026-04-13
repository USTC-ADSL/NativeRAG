#include "vector_db/IngestUtils.hpp"

#include <cstring>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <sqlite3.h>

#include "vector_db/SqliteVectorDB.hpp"
#include "vector_Index/FaissIndex.hpp"

namespace mobile_rag {

namespace {
std::vector<std::vector<float>> load_vectors_file(const std::string& file_path) {
  std::vector<std::vector<float>> vectors;
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "[Ingest] Failed to open vectors file: " << file_path << '\n';
    return vectors;
  }
  int count = 0;
  int dimension = 0;
  file.read(reinterpret_cast<char*>(&count), sizeof(int));
  file.read(reinterpret_cast<char*>(&dimension), sizeof(int));
  if (!file.good() || count <= 0 || dimension <= 0) {
    std::cerr << "[Ingest] Invalid vectors header in: " << file_path << '\n';
    return vectors;
  }
  vectors.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    std::vector<float> v(static_cast<size_t>(dimension));
    file.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(dimension * sizeof(float)));
    if (!file.good()) {
      std::cerr << "[Ingest] Failed reading vector #" << i << " from " << file_path << '\n';
      vectors.clear();
      return vectors;
    }
    vectors.emplace_back(std::move(v));
  }
  return vectors;
}

std::vector<std::string> load_metadata_file(const std::string& file_path) {
  std::vector<std::string> metadata;
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "[Ingest] Failed to open metadata file: " << file_path << '\n';
    return metadata;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty()) metadata.push_back(line);
  }
  return metadata;
}
}  // namespace

bool ingest_from_files_to_sqlite(const std::string& vectors_path,
                                 const std::string& metadata_path,
                                 const std::string& sqlite_db_path) {
  // Load metadata only for text mapping; vectors are used for FAISS building, not stored in DB.
  auto vectors = load_vectors_file(vectors_path);  // still validate sizes with metadata
  auto metadata = load_metadata_file(metadata_path);
  if (vectors.empty()) {
    std::cerr << "[Ingest] No vectors loaded from " << vectors_path << '\n';
    return false;
  }
  if (metadata.size() != vectors.size()) {
    std::cerr << "[Ingest] Metadata size (" << metadata.size()
              << ") does not match vectors size (" << vectors.size() << ")"
              << '\n';
    return false;
  }

  // Prepare IDs
  std::vector<int64_t> ids;
  ids.reserve(vectors.size());
  for (int64_t i = 0; i < static_cast<int64_t>(vectors.size()); ++i) {
    ids.push_back(i);
  }

  // Insert only text mapping into SQLite
  SqliteVectorDB db(sqlite_db_path);
  if (!db.add_texts(metadata, ids)) {
    std::cerr << "[Ingest] Failed to insert texts into SQLite." << '\n';
    return false;
  }
  std::cout << "[Ingest] Inserted " << vectors.size()
            << " id->text mappings into SQLite: " << sqlite_db_path
            << '\n';
  return true;
}

bool build_faiss_index_from_vectors_file(const std::string& vectors_path,
                                         const std::string& faiss_index_path) {
  auto vectors = load_vectors_file(vectors_path);
  if (vectors.empty()) {
    std::cerr << "[Ingest] No vectors loaded from " << vectors_path << '\n';
    return false;
  }
  std::vector<int64_t> ids;
  ids.reserve(vectors.size());
  for (int64_t i = 0; i < static_cast<int64_t>(vectors.size()); ++i) {
    ids.push_back(i);
  }
  auto index = std::make_shared<FaissIndex>();
  if (!index->add_vectors(vectors, ids)) {
    std::cerr << "[Ingest] Failed to add vectors to FAISS index." << '\n';
    return false;
  }
  if (!index->save_index(faiss_index_path)) {
    std::cerr << "[Ingest] Failed to save FAISS index: " << faiss_index_path
              << '\n';
    return false;
  }
  std::cout << "[Ingest] FAISS index saved to " << faiss_index_path << '\n';
  return true;
}

bool sqlite_from_pairs(
    const std::vector<std::pair<std::vector<float>, std::string>>& items,
    const std::string& sqlite_db_path) {
  if (items.empty()) {
    std::cerr << "[Ingest] sqlite_from_pairs: empty items" << '\n';
    return false;
  }
  // Extract texts and validate dims (optional)
  const int d = static_cast<int>(items.front().first.size());
  std::vector<std::string> texts;
  texts.reserve(items.size());
  for (const auto& it : items) {
    if (static_cast<int>(it.first.size()) != d) {
      std::cerr << "[Ingest] sqlite_from_pairs: inconsistent dims" << '\n';
      return false;
    }
    texts.push_back(it.second);
  }
  std::vector<int64_t> ids;
  ids.reserve(items.size());
  for (int64_t i = 0; i < static_cast<int64_t>(items.size()); ++i) {
    ids.push_back(i);
  }
  SqliteVectorDB db(sqlite_db_path);
  if (!db.add_texts(texts, ids)) {
    std::cerr << "[Ingest] sqlite_from_pairs: add_texts failed" << '\n';
    return false;
  }
  std::cout << "[Ingest] sqlite_from_pairs: wrote " << items.size()
            << " id->text mappings to " << sqlite_db_path << '\n';
  return true;
}

bool faiss_from_pairs(
    const std::vector<std::pair<std::vector<float>, std::string>>& items,
    const std::string& faiss_index_path) {
  if (items.empty()) {
    std::cerr << "[Ingest] faiss_from_pairs: empty items" << '\n';
    return false;
  }
  const int d = static_cast<int>(items.front().first.size());
  std::vector<std::vector<float>> vectors;
  vectors.reserve(items.size());
  for (const auto& it : items) {
    if (static_cast<int>(it.first.size()) != d) {
      std::cerr << "[Ingest] faiss_from_pairs: inconsistent dims" << '\n';
      return false;
    }
    vectors.push_back(it.first);
  }
  std::vector<int64_t> ids;
  ids.reserve(items.size());
  for (int64_t i = 0; i < static_cast<int64_t>(items.size()); ++i) {
    ids.push_back(i);
  }
  auto index = std::make_shared<FaissIndex>(); // default Flat, IP
  if (!index->add_vectors(vectors, ids)) {
    std::cerr << "[Ingest] faiss_from_pairs: add_vectors failed" << '\n';
    return false;
  }
  if (!index->save_index(faiss_index_path)) {
    std::cerr << "[Ingest] faiss_from_pairs: save_index failed" << '\n';
    return false;
  }
  std::cout << "[Ingest] faiss_from_pairs: index saved to "
            << faiss_index_path << '\n';
  return true;
}

bool faiss_from_pairs(
    const std::vector<std::pair<std::vector<float>, std::string>>& items,
    const std::string& faiss_index_path,
    const std::string& factory_desc,
    faiss::MetricType metric) {
  if (items.empty()) {
    std::cerr << "[Ingest] faiss_from_pairs(cfg): empty items" << '\n';
    return false;
  }
  const int d = static_cast<int>(items.front().first.size());
  std::vector<std::vector<float>> vectors;
  vectors.reserve(items.size());
  for (const auto& it : items) {
    if (static_cast<int>(it.first.size()) != d) {
      std::cerr << "[Ingest] faiss_from_pairs(cfg): inconsistent dims" << '\n';
      return false;
    }
    vectors.push_back(it.first);
  }
  std::vector<int64_t> ids;
  ids.reserve(items.size());
  for (int64_t i = 0; i < static_cast<int64_t>(items.size()); ++i) {
    ids.push_back(i);
  }
  auto index = std::make_shared<FaissIndex>(factory_desc, metric);
  if (!index->add_vectors(vectors, ids)) {
    std::cerr << "[Ingest] faiss_from_pairs(cfg): add_vectors failed" << '\n';
    return false;
  }
  if (!index->save_index(faiss_index_path)) {
    std::cerr << "[Ingest] faiss_from_pairs(cfg): save_index failed" << '\n';
    return false;
  }
  std::cout << "[Ingest] faiss_from_pairs(cfg): index saved to "
            << faiss_index_path << " with factory='" << factory_desc << "'\n";
  return true;
}

bool build_and_persist_from_pairs(
    const std::vector<std::pair<std::vector<float>, std::string>>& items,
    const std::string& sqlite_db_path,
    const std::string& faiss_index_path) {
  if (!sqlite_from_pairs(items, sqlite_db_path)) {
    return false;
  }
  if (!faiss_from_pairs(items, faiss_index_path)) {
    return false;
  }
  return true;
}

bool build_and_persist_from_pairs(
    const std::vector<std::pair<std::vector<float>, std::string>>& items,
    const std::string& sqlite_db_path,
    const std::string& faiss_index_path,
    const std::string& factory_desc,
    faiss::MetricType metric) {
  if (!sqlite_from_pairs(items, sqlite_db_path)) {
    return false;
  }
  if (!faiss_from_pairs(items, faiss_index_path, factory_desc, metric)) {
    return false;
  }
  return true;
}

bool build_faiss_index_from_vectors_file(
    const std::string& vectors_path,
    const std::string& faiss_index_path,
    const std::string& factory_desc,
    faiss::MetricType metric) {
  auto vectors = load_vectors_file(vectors_path);
  if (vectors.empty()) {
    std::cerr << "[Ingest] No vectors loaded from " << vectors_path << '\n';
    return false;
  }
  std::vector<int64_t> ids;
  ids.reserve(vectors.size());
  for (int64_t i = 0; i < static_cast<int64_t>(vectors.size()); ++i) {
    ids.push_back(i);
  }
  auto index = std::make_shared<FaissIndex>(factory_desc, metric);
  if (!index->add_vectors(vectors, ids)) {
    std::cerr << "[Ingest] build_faiss_index_from_vectors_file(cfg): add_vectors failed" << '\n';
    return false;
  }
  if (!index->save_index(faiss_index_path)) {
    std::cerr << "[Ingest] build_faiss_index_from_vectors_file(cfg): save_index failed" << '\n';
    return false;
  }
  std::cout << "[Ingest] FAISS index saved to " << faiss_index_path
            << " with factory='" << factory_desc << "'\n";
  return true;
}

bool rebuild_faiss_index_from_sqlite_by_chunk_states(
    const std::string& sqlite_db_path,
    const std::string& faiss_index_path,
    const std::vector<ChunkState>& allowed_states,
    const std::string& factory_desc,
    faiss::MetricType metric) {
  if (sqlite_db_path.empty() || faiss_index_path.empty() || allowed_states.empty()) {
    std::cerr << "[Ingest] rebuild_faiss_index_from_sqlite_by_chunk_states: invalid arguments"
              << '\n';
    return false;
  }

  SqliteVectorDB db(sqlite_db_path);
  const int dimension = db.get_vector_dimension();
  if (dimension <= 0) {
    std::cerr << "[Ingest] rebuild_faiss_index_from_sqlite_by_chunk_states: no vectors in SQLite"
              << '\n';
    return false;
  }

  const auto filtered_rows = db.load_vectors_by_chunk_states(allowed_states);
  std::vector<std::vector<float>> vectors;
  std::vector<int64_t> ids;
  vectors.reserve(filtered_rows.size());
  ids.reserve(filtered_rows.size());
  for (const auto& [id, vector] : filtered_rows) {
    ids.push_back(id);
    vectors.push_back(vector);
  }

  auto index = std::make_shared<FaissIndex>(factory_desc, metric);
  bool initialized = false;
  if (vectors.empty()) {
    initialized = index->initialize_empty(dimension);
  } else {
    initialized = index->add_vectors(vectors, ids);
  }
  if (!initialized) {
    std::cerr << "[Ingest] rebuild_faiss_index_from_sqlite_by_chunk_states: failed to initialize Faiss index"
              << '\n';
    return false;
  }

  if (!index->save_index(faiss_index_path)) {
    std::cerr << "[Ingest] rebuild_faiss_index_from_sqlite_by_chunk_states: failed to save index"
              << '\n';
    return false;
  }

  std::cout << "[Ingest] Rebuilt state-filtered FAISS index at "
            << faiss_index_path
            << " with " << filtered_rows.size() << " vectors\n";
  return true;
}

}  // namespace mobile_rag


