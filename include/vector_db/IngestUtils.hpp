#pragma once

#include <string>
#include <utility>
#include <vector>

#include "faiss/MetricType.h"
#include "vector_db/SqliteVectorDB.hpp"

namespace mobile_rag {

// Ingest vectors and metadata strings into SQLite database.
// - vectors_path: path to vectors.bin produced by VectorDataGenerator
// - metadata_path: path to metadata.txt produced by VectorDataGenerator
// - sqlite_db_path: path to SQLite database file to create/populate
bool ingest_from_files_to_sqlite(const std::string& vectors_path,
                                 const std::string& metadata_path,
                                 const std::string& sqlite_db_path);

// Build a FAISS index directly from vectors.bin and persist it.
// - vectors_path: path to vectors.bin (count + dim header, then raw floats)
// - faiss_index_path: output path for FAISS index file
bool build_faiss_index_from_vectors_file(const std::string& vectors_path,
                                         const std::string& faiss_index_path);

// Persist id->text mapping to SQLite from pairs {embedding, text}.
// IDs assigned in ascending order [0..n-1].
bool sqlite_from_pairs(
    const std::vector<std::pair<std::vector<float>, std::string>>& items,
    const std::string& sqlite_db_path);

// Build a FAISS index from pairs {embedding, text} and persist it.
// IDs assigned in ascending order [0..n-1].
bool faiss_from_pairs(
    const std::vector<std::pair<std::vector<float>, std::string>>& items,
    const std::string& faiss_index_path);

// Convenience: Do both steps (SQLite mapping + FAISS index save).
bool build_and_persist_from_pairs(
    const std::vector<std::pair<std::vector<float>, std::string>>& items,
    const std::string& sqlite_db_path,
    const std::string& faiss_index_path);

// Overloads with index config: factory description and metric
bool faiss_from_pairs(
    const std::vector<std::pair<std::vector<float>, std::string>>& items,
    const std::string& faiss_index_path,
    const std::string& factory_desc,
    faiss::MetricType metric);

bool build_and_persist_from_pairs(
    const std::vector<std::pair<std::vector<float>, std::string>>& items,
    const std::string& sqlite_db_path,
    const std::string& faiss_index_path,
    const std::string& factory_desc,
    faiss::MetricType metric);

bool build_faiss_index_from_vectors_file(
    const std::string& vectors_path,
    const std::string& faiss_index_path,
    const std::string& factory_desc,
    faiss::MetricType metric);

bool rebuild_faiss_index_from_sqlite_by_chunk_states(
    const std::string& sqlite_db_path,
    const std::string& faiss_index_path,
    const std::vector<ChunkState>& allowed_states,
    const std::string& factory_desc = "Flat",
    faiss::MetricType metric = faiss::METRIC_INNER_PRODUCT);

}  // namespace mobile_rag


