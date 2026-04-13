#include "vector_Index/FaissIndex.hpp"

#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include "faiss/Index.h"
#include "faiss/index_factory.h"
#include "faiss/IndexIDMap.h"

namespace mobile_rag {

FaissIndex::FaissIndex()
    : factory_desc_("Flat"), metric_(faiss::METRIC_INNER_PRODUCT) {}

FaissIndex::FaissIndex(std::string factory_desc, faiss::MetricType metric)
    : factory_desc_(std::move(factory_desc)), metric_(metric) {
  if (factory_desc_.empty()) {
    factory_desc_ = "Flat";
  }
}

bool FaissIndex::ensure_index_created(int dimension) {
  if (index_) return true;
  try {
    // Build base index via factory and wrap with IDMap to support add_with_ids
    faiss::Index* base = faiss::index_factory(dimension, factory_desc_.c_str(), metric_);
    if (!base) {
      std::cerr << "[FaissIndex] index_factory returned null for: " << factory_desc_ << '\n';
      return false;
    }
    auto idmap = std::make_unique<faiss::IndexIDMap>(base);
    index_.reset(idmap.release());
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[FaissIndex] ensure_index_created failed: " << e.what() << '\n';
  } catch (...) {
    std::cerr << "[FaissIndex] ensure_index_created failed: unknown error" << '\n';
  }
  return false;
}

bool FaissIndex::train_if_needed(const std::vector<std::vector<float>>& vectors) {
  if (!index_) return false;
  if (index_->is_trained) return true;
  // Flatten training data
  const int d = index_->d;
  const size_t n = vectors.size();
  std::vector<float> flat;
  flat.reserve(n * static_cast<size_t>(d));
  for (const auto& v : vectors) {
    if (static_cast<int>(v.size()) != d) {
      std::cerr << "[FaissIndex] train_if_needed: dim mismatch during training" << '\n';
      return false;
    }
    flat.insert(flat.end(), v.begin(), v.end());
  }
  try {
    index_->train(static_cast<faiss::idx_t>(n), flat.data());
    if (!index_->is_trained) {
      std::cerr << "[FaissIndex] Training did not complete." << '\n';
      return false;
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[FaissIndex] train failed: " << e.what() << '\n';
  } catch (...) {
    std::cerr << "[FaissIndex] train failed: unknown error" << '\n';
  }
  return false;
}

bool FaissIndex::add_vectors(const std::vector<std::vector<float>>& vectors,
                             const std::vector<int64_t>& ids) {
  if (vectors.empty()) {
    std::cerr << "[FaissIndex] No vectors to add." << '\n';
    return false;
  }
  if (vectors.size() != ids.size()) {
    std::cerr << "[FaissIndex] vectors.size() != ids.size()" << '\n';
    return false;
  }

  const int d = static_cast<int>(vectors.front().size());
  for (const auto& v : vectors) {
    if (static_cast<int>(v.size()) != d) {
      std::cerr << "[FaissIndex] Inconsistent vector dimensionality." << '\n';
      return false;
    }
  }

  // Initialize index lazily (using factory)
  if (!ensure_index_created(d)) {
    return false;
  } else {
    if (index_->d != d) {
      std::cerr << "[FaissIndex] Dimension mismatch. Index d=" << index_->d
                << ", vectors d=" << d << '\n';
      return false;
    }
  }

  // Train if needed (IVF/PQ, etc.)
  if (!train_if_needed(vectors)) {
    if (!index_->is_trained) {
      std::cerr << "[FaissIndex] Index is not trained and training failed." << '\n';
      return false;
    }
  }

  const size_t n = vectors.size();
  std::vector<float> flat;
  flat.reserve(n * static_cast<size_t>(d));
  for (const auto& v : vectors) {
    flat.insert(flat.end(), v.begin(), v.end());
  }

  // faiss::idx_t may differ from int64_t on some platforms
  std::vector<faiss::idx_t> faiss_ids(ids.begin(), ids.end());

  try {
    index_->add_with_ids(static_cast<faiss::idx_t>(n), flat.data(),
                         faiss_ids.data());
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[FaissIndex] add_with_ids failed: " << e.what() << '\n';
  } catch (...) {
    std::cerr << "[FaissIndex] add_with_ids failed: unknown error" << '\n';
  }
  return false;
}

std::vector<std::pair<int64_t, float>> FaissIndex::search(
    const std::vector<float>& query_vector, int k) {
  std::vector<std::pair<int64_t, float>> results;

  if (!index_) {
    std::cerr << "[FaissIndex] search called with no index built." << '\n';
    return results;
  }
  if (k <= 0) return results;
  if (static_cast<int>(query_vector.size()) != index_->d) {
    std::cerr << "[FaissIndex] Query dimension mismatch. Index d=" << index_->d
              << ", query d=" << query_vector.size() << '\n';
    return results;
  }

  std::vector<float> distances(static_cast<size_t>(k));
  std::vector<faiss::idx_t> labels(static_cast<size_t>(k));
  try {
    index_->search(1, query_vector.data(), k, distances.data(), labels.data());
  } catch (const std::exception& e) {
    std::cerr << "[FaissIndex] search failed: " << e.what() << '\n';
    return results;
  } catch (...) {
    std::cerr << "[FaissIndex] search failed: unknown error" << '\n';
    return results;
  }

  results.reserve(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) {
    if (labels[static_cast<size_t>(i)] < 0) continue;  // invalid label
    results.emplace_back(static_cast<int64_t>(labels[static_cast<size_t>(i)]),
                         distances[static_cast<size_t>(i)]);
  }
  return results;
}

bool FaissIndex::initialize_empty(int dimension) {
  if (dimension <= 0) {
    std::cerr << "[FaissIndex] Refusing to initialize empty index with invalid dimension." << '\n';
    return false;
  }
  return ensure_index_created(dimension);
}

bool FaissIndex::save_index(const std::string& index_path) {
  if (!index_) {
    std::cerr << "[FaissIndex] No index to save." << '\n';
    return false;
  }
  try {
    faiss::write_index(index_.get(), index_path.c_str());
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[FaissIndex] Failed to save index: " << e.what() << '\n';
  } catch (...) {
    std::cerr << "[FaissIndex] Failed to save index: unknown error" << '\n';
  }
  return false;
}

bool FaissIndex::load_index(const std::string& index_path) {
  try {
    faiss::Index* idx = faiss::read_index(index_path.c_str());
    index_.reset(idx);
    return index_ != nullptr;
  } catch (const std::exception& e) {
    std::cerr << "[FaissIndex] Failed to load index: " << e.what() << '\n';
  } catch (...) {
    std::cerr << "[FaissIndex] Failed to load index: unknown error" << '\n';
  }
  return false;
}

}  // namespace mobile_rag


