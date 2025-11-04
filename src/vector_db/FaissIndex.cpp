#include "vector_Index/FaissIndex.hpp"

#include <exception>
#include <iostream>

namespace mobile_rag {

bool FaissIndex::add_vectors(const std::vector<std::vector<float>>& /*vectors*/,
                             const std::vector<int64_t>& /*ids*/) {
  // TODO: to be implemented later
  return true;
}

std::vector<std::pair<int64_t, float>> FaissIndex::search(
    const std::vector<float>& /*query_vector*/, int /*k*/) {
  // TODO: to be implemented later
  return {};
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



