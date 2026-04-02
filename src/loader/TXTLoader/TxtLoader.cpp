#include "TxtLoader.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <vector>

namespace mobile_rag {

namespace {

static inline bool has_txt_extension(const std::string& path) {
  auto dot_pos = path.find_last_of('.');
  if (dot_pos == std::string::npos) return false;
  std::string ext = path.substr(dot_pos);
  std::string lower_ext;
  lower_ext.resize(ext.size());
  std::transform(ext.begin(), ext.end(), lower_ext.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return lower_ext == ".txt";
}

static inline std::string read_file_text(const std::string& file_path) {
  std::ifstream in(file_path, std::ios::in | std::ios::binary);
  if (!in) return {};
  in.seekg(0, std::ios::end);
  std::streampos size = in.tellg();
  if (size < 0) size = 0;
  std::string content;
  content.resize(static_cast<size_t>(size));
  in.seekg(0, std::ios::beg);
  if (size > 0) in.read(&content[0], size);
  return content;
}

}  // namespace

std::vector<std::string> TxtLoader::load_texts(const std::string& path) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path p(path);

  if (fs::is_regular_file(p, ec)) {
    std::vector<std::string> texts;
    if (has_txt_extension(path)) {
      auto content = read_file_text(path);
      if (!content.empty()) texts.push_back(std::move(content));
    }
    return texts;
  }

  if (fs::is_directory(p, ec)) {
    std::vector<std::string> files;
    for (auto& entry : fs::recursive_directory_iterator(p, ec)) {
      if (entry.is_regular_file(ec)) {
        auto file_path = entry.path().string();
        if (has_txt_extension(file_path)) {
          files.push_back(std::move(file_path));
        }
      }
    }

    if (files.empty()) {
      return {};
    }

    std::sort(files.begin(), files.end());

    const size_t worker_count =
        std::min(static_cast<size_t>(std::max(1u, num_threads_)), files.size());

    if (worker_count <= 1) {
      std::vector<std::string> texts;
      texts.reserve(files.size());
      for (const auto& file_path : files) {
        auto content = read_file_text(file_path);
        if (!content.empty()) {
          texts.push_back(std::move(content));
        }
      }
      return texts;
    }

    std::vector<std::future<std::vector<std::string>>> futures;
    futures.reserve(worker_count);

    const size_t batch_size = (files.size() + worker_count - 1) / worker_count;
    for (size_t worker = 0; worker < worker_count; ++worker) {
      const size_t begin = worker * batch_size;
      const size_t end = std::min(begin + batch_size, files.size());
      if (begin >= end) {
        break;
      }

      futures.push_back(std::async(std::launch::async, [begin, end, &files]() {
        std::vector<std::string> batch_texts;
        batch_texts.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
          auto content = read_file_text(files[i]);
          if (!content.empty()) {
            batch_texts.push_back(std::move(content));
          }
        }
        return batch_texts;
      }));
    }

    std::vector<std::string> texts;
    texts.reserve(files.size());
    for (auto& future : futures) {
      auto batch_texts = future.get();
      texts.insert(texts.end(),
                   std::make_move_iterator(batch_texts.begin()),
                   std::make_move_iterator(batch_texts.end()));
    }

    return texts;
  }

  return {};
}

}  // namespace mobile_rag


