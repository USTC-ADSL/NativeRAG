#include "TxtLoader.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
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
  std::vector<std::string> texts;

  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path p(path);

  if (fs::is_regular_file(p, ec)) {
    if (has_txt_extension(path)) {
      auto content = read_file_text(path);
      if (!content.empty()) texts.push_back(std::move(content));
    }
    return texts;
  }

  if (fs::is_directory(p, ec)) {
    for (auto& entry : fs::recursive_directory_iterator(p, ec)) {
      if (entry.is_regular_file(ec)) {
        auto file_path = entry.path().string();
        if (has_txt_extension(file_path)) {
          auto content = read_file_text(file_path);
          if (!content.empty()) texts.push_back(std::move(content));
        }
      }
    }
  }

  return texts;
}

}  // namespace mobile_rag



