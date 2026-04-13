#pragma once

#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace mobile_rag {

inline std::string trim_copy(std::string text) {
  auto is_space = [](unsigned char ch) {
    return std::isspace(ch) != 0;
  };

  size_t begin = 0;
  while (begin < text.size() && is_space(static_cast<unsigned char>(text[begin]))) {
    ++begin;
  }

  size_t end = text.size();
  while (end > begin && is_space(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }

  return text.substr(begin, end - begin);
}

inline std::string remove_tag_blocks(std::string text,
                                     std::string_view open_tag,
                                     std::string_view close_tag) {
  size_t search_pos = 0;
  while (true) {
    const size_t open_pos = text.find(open_tag, search_pos);
    if (open_pos == std::string::npos) {
      break;
    }

    const size_t close_pos = text.find(close_tag, open_pos + open_tag.size());
    if (close_pos == std::string::npos) {
      text.erase(open_pos);
      break;
    }

    text.erase(open_pos, close_pos + close_tag.size() - open_pos);
    search_pos = open_pos;
  }

  return text;
}

inline std::string to_lower_copy(std::string text) {
  for (char& ch : text) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return text;
}

inline bool looks_like_meta_reasoning_line(const std::string& line) {
  const std::string lowered = to_lower_copy(trim_copy(line));
  if (lowered.empty()) {
    return false;
  }

  return lowered.rfind("okay,", 0) == 0 ||
         lowered.rfind("ok,", 0) == 0 ||
         lowered.rfind("the user is", 0) == 0 ||
         lowered.rfind("the user wants", 0) == 0 ||
         lowered.rfind("looking at", 0) == 0 ||
         lowered.rfind("let me", 0) == 0 ||
         lowered.rfind("i should", 0) == 0 ||
         lowered.rfind("i need", 0) == 0 ||
         lowered.rfind("we need", 0) == 0;
}

inline std::string strip_meta_reasoning_tail(const std::string& text) {
  std::string kept;
  std::string current_line;
  bool has_answer_content = false;

  auto flush_line = [&]() {
    const std::string trimmed = trim_copy(current_line);
    if (trimmed.empty()) {
      if (!kept.empty() && kept.back() != '\n') {
        kept.push_back('\n');
      }
      current_line.clear();
      return false;
    }

    if (has_answer_content && looks_like_meta_reasoning_line(trimmed)) {
      current_line.clear();
      return true;
    }

    if (!kept.empty() && kept.back() != '\n') {
      kept.push_back('\n');
    }
    kept += trimmed;
    has_answer_content = true;
    current_line.clear();
    return false;
  };

  for (char ch : text) {
    if (ch == '\n' || ch == '\r') {
      if (flush_line()) {
        break;
      }
      continue;
    }
    current_line.push_back(ch);
  }

  if (!current_line.empty()) {
    flush_line();
  }

  return trim_copy(kept);
}

inline std::string cleanup_generation_output(const std::string & raw_text) {
  const std::string raw = trim_copy(raw_text);
  if (raw.empty()) {
    return raw;
  }

  std::string cleaned = raw;
  cleaned = remove_tag_blocks(cleaned, "<think>", "</think>");
  cleaned = remove_tag_blocks(cleaned, "<analysis>", "</analysis>");
  cleaned = trim_copy(cleaned);

  const std::string answer_prefix = "Answer:";
  if (cleaned.rfind(answer_prefix, 0) == 0) {
    cleaned = trim_copy(cleaned.substr(answer_prefix.size()));
  }

  cleaned = strip_meta_reasoning_tail(cleaned);

  if (!cleaned.empty()) {
    return cleaned;
  }

  if (raw.find("<think>") == std::string::npos &&
      raw.find("<analysis>") == std::string::npos) {
    return raw;
  }

  return {};
}

inline std::string build_rag_prompt(const std::string& query,
                                    const std::vector<std::string>& contexts) {
  std::string prompt;
  prompt.reserve(query.size() + contexts.size() * 256);

  prompt += "You are a retrieval-augmented assistant.\n";
  prompt += "Use only the reference documents below to answer the question.\n";
  prompt += "If the documents do not contain enough information, say: "
            "\"I don't know based on the provided documents.\"\n";
  prompt += "Do not output reasoning, analysis, or <think> tags.\n";
  prompt += "Return only the final answer in plain text.\n\n";

  if (contexts.empty()) {
    prompt += "Reference documents:\n";
    prompt += "[none]\n\n";
  } else {
    prompt += "Reference documents:\n";
    for (size_t i = 0; i < contexts.size(); ++i) {
      prompt += "[Document " + std::to_string(i + 1) + "]\n";
      prompt += contexts[i];
      prompt += "\n\n";
    }
  }

  prompt += "Question: ";
  prompt += query;
  prompt += "\nAnswer:";

  return prompt;
}

}  // namespace mobile_rag
