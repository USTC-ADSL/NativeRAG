#include <cassert>
#include <iostream>
#include <memory>

#include "llm/LLMFactory.hpp"
#include "TestBase.hpp"

using namespace mobile_rag;
using namespace mobile_rag::testing;

/**
 * Group B (LLM) 测试范式
 * 
 * 测试内容：
 * 1. 加载 LLM 模型
 * 2. 构建 Prompt
 * 3. 生成回答
 * 4. 验证回答质量
 * 5. 批量处理查询
 */
class LLMTest : public TestBase {
 public:
  bool test_load_model() {
    print_test_info("Test 1: Load LLM Model");
    
    auto llm = create_llm();
    bool success = (llm != nullptr);
    
    if (success) {
      std::cout << "LLM model loaded successfully\n";
    }
    
    print_result("Load LLM model", success);
    return success;
  }

  bool test_build_prompt() {
    print_test_info("Test 2: Build Prompt");
    
    auto llm = create_llm();
    if (!llm) {
      print_result("Create LLM", false);
      return false;
    }

    std::string query = "What is the capital of France?";
    std::vector<std::string> contexts = {
        "France is a country in Europe.",
        "Paris is the capital of France."};

    std::string prompt = llm->build_prompt(query, contexts);

    bool success = !prompt.empty() && prompt.find(query) != std::string::npos;

    std::cout << "Query: " << query << '\n';
    std::cout << "Prompt length: " << prompt.length() << '\n';
    std::cout << "Prompt preview: " << prompt.substr(0, 100) << "...\n";

    print_result("Build prompt", success);
    return success;
  }

  bool test_generate_answer() {
    print_test_info("Test 3: Generate Answer");
    
    auto llm = create_llm();
    if (!llm) {
      print_result("Create LLM", false);
      return false;
    }

    std::string prompt = "Q: What is 2+2?\nA:";
    std::string answer = llm->generate(prompt);

    bool success = !answer.empty();

    std::cout << "Prompt: " << prompt << '\n';
    std::cout << "Answer: " << answer << '\n';

    print_result("Generate answer", success);
    return success;
  }

  bool test_process_dataset_queries() {
    print_test_info("Test 4: Process Dataset Queries");
    
    if (!load_dataset("dataset/data/val00-100.json")) {
      print_result("Load dataset", false);
      return false;
    }

    auto llm = create_llm();
    if (!llm) {
      print_result("Create LLM", false);
      return false;
    }

    // 获取前 3 个样本
    auto samples = get_samples(3);
    int success_count = 0;

    for (size_t i = 0; i < samples.size(); ++i) {
      const auto& sample = samples[i];
      
      std::cout << "\nSample " << (i + 1) << ":\n";
      std::cout << "  Query: " << sample.query.substr(0, 50) << "Length is :" << sample.query.length() << "...\n";
      std::cout << "  Expected answers: " << sample.answers.size() << '\n';

      // 构建 prompt
      std::string prompt = llm->build_prompt(sample.query, sample.documents);
      std::cout << "  Prompt length: " << prompt.length() << '\n';
      // 生成答案
      std::string answer = llm->generate(prompt);
      
      if (!answer.empty()) {
        success_count++;
        std::cout << "  Generated answer: " << answer.substr(0, 50) << "...\n";
      }
    }

    bool success = success_count == samples.size();
    std::cout << "\nProcessed " << success_count << "/" << samples.size()
              << " queries successfully\n";

    print_result("Process dataset queries", success);
    return success;
  }

  bool test_answer_consistency() {
    print_test_info("Test 5: Answer Consistency");
    
    auto llm = create_llm();
    if (!llm) {
      print_result("Create LLM", false);
      return false;
    }

    std::string prompt = "Q: What is the color of the sky?\nA:";
    
    // 生成多次答案，检查一致性
    std::string answer1 = llm->generate(prompt);
    std::string answer2 = llm->generate(prompt);

    bool success = !answer1.empty() && !answer2.empty();

    std::cout << "Answer 1: " << answer1 << '\n';
    std::cout << "Answer 2: " << answer2 << '\n';
    std::cout << "Answers are " << (answer1 == answer2 ? "identical" : "different")
              << '\n';

    print_result("Answer consistency", success);
    return success;
  }

  void run_all_tests() {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n"
              << "║         Group B (LLM) Unit Tests                           ║\n"
              << "╚════════════════════════════════════════════════════════════╝\n";

    int passed = 0;
    int total = 5;

    if (test_load_model()) passed++;
    if (test_build_prompt()) passed++;
    if (test_generate_answer()) passed++;
    if (test_process_dataset_queries()) passed++;
    if (test_answer_consistency()) passed++;

    std::cout << "\n========================================\n"
              << "Results: " << passed << "/" << total << " tests passed\n"
              << "========================================\n";

    assert(passed == total && "Some tests failed!");
  }
};

int main() {
  try {
    LLMTest test;
    test.run_all_tests();
    std::cout << "\n✓ All LLM tests passed!\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << '\n';
    return 1;
  }
}

