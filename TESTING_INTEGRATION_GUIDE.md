# NativeRAG 测试集成指南

本文档说明如何在 NativeRAG 项目中集成数据集和进行单元测试。

## 快速开始 (5 分钟)

### 编译

```bash
cd /path/to/NativeRAG
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
```

### 运行测试

```bash
# 运行所有测试
ctest --output-on-failure

# 或运行特定测试
./tests/test_embedding          # Group A
./tests/test_llm                # Group B
./tests/test_vectordb_integrated # Group C
./tests/test_rag_integration    # 集成测试
```

## 概述

项目提供了一套完整的测试框架，支持各个 Group 的独立测试和集成测试：

- **Group A (Embedding)**: 嵌入模型测试
- **Group B (LLM)**: 大语言模型测试
- **Group C (VectorDB)**: 向量数据库测试
- **集成测试**: 完整的 RAG Pipeline 测试

## 项目结构

```
include/
├── RAGPipelineWithDataset.hpp      # 支持数据集的 RAG Pipeline
├── dataset/
│   ├── IDataset.hpp                # 数据集接口
│   ├── TrivialQADataset.hpp        # TrivialQA 实现
│   └── VectorDataGenerator.hpp     # 向量生成工具
└── ...

src/
├── RAGPipelineWithDataset.cpp      # 实现
└── ...

tests/
├── TestBase.hpp                    # 测试基类
├── test_rag_pipeline_integration.cpp  # 集成测试
├── group_a_embedding/
│   └── test_embedding.cpp          # Group A 测试
├── group_b_llm/
│   └── test_llm.cpp                # Group B 测试
├── group_c_vectordb/
│   ├── test_vectordb.cpp           # Group C 基础测试
│   └── test_vectordb_integrated.cpp # Group C 集成测试
└── CMakeLists.txt                  # 测试编译配置

dataset/
├── data/
│   └── val00-100.json              # TrivialQA 数据集 (100 个样本，643 个文档)
└── src/
    ├── TrivialQADataset.cpp
    └── VectorDataGenerator.cpp
```

## 数据集说明

### TrivialQA 数据集

**位置**: `dataset/data/val00-100.json`

**支持的格式**:

**格式1（新格式）**:
```json
{
  "docs": ["doc1", "doc2", ...],
  "questions": ["question1", "question2", ...],
  "answers": [["answer1", "answer2"], ["answer3"], ...]
}
```

**格式2（旧格式）**:
```json
{
  "docs": ["doc1", "doc2", ...],
  "qas": [
    {
      "id": "q1",
      "question": "What is...",
      "answers": ["answer1", "answer2"],
      "supporting_facts": [[0, 1], [1, 2]]
    }
  ]
}
```

### DataSample 结构

```cpp
struct DataSample {
    std::string id;                      // 样本 ID
    std::string query;                   // 问题
    std::vector<std::string> documents;  // 相关文档
    std::vector<std::string> answers;    // 可能的答案
};
```



## 测试范式

### 1. Group A (Embedding) 测试范式

```cpp
#include "tests/TestBase.hpp"
#include "embedding/MNNEmbedding.hpp"

class EmbeddingTest : public TestBase {
public:
    bool test_embed_documents() {
        // 加载数据集
        if (!load_dataset("dataset/data/val00-100.json")) {
            return false;
        }
        
        // 获取样本
        auto samples = get_samples(5);
        
        // 创建嵌入模型
        auto embedder = std::make_shared<MNNEmbedding>();
        embedder->load_model("");
        
        // 嵌入文档
        std::vector<std::string> documents;
        for (const auto& sample : samples) {
            for (const auto& doc : sample.documents) {
                documents.push_back(doc);
            }
        }
        
        auto embeddings = embedder->embed_documents(documents);
        return embeddings.size() == documents.size();
    }
};
```

### 2. Group B (LLM) 测试范式

```cpp
#include "tests/TestBase.hpp"
#include "llm/LLMFactory.hpp"

class LLMTest : public TestBase {
public:
    bool test_process_queries() {
        // 加载数据集
        if (!load_dataset("dataset/data/val00-100.json")) {
            return false;
        }
        
        // 创建 LLM
        auto llm = create_llm();
        
        // 获取样本
        auto samples = get_samples(3);
        
        // 处理每个样本
        for (const auto& sample : samples) {
            // 构建 prompt
            std::string prompt = llm->build_prompt(
                sample.query, 
                sample.documents
            );
            
            // 生成答案
            std::string answer = llm->generate(prompt);
            
            // 验证答案
            if (answer.empty()) return false;
        }
        
        return true;
    }
};
```

### 3. Group C (VectorDB) 测试范式

```cpp
#include "tests/TestBase.hpp"
#include "vector_Index/FaissIndex.hpp"
#include "dataset/VectorDataGenerator.hpp"

class VectorDBTest : public TestBase {
public:
    bool test_vector_search() {
        // 加载数据集
        if (!load_dataset("dataset/data/val00-100.json")) {
            return false;
        }
        
        // 生成向量
        VectorDataGenerator::generate_from_dataset(
            dataset_, 
            "/tmp/vectors", 
            384
        );
        
        // 加载向量
        auto vectors = VectorDataGenerator::load_vectors(
            "/tmp/vectors/vectors.bin"
        );
        
        // 创建索引
        auto index = std::make_shared<FaissIndex>();
        
        // 添加向量
        std::vector<int64_t> ids;
        for (int64_t i = 0; i < vectors.size(); ++i) {
            ids.push_back(i);
        }
        index->add_vectors(vectors, ids);
        
        // 搜索
        auto results = index->search(vectors[0], 10);
        return !results.empty();
    }
};
```

### 4. 集成测试范式

```cpp
#include "RAGPipelineWithDataset.hpp"
#include "tests/TestBase.hpp"

class IntegrationTest : public TestBase {
public:
    bool test_full_pipeline() {
        // 加载数据集
        if (!load_dataset("dataset/data/val00-100.json")) {
            return false;
        }
        
        // 创建 Pipeline
        auto pipeline = std::make_shared<RAGPipelineWithDataset>(
            loader, embedder, index, llm
        );
        
        // 从数据集构建索引
        pipeline->build_index_from_dataset(dataset_, true);
        
        // 获取查询
        auto queries = pipeline->get_queries_from_dataset(dataset_);
        
        // 测试查询
        for (const auto& query : queries) {
            std::string answer = pipeline->answer_query(query);
            if (answer.empty()) return false;
        }
        
        return true;
    }
};
```

## TestBase 类 API

### 加载数据集

```cpp
bool load_dataset(const std::string& dataset_path);
```

### 获取数据集

```cpp
std::shared_ptr<IDataset> get_dataset() const;
```

### 获取样本

```cpp
// 获取所有样本
std::vector<DataSample> get_samples() const;

// 获取前 N 个样本
std::vector<DataSample> get_samples(size_t count) const;
```

### 打印信息

```cpp
void print_test_info(const std::string& test_name);
void print_result(const std::string& test_name, bool passed);
```

## RAGPipelineWithDataset API

### 从数据集构建索引

```cpp
void build_index_from_dataset(
    const std::shared_ptr<IDataset>& dataset,
    bool use_documents = true
);
```

### 获取查询

```cpp
std::vector<std::string> get_queries_from_dataset(
    const std::shared_ptr<IDataset>& dataset
);
```

### 获取答案

```cpp
std::vector<std::vector<std::string>> get_answers_from_dataset(
    const std::shared_ptr<IDataset>& dataset
);
```



## 常见问题

### Q: 如何添加新的测试？

A: 在 `tests/` 目录下创建新的测试文件，继承 `TestBase` 类，然后在 `tests/CMakeLists.txt` 中添加编译配置。

### Q: 如何使用自己的数据集？

A: 在 `include/dataset/` 中创建新的数据集类，继承 `IDataset` 接口，实现 `load()` 方法。

### Q: 如何跳过某些测试？

A: 在测试中添加条件检查，或者在 CMakeLists.txt 中注释掉相应的测试。

### Q: 测试失败了怎么办？

A: 查看测试输出，检查：
1. 数据集文件是否存在
2. 模型文件是否存在
3. 依赖库是否正确链接
4. 是否有权限访问文件

## 性能指标

- 加载 100 个样本的 TrivialQA 数据集：~100ms
- 生成 64,300 个 384 维向量：~500ms
- 向量搜索 (Top-10)：~20ms
- 嵌入 100 个文档：~1-2s（取决于模型）
- 生成答案：~1-5s（取决于模型）

## 下一步

1. 为各个 Group 实现具体的功能
2. 添加更多的测试用例
3. 集成 CI/CD 流程
4. 性能优化和基准测试

