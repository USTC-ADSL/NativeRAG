# Native RAG

NativeRAG：移动端 RAG 系统

支持多种大语言模型后端（Llama.cpp / MNN / MLLM）

支持多种向量索引方式（基于 Faiss）

## 0. ✅ 使用前提与快速开始

### 使用前提
- **运行环境**: CMake ≥ 3.15，GCC/Clang，Linux/macOS/Android NDK(仅安卓)。
- **第三方库**: 默认使用 `prebuilt/` 中的预编译库；如需自编译，参考下文“编译与构建”。
- **模型文件**:
  - Embedding 模型：通过 `--embedding-model` 指定路径。
  - LLM 模型：通过 `--llm-model` 指定路径（查询阶段需要）。
- **数据源**:
  - 文本文件模式：当前支持 `.txt`；会在构建索引时按字符切分。
  - 数据集模式：输入 JSON（TrivialQA 类似），每条样本包含 `id`、`query`、`answers`、`documents`（字符串数组）。
    例如：
    ```json
    {
      "id": "sample-001",
      "query": "What is AI?",
      "answers": ["Artificial Intelligence", "AI"],
      "documents": ["doc paragraph 1 ...", "doc paragraph 2 ..."]
    }
    ```
- **模型下载**:
  - 推荐使用 `huggingface` CLI（`pip install -U huggingface_hub` 并登录 `huggingface-cli login`），可直接把模型拉取到本地。
  - 下载完成后，将 `config.json`/权重文件路径分别传给 `--embedding-model` 与 `--llm-model` 参数。

  **Embedding（MNN）示例：**

  ```bash
  # shell download
  huggingface download \
    --model 'taobao-mnn/Qwen3-Embedding-0.6B-MNN' \
    --local_dir './models/emb_qwen3_mnn'

  # 构建/查询时指向模型目录
  ./mobile_rag_cli --build \
    --text-path documents.txt \
    --embedding-model ./models/emb_qwen3_mnn/config.json \
    --index-path ./faiss_index.bin
  ```

  **LLM（MNN / llama.cpp）示例：**

  ```bash
  # MNN 版 Qwen 指令模型
  huggingface download \
    --model 'taobao-mnn/Qwen2-1.5B-Instruct-MNN' \
    --local_dir './models/llm_qwen2_mnn'

  # llama.cpp GGUF 权重
  huggingface download \
    --model 'QuantFactory/Meta-Llama-3-Int4' \
    --local_dir './models/llm_llama3_gguf'

  # 运行查询
  ./mobile_rag_cli --query "What is AI?" \
    --llm-model ./models/llm_llama3_gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    --embedding-model ./models/emb_qwen3_mnn/config.json \
    --index-path ./faiss_index.bin
  ```

  > 若使用 MLLM 后端，请确保下载后的权重路径传入 `--llm-model`，并在 `cmake .. -DLLM_BACKEND=MLLM` 时启用对应后端。
- **切分与进度**:
  - 字符切分：默认 `chunk_size=1000`，`overlap=200`（与 `TextFileLoader` 一致）。
  - 嵌入批量：默认批大小 `64`；构建索引时会打印进度日志：`Embedded X/Y (Z%)`。
- **向量索引与持久化**: 使用 Faiss 建索引；(ChunkID -> ChunkText) 会写入 SQLite，便于检索原文。

### 快速开始
- 使用预编译库快速构建：
  ```bash
  mkdir -p build && cd build
  cmake .. -DUSE_PREBUILT=ON
  make -j$(nproc)
  ```

- 从文本文件构建索引（自动切分 + 进度提示）：
  ```bash
  ./mobile_rag_cli --build \
    --text-path documents.txt \
    --embedding-model ./models/emb/config.json \
    --index-path ./faiss_index.bin \
    --verbose
  ```
- 从数据集构建索引（对 documents 逐条切分 + 进度提示）：
  ```bash
  ./mobile_rag_dataset --build \
    --dataset-path dataset/data.json \
    --embedding-model ./models/emb/config.json \
    --index-path ./faiss_index.bin \
    --verbose
  ```
- 执行查询（复用索引 + LLM 推理）：
  ```bash
  ./mobile_rag_cli --query "What is AI?" \
    --llm-model ./models/qwen/config.json \
    --embedding-model ./models/emb/config.json \
    --index-path ./faiss_index.bin \
    --verbose
  ```

## 1. 🎯 目标与背景

### 1.1 项目目标

- 原生实现： 构建一个**纯C++**的、面向端侧（On-Device）的原生RAG（Retrieval-Augmented Generation）框架。

- 可评估性： 框架设计必须支持插拔式评估。能够灵活替换和评测不同的向量索引库（如 Faiss, sqlite-vec）和LLM推理后端（如 MLLM, MNN, Llama.cpp）。

- 交付物： 第一阶段交付一个功能完整的C++命令行工具（CLI），用于验证完整的RAG工作流。

## 2. 核心架构：抽象接口

为实现可评估性与解耦，项目将基于以下四大核心C++抽象接口（Interfaces）构建：

- IDocumentLoader

    - 职责： 负责从文件加载原始数据，并将其切分为文本块（Chunks）。

    - 实现： StandardDocumentLoader (封装 purecpp 库)。

- IEmbeddingModel

    - 职责： 负责加载嵌入模型，并将文本（Chunks或Query）转换为Embedding向量。

    - 实现： MNNEmbedding (使用 MNN 作为推理后端)。

- IVectorDB

    - 职责： 负责向量的存储、索引构建、持久化加载以及搜索。

    - 实现： FaissDB, SqliteVecDB。

- ILargeLanguageModel

    - 职责： 负责加载LLM，并根据Full Prompt推理生成最终答案。

    - 实现： MLLMBackend, LlamaCppBackend。

## 3. 🔨 编译与构建

### 3.1 两种集成方式

项目支持两种集成方式，用户可根据需求选择：

#### **方式1: 使用 Prebuilt 库（推荐，快速）**

项目已预编译了所有第三方库，存放在 `prebuilt/` 目录中。这是最快的方式，适合大多数用户。

**目录结构：**
```
prebuilt/
├── include/              # 统一的头文件（所有平台共用）
│   ├── faiss/           # Faiss 向量索引库头文件
│   ├── MNN/             # MNN 推理框架头文件
│   └── llama/           # llama.cpp 推理框架头文件
├── linux-x86_64/        # Linux x86_64 平台库
│   ├── faiss/           # libfaiss.so
│   ├── MNN/             # libMNN.so, libMNN_Express.so
│   └── llama/           # libllama.so, libggml*.so
└── android-aarch64/     # Android aarch64 平台库
    ├── faiss/           # libfaiss.so
    ├── MNN/             # libMNN_Express.so
    └── llama/           # libllama.so, libggml*.so
```

**快速编译：**

```bash
mkdir build && cd build
cmake .. -DUSE_PREBUILT=ON
make -j$(nproc)
```

#### **方式2: 手动编译第三方库**

如需自定义编译选项或修改第三方库源码，可使用手动编译方式。

**编译脚本：**

```bash
# 编译所有第三方库
./build_thirdparty.sh --linux --all

# 编译 Android 平台
./build_thirdparty.sh --android --all

# 仅编译特定库
./build_thirdparty.sh --linux --faiss
./build_thirdparty.sh --linux --mnn
./build_thirdparty.sh --linux --llama

# 查看帮助
./build_thirdparty.sh --help
```

**编译依赖：**

- CMake >= 3.15
- GCC/Clang 编译器
- Android NDK（仅编译 Android 平台时需要）
- OpenBLAS（用于 Faiss 编译）

**编译项目：**

```bash
mkdir build && cd build
cmake .. -DUSE_PREBUILT=OFF
make -j$(nproc)
```

### 3.2 编译选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `USE_PREBUILT` | ON | 使用 prebuilt 库（ON）或 third_party 库（OFF） |
| `LLM_BACKEND` | MNN | LLM 后端：MNN / MLLM / LlamaCpp |
| `VECTOR_INDEX` | Faiss | 向量索引：Faiss / None |
| `BUILD_TESTS` | ON | 编译单元测试 |

**编译示例：**

```bash
# 使用 Prebuilt + MNN + Faiss（默认）
mkdir build && cd build
cmake .. -DUSE_PREBUILT=ON
make -j$(nproc)

# 使用 Prebuilt + LlamaCpp + Faiss
mkdir build && cd build
cmake .. -DUSE_PREBUILT=ON -DLLM_BACKEND=LlamaCpp
make -j$(nproc)

# 手动编译 + MLLM + Faiss
./build_thirdparty.sh --linux --all
mkdir build && cd build
cmake .. -DUSE_PREBUILT=OFF -DLLM_BACKEND=MLLM
make -j$(nproc)
```

### 3.3 运行

```bash
# 运行主程序
./mobile_rag_cli
```

### 3.4 两阶段 Pipeline 使用指南

NativeRAG 采用**两阶段设计**，将 RAG 工作流分为离线阶段和查询阶段，支持索引的持久化和复用。

#### **阶段一：离线构建 (Offline Indexing Phase)**

离线阶段执行 Step 1-3，构建并保存索引到磁盘。

```bash
# 从文本文件构建索引
./mobile_rag_cli --build \
  --text-path documents.txt \
  --llm-model ./models/qwen/config.json \
  --embedding-model ./models/emb/config.json \
  --index-path ./faiss_index.bin \
  --verbose

# 从数据集构建索引
./mobile_rag_dataset --build \
  --dataset-path dataset/data.json \
  --llm-model ./models/qwen/config.json \
  --embedding-model ./models/emb/config.json \
  --index-path ./faiss_index.bin \
  --verbose
```

**输出：**
- `./faiss_index.bin` - 持久化的 Faiss 向量索引
- 内存中的元数据映射（ChunkID -> ChunkText）

#### **阶段二：在线查询 (Online Query Phase)**

查询阶段执行 Step 4-7，从磁盘加载索引并处理用户查询。

```bash
# 单次查询
./mobile_rag_cli --query "What is AI?" \
  --llm-model ./models/qwen/config.json \
  --embedding-model ./models/emb/config.json \
  --index-path ./faiss_index.bin \
  --verbose

# 交互式查询
./mobile_rag_cli --interactive \
  --llm-model ./models/qwen/config.json \
  --embedding-model ./models/emb/config.json \
  --index-path ./faiss_index.bin \
  --verbose
```

#### **关键参数说明**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--index-path <path>` | 索引文件保存/加载路径 | `./faiss_index.bin` |
| `--save-index` | 构建后是否保存索引（默认启用） | 启用 |
| `--no-save-index` | 禁用索引保存 | - |
| `--load-index` | 查询前是否加载索引（默认启用） | 启用 |
| `--no-load-index` | 禁用索引加载 | - |

#### **工作流示例**

```bash
# 1. 离线阶段：构建索引（一次性）
./mobile_rag_cli --build \
  --text-path documents.txt \
  --llm-model ./models/qwen/config.json \
  --embedding-model ./models/emb/config.json \
  --index-path ./my_index.bin

# 2. 查询阶段：多次查询（复用索引）
./mobile_rag_cli --query "Question 1?" \
  --llm-model ./models/qwen/config.json \
  --embedding-model ./models/emb/config.json \
  --index-path ./my_index.bin

./mobile_rag_cli --query "Question 2?" \
  --llm-model ./models/qwen/config.json \
  --embedding-model ./models/emb/config.json \
  --index-path ./my_index.bin

# 3. 交互式查询（复用索引）
./mobile_rag_cli --interactive \
  --llm-model ./models/qwen/config.json \
  --embedding-model ./models/emb/config.json \
  --index-path ./my_index.bin
```

## 4. ⚙️ RAG 工作流 (Workflow)

整个工作流分为“构建”和“查询”两个阶段：

### 4.1 阶段一：构建 (Indexing / Build Phase)

此阶段为离线处理，用于构建索引。

- Step 1: 文档加载与切分 (Loading & Splitting)

    - 输入： 原始文件 (File)

    - 动作： 调用 IDocumentLoader 接口（如 StandardDocumentLoader），读取文件内容并将其切分为 Document Chunks。

    - 输出： std::vector<std::string> (文本块列表)

- Step 2: 文本向量化 (Text Embedding)

    - 输入： Document Chunks (文本块列表)

    - 动作： 调用 IEmbeddingModel 接口（如 MNNEmbedding），将所有文本块批量转换为Embedding向量。

    - 输出： `std::vector<std::vector<float>>` (Embedding向量列表)

- Step 3: 向量索引与存储 (Vector Indexing & Storage)

    - 输入： Embedding向量列表 及对应的 Document Chunks

    - 动作：

        1. 元数据存储： 使用 SQLite 存储 (ChunkID, ChunkText) 键值对，用于后续原文检索。

        2. 向量索引： 调用 IVectorDB 接口（如 FaissDB），将 (ChunkID, Embedding) 批量添加到索引中（例如使用PQ量化）。

        3. 持久化： 将构建好的Faiss索引 (.index) 和SQLite数据库 (.db) 保存到磁盘。

    - 输出： 持久化的索引文件和元数据文件。

### 4.2 阶段二：查询 (Query Phase)

此阶段为在线推理，用于响应用户查询。

- Step 4: 查询向量化 (Query Embedding)

    - 输入： 用户的 Query (字符串)

    - 动作： 复用 IEmbeddingModel 接口 (与Step 2相同)，将单条Query文本转换为Embedding向量。

    - 输出： std::vector<float> (Query向量)

- Step 5: 向量检索 (Vector Search)

    - 输入： Query向量

    - 动作： 调用 IVectorDB->search() 接口，从加载的Faiss索引中检索Top-K个最相似向量，并返回它们对应的 ChunkID。

    - 输出： std::vector<int64_t> (ChunkID 列表)

- Step 6: 上下文拼接 (Context Retrieval & Prompting)

    - 输入： ChunkID 列表 和 原始 Query

    - 动作：

        1. 检索原文： 遍历 ChunkID 列表，从SQLite数据库中查询（Lookup）对应的 ChunkText。

        2. 构建Prompt： 将检索到的所有 ChunkText（作为上下文）与原始 Query 按照LLM模板拼接成 Full Prompt。

    - 输出： Full Prompt (字符串)

- Step 7: LLM 推理生成 (LLM Inference)

- 输入： Full Prompt

- 动作： 调用 ILargeLanguageModel 接口（如 LlamaCppBackend），执行LLM推理。

- 输出： 最终答案 (字符串)

## 5. 🔜 后续规划 & 协作方式

核心链路已打通，现阶段聚焦特性演进与体验迭代，可按“功能模组负责人”轻量协作：

- **检索与索引负责人**：评估并扩展 Faiss 配置（IVF/HNSW、量化策略）、补齐 sqlite-vec/SQLite 混合索引能力，并沉淀性能对比数据。
- **模型与推理负责人**：维护 MNN/MLLM/llama.cpp 三类后端，新增主流 GGUF/MNN 模型范式、自动量化脚本，以及模型管理（下载/校验）自动化。
- **产品体验负责人**：完善数据导入工具（Markdown/HTML/多语），补充评测脚本、交互式 CLI/UI 体验以及 Android Demo 打包。

每个负责人面向 feature backlog 自主迭代，保持 issue/PR 模式协同即可，无需再划分大规模小组。
