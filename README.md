# Native RAG

Native RAG System For Phone

Support different LLM Backend(Llama.cpp/MNN/mllm)

Support different vector index(base on Faiss)

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

## 3. ⚙️ RAG 工作流 (Workflow)

整个工作流分为“构建”和“查询”两个阶段：

### 阶段一：构建 (Indexing / Build Phase)

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

### 阶段二：查询 (Query Phase)

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

4. 👥 预期分工 (Team Breakdown)

为并行推进项目，按以下分工：

- Group A (数据源 -> 向量)

    - 任务： 打通从 File 到 Embedding 的数据通路。

    - 职责： 负责 IDocumentLoader 和 IEmbeddingModel 接口的实现（即 Step 1 & 2）。

    - 技术栈： purecpp, MNN。

- Group B (LLM 后端)

    - 任务： 打通 LLM Backend 的调用通路。

    - 职责： 负责 ILargeLanguageModel 接口的实现（即 Step 7）。

    - 技术栈： MLLM, Llama.cpp。

- Group C (向量 -> 数据库)

    - 任务： 打通从 Embedding 到 向量数据库/索引 的数据通路。

    - 职责： 负责 IVectorDB 接口的实现，支持插入、构建、持久化（即 Step 3）。

    - 技术栈： Faiss, sqlite-vec, SQLite。

- Group D (RAG 核心逻辑)

    - 任务： 打通从 Query 到 Full Prompt 的逻辑通路。

    - 职责： 调用 Group A 和 C 的接口，串联查询阶段的核心逻辑（即 Step 4, 5, 6）。

    - 技术栈： C++ 业务逻辑。

- 系统集成 (Integration)

    - 职责： 负责定义并维护上述C++抽象接口（.hpp文件），编写 RAGPipeline 胶水代码，以及 main.cpp 命令行程序。