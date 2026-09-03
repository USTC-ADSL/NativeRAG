# NativeRAG

NativeRAG 是一个面向 Android `arm64-v8a` 的纯 C++ Local RAG 命令行程序。Embedding、Reranker 与文本生成均由外部提供的同一份 llama.cpp 源码和动态库执行；Faiss 负责第一阶段向量最近邻召回，SQLite 只保存向量 ID 到原始文本块的映射。

```text
文档 .txt
  -> 文本切块
  -> llama.cpp Embedding
  -> Faiss 向量索引
  -> SQLite ID -> 文本元数据

用户问题
  -> llama.cpp Query Embedding
  -> Faiss Candidate ID / cosine score
  -> SQLite 按 ID 取回文本
  -> llama.cpp Reranker 相关性评分
  -> 重排后的最终 Top-K
  -> RAG Prompt
  -> llama.cpp LLM
  -> 最终答案
```

## 支持范围

仓库只保留三种彼此独立的 Android 编译期 profile：

| Profile | CMake 值 | llama.cpp 目标设备 | 额外运行库 |
|---|---|---|---|
| RAG + CPU | `CPU` | `CPU` | 无加速后端库 |
| RAG + OpenCL GPU | `OPENCL` | `GPUOpenCL` | `libggml-opencl.so`、设备 OpenCL runtime |
| RAG + FastRPC/HTP | `HEXAGON` | `HTP0` | `libggml-hexagon.so`、`libggml-htp-vXX.so` |

三种 profile 都使用 llama.cpp Embedding GGUF、Reranker GGUF、生成模型 GGUF、Faiss 和 SQLite，并固定为 Android `arm64-v8a`、API 31 及以上。

这里的 GPU-only/HTP-only 表示 Embedding、Reranker 和 LLM 的模型层固定选择目标 accelerator，不表示整个进程完全不使用 CPU。Faiss、SQLite、文件读取、tokenizer、采样和 OpenCL/FastRPC host 调度仍在 CPU 上执行，`libggml-cpu.so` 也仍是 llama.cpp 的必要运行库。

## 仓库结构

```text
NativeRAG/
├── CMakeLists.txt
├── include/                  # RAG、Embedding、Reranker、LLM、Faiss、SQLite 接口
├── src/                      # mobile_rag_cli 实现
├── prebuilt/
│   ├── include/faiss/        # 与 libfaiss.so 匹配的头文件
│   └── android-aarch64/
│       ├── faiss/libfaiss.so
│       └── openblas/libopenblas.so
├── scripts/                  # CPU/OpenCL/HTP 构建、部署和测试脚本
├── tests/data/               # 确定性设备端冒烟测试语料
└── third_party/sqlite/       # Android 静态编译的 SQLite amalgamation
```

## 依赖准备

需要 CMake、Ninja、Android NDK、adb、Embedding GGUF、Reranker GGUF，以及完整 RAG 测试所需的生成模型 GGUF。

构建与部署脚本要求显式设置：

```bash
export ANDROID_NDK_ROOT=/path/to/android-ndk
```

脚本默认在 NativeRAG 同级目录寻找 `llama.cpp`：

```text
workspace/
├── NativeRAG/
└── llama.cpp/
```

也可以覆盖：

```bash
export LLAMA_CPP_ROOT=/path/to/llama.cpp
```

llama.cpp 头文件与动态库必须来自同一提交。NativeRAG 至少使用：

```text
<LLAMA_CPP_ROOT>/include/llama.h
<LLAMA_CPP_ROOT>/ggml/include/ggml-backend.h

<LLAMA_LIBRARY_DIR>/libllama.so
<LLAMA_LIBRARY_DIR>/libggml.so
<LLAMA_LIBRARY_DIR>/libggml-base.so
<LLAMA_LIBRARY_DIR>/libggml-cpu.so
```

默认模型文件名为 `Qwen3-Embedding-0.6B-Q8_0.gguf` 和 `Qwen3-Reranker-0.6B-Q8_0.gguf`。部署时建议显式设置：

```bash
export EMBEDDING_MODEL_LOCAL=/path/to/embedding.gguf
export RERANKER_MODEL_LOCAL=/path/to/reranker.gguf
export LLM_MODEL_LOCAL=/path/to/generation-model.gguf
```

`EMBEDDING_MODEL_LOCAL` 和 `RERANKER_MODEL_LOCAL` 未设置时，部署脚本分别从 NativeRAG 同级的 `models/Qwen3-Embedding-0.6B-Q8_0.gguf` 和 `models/Qwen3-Reranker-0.6B-Q8_0.gguf` 读取。`LLM_MODEL_LOCAL` 可省略；如果生成模型已在设备上，测试时设置 `LLM_MODEL_REMOTE` 即可。

当前指定的 Qwen3 Reranker GGUF 是完整 LM head 格式，没有 `rerank pooling metadata`。NativeRAG 会按 Qwen3 官方 prompt 读取最后位置的 `yes`/`no` logits，并以 `P(yes)` 作为相关性分数；如果换用带命名 `rerank template` 与 `rank classification head` 的新版 GGUF，则自动使用 llama.cpp 的 `LLAMA_POOLING_TYPE_RANK`。两种路径都不需要修改 llama.cpp 源码。

## 构建

三个构建脚本默认先构建对应配置的 llama.cpp，再构建 NativeRAG。已有匹配的 llama.cpp 动态库时，可以设置 `NATIVERAG_SKIP_LLAMA_BUILD=1` 并通过 `LLAMA_LIBRARY_DIR` 指定库目录。

### CPU

```bash
cd /path/to/NativeRAG
export ANDROID_NDK_ROOT=/path/to/android-ndk
export LLAMA_CPP_ROOT=/path/to/llama.cpp
./scripts/build_android_cpu.sh
```

默认输出为 `<workspace>/build-nativerag-android-cpu/mobile_rag_cli`。

使用已有库：

```bash
export NATIVERAG_SKIP_LLAMA_BUILD=1
export LLAMA_LIBRARY_DIR=/path/to/llama-cpu-libs
./scripts/build_android_cpu.sh
```

### OpenCL GPU

设备需要兼容的 OpenCL runtime。构建机需要 Android OpenCL 头文件和链接库；默认从 NDK sysroot 查找，也可以覆盖：

```bash
export ANDROID_NDK_ROOT=/path/to/android-ndk
export LLAMA_CPP_ROOT=/path/to/llama.cpp
export OPENCL_INCLUDE_DIR=/path/to/android/opencl/include
export OPENCL_LIBRARY=/path/to/android/libOpenCL.so
./scripts/build_android_opencl.sh
```

默认输出为 `<workspace>/build-nativerag-android-opencl/mobile_rag_cli`。llama.cpp 库目录必须额外包含 `libggml-opencl.so`。

### FastRPC/Hexagon HTP

重新构建 llama.cpp HTP 后端时需要 Qualcomm Hexagon SDK 与 Tools：

```bash
export ANDROID_NDK_ROOT=/path/to/android-ndk
export LLAMA_CPP_ROOT=/path/to/llama.cpp
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK
export HEXAGON_TOOLS_ROOT=/path/to/HEXAGON_Tools/19.0.04
./scripts/build_android_htp.sh
```

默认输出为 `<workspace>/build-nativerag-android-htp/mobile_rag_cli`。库目录必须额外包含 `libggml-hexagon.so`，DSP 侧还需要设备架构对应的 `libggml-htp-vXX.so`。

已有 HTP 产物时：

```bash
export NATIVERAG_SKIP_LLAMA_BUILD=1
export LLAMA_LIBRARY_DIR=/path/to/llama-htp-libs
export HTP_SKEL_DIR=/path/to/htp-skeletons
export HTP_ARCHES="v79 v81"
./scripts/build_android_htp.sh
```

`HTP_ARCHES` 必须包含目标设备实际使用的架构。

## 部署

```bash
adb devices
export ADB_SERIAL=YOUR_DEVICE_SERIAL       # 多设备时设置
export ANDROID_NDK_ROOT=/path/to/android-ndk
export EMBEDDING_MODEL_LOCAL=/path/to/embedding.gguf
export RERANKER_MODEL_LOCAL=/path/to/reranker.gguf
export LLM_MODEL_LOCAL=/path/to/llm.gguf  # 可选
```

按 profile 执行：

```bash
./scripts/deploy_android_cpu.sh
./scripts/deploy_android_opencl.sh
./scripts/deploy_android_htp.sh
```

OpenCL 系统路径不能提供 `libOpenCL.so` 时，设置与目标设备兼容的 `OPENCL_RUNTIME_LOCAL`。HTP 部署可用 `HTP_ARCHES="v79 v81"` 缩小 skeleton 集合；设备还必须具备可工作的 CDSP/FastRPC runtime、firmware、权限和 SELinux 策略。

部署目录包含 CLI、Faiss/OpenBLAS/OpenMP、匹配 profile 的 llama.cpp 库、Embedding/Reranker 模型、可选生成模型和 `tests/data/*.txt`。部署结束自动执行 `--backend-info`：OpenCL 必须看到 `selected_device=GPUOpenCL`，HTP 必须看到 `selected_device=HTP0`；目标设备不存在时程序直接失败，不会切换到 CPU profile。

## 一键冒烟测试

```bash
# CPU
export NATIVERAG_REMOTE_DIR=/data/local/tmp/nativerag-cpu
export LLM_MODEL_REMOTE="$NATIVERAG_REMOTE_DIR/models/llm.gguf"
export RERANKER_MODEL_REMOTE="$NATIVERAG_REMOTE_DIR/models/Qwen3-Reranker-0.6B-Q8_0.gguf"
export NATIVERAG_RERANK_CANDIDATES=4
./scripts/test_android_cpu.sh

# OpenCL
export NATIVERAG_REMOTE_DIR=/data/local/tmp/nativerag-opencl
export LLM_MODEL_REMOTE="$NATIVERAG_REMOTE_DIR/models/llm.gguf"
export RERANKER_MODEL_REMOTE="$NATIVERAG_REMOTE_DIR/models/Qwen3-Reranker-0.6B-Q8_0.gguf"
export NATIVERAG_RERANK_CANDIDATES=4
./scripts/test_android_opencl.sh

# HTP
export NATIVERAG_REMOTE_DIR=/data/local/tmp/nativerag-htp
export LLM_MODEL_REMOTE="$NATIVERAG_REMOTE_DIR/models/llm.gguf"
export RERANKER_MODEL_REMOTE="$NATIVERAG_REMOTE_DIR/models/Qwen3-Reranker-0.6B-Q8_0.gguf"
export NATIVERAG_RERANK_CANDIDATES=4
./scripts/test_android_htp.sh
```

`RERANKER_MODEL_REMOTE` 和 `NATIVERAG_RERANK_CANDIDATES` 均有上例所示的默认值，可省略。测试依次验证 profile 和目标设备、Embedding 建库、Faiss/SQLite 文件、新进程加载索引、Qwen3 `yes-no-logits` 模式、Reranker 输出、重排后的 Top-1、完整 RAG 答案以及 OpenCL/HTP offload 日志。检索用例特意使用否定语义：Faiss 原始第 2 名必须被 Reranker 提升到最终第 1 名，从而证明第二阶段确实改变了排序。设置 `NATIVERAG_RUN_NEGATIVE_TEST=1` 时，加速测试还会临时移走 backend `.so`，确认程序无法静默回退，然后自动恢复。

## 手动测试

进入设备，以 CPU profile 为例：

```bash
adb -s YOUR_DEVICE_SERIAL shell
```

```sh
cd /data/local/tmp/nativerag-cpu
export LD_LIBRARY_PATH="$PWD/lib"

./mobile_rag_cli --backend-info

./mobile_rag_cli --build \
  --text-path "$PWD/data" \
  --embedding-model "$PWD/models/Qwen3-Embedding-0.6B-Q8_0.gguf" \
  --faiss-type Flat \
  --index-path "$PWD/faiss.bin" \
  --sqlite-db "$PWD/texts.sqlite3" \
  --threads 4 --verbose

./mobile_rag_cli --query "NativeRAG 的设备端测试口令是什么？" \
  --retrieve-only \
  --embedding-model "$PWD/models/Qwen3-Embedding-0.6B-Q8_0.gguf" \
  --reranker-model "$PWD/models/Qwen3-Reranker-0.6B-Q8_0.gguf" \
  --rerank-candidates 4 \
  --index-path "$PWD/faiss.bin" \
  --sqlite-db "$PWD/texts.sqlite3" \
  --top-k 2 --threads 4 --verbose

./mobile_rag_cli --query "NativeRAG 的设备端测试口令是什么？只回答口令。" \
  --embedding-model "$PWD/models/Qwen3-Embedding-0.6B-Q8_0.gguf" \
  --reranker-model "$PWD/models/Qwen3-Reranker-0.6B-Q8_0.gguf" \
  --rerank-candidates 4 \
  --llm-model "$PWD/models/llm.gguf" \
  --index-path "$PWD/faiss.bin" \
  --sqlite-db "$PWD/texts.sqlite3" \
  --top-k 2 --threads 4 --max-tokens 128 --verbose
```

OpenCL shell 还应将 vendor library 目录加入 `LD_LIBRARY_PATH`。HTP shell 还应设置部署脚本中的 `ADSP_LIBRARY_PATH` 和 `GGML_HEXAGON_*` 变量。

### HNSW

```sh
./mobile_rag_cli --build \
  --text-path "$PWD/data" \
  --embedding-model "$PWD/models/Qwen3-Embedding-0.6B-Q8_0.gguf" \
  --faiss-type HNSW32 \
  --index-path "$PWD/faiss.hnsw32.bin" \
  --sqlite-db "$PWD/texts.hnsw32.sqlite3" \
  --threads 4 --verbose
```

索引类型已经序列化在 `faiss.hnsw32.bin` 中；`--faiss-type` 只影响新建索引，加载已有索引时不改变其结构。

## CLI 参考

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--backend-info` | - | 打印 profile、设备列表和目标设备 |
| `--build [PATH]` | - | 从单个 `.txt` 或目录递归构建索引 |
| `--query "QUESTION"` | - | 单次查询 |
| `--interactive`, `-i` | - | 交互查询 |
| `--embedding-model PATH` | - | Build、Query、Interactive 必需 |
| `--reranker-model PATH` | - | 可选；Query、Interactive 的 llama.cpp 二阶段重排模型 |
| `--llm-model PATH` | - | 完整 Query、Interactive 必需 |
| `--text-path PATH` | - | Build 输入，也可作为位置参数 |
| `--index-path PATH` | `./faiss_index.bin` | Faiss 索引路径 |
| `--sqlite-db PATH` | `./vector_store.sqlite3` | SQLite 文本元数据；`--vector-db` 是别名 |
| `--faiss-type DESC` | `Flat` | 新建索引的 Faiss factory 描述 |
| `--top-k N` | `5` | Reranker 重排后最终返回的文本块数量 |
| `--rerank-candidates N` | `20` | 启用 Reranker 时由 Faiss 召回并参与重排的候选数，必须不小于 `top-k` |
| `--threads N` | `4` | llama.cpp 线程数 |
| `--max-tokens N` | `256` | 最大生成 token 数 |
| `--retrieve-only` | 关闭 | 不加载生成 LLM；运行 Embedding、Faiss、SQLite lookup，以及已启用的 Reranker |
| `--verbose`, `-v` | 关闭 | 输出阶段信息 |
| `--no-save-index` | 关闭 | Build 后不保存 Faiss 索引 |
| `--no-load-index` | 关闭 | 不加载索引，通常用于错误路径测试 |

## 文本切块限制

- 只读取 `.txt`，目录模式递归扫描；
- 每个文件单独切块；
- 固定 `chunk_size=1000`、`overlap=200`；
- 按 `std::string` 字节位置切割，不是按 Unicode 字符或 token；
- 目录遍历未排序，chunk ID 不保证跨平台稳定。

如果多个事实合并在一个不足 1000 字节的文件中，它们只会产生一个向量。查询任何事实都只能返回同一个 chunk，预览从文件开头开始时可能看起来像语义错误。确定性测试应像 `tests/data/` 一样，把独立事实放在独立 `.txt` 文件中。
