#!/usr/bin/env bash
set -euo pipefail

remote_dir=${NATIVERAG_REMOTE_DIR:-/data/local/tmp/nativerag-cpu}
llm_model_remote=${LLM_MODEL_REMOTE:-${remote_dir}/models/llm.gguf}
reranker_model_remote=${RERANKER_MODEL_REMOTE:-${remote_dir}/models/Qwen3-Reranker-0.6B-Q8_0.gguf}
rerank_candidates=${NATIVERAG_RERANK_CANDIDATES:-4}
threads=${NATIVERAG_THREADS:-4}

adb_args=()
if [[ -n "${ADB_SERIAL:-}" ]]; then
  adb_args=(-s "${ADB_SERIAL}")
fi

run_remote() {
  adb "${adb_args[@]}" shell \
    "cd '${remote_dir}' && export LD_LIBRARY_PATH='${remote_dir}/lib' && $*"
}

adb "${adb_args[@]}" get-state >/dev/null
run_remote "test -f '${reranker_model_remote}'"

backend_output=$(run_remote "./mobile_rag_cli --backend-info")
printf '%s\n' "${backend_output}"
grep -F 'accelerator=CPU' <<<"${backend_output}"

run_remote "./mobile_rag_cli --build \
  --text-path '${remote_dir}/data' \
  --embedding-model '${remote_dir}/models/Qwen3-Embedding-0.6B-Q8_0.gguf' \
  --index-path '${remote_dir}/faiss.bin' \
  --sqlite-db '${remote_dir}/texts.sqlite3' \
  --threads '${threads}' --verbose"

run_remote "test -s '${remote_dir}/faiss.bin' && test -s '${remote_dir}/texts.sqlite3' && ls -l '${remote_dir}/faiss.bin' '${remote_dir}/texts.sqlite3'"

retrieval_output=$(run_remote "./mobile_rag_cli --query '哪一段内容明确说它与测试口令的具体答案无关？' \
  --retrieve-only \
  --embedding-model '${remote_dir}/models/Qwen3-Embedding-0.6B-Q8_0.gguf' \
  --reranker-model '${reranker_model_remote}' \
  --rerank-candidates '${rerank_candidates}' \
  --index-path '${remote_dir}/faiss.bin' \
  --sqlite-db '${remote_dir}/texts.sqlite3' \
  --top-k 2 --threads '${threads}'")
printf '%s\n' "${retrieval_output}"
grep -F '[LlamaCppReranker] Loaded CPU model on CPU' <<<"${retrieval_output}"
grep -F 'mode=yes-no-logits' <<<"${retrieval_output}"
grep -F '[RERANK] Scored' <<<"${retrieval_output}"
grep -F 'rerank_score=' <<<"${retrieval_output}"
grep -F '[TOP-1]' <<<"${retrieval_output}" | grep -F 'faiss_rank=2' | \
  grep -F '端侧测试设备可以通过 adb shell 运行命令'

generation_output=$(run_remote "./mobile_rag_cli --query 'NativeRAG 的设备端测试口令是什么？' \
  --llm-model '${llm_model_remote}' \
  --embedding-model '${remote_dir}/models/Qwen3-Embedding-0.6B-Q8_0.gguf' \
  --reranker-model '${reranker_model_remote}' \
  --rerank-candidates '${rerank_candidates}' \
  --index-path '${remote_dir}/faiss.bin' \
  --sqlite-db '${remote_dir}/texts.sqlite3' \
  --top-k 2 --threads '${threads}' --max-tokens 128")
printf '%s\n' "${generation_output}"
grep -F '[RERANK] Scored' <<<"${generation_output}"
grep -F '[ANSWER]' <<<"${generation_output}" | grep -F 'blue-orchid-7319'

echo "Android CPU Local RAG smoke test passed"
