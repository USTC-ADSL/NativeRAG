#!/usr/bin/env bash
set -euo pipefail

remote_dir=${NATIVERAG_REMOTE_DIR:-/data/local/tmp/nativerag-htp}
llm_model_remote=${LLM_MODEL_REMOTE:-${remote_dir}/models/llm.gguf}
reranker_model_remote=${RERANKER_MODEL_REMOTE:-${remote_dir}/models/Qwen3-Reranker-0.6B-Q8_0.gguf}
rerank_candidates=${NATIVERAG_RERANK_CANDIDATES:-4}
threads=${NATIVERAG_THREADS:-4}
require_full_offload=${NATIVERAG_REQUIRE_FULL_OFFLOAD:-1}

adb_args=()
if [[ -n "${ADB_SERIAL:-}" ]]; then
  adb_args=(-s "${ADB_SERIAL}")
fi

run_remote() {
  adb "${adb_args[@]}" shell \
    "cd '${remote_dir}' && export LD_LIBRARY_PATH='${remote_dir}/lib:/vendor/lib64:/system/vendor/lib64' && export ADSP_LIBRARY_PATH='${remote_dir}/lib;/vendor/lib/rfsa/adsp;/vendor/lib/rfsa/dsp;/vendor/dsp;' && export GGML_HEXAGON_NDEV='${GGML_HEXAGON_NDEV:-1}' && export GGML_HEXAGON_NHVX='${GGML_HEXAGON_NHVX:-0}' && export GGML_HEXAGON_HOSTBUF='${GGML_HEXAGON_HOSTBUF:-1}' && export GGML_HEXAGON_VERBOSE='${GGML_HEXAGON_VERBOSE:-1}' && $*" 2>&1
}

restore_backend() {
  adb "${adb_args[@]}" shell \
    "if [ -f '${remote_dir}/lib/libggml-hexagon.so.disabled' ]; then mv '${remote_dir}/lib/libggml-hexagon.so.disabled' '${remote_dir}/lib/libggml-hexagon.so'; fi" \
    >/dev/null 2>&1 || true
}

adb "${adb_args[@]}" get-state >/dev/null
run_remote "test -f '${reranker_model_remote}'"

backend_output=$(run_remote "./mobile_rag_cli --backend-info")
printf '%s\n' "${backend_output}"
grep -F 'accelerator=HEXAGON' <<<"${backend_output}"
grep -F 'selected_device=HTP0' <<<"${backend_output}"

build_output=$(run_remote "./mobile_rag_cli --build \
  --text-path '${remote_dir}/data' \
  --embedding-model '${remote_dir}/models/Qwen3-Embedding-0.6B-Q8_0.gguf' \
  --faiss-type Flat \
  --index-path '${remote_dir}/faiss.bin' \
  --sqlite-db '${remote_dir}/texts.sqlite3' \
  --threads '${threads}' --verbose")
printf '%s\n' "${build_output}"
grep -F '[LlamaCppEmbedding] Loaded HEXAGON model on HTP0' <<<"${build_output}"
grep -E 'HTP0 .*new session|new session: HTP0' <<<"${build_output}"
if grep -Eiq 'will fall back|fall back to CPU|fallback to CPU' <<<"${build_output}"; then
  echo "HTP build reported an accelerator fallback" >&2
  exit 1
fi
if [[ "${require_full_offload}" == "1" ]]; then
  grep -E 'offloaded [1-9][0-9]*/[1-9][0-9]* layers to GPU' <<<"${build_output}"
fi

run_remote "test -s '${remote_dir}/faiss.bin' && test -s '${remote_dir}/texts.sqlite3'"

retrieval_output=$(run_remote "./mobile_rag_cli --query '哪一段内容明确说它与测试口令的具体答案无关？' \
  --retrieve-only \
  --embedding-model '${remote_dir}/models/Qwen3-Embedding-0.6B-Q8_0.gguf' \
  --reranker-model '${reranker_model_remote}' \
  --rerank-candidates '${rerank_candidates}' \
  --index-path '${remote_dir}/faiss.bin' \
  --sqlite-db '${remote_dir}/texts.sqlite3' \
  --top-k 2 --threads '${threads}'")
printf '%s\n' "${retrieval_output}"
grep -F '[LlamaCppReranker] Loaded HEXAGON model on HTP0' <<<"${retrieval_output}"
grep -F 'mode=yes-no-logits' <<<"${retrieval_output}"
grep -F '[RERANK] Scored' <<<"${retrieval_output}"
grep -F 'rerank_score=' <<<"${retrieval_output}"
if grep -Eiq 'will fall back|fall back to CPU|fallback to CPU' <<<"${retrieval_output}"; then
  echo "HTP retrieval/reranking reported an accelerator fallback" >&2
  exit 1
fi
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
grep -F '[LlamaCppModel] Loaded HEXAGON model on HTP0' <<<"${generation_output}"
grep -E 'HTP0 .*new session|new session: HTP0' <<<"${generation_output}"
if grep -Eiq 'will fall back|fall back to CPU|fallback to CPU' <<<"${generation_output}"; then
  echo "HTP generation reported an accelerator fallback" >&2
  exit 1
fi
grep -F '[ANSWER]' <<<"${generation_output}" | grep -F 'blue-orchid-7319'

if [[ "${NATIVERAG_RUN_NEGATIVE_TEST:-0}" == "1" ]]; then
  trap restore_backend EXIT
  adb "${adb_args[@]}" shell \
    "mv '${remote_dir}/lib/libggml-hexagon.so' '${remote_dir}/lib/libggml-hexagon.so.disabled'"
  if run_remote "./mobile_rag_cli --backend-info"; then
    echo "HTP negative test failed: program ran without libggml-hexagon.so" >&2
    exit 1
  fi
  restore_backend
  trap - EXIT
fi

echo "Android FastRPC/HTP Local RAG smoke test passed"
