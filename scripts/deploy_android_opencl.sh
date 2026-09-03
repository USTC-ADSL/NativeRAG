#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
workspace_root=$(cd "${repo_root}/.." && pwd)

android_ndk_root=${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT to your Android NDK}
llama_cpp_root=${LLAMA_CPP_ROOT:-${workspace_root}/llama.cpp}
llama_build_dir=${LLAMA_ANDROID_BUILD_DIR:-${llama_cpp_root}/build-android-opencl-nativerag}
llama_library_dir=${LLAMA_LIBRARY_DIR:-${llama_build_dir}/bin}
build_dir=${NATIVERAG_ANDROID_BUILD_DIR:-${workspace_root}/build-nativerag-android-opencl}
embedding_model=${EMBEDDING_MODEL_LOCAL:-${workspace_root}/models/Qwen3-Embedding-0.6B-Q8_0.gguf}
reranker_model=${RERANKER_MODEL_LOCAL:-${workspace_root}/models/Qwen3-Reranker-0.6B-Q8_0.gguf}
remote_dir=${NATIVERAG_REMOTE_DIR:-/data/local/tmp/nativerag-opencl}

adb_args=()
if [[ -n "${ADB_SERIAL:-}" ]]; then
  adb_args=(-s "${ADB_SERIAL}")
fi

libomp_path=$(find "${android_ndk_root}/toolchains/llvm/prebuilt" \
  -path '*/lib/clang/*/lib/linux/aarch64/libomp.so' -print -quit)

test -x "${build_dir}/mobile_rag_cli"
test -f "${embedding_model}"
test -f "${reranker_model}"
test -n "${libomp_path}"
for library in libllama.so libggml.so libggml-base.so libggml-cpu.so libggml-opencl.so; do
  test -f "${llama_library_dir}/${library}"
done

adb "${adb_args[@]}" get-state >/dev/null
adb "${adb_args[@]}" shell "mkdir -p '${remote_dir}/lib' '${remote_dir}/models' '${remote_dir}/data'"

adb "${adb_args[@]}" push "${build_dir}/mobile_rag_cli" "${remote_dir}/mobile_rag_cli"
for library in libllama.so libggml.so libggml-base.so libggml-cpu.so libggml-opencl.so; do
  adb "${adb_args[@]}" push "${llama_library_dir}/${library}" "${remote_dir}/lib/${library}"
done
if [[ -n "${OPENCL_RUNTIME_LOCAL:-}" ]]; then
  test -f "${OPENCL_RUNTIME_LOCAL}"
  adb "${adb_args[@]}" push "${OPENCL_RUNTIME_LOCAL}" "${remote_dir}/lib/libOpenCL.so"
fi
adb "${adb_args[@]}" push "${repo_root}/prebuilt/android-aarch64/faiss/libfaiss.so" "${remote_dir}/lib/libfaiss.so"
adb "${adb_args[@]}" push "${repo_root}/prebuilt/android-aarch64/openblas/libopenblas.so" "${remote_dir}/lib/libopenblas.so"
adb "${adb_args[@]}" push "${libomp_path}" "${remote_dir}/lib/libomp.so"
adb "${adb_args[@]}" push "${embedding_model}" "${remote_dir}/models/Qwen3-Embedding-0.6B-Q8_0.gguf"
adb "${adb_args[@]}" push "${reranker_model}" "${remote_dir}/models/Qwen3-Reranker-0.6B-Q8_0.gguf"
adb "${adb_args[@]}" push "${repo_root}/tests/data/." "${remote_dir}/data/"
adb "${adb_args[@]}" shell "chmod 755 '${remote_dir}/mobile_rag_cli'"

if [[ -n "${LLM_MODEL_LOCAL:-}" ]]; then
  adb "${adb_args[@]}" push "${LLM_MODEL_LOCAL}" "${remote_dir}/models/llm.gguf"
fi

adb "${adb_args[@]}" shell \
  "cd '${remote_dir}' && export LD_LIBRARY_PATH='${remote_dir}/lib:/vendor/lib64:/system/vendor/lib64' && ./mobile_rag_cli --backend-info"
echo "Deployed OpenCL profile to ${remote_dir}"
