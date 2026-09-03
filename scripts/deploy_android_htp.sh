#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
workspace_root=$(cd "${repo_root}/.." && pwd)

android_ndk_root=${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT to your Android NDK}
llama_cpp_root=${LLAMA_CPP_ROOT:-${workspace_root}/llama.cpp}
llama_build_dir=${LLAMA_ANDROID_BUILD_DIR:-${llama_cpp_root}/build-android-htp-nativerag}
llama_library_dir=${LLAMA_LIBRARY_DIR:-${llama_build_dir}/bin}
htp_skel_dir=${HTP_SKEL_DIR:-${llama_build_dir}/ggml/src/ggml-hexagon}
read -r -a htp_arches <<<"${HTP_ARCHES:-v73 v75 v79 v81}"
build_dir=${NATIVERAG_ANDROID_BUILD_DIR:-${workspace_root}/build-nativerag-android-htp}
embedding_model=${EMBEDDING_MODEL_LOCAL:-${workspace_root}/models/Qwen3-Embedding-0.6B-Q8_0.gguf}
reranker_model=${RERANKER_MODEL_LOCAL:-${workspace_root}/models/Qwen3-Reranker-0.6B-Q8_0.gguf}
remote_dir=${NATIVERAG_REMOTE_DIR:-/data/local/tmp/nativerag-htp}

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
for library in libllama.so libggml.so libggml-base.so libggml-cpu.so libggml-hexagon.so; do
  test -f "${llama_library_dir}/${library}"
done
for arch in "${htp_arches[@]}"; do
  [[ "${arch}" =~ ^v[0-9]+$ ]]
  test -f "${htp_skel_dir}/libggml-htp-${arch}.so"
done

adb "${adb_args[@]}" get-state >/dev/null
adb "${adb_args[@]}" shell "mkdir -p '${remote_dir}/lib' '${remote_dir}/models' '${remote_dir}/data'"

adb "${adb_args[@]}" push "${build_dir}/mobile_rag_cli" "${remote_dir}/mobile_rag_cli"
for library in libllama.so libggml.so libggml-base.so libggml-cpu.so libggml-hexagon.so; do
  adb "${adb_args[@]}" push "${llama_library_dir}/${library}" "${remote_dir}/lib/${library}"
done
for arch in "${htp_arches[@]}"; do
  adb "${adb_args[@]}" push "${htp_skel_dir}/libggml-htp-${arch}.so" \
    "${remote_dir}/lib/libggml-htp-${arch}.so"
done
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
  "cd '${remote_dir}' && export LD_LIBRARY_PATH='${remote_dir}/lib:/vendor/lib64:/system/vendor/lib64' && export ADSP_LIBRARY_PATH='${remote_dir}/lib;/vendor/lib/rfsa/adsp;/vendor/lib/rfsa/dsp;/vendor/dsp;' && export GGML_HEXAGON_NDEV='${GGML_HEXAGON_NDEV:-1}' && export GGML_HEXAGON_NHVX='${GGML_HEXAGON_NHVX:-0}' && export GGML_HEXAGON_HOSTBUF='${GGML_HEXAGON_HOSTBUF:-1}' && ./mobile_rag_cli --backend-info"
echo "Deployed FastRPC/HTP profile to ${remote_dir}"
