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
hexagon_sdk_root=${HEXAGON_SDK_ROOT:-}
hexagon_tools_root=${HEXAGON_TOOLS_ROOT:-${hexagon_sdk_root:+${hexagon_sdk_root}/tools/HEXAGON_Tools/19.0.04}}
hexagon_prebuilt_lib_dir=${HEXAGON_PREBUILT_LIB_DIR:-android_aarch64}

test -f "${android_ndk_root}/build/cmake/android.toolchain.cmake"
test -f "${llama_cpp_root}/include/llama.h"

if [[ "${NATIVERAG_SKIP_LLAMA_BUILD:-0}" != "1" ]]; then
  test -n "${hexagon_sdk_root}"
  test -n "${hexagon_tools_root}"
  test -d "${hexagon_sdk_root}"
  test -d "${hexagon_tools_root}"
  cmake -S "${llama_cpp_root}" -B "${llama_build_dir}" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${android_ndk_root}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-31 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_CPU=ON \
    -DGGML_OPENCL=OFF \
    -DGGML_HEXAGON=ON \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_BUILD_MTMD=OFF \
    -DLLAMA_TOOLS_INSTALL=OFF \
    -DHEXAGON_SDK_ROOT="${hexagon_sdk_root}" \
    -DHEXAGON_TOOLS_ROOT="${hexagon_tools_root}" \
    -DPREBUILT_LIB_DIR="${hexagon_prebuilt_lib_dir}"
  cmake --build "${llama_build_dir}" --parallel
fi

for library in libllama.so libggml.so libggml-base.so libggml-cpu.so libggml-hexagon.so; do
  test -f "${llama_library_dir}/${library}"
done
for arch in "${htp_arches[@]}"; do
  [[ "${arch}" =~ ^v[0-9]+$ ]]
  test -f "${htp_skel_dir}/libggml-htp-${arch}.so"
done

cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="${android_ndk_root}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-31 \
  -DCMAKE_BUILD_TYPE=Release \
  -DNATIVERAG_ACCELERATOR=HEXAGON \
  -DLLAMA_CPP_ROOT="${llama_cpp_root}" \
  -DLLAMA_LIBRARY_DIR="${llama_library_dir}"

cmake --build "${build_dir}" --parallel
echo "Android FastRPC/HTP binary: ${build_dir}/mobile_rag_cli"
echo "Matching llama.cpp libraries: ${llama_library_dir}"
echo "HTP skeletons: ${htp_skel_dir}"
