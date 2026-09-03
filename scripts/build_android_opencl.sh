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
opencl_include_dir=${OPENCL_INCLUDE_DIR:-${android_ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include}
opencl_library=${OPENCL_LIBRARY:-${android_ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libOpenCL.so}

test -f "${android_ndk_root}/build/cmake/android.toolchain.cmake"
test -f "${llama_cpp_root}/include/llama.h"

if [[ "${NATIVERAG_SKIP_LLAMA_BUILD:-0}" != "1" ]]; then
  test -f "${opencl_include_dir}/CL/opencl.h"
  test -f "${opencl_library}"
  cmake -S "${llama_cpp_root}" -B "${llama_build_dir}" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${android_ndk_root}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-31 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_CPU=ON \
    -DGGML_OPENCL=ON \
    -DGGML_OPENCL_EMBED_KERNELS=ON \
    -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
    -DGGML_OPENCL_TARGET_VERSION=300 \
    -DGGML_HEXAGON=OFF \
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
    -DOpenCL_INCLUDE_DIR="${opencl_include_dir}" \
    -DOpenCL_LIBRARY="${opencl_library}"
  cmake --build "${llama_build_dir}" --parallel
fi

for library in libllama.so libggml.so libggml-base.so libggml-cpu.so libggml-opencl.so; do
  test -f "${llama_library_dir}/${library}"
done

cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="${android_ndk_root}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-31 \
  -DCMAKE_BUILD_TYPE=Release \
  -DNATIVERAG_ACCELERATOR=OPENCL \
  -DLLAMA_CPP_ROOT="${llama_cpp_root}" \
  -DLLAMA_LIBRARY_DIR="${llama_library_dir}"

cmake --build "${build_dir}" --parallel
echo "Android OpenCL binary: ${build_dir}/mobile_rag_cli"
echo "Matching llama.cpp libraries: ${llama_library_dir}"
