#!/bin/bash
set -e

# Build script for compiling third-party libraries (Faiss, MNN, llama.cpp)
# for both Linux x86_64 and Android aarch64

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${SCRIPT_DIR}/third_party"
PREBUILT_DIR="${SCRIPT_DIR}/prebuilt"

# New directory structure (each library has its own lib and include)
# Linux x86_64
PREBUILT_LINUX_FAISS="${PREBUILT_DIR}/linux-x86_64/faiss"
PREBUILT_LINUX_MNN="${PREBUILT_DIR}/linux-x86_64/MNN"
PREBUILT_LINUX_LLAMA="${PREBUILT_DIR}/linux-x86_64/llama"
# Android aarch64
PREBUILT_ANDROID_FAISS="${PREBUILT_DIR}/android-aarch64/faiss"
PREBUILT_ANDROID_MNN="${PREBUILT_DIR}/android-aarch64/MNN"
PREBUILT_ANDROID_LLAMA="${PREBUILT_DIR}/android-aarch64/llama"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Android NDK is available
check_android_ndk() {
    if [ -z "$ANDROID_NDK" ]; then
        print_warn "ANDROID_NDK environment variable not set"
        if [ -d "$HOME/Android/Sdk/ndk" ]; then
            # Try to find the latest NDK version
            ANDROID_NDK=$(ls -d $HOME/Android/Sdk/ndk/* | sort -V | tail -n 1)
            print_info "Found Android NDK at: $ANDROID_NDK"
            export ANDROID_NDK
        else
            print_error "Android NDK not found. Please set ANDROID_NDK environment variable"
            return 1
        fi
    fi
    print_info "Using Android NDK: $ANDROID_NDK"
    return 0
}

# Parse command line arguments
BUILD_LINUX=false
BUILD_ANDROID=false
BUILD_FAISS=false
BUILD_MNN=false
BUILD_LLAMA=false

if [ $# -eq 0 ]; then
    # Build everything by default
    BUILD_LINUX=true
    BUILD_ANDROID=true
    BUILD_FAISS=true
    BUILD_MNN=true
    BUILD_LLAMA=true
else
    for arg in "$@"; do
        case $arg in
            --linux)
                BUILD_LINUX=true
                ;;
            --android)
                BUILD_ANDROID=true
                ;;
            --faiss)
                BUILD_FAISS=true
                ;;
            --mnn)
                BUILD_MNN=true
                ;;
            --llama)
                BUILD_LLAMA=true
                ;;
            --all)
                BUILD_LINUX=true
                BUILD_ANDROID=true
                BUILD_FAISS=true
                BUILD_MNN=true
                BUILD_LLAMA=true
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --linux      Build for Linux x86_64"
                echo "  --android    Build for Android aarch64"
                echo "  --faiss      Build Faiss library"
                echo "  --mnn        Build MNN library"
                echo "  --llama      Build llama.cpp library"
                echo "  --all        Build all libraries for all platforms (default)"
                echo "  --help       Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $arg"
                exit 1
                ;;
        esac
    done
fi

# If platform not specified but library is, build for both platforms
if [ "$BUILD_FAISS" = true ] || [ "$BUILD_MNN" = true ] || [ "$BUILD_LLAMA" = true ]; then
    if [ "$BUILD_LINUX" = false ] && [ "$BUILD_ANDROID" = false ]; then
        BUILD_LINUX=true
        BUILD_ANDROID=true
    fi
fi

# If platform specified but no library, build all libraries
if [ "$BUILD_LINUX" = true ] || [ "$BUILD_ANDROID" = true ]; then
    if [ "$BUILD_FAISS" = false ] && [ "$BUILD_MNN" = false ] && [ "$BUILD_LLAMA" = false ]; then
        BUILD_FAISS=true
        BUILD_MNN=true
        BUILD_LLAMA=true
    fi
fi

print_info "Build configuration:"
print_info "  Linux x86_64: $BUILD_LINUX"
print_info "  Android aarch64: $BUILD_ANDROID"
print_info "  Faiss: $BUILD_FAISS"
print_info "  MNN: $BUILD_MNN"
print_info "  llama.cpp: $BUILD_LLAMA"

# Check Android NDK if building for Android
if [ "$BUILD_ANDROID" = true ]; then
    check_android_ndk || exit 1
fi

# Create prebuilt directories (each library has its own directory with lib and include subdirectories)
# Linux x86_64
mkdir -p "${PREBUILT_LINUX_FAISS}/lib"
mkdir -p "${PREBUILT_LINUX_FAISS}/include"
mkdir -p "${PREBUILT_LINUX_MNN}/lib"
mkdir -p "${PREBUILT_LINUX_MNN}/include"
mkdir -p "${PREBUILT_LINUX_LLAMA}/lib"
mkdir -p "${PREBUILT_LINUX_LLAMA}/include"
# Android aarch64
mkdir -p "${PREBUILT_ANDROID_FAISS}/lib"
mkdir -p "${PREBUILT_ANDROID_FAISS}/include"
mkdir -p "${PREBUILT_ANDROID_MNN}/lib"
mkdir -p "${PREBUILT_ANDROID_MNN}/include"
mkdir -p "${PREBUILT_ANDROID_LLAMA}/lib"
mkdir -p "${PREBUILT_ANDROID_LLAMA}/include"

#################
# Build Faiss   #
#################
build_faiss_linux() {
    print_info "Building Faiss for Linux x86_64..."
    cd "${THIRD_PARTY_DIR}/faiss"

    rm -rf build-linux
    mkdir -p build-linux
    cd build-linux

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=OFF \
        -DCMAKE_INSTALL_PREFIX="${PWD}/install"

    make -j$(nproc)
    make install

    # Copy libraries to prebuilt/linux-x86_64/faiss/lib/
    cp install/lib/libfaiss.so "${PREBUILT_LINUX_FAISS}/lib/"


    # Copy headers to prebuilt/linux-x86_64/faiss/include/
    cp -r install/include/faiss/* "${PREBUILT_LINUX_FAISS}/include/"

    print_info "Faiss for Linux built successfully"
}

build_openblas_android() {
    print_info "Building OpenBLAS for Android aarch64..."
    cd "${THIRD_PARTY_DIR}/OpenBLAS"

    # Clean previous build
    rm -rf build-android
    mkdir -p build-android
    cd build-android

    # Build OpenBLAS for Android using CMake
    cmake .. \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-24 \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX="${PWD}/install"

    make -j$(nproc)
    make install

    # Copy libraries to prebuilt/android-aarch64/openblas/lib/
    mkdir -p "${PREBUILT_DIR}/android-aarch64/openblas/lib"
    mkdir -p "${PREBUILT_DIR}/android-aarch64/openblas/include"

    if [ -f "install/lib/libopenblas.so" ]; then
        cp install/lib/libopenblas.so "${PREBUILT_DIR}/android-aarch64/openblas/lib/"
        print_info "Copied libopenblas.so to prebuilt/android-aarch64/openblas/lib/"
    fi

    # Copy headers
    cp -r install/include/* "${PREBUILT_DIR}/android-aarch64/openblas/include/" 2>/dev/null || true

    print_info "OpenBLAS for Android built successfully"
}

build_faiss_android() {
    print_info "Building Faiss for Android aarch64..."

    # First build OpenBLAS if not already built
    if [ ! -f "${PREBUILT_DIR}/android-aarch64/openblas/lib/libopenblas.so" ]; then
        print_info "OpenBLAS not found, building it first..."
        build_openblas_android
    fi

    cd "${THIRD_PARTY_DIR}/faiss"

    rm -rf build-android
    mkdir -p build-android
    cd build-android

    # Build Faiss for Android with OpenBLAS support
    cmake .. \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-24 \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DFAISS_ENABLE_MKL=OFF \
        -DBUILD_TESTING=OFF \
        -DBLAS_LIBRARIES="${PREBUILT_DIR}/android-aarch64/openblas/lib/libopenblas.so" \
        -DBLAS_INCLUDE_DIRS="${PREBUILT_DIR}/android-aarch64/openblas/include" \
        -DCMAKE_INSTALL_PREFIX="${PWD}/install"

    make -j$(nproc)
    make install

    # Copy libraries to prebuilt/android-aarch64/faiss/lib/
    cp install/lib/libfaiss.so "${PREBUILT_ANDROID_FAISS}/lib/"

    # Also copy OpenBLAS dependency
    cp "${PREBUILT_DIR}/android-aarch64/openblas/lib/libopenblas.so" "${PREBUILT_ANDROID_FAISS}/lib/"

    # Copy headers to prebuilt/android-aarch64/faiss/include/
    cp -r install/include/faiss/* "${PREBUILT_ANDROID_FAISS}/include/"

    print_info "Faiss for Android built successfully"
}

#################
# Build MNN     #
#################
build_mnn_linux() {
    print_info "Building MNN for Linux x86_64..."
    cd "${THIRD_PARTY_DIR}/MNN"

    rm -rf build-linux
    mkdir -p build-linux
    cd build-linux

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DMNN_BUILD_CONVERTER=OFF \
        -DMNN_BUILD_TRAIN=OFF \
        -DMNN_BUILD_QUANTOOLS=OFF \
        -DMNN_BUILD_BENCHMARK=OFF \
        -DMNN_BUILD_TEST=OFF \
        -DMNN_BUILD_DEMO=OFF \
        -DMNN_BUILD_LLM=ON \
        -DMNN_BUILD_LLM_OMNI=ON \
        -DCMAKE_INSTALL_PREFIX="${PWD}/install"

    make -j$(nproc)
    make install

    # Copy libraries to prebuilt/linux-x86_64/MNN/lib/
    cp install/lib/libMNN.so "${PREBUILT_LINUX_MNN}/lib/"
    cp install/lib/libMNN_Express.so "${PREBUILT_LINUX_MNN}/lib/"

    # Copy libllm.so if it exists (generated when MNN_BUILD_LLM=ON)
    # Check both in current directory and in install/lib
    if [ -f "libllm.so" ]; then
        cp libllm.so "${PREBUILT_LINUX_MNN}/lib/"
        print_info "Copied libllm.so to prebuilt/linux-x86_64/MNN/lib/"
    elif [ -f "install/lib/libllm.so" ]; then
        cp install/lib/libllm.so "${PREBUILT_LINUX_MNN}/lib/"
        print_info "Copied libllm.so to prebuilt/linux-x86_64/MNN/lib/"
    fi

    # Copy headers to prebuilt/linux-x86_64/MNN/include/
    cp -r install/include/* "${PREBUILT_LINUX_MNN}/include/" 2>/dev/null || true

    print_info "MNN for Linux built successfully"
}

build_mnn_android() {
    print_info "Building MNN for Android aarch64..."
    cd "${THIRD_PARTY_DIR}/MNN"

    rm -rf build-android
    mkdir -p build-android
    cd build-android

    cmake .. \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-24 \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DMNN_BUILD_CONVERTER=OFF \
        -DMNN_BUILD_TRAIN=OFF \
        -DMNN_BUILD_QUANTOOLS=OFF \
        -DMNN_BUILD_BENCHMARK=OFF \
        -DMNN_BUILD_TEST=OFF \
        -DMNN_BUILD_DEMO=OFF \
        -DMNN_BUILD_LLM=ON \
        -DMNN_BUILD_LLM_OMNI=ON \
        -DCMAKE_INSTALL_PREFIX="${PWD}/install"

    make -j$(nproc)
    make install

    # Copy libraries to prebuilt/android-aarch64/MNN/lib/
    cp install/lib/libMNN_Express.so "${PREBUILT_ANDROID_MNN}/lib/"

    # Copy libMNN.so and libllm.so from build output directory
    # These are generated in OFF/arm64-v8a/ subdirectory
    if [ -f "OFF/arm64-v8a/libMNN.so" ]; then
        cp OFF/arm64-v8a/libMNN.so "${PREBUILT_ANDROID_MNN}/lib/"
        print_info "Copied libMNN.so to prebuilt/android-aarch64/MNN/lib/"
    fi

    if [ -f "OFF/arm64-v8a/libllm.so" ]; then
        cp OFF/arm64-v8a/libllm.so "${PREBUILT_ANDROID_MNN}/lib/"
        print_info "Copied libllm.so to prebuilt/android-aarch64/MNN/lib/"
    fi

    # Copy headers to prebuilt/android-aarch64/MNN/include/
    cp -r MNN "${PREBUILT_LINUX_MNN}/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/MNN/tools/cv/include"/* "${PREBUILT_ANDROID_MNN}/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/MNN/tools/audio/include"/* "${PREBUILT_ANDROID_MNN}/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/MNN/transformers/llm/engine/include"/* "${PREBUILT_ANDROID_MNN}/include/" 2>/dev/null || true

    print_info "MNN for Android built successfully"
}

#################
# Build llama.cpp #
#################
build_llama_linux() {
    print_info "Building llama.cpp for Linux x86_64..."
    cd "${THIRD_PARTY_DIR}/llama.cpp"

    rm -rf build-linux
    mkdir -p build-linux
    cd build-linux

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_CUDA=OFF \
        -DGGML_METAL=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_INSTALL_PREFIX="${PWD}/install"

    make -j$(nproc)
    make install

    # Copy libraries to prebuilt/linux-x86_64/llama/lib/
    if [ -f "install/lib/libllama.so" ]; then
        cp install/lib/libllama.so "${PREBUILT_LINUX_LLAMA}/lib/"
        print_info "Copied libllama.so to prebuilt/linux-x86_64/llama/lib/"
    fi

    # Copy all other libraries (ggml, etc.)
    if [ -d "install/lib" ]; then
        cp install/lib/*.so* "${PREBUILT_LINUX_LLAMA}/lib/" 2>/dev/null || true
    fi

    # Copy headers to prebuilt/linux-x86_64/llama/include/
    cp -r install/include/* "${PREBUILT_LINUX_LLAMA}/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/llama.cpp/include"/* "${PREBUILT_LINUX_LLAMA}/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/llama.cpp/ggml/include"/* "${PREBUILT_LINUX_LLAMA}/include/" 2>/dev/null || true

    print_info "llama.cpp for Linux built successfully"
}

build_llama_android() {
    print_info "Building llama.cpp for Android aarch64..."
    cd "${THIRD_PARTY_DIR}/llama.cpp"

    rm -rf build-android
    mkdir -p build-android
    cd build-android

    cmake .. \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-24 \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_CUDA=OFF \
        -DGGML_METAL=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_INSTALL_PREFIX="${PWD}/install"

    make -j$(nproc)
    make install

    # Copy libraries to prebuilt/android-aarch64/llama/lib/
    if [ -f "install/lib/libllama.so" ]; then
        cp install/lib/libllama.so "${PREBUILT_ANDROID_LLAMA}/lib/"
        print_info "Copied libllama.so to prebuilt/android-aarch64/llama/lib/"
    fi

    # Copy all other libraries (ggml, etc.)
    if [ -d "install/lib" ]; then
        cp install/lib/*.so* "${PREBUILT_ANDROID_LLAMA}/lib/" 2>/dev/null || true
    fi

    # Copy headers to prebuilt/android-aarch64/llama/include/
    cp -r install/include/* "${PREBUILT_ANDROID_LLAMA}/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/llama.cpp/include"/* "${PREBUILT_ANDROID_LLAMA}/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/llama.cpp/ggml/include"/* "${PREBUILT_ANDROID_LLAMA}/include/" 2>/dev/null || true

    print_info "llama.cpp for Android built successfully"
}

# Main build process
main() {
    print_info "Starting build process..."
    
    # Build for Linux
    if [ "$BUILD_LINUX" = true ]; then
        print_info "=== Building for Linux x86_64 ==="
        
        if [ "$BUILD_FAISS" = true ]; then
            build_faiss_linux
        fi
        
        if [ "$BUILD_MNN" = true ]; then
            build_mnn_linux
        fi
        
        if [ "$BUILD_LLAMA" = true ]; then
            build_llama_linux
        fi
    fi
    
    # Build for Android
    if [ "$BUILD_ANDROID" = true ]; then
        print_info "=== Building for Android aarch64 ==="
        
        if [ "$BUILD_FAISS" = true ]; then
            build_faiss_android
        fi
        
        if [ "$BUILD_MNN" = true ]; then
            build_mnn_android
        fi
        
        if [ "$BUILD_LLAMA" = true ]; then
            build_llama_android
        fi
    fi
    
    print_info "=== Build Summary ==="
    print_info "All builds completed successfully!"
    print_info "Prebuilt libraries are located in: ${PREBUILT_DIR}"

    if [ "$BUILD_LINUX" = true ]; then
        print_info "Linux x86_64 libraries:"
        if [ "$BUILD_MNN" = true ]; then
            print_info "  MNN: ${PREBUILT_LINUX_MNN}/"
            ls -lh "${PREBUILT_LINUX_MNN}/lib"/*.so 2>/dev/null || print_warn "No MNN libraries found"
        fi
        if [ "$BUILD_LLAMA" = true ]; then
            print_info "  llama.cpp: ${PREBUILT_LINUX_LLAMA}/"
            ls -lh "${PREBUILT_LINUX_LLAMA}/lib"/*.so 2>/dev/null || print_warn "No llama libraries found"
        fi
        if [ "$BUILD_FAISS" = true ]; then
            print_info "  Faiss: ${PREBUILT_LINUX_FAISS}/"
            ls -lh "${PREBUILT_LINUX_FAISS}/lib"/*.so 2>/dev/null || print_warn "No Faiss libraries found"
        fi
    fi

    if [ "$BUILD_ANDROID" = true ]; then
        print_info "Android aarch64 libraries:"
        if [ "$BUILD_MNN" = true ]; then
            print_info "  MNN: ${PREBUILT_ANDROID_MNN}/"
            ls -lh "${PREBUILT_ANDROID_MNN}/lib"/*.so 2>/dev/null || print_warn "No MNN libraries found"
        fi
        if [ "$BUILD_LLAMA" = true ]; then
            print_info "  llama.cpp: ${PREBUILT_ANDROID_LLAMA}/"
            ls -lh "${PREBUILT_ANDROID_LLAMA}/lib"/*.so 2>/dev/null || print_warn "No llama libraries found"
        fi
        if [ "$BUILD_FAISS" = true ]; then
            print_info "  Faiss: ${PREBUILT_ANDROID_FAISS}/"
            ls -lh "${PREBUILT_ANDROID_FAISS}/lib"/*.so 2>/dev/null || print_warn "No Faiss libraries found"
        fi
    fi
}

# Run main function
main

