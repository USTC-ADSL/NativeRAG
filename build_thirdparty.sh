#!/bin/bash
set -e

# Build script for compiling third-party libraries (Faiss, MNN, llama.cpp)
# for both Linux x86_64 and Android aarch64

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${SCRIPT_DIR}/third_party"
PREBUILT_DIR="${SCRIPT_DIR}/prebuilt"

# New directory structure
PREBUILT_INCLUDE="${PREBUILT_DIR}/include"
PREBUILT_LINUX_FAISS="${PREBUILT_DIR}/linux-x86_64/faiss"
PREBUILT_LINUX_MNN="${PREBUILT_DIR}/linux-x86_64/MNN"
PREBUILT_LINUX_LLAMA="${PREBUILT_DIR}/linux-x86_64/llama"
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

# Create prebuilt directories
mkdir -p "${PREBUILT_DIR}/linux-x86_64"/{lib,include}
mkdir -p "${PREBUILT_DIR}/android-aarch64"/{lib,include}

#################
# Build Faiss   #
#################
build_faiss_linux() {
    print_info "Building Faiss for Linux x86_64..."
    cd "${THIRD_PARTY_DIR}/faiss"

    rm -rf build-linux
    mkdir -p build-linux
    cd build-linux

    # Create directories
    mkdir -p "${PREBUILT_LINUX_FAISS}"
    mkdir -p "${PREBUILT_INCLUDE}/faiss"

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=OFF \
        -DCMAKE_INSTALL_PREFIX="${PWD}/install"

    make -j$(nproc)
    make install

    # Copy libraries and headers to new structure
    cp install/lib/libfaiss.so "${PREBUILT_LINUX_FAISS}/"
    cp -r install/include/faiss/* "${PREBUILT_INCLUDE}/faiss/"

    print_info "Faiss for Linux built successfully"
}

build_faiss_android() {
    print_info "Building Faiss for Android aarch64..."
    print_warn "Skipping Faiss for Android - requires BLAS library which is not readily available on Android"
    print_warn "For Android, consider using a lighter vector search library or cross-compile OpenBLAS"
    return 0
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
        -DCMAKE_INSTALL_PREFIX="${PREBUILT_DIR}/linux-x86_64"
    
    make -j$(nproc)
    make install
    
    # Copy additional headers that MNN needs
    cp -r "${THIRD_PARTY_DIR}/MNN/include"/* "${PREBUILT_DIR}/linux-x86_64/include/"
    mkdir -p "${PREBUILT_DIR}/linux-x86_64/include/MNN"
    cp -r "${THIRD_PARTY_DIR}/MNN/tools/cv/include"/* "${PREBUILT_DIR}/linux-x86_64/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/MNN/tools/audio/include"/* "${PREBUILT_DIR}/linux-x86_64/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/MNN/transformers/llm/engine/include"/* "${PREBUILT_DIR}/linux-x86_64/include/" 2>/dev/null || true
    
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
        -DCMAKE_INSTALL_PREFIX="${PREBUILT_DIR}/android-aarch64"
    
    make -j$(nproc)
    make install
    
    # Copy additional headers
    cp -r "${THIRD_PARTY_DIR}/MNN/include"/* "${PREBUILT_DIR}/android-aarch64/include/"
    mkdir -p "${PREBUILT_DIR}/android-aarch64/include/MNN"
    cp -r "${THIRD_PARTY_DIR}/MNN/tools/cv/include"/* "${PREBUILT_DIR}/android-aarch64/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/MNN/tools/audio/include"/* "${PREBUILT_DIR}/android-aarch64/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/MNN/transformers/llm/engine/include"/* "${PREBUILT_DIR}/android-aarch64/include/" 2>/dev/null || true
    
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
        -DCMAKE_INSTALL_PREFIX="${PREBUILT_DIR}/linux-x86_64"

    make -j$(nproc)
    make install

    # Copy headers
    cp -r "${THIRD_PARTY_DIR}/llama.cpp/include"/* "${PREBUILT_DIR}/linux-x86_64/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/llama.cpp/ggml/include"/* "${PREBUILT_DIR}/linux-x86_64/include/" 2>/dev/null || true

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
        -DCMAKE_INSTALL_PREFIX="${PREBUILT_DIR}/android-aarch64"

    make -j$(nproc)
    make install

    # Copy headers
    cp -r "${THIRD_PARTY_DIR}/llama.cpp/include"/* "${PREBUILT_DIR}/android-aarch64/include/" 2>/dev/null || true
    cp -r "${THIRD_PARTY_DIR}/llama.cpp/ggml/include"/* "${PREBUILT_DIR}/android-aarch64/include/" 2>/dev/null || true

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
        ls -lh "${PREBUILT_DIR}/linux-x86_64/lib/" 2>/dev/null || print_warn "No libraries found"
    fi
    
    if [ "$BUILD_ANDROID" = true ]; then
        print_info "Android aarch64 libraries:"
        ls -lh "${PREBUILT_DIR}/android-aarch64/lib/" 2>/dev/null || print_warn "No libraries found"
    fi
}

# Run main function
main

