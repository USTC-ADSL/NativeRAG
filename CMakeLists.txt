cmake_minimum_required(VERSION 3.15)

project(MobileRAG LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# ============================================================================
# 集成方式选择
# ============================================================================
# 选项1: 使用 prebuilt 库（推荐，快速）
# 选项2: 手动编译（需要运行 build_thirdparty.sh）
# ============================================================================
option(USE_PREBUILT "Use prebuilt libraries from prebuilt/ directory" ON)

if(USE_PREBUILT)
  message(STATUS "Using prebuilt libraries from prebuilt/ directory")
  set(PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt")
else()
  message(STATUS "Using third_party/ directory (requires manual compilation)")
  message(STATUS "To build third-party libraries, run: ./build_thirdparty.sh")
endif()

# Options to select vector index and LLM backend
set(VECTOR_INDEX "Faiss" CACHE STRING "Vector index implementation (Faiss or None)")
set_property(CACHE VECTOR_INDEX PROPERTY STRINGS Faiss None)

set(LLM_BACKEND "MNN" CACHE STRING "LLM backend (MNN, MLLM, or LlamaCpp)")
set_property(CACHE LLM_BACKEND PROPERTY STRINGS MNN MLLM LlamaCpp)

# 检测目标平台
if(ANDROID)
  set(TARGET_PLATFORM "android-aarch64")
  message(STATUS "Target platform: Android aarch64")
else()
  set(TARGET_PLATFORM "linux-x86_64")
  message(STATUS "Target platform: Linux x86_64")
endif()

# ============================================================================
# 辅助函数：查找库（支持 prebuilt 和 third_party 两种方式）
# ============================================================================
function(find_library_with_fallback LIB_NAME LIB_VAR INCLUDE_VAR)
  if(USE_PREBUILT)
    # 从 prebuilt 目录查找
    find_library(${LIB_VAR} NAMES ${LIB_NAME}
      HINTS "${PREBUILT_DIR}/include"
            "${PREBUILT_DIR}/${TARGET_PLATFORM}/${LIB_NAME}"
      NO_DEFAULT_PATH
    )
    find_path(${INCLUDE_VAR} NAMES ${ARGN}
      HINTS "${PREBUILT_DIR}/include"
      NO_DEFAULT_PATH
    )
  else()
    # 从 third_party 目录查找
    find_library(${LIB_VAR} NAMES ${LIB_NAME}
      HINTS "${THIRD_PARTY_DIR}/${LIB_NAME}"
            "${THIRD_PARTY_DIR}/${LIB_NAME}/build"
            "${THIRD_PARTY_DIR}/${LIB_NAME}/lib"
      NO_DEFAULT_PATH
    )
    find_path(${INCLUDE_VAR} NAMES ${ARGN}
      HINTS "${THIRD_PARTY_DIR}/${LIB_NAME}"
            "${THIRD_PARTY_DIR}/${LIB_NAME}/include"
      NO_DEFAULT_PATH
    )
  endif()
endfunction()

# Collect all sources in src/ recursively (including main.cpp)
file(GLOB_RECURSE MOBILERAG_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
# Exclude llm sources by default; add per-backend below
list(FILTER MOBILERAG_SOURCES EXCLUDE REGEX ".*/src/llm/.*\\.cpp$")

# Add CLI module sources
file(GLOB CLI_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cli/*.cpp"
)
list(APPEND MOBILERAG_SOURCES ${CLI_SOURCES})

add_executable(mobile_rag_cli ${MOBILERAG_SOURCES})

target_include_directories(mobile_rag_cli
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

############
# SQLite3 #
############
find_package(SQLite3 REQUIRED)
if (TARGET SQLite::SQLite3)
  target_link_libraries(mobile_rag_cli PRIVATE SQLite::SQLite3)
else()
  target_link_libraries(mobile_rag_cli PRIVATE ${SQLite3_LIBRARIES})
  target_include_directories(mobile_rag_cli PRIVATE ${SQLite3_INCLUDE_DIRS})
endif()

# Prebuilt third-party libraries (MNN and llama.cpp) in third_party/
#######################
# Vector Index Select #
#######################
if (VECTOR_INDEX STREQUAL "Faiss")
  message(STATUS "Using VECTOR_INDEX=Faiss")
  target_compile_definitions(mobile_rag_cli PRIVATE VECTOR_INDEX_FAISS=1)

  if(USE_PREBUILT)
    # 使用 prebuilt 库
    message(STATUS "Looking for Faiss in prebuilt/")
    set(FAISS_PREBUILT_DIR "${PREBUILT_DIR}/${TARGET_PLATFORM}/faiss")

    # Add Faiss include directory directly (headers are in prebuilt/include/faiss/)
    target_include_directories(mobile_rag_cli SYSTEM PRIVATE "${PREBUILT_DIR}/include")

    # Find library in prebuilt/platform/faiss/ (libraries are directly in the directory)
    find_library(FAISS_LIBRARY NAMES faiss
      HINTS "${FAISS_PREBUILT_DIR}"
      NO_DEFAULT_PATH
    )

    # Also link OpenBLAS if it exists (for Android)
    if(TARGET_PLATFORM STREQUAL "android-aarch64")
      find_library(OPENBLAS_LIBRARY NAMES openblas
        HINTS "${PREBUILT_DIR}/${TARGET_PLATFORM}/openblas/lib"
        NO_DEFAULT_PATH
      )
      if(OPENBLAS_LIBRARY)
        message(STATUS "Found OpenBLAS library: ${OPENBLAS_LIBRARY}")
        target_link_libraries(mobile_rag_cli PRIVATE ${OPENBLAS_LIBRARY})
      endif()
    endif()
  else()
    # 使用 third_party 库
    message(STATUS "Looking for Faiss in third_party/")
    target_include_directories(mobile_rag_cli PRIVATE "${THIRD_PARTY_DIR}/faiss")

    find_library(FAISS_LIBRARY NAMES faiss Faiss
      HINTS
        "${THIRD_PARTY_DIR}/faiss/build"
        "${THIRD_PARTY_DIR}/faiss/build/lib"
        "${THIRD_PARTY_DIR}/faiss/lib"
        "${THIRD_PARTY_DIR}/faiss"
      NO_DEFAULT_PATH
    )
  endif()

  if (FAISS_LIBRARY)
    message(STATUS "Found Faiss library: ${FAISS_LIBRARY}")
    target_link_libraries(mobile_rag_cli PRIVATE ${FAISS_LIBRARY})
  else()
    message(WARNING "Faiss library not found. Please ensure it is built or available in prebuilt/")
  endif()
elseif (VECTOR_INDEX STREQUAL "None")
  message(STATUS "Using VECTOR_INDEX=None (no external index library)")
else()
  message(FATAL_ERROR "Unsupported VECTOR_INDEX='${VECTOR_INDEX}'. Use: Faiss or None")
endif()

#################
# LLM Backend   #
#################
if (LLM_BACKEND STREQUAL "MNN")
  message(STATUS "Using LLM_BACKEND=MNN")
  target_compile_definitions(mobile_rag_cli PRIVATE LLM_BACKEND_MNN=1)

  if(USE_PREBUILT)
    # 使用 prebuilt 库
    message(STATUS "Looking for MNN in prebuilt/")
    set(MNN_PREBUILT_DIR "${PREBUILT_DIR}/${TARGET_PLATFORM}/MNN")

    # Include headers from prebuilt/include/MNN/
    target_include_directories(mobile_rag_cli PRIVATE "${PREBUILT_DIR}/include/MNN")

    # Find library in prebuilt/platform/MNN/ (libraries are directly in the directory)
    find_library(MNN_LIBRARY NAMES MNN
      HINTS "${MNN_PREBUILT_DIR}"
      NO_DEFAULT_PATH
    )
  else()
    # 使用 third_party 库
    message(STATUS "Looking for MNN in third_party/")
    find_path(MNN_INCLUDE_DIR NAMES MNN/Interpreter.hpp
      HINTS "${THIRD_PARTY_DIR}/MNN" "${THIRD_PARTY_DIR}/MNN/include"
      NO_DEFAULT_PATH
    )
    find_library(MNN_LIBRARY NAMES MNN libMNN
      HINTS "${THIRD_PARTY_DIR}/MNN" "${THIRD_PARTY_DIR}/MNN/lib" "${THIRD_PARTY_DIR}/MNN/build"
      NO_DEFAULT_PATH
    )
    if (MNN_INCLUDE_DIR)
      target_include_directories(mobile_rag_cli PRIVATE "${MNN_INCLUDE_DIR}")
    endif()
    # Additional include directories requested for MNN
    target_include_directories(mobile_rag_cli PRIVATE
      "${THIRD_PARTY_DIR}/MNN/tools/audio/include"
      "${THIRD_PARTY_DIR}/MNN/transformers/llm/engine/include"
      "${THIRD_PARTY_DIR}/MNN/include"
      "${THIRD_PARTY_DIR}/MNN/tools/cv/include"
      "${THIRD_PARTY_DIR}/MNN/3rd_party"
    )
  endif()

  if (MNN_LIBRARY)
    message(STATUS "Found MNN library: ${MNN_LIBRARY}")
    target_link_libraries(mobile_rag_cli PRIVATE "${MNN_LIBRARY}")

    # Also link MNN_Express if using prebuilt
    if(USE_PREBUILT)
      find_library(MNN_EXPRESS_LIBRARY NAMES MNN_Express
        HINTS "${MNN_PREBUILT_DIR}"
        NO_DEFAULT_PATH
      )
      if(MNN_EXPRESS_LIBRARY)
        message(STATUS "Found MNN_Express library: ${MNN_EXPRESS_LIBRARY}")
        target_link_libraries(mobile_rag_cli PRIVATE "${MNN_EXPRESS_LIBRARY}")
      endif()

      # Link libllm if it exists (generated when MNN_BUILD_LLM=ON)
      find_library(MNN_LLM_LIBRARY NAMES llm
        HINTS "${MNN_PREBUILT_DIR}"
        NO_DEFAULT_PATH
      )
      if(MNN_LLM_LIBRARY)
        message(STATUS "Found MNN LLM library: ${MNN_LLM_LIBRARY}")
        target_link_libraries(mobile_rag_cli PRIVATE "${MNN_LLM_LIBRARY}")
      else()
        message(WARNING "MNN LLM library not found. Make sure to compile MNN with -DMNN_BUILD_LLM=ON")
      endif()
    endif()
  else()
    message(WARNING "MNN library not found. Please build it or use prebuilt libraries.")
  endif()

  # Add backend-specific LLM sources
  target_sources(mobile_rag_cli PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llm/MNNModel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llm/LLMFactory_MNN.cpp"
  )
elseif (LLM_BACKEND STREQUAL "MLLM")
  message(STATUS "Using LLM_BACKEND=MLLM")
  target_compile_definitions(mobile_rag_cli PRIVATE LLM_BACKEND_MLLM=1)

  if(USE_PREBUILT)
    # 使用 prebuilt 库
    message(STATUS "Looking for MLLM in prebuilt/")
    set(MLLM_PREBUILT_DIR "${PREBUILT_DIR}/${TARGET_PLATFORM}/mllm")

    # Include headers from prebuilt/include/mllm/
    target_include_directories(mobile_rag_cli PRIVATE "${PREBUILT_DIR}/include/mllm")

    # Find library in prebuilt/platform/mllm/ (libraries are directly in the directory)
    find_library(MLLM_LIBRARY NAMES mllm
      HINTS "${MLLM_PREBUILT_DIR}"
      NO_DEFAULT_PATH
    )
  else()
    # 使用 third_party 库
    message(STATUS "Looking for MLLM in third_party/")
    find_path(MLLM_INCLUDE_DIR NAMES mllm/mllm.h
      HINTS "${THIRD_PARTY_DIR}/mllm" "${THIRD_PARTY_DIR}/mllm/include"
      NO_DEFAULT_PATH)
    find_library(MLLM_LIBRARY NAMES mllm libmllm
      HINTS "${THIRD_PARTY_DIR}/mllm/build" "${THIRD_PARTY_DIR}/mllm/lib" "${THIRD_PARTY_DIR}/mllm"
      NO_DEFAULT_PATH)
    if (MLLM_INCLUDE_DIR)
      target_include_directories(mobile_rag_cli PRIVATE "${MLLM_INCLUDE_DIR}")
    endif()
  endif()

  if (MLLM_LIBRARY)
    message(STATUS "Found MLLM library: ${MLLM_LIBRARY}")
    target_link_libraries(mobile_rag_cli PRIVATE "${MLLM_LIBRARY}")
  else()
    message(WARNING "MLLM library not found. Please build it or use prebuilt libraries.")
  endif()

  target_sources(mobile_rag_cli PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llm/LLMFactory_MLLM.cpp"
  )
elseif (LLM_BACKEND STREQUAL "LlamaCpp")
  message(STATUS "Using LLM_BACKEND=LlamaCpp")

  if(USE_PREBUILT)
    # 使用 prebuilt 库
    message(STATUS "Looking for llama.cpp in prebuilt/")
    set(LLAMA_PREBUILT_DIR "${PREBUILT_DIR}/${TARGET_PLATFORM}/llama")

    # Include headers from prebuilt/include/llama/
    target_include_directories(mobile_rag_cli PRIVATE "${PREBUILT_DIR}/include/llama")

    # Find library in prebuilt/platform/llama/ (libraries are directly in the directory)
    find_library(LLAMA_LIBRARY NAMES llama
      HINTS "${LLAMA_PREBUILT_DIR}"
      NO_DEFAULT_PATH
    )
  else()
    # 使用 third_party 库
    message(STATUS "Looking for llama.cpp in third_party/")
    find_path(LLAMA_INCLUDE_DIR NAMES llama.h
      HINTS "${THIRD_PARTY_DIR}/llama.cpp" "${THIRD_PARTY_DIR}/llama.cpp/include"
      NO_DEFAULT_PATH
    )
    find_library(LLAMA_LIBRARY NAMES llama libllama
      HINTS "${THIRD_PARTY_DIR}/llama.cpp" "${THIRD_PARTY_DIR}/llama.cpp/lib" "${THIRD_PARTY_DIR}/llama.cpp/build"
      NO_DEFAULT_PATH
    )
    if (LLAMA_INCLUDE_DIR)
      target_include_directories(mobile_rag_cli PRIVATE "${LLAMA_INCLUDE_DIR}")
    endif()
    # Always add bundled llama.cpp include directory
    target_include_directories(mobile_rag_cli PRIVATE "${THIRD_PARTY_DIR}/llama.cpp/include")
    # Add ggml headers for nested includes from llama.h
    target_include_directories(mobile_rag_cli PRIVATE "${THIRD_PARTY_DIR}/llama.cpp/ggml/include")
  endif()

  if (LLAMA_LIBRARY)
    message(STATUS "Found llama.cpp library: ${LLAMA_LIBRARY}")
    target_link_libraries(mobile_rag_cli PRIVATE "${LLAMA_LIBRARY}")
  else()
    message(WARNING "llama.cpp library not found. Please build it or use prebuilt libraries.")
  endif()

  target_sources(mobile_rag_cli PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llm/LlamaCppModel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llm/LLMFactory_Llama.cpp"
  )
else()
  message(FATAL_ERROR "Unsupported LLM_BACKEND='${LLM_BACKEND}'. Use: MNN, MLLM, or LlamaCpp")
endif()

# 测试支持
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

# ============================================================================
# 使用说明
# ============================================================================
#
# 方式1: 使用 prebuilt 库（推荐，快速）
#   mkdir build && cd build
#   cmake .. -DUSE_PREBUILT=ON
#   make -j$(nproc)
#
# 方式2: 手动编译第三方库（需要更多时间）
#   1. 首先编译第三方库:
#      ./build_thirdparty.sh
#   2. 然后编译项目:
#      mkdir build && cd build
#      cmake .. -DUSE_PREBUILT=OFF
#      make -j$(nproc)
#
# 选择 LLM 后端:
#   cmake .. -DLLM_BACKEND=MNN      # 默认
#   cmake .. -DLLM_BACKEND=MLLM
#   cmake .. -DLLM_BACKEND=LlamaCpp
#
# 选择向量索引:
#   cmake .. -DVECTOR_INDEX=Faiss   # 默认
#   cmake .. -DVECTOR_INDEX=None
#
# ============================================================================
