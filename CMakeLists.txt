cmake_minimum_required(VERSION 3.15)

project(NativeRAG LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT ANDROID)
  message(FATAL_ERROR "NativeRAG currently supports Android builds only")
endif()

if(CMAKE_SYSTEM_PROCESSOR AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
  message(FATAL_ERROR "NativeRAG currently supports Android arm64-v8a only")
endif()

string(REGEX REPLACE "^android-" "" ANDROID_API_LEVEL "${ANDROID_PLATFORM}")
if(ANDROID_API_LEVEL VERSION_LESS 31)
  message(FATAL_ERROR "NativeRAG requires ANDROID_PLATFORM=android-31 or newer")
endif()

set(NATIVERAG_ACCELERATOR "CPU" CACHE STRING
    "Compile-time llama.cpp accelerator profile: CPU, OPENCL, or HEXAGON")
set_property(CACHE NATIVERAG_ACCELERATOR PROPERTY STRINGS CPU OPENCL HEXAGON)
string(TOUPPER "${NATIVERAG_ACCELERATOR}" NATIVERAG_ACCELERATOR)
if(NOT NATIVERAG_ACCELERATOR MATCHES "^(CPU|OPENCL|HEXAGON)$")
  message(FATAL_ERROR
    "Unsupported NATIVERAG_ACCELERATOR='${NATIVERAG_ACCELERATOR}'. "
    "Expected CPU, OPENCL, or HEXAGON")
endif()

set(LLAMA_CPP_ROOT "" CACHE PATH
    "Path to the matching official llama.cpp source tree")
set(LLAMA_LIBRARY_DIR "" CACHE PATH
    "Directory containing matching Android llama.cpp shared libraries")

if(NOT EXISTS "${LLAMA_CPP_ROOT}/include/llama.h" OR
   NOT EXISTS "${LLAMA_CPP_ROOT}/ggml/include/ggml-backend.h")
  message(FATAL_ERROR
    "LLAMA_CPP_ROOT must point to the source tree used to build the llama.cpp libraries")
endif()
if(NOT IS_DIRECTORY "${LLAMA_LIBRARY_DIR}")
  message(FATAL_ERROR
    "LLAMA_LIBRARY_DIR must contain the matching Android llama.cpp shared libraries")
endif()

set(PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt")
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(TARGET_PLATFORM "android-aarch64")

set(NATIVERAG_SOURCES
  src/main.cpp
  src/RAGPipeline.cpp
  src/cli/CommandLineArgs.cpp
  src/loader/TXTLoader/TextFileLoader.cpp
  src/loader/TXTLoader/TxtLoader.cpp
  src/loader/Chunk/CharacterSplitter.cpp
  src/vector_db/SqliteVectorDB.cpp
  src/vector_db/FaissIndex.cpp
  src/embedding/LlamaCppEmbedding.cpp
  src/embedding/EmbeddingFactory_Llama.cpp
  src/reranker/LlamaCppReranker.cpp
  src/reranker/RerankerFactory_Llama.cpp
  src/llm/LlamaCppModel.cpp
  src/llm/LLMFactory_Llama.cpp
  src/llama/LlamaRuntime.cpp
)

add_executable(mobile_rag_cli ${NATIVERAG_SOURCES})
target_include_directories(mobile_rag_cli PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_include_directories(mobile_rag_cli SYSTEM PRIVATE
  "${PREBUILT_DIR}/include"
  "${LLAMA_CPP_ROOT}/include"
  "${LLAMA_CPP_ROOT}/ggml/include"
)
target_compile_definitions(mobile_rag_cli PRIVATE
  NATIVERAG_USE_LLAMA=1
  "NATIVERAG_ACCEL_${NATIVERAG_ACCELERATOR}=1"
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(mobile_rag_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NOT EXISTS "${THIRD_PARTY_DIR}/sqlite/sqlite3.c")
  message(FATAL_ERROR "Bundled SQLite source is missing: third_party/sqlite/sqlite3.c")
endif()
add_library(sqlite3_local STATIC third_party/sqlite/sqlite3.c)
target_include_directories(sqlite3_local PUBLIC "${THIRD_PARTY_DIR}/sqlite")
target_compile_definitions(sqlite3_local PRIVATE
  SQLITE_THREADSAFE=1
  SQLITE_OMIT_LOAD_EXTENSION=1
  SQLITE_DEFAULT_MEMSTATUS=0
)

if(NOT EXISTS "${PREBUILT_DIR}/include/faiss/Index.h")
  message(FATAL_ERROR "Bundled Faiss headers are missing: prebuilt/include/faiss")
endif()
find_library(FAISS_LIBRARY NAMES faiss
  HINTS "${PREBUILT_DIR}/${TARGET_PLATFORM}/faiss"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)
find_library(OPENBLAS_LIBRARY NAMES openblas
  HINTS "${PREBUILT_DIR}/${TARGET_PLATFORM}/openblas"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)
if(NOT FAISS_LIBRARY)
  message(FATAL_ERROR "Bundled Android libfaiss.so was not found")
endif()
if(NOT OPENBLAS_LIBRARY)
  message(FATAL_ERROR "Bundled Android libopenblas.so was not found")
endif()

file(GLOB NDK_OPENMP_CANDIDATES
  "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/*/lib/clang/*/lib/linux/aarch64/libomp.so"
)
if(NOT NDK_OPENMP_CANDIDATES)
  message(FATAL_ERROR "Android libomp.so was not found in ${CMAKE_ANDROID_NDK}")
endif()
list(GET NDK_OPENMP_CANDIDATES 0 NDK_OPENMP_LIBRARY)
add_library(android_openmp SHARED IMPORTED)
set_target_properties(android_openmp PROPERTIES
  IMPORTED_LOCATION "${NDK_OPENMP_LIBRARY}"
)

find_library(LLAMA_LIBRARY NAMES llama
  HINTS "${LLAMA_LIBRARY_DIR}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
find_library(GGML_LIBRARY NAMES ggml
  HINTS "${LLAMA_LIBRARY_DIR}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
find_library(GGML_BASE_LIBRARY NAMES ggml-base
  HINTS "${LLAMA_LIBRARY_DIR}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
find_library(GGML_CPU_LIBRARY NAMES ggml-cpu
  HINTS "${LLAMA_LIBRARY_DIR}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

foreach(required_library
    LLAMA_LIBRARY GGML_LIBRARY GGML_BASE_LIBRARY GGML_CPU_LIBRARY)
  if(NOT ${required_library})
    message(FATAL_ERROR
      "${required_library} was not found in LLAMA_LIBRARY_DIR=${LLAMA_LIBRARY_DIR}")
  endif()
endforeach()

set(LLAMA_ACCELERATOR_LIBRARY "")
if(NATIVERAG_ACCELERATOR STREQUAL "OPENCL")
  find_library(GGML_OPENCL_LIBRARY NAMES ggml-opencl
    HINTS "${LLAMA_LIBRARY_DIR}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  if(NOT GGML_OPENCL_LIBRARY)
    message(FATAL_ERROR "libggml-opencl.so was not found in LLAMA_LIBRARY_DIR")
  endif()
  set(LLAMA_ACCELERATOR_LIBRARY "${GGML_OPENCL_LIBRARY}")
elseif(NATIVERAG_ACCELERATOR STREQUAL "HEXAGON")
  find_library(GGML_HEXAGON_LIBRARY NAMES ggml-hexagon
    HINTS "${LLAMA_LIBRARY_DIR}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  if(NOT GGML_HEXAGON_LIBRARY)
    message(FATAL_ERROR "libggml-hexagon.so was not found in LLAMA_LIBRARY_DIR")
  endif()
  set(LLAMA_ACCELERATOR_LIBRARY "${GGML_HEXAGON_LIBRARY}")
endif()

target_link_libraries(mobile_rag_cli PRIVATE
  sqlite3_local
  "${FAISS_LIBRARY}"
  "${OPENBLAS_LIBRARY}"
  android_openmp
  "${LLAMA_LIBRARY}"
  "${GGML_LIBRARY}"
  ${LLAMA_ACCELERATOR_LIBRARY}
  "${GGML_CPU_LIBRARY}"
  "${GGML_BASE_LIBRARY}"
  ${CMAKE_DL_LIBS}
)

set_target_properties(mobile_rag_cli PROPERTIES
  BUILD_RPATH
    "${PREBUILT_DIR}/${TARGET_PLATFORM}/faiss;${PREBUILT_DIR}/${TARGET_PLATFORM}/openblas;${LLAMA_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN/lib"
)

message(STATUS "Target platform: Android arm64-v8a (API ${ANDROID_API_LEVEL})")
message(STATUS "Embedding backend: llama.cpp")
message(STATUS "LLM backend: llama.cpp")
message(STATUS "Vector index: Faiss")
message(STATUS "Metadata store: bundled SQLite")
message(STATUS "NativeRAG accelerator profile: ${NATIVERAG_ACCELERATOR}")
message(STATUS "llama.cpp library path: ${LLAMA_LIBRARY_DIR}")
