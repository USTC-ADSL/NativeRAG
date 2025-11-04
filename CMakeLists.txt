cmake_minimum_required(VERSION 3.15)

project(MobileRAG LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# Options to select vector index and LLM backend
set(VECTOR_INDEX "Faiss" CACHE STRING "Vector index implementation (Faiss or None)")
set_property(CACHE VECTOR_INDEX PROPERTY STRINGS Faiss None)

set(LLM_BACKEND "MNN" CACHE STRING "LLM backend (MNN, MLLM, or LlamaCpp)")
set_property(CACHE LLM_BACKEND PROPERTY STRINGS MNN MLLM LlamaCpp)

# Collect all sources in src/ recursively (including main.cpp)
file(GLOB_RECURSE MOBILERAG_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
# Exclude llm sources by default; add per-backend below
list(FILTER MOBILERAG_SOURCES EXCLUDE REGEX ".*/src/llm/.*\\.cpp$")

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

  # Headers: third_party/faiss contains folder 'faiss' for includes
  target_include_directories(mobile_rag_cli PRIVATE "${THIRD_PARTY_DIR}/faiss")

  # Try to find a prebuilt faiss library under third_party
  find_library(FAISS_LIBRARY NAMES faiss Faiss
    HINTS
      "${THIRD_PARTY_DIR}/faiss/build"
      "${THIRD_PARTY_DIR}/faiss/build/lib"
      "${THIRD_PARTY_DIR}/faiss/lib"
      "${THIRD_PARTY_DIR}/faiss"
  )
  if (FAISS_LIBRARY)
    target_link_libraries(mobile_rag_cli PRIVATE ${FAISS_LIBRARY})
  else()
    message(STATUS "Faiss library not found in third_party. If needed, set FAISS_LIBRARY or add_subdirectory.")
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
  if (MNN_LIBRARY)
    target_link_libraries(mobile_rag_cli PRIVATE "${MNN_LIBRARY}")
  else()
    message(STATUS "MNN library not found in third_party. Build MNN or set MNN_LIBRARY.")
  endif()
  # Add backend-specific LLM sources
  target_sources(mobile_rag_cli PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llm/MNNModel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llm/LLMFactory_MNN.cpp"
  )
elseif (LLM_BACKEND STREQUAL "MLLM")
  message(STATUS "Using LLM_BACKEND=MLLM")
  target_compile_definitions(mobile_rag_cli PRIVATE LLM_BACKEND_MLLM=1)
  # Try to locate mllm includes and library
  find_path(MLLM_INCLUDE_DIR NAMES mllm/mllm.h
    HINTS "${THIRD_PARTY_DIR}/mllm" "${THIRD_PARTY_DIR}/mllm/include"
    NO_DEFAULT_PATH)
  find_library(MLLM_LIBRARY NAMES mllm libmllm
    HINTS "${THIRD_PARTY_DIR}/mllm/build" "${THIRD_PARTY_DIR}/mllm/lib" "${THIRD_PARTY_DIR}/mllm"
    NO_DEFAULT_PATH)
  if (MLLM_INCLUDE_DIR)
    target_include_directories(mobile_rag_cli PRIVATE "${MLLM_INCLUDE_DIR}")
  endif()
  if (MLLM_LIBRARY)
    target_link_libraries(mobile_rag_cli PRIVATE "${MLLM_LIBRARY}")
  else()
    message(STATUS "MLLM library not found in third_party. Build mllm or set MLLM_LIBRARY.")
  endif()
  target_sources(mobile_rag_cli PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llm/LLMFactory_MLLM.cpp"
  )
elseif (LLM_BACKEND STREQUAL "LlamaCpp")
  message(STATUS "Using LLM_BACKEND=LlamaCpp")
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
  if (LLAMA_LIBRARY)
    target_link_libraries(mobile_rag_cli PRIVATE "${LLAMA_LIBRARY}")
  else()
    message(STATUS "llama.cpp library not found in third_party. Build or set LLAMA_LIBRARY.")
  endif()
  target_sources(mobile_rag_cli PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llm/LlamaCppModel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llm/LLMFactory_Llama.cpp"
  )
else()
  message(FATAL_ERROR "Unsupported LLM_BACKEND='${LLM_BACKEND}'. Use: MNN, MLLM, or LlamaCpp")
endif()


