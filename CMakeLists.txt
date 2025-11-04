cmake_minimum_required(VERSION 3.15)

project(MobileRAG LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect all sources in src/ recursively (including main.cpp)
file(GLOB_RECURSE MOBILERAG_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(mobile_rag_cli ${MOBILERAG_SOURCES})

target_include_directories(mobile_rag_cli
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Dependencies: Faiss and SQLite3 via find_package
find_package(Faiss REQUIRED)
find_package(SQLite3 REQUIRED)

# Link Faiss - prefer imported target if available
if (TARGET Faiss::faiss)
  target_link_libraries(mobile_rag_cli PRIVATE Faiss::faiss)
elseif (TARGET faiss)
  target_link_libraries(mobile_rag_cli PRIVATE faiss)
else()
  # Fallback to library name if the package provides no target
  target_link_libraries(mobile_rag_cli PRIVATE faiss)
endif()

# Link SQLite3 imported target
target_link_libraries(mobile_rag_cli PRIVATE SQLite::SQLite3)

# Prebuilt third-party libraries (MNN and llama.cpp) in third_party/
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# MNN
find_path(MNN_INCLUDE_DIR NAMES MNN/Interpreter.hpp
  HINTS "${THIRD_PARTY_DIR}/MNN" "${THIRD_PARTY_DIR}/MNN/include"
  NO_DEFAULT_PATH
)
find_library(MNN_LIBRARY NAMES MNN libMNN
  HINTS "${THIRD_PARTY_DIR}/MNN" "${THIRD_PARTY_DIR}/MNN/lib"
  NO_DEFAULT_PATH
)
if (MNN_INCLUDE_DIR AND MNN_LIBRARY)
  target_include_directories(mobile_rag_cli PRIVATE "${MNN_INCLUDE_DIR}")
  target_link_libraries(mobile_rag_cli PRIVATE "${MNN_LIBRARY}")
else()
  message(STATUS "MNN not found in third_party. Skipping explicit link.")
endif()

# llama.cpp
find_path(LLAMA_INCLUDE_DIR NAMES llama.h
  HINTS "${THIRD_PARTY_DIR}/llama.cpp" "${THIRD_PARTY_DIR}/llama.cpp/include"
  NO_DEFAULT_PATH
)
find_library(LLAMA_LIBRARY NAMES llama libllama
  HINTS "${THIRD_PARTY_DIR}/llama.cpp" "${THIRD_PARTY_DIR}/llama.cpp/lib"
  NO_DEFAULT_PATH
)
if (LLAMA_INCLUDE_DIR AND LLAMA_LIBRARY)
  target_include_directories(mobile_rag_cli PRIVATE "${LLAMA_INCLUDE_DIR}")
  target_link_libraries(mobile_rag_cli PRIVATE "${LLAMA_LIBRARY}")
else()
  message(STATUS "llama.cpp not found in third_party. Skipping explicit link.")
endif()


