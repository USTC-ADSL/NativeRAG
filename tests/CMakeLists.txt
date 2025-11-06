# Tests CMakeLists.txt

# Get prebuilt directory from parent CMakeLists
if(USE_PREBUILT)
    set(PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt")
else()
    set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party")
endif()

# Find required libraries for tests
if(USE_PREBUILT)
    set(TARGET_PLATFORM "linux-x86_64")
    if(ANDROID)
        set(TARGET_PLATFORM "android-aarch64")
    endif()

    set(FAISS_PREBUILT_DIR "${PREBUILT_DIR}/${TARGET_PLATFORM}/faiss")

    # Add Faiss include directory directly (headers are in prebuilt/include/faiss/)
    set(FAISS_INCLUDE_WRAPPER "${PREBUILT_DIR}/include")

    # Find Faiss library in prebuilt/platform/faiss/ (libraries are directly in the directory)
    find_library(FAISS_LIBRARY NAMES faiss
        HINTS "${FAISS_PREBUILT_DIR}"
        NO_DEFAULT_PATH
    )

    # Find MNN library in prebuilt/platform/MNN/ (libraries are directly in the directory)
    find_library(MNN_LIBRARY NAMES MNN
        HINTS "${PREBUILT_DIR}/${TARGET_PLATFORM}/MNN"
        NO_DEFAULT_PATH
    )

    # Find MNN_Express library in prebuilt/platform/MNN/ (libraries are directly in the directory)
    find_library(MNN_EXPRESS_LIBRARY NAMES MNN_Express
        HINTS "${PREBUILT_DIR}/${TARGET_PLATFORM}/MNN"
        NO_DEFAULT_PATH
    )

    # Find llm library in prebuilt/platform/MNN/ (libraries are directly in the directory)
    find_library(MNN_LLM_LIBRARY NAMES llm
        HINTS "${PREBUILT_DIR}/${TARGET_PLATFORM}/MNN"
        NO_DEFAULT_PATH
    )

    # Find llama.cpp library in prebuilt/platform/llama/ (libraries are directly in the directory)
    find_library(LLAMA_LIBRARY NAMES llama
        HINTS "${PREBUILT_DIR}/${TARGET_PLATFORM}/llama"
        NO_DEFAULT_PATH
    )
endif()

# Build dataset library
add_library(dataset_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../dataset/src/TrivialQADataset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../dataset/src/VectorDataGenerator.cpp
)

target_include_directories(dataset_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../dataset/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Find nlohmann_json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    # Try to find it in system paths
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
    if(NLOHMANN_JSON_INCLUDE_DIR)
        target_include_directories(dataset_lib PUBLIC ${NLOHMANN_JSON_INCLUDE_DIR})
    else()
        message(WARNING "nlohmann_json not found. Tests may fail to compile.")
    endif()
else()
    target_link_libraries(dataset_lib PUBLIC nlohmann_json::nlohmann_json)
endif()

# Group C VectorDB tests
add_executable(test_vectordb
    ${CMAKE_CURRENT_SOURCE_DIR}/group_c_vectordb/test_vectordb.cpp
)

target_include_directories(test_vectordb
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../dataset/include
)

target_link_libraries(test_vectordb
    PRIVATE
        dataset_lib
)

# Core RAG library (implementation files)
add_library(rag_core STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/RAGPipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/embedding/MNNEmbedding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/embedding/LlamaCppEmbedding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/vector_db/FaissIndex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/vector_db/SqliteVectorDB.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/llm/MNNModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/llm/LLMFactory_MNN.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/loader/TXTLoader/TxtLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/loader/TXTLoader/TextFileLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/loader/Chunk/CharacterSplitter.cpp
)

target_include_directories(rag_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Add Faiss headers as system include to handle both "faiss/Index.h" and <faiss/MetricType.h>
target_include_directories(rag_core SYSTEM PUBLIC
    ${FAISS_INCLUDE_WRAPPER}
    ${PREBUILT_DIR}/include/MNN
    ${PREBUILT_DIR}/include/llama
)

target_link_libraries(rag_core PUBLIC
    ${MNN_LIBRARY}
    ${MNN_EXPRESS_LIBRARY}
    ${MNN_LLM_LIBRARY}
    ${FAISS_LIBRARY}
    ${LLAMA_LIBRARY}
)

# RAGPipelineWithDataset library
add_library(rag_pipeline_dataset STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/RAGPipelineWithDataset.cpp
)

target_include_directories(rag_pipeline_dataset PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(rag_pipeline_dataset PUBLIC
    dataset_lib
    rag_core
)

# Group A (Embedding) tests
add_executable(test_embedding
    ${CMAKE_CURRENT_SOURCE_DIR}/group_a_embedding/test_embedding.cpp
)

target_include_directories(test_embedding PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PREBUILT_DIR}/include/faiss
    ${PREBUILT_DIR}/include/MNN
    ${PREBUILT_DIR}/include/llama
)

target_link_libraries(test_embedding PRIVATE
    dataset_lib
    rag_core
)

# Group B (LLM) tests
add_executable(test_llm
    ${CMAKE_CURRENT_SOURCE_DIR}/group_b_llm/test_llm.cpp
)

target_include_directories(test_llm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PREBUILT_DIR}/include/faiss
    ${PREBUILT_DIR}/include/MNN
    ${PREBUILT_DIR}/include/llama
)

target_link_libraries(test_llm PRIVATE
    dataset_lib
    rag_core
)

# Group C (VectorDB) integrated tests
add_executable(test_vectordb_integrated
    ${CMAKE_CURRENT_SOURCE_DIR}/group_c_vectordb/test_vectordb_integrated.cpp
)

target_include_directories(test_vectordb_integrated PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PREBUILT_DIR}/include/faiss
    ${PREBUILT_DIR}/include/MNN
    ${PREBUILT_DIR}/include/llama
)

target_link_libraries(test_vectordb_integrated PRIVATE
    dataset_lib
    rag_core
)

# RAG Pipeline integration tests
add_executable(test_rag_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_rag_pipeline_integration.cpp
)

target_include_directories(test_rag_integration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PREBUILT_DIR}/include/faiss
    ${PREBUILT_DIR}/include/MNN
    ${PREBUILT_DIR}/include/llama
)

target_link_libraries(test_rag_integration PRIVATE
    rag_pipeline_dataset
    dataset_lib
    rag_core
)

# Enable testing
enable_testing()
add_test(NAME VectorDBTest COMMAND test_vectordb)
add_test(NAME EmbeddingTest COMMAND test_embedding)
add_test(NAME LLMTest COMMAND test_llm)
add_test(NAME VectorDBIntegratedTest COMMAND test_vectordb_integrated)
add_test(NAME RAGIntegrationTest COMMAND test_rag_integration)

# Set test properties
set_tests_properties(
    EmbeddingTest
    LLMTest
    VectorDBTest
    VectorDBIntegratedTest
    RAGIntegrationTest
    PROPERTIES
    TIMEOUT 120
    LABELS "integration"
)

message(STATUS "Tests configured:")
message(STATUS "  - test_embedding (Group A)")
message(STATUS "  - test_llm (Group B)")
message(STATUS "  - test_vectordb (Group C - basic)")
message(STATUS "  - test_vectordb_integrated (Group C - integrated)")
message(STATUS "  - test_rag_integration (Full pipeline)")

