# Tests CMakeLists.txt

# Build dataset library
add_library(dataset_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../dataset/src/TrivialQADataset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../dataset/src/VectorDataGenerator.cpp
)

target_include_directories(dataset_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../dataset/include
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
)

target_link_libraries(test_embedding PRIVATE
    rag_core
    dataset_lib
)

# Group B (LLM) tests
add_executable(test_llm
    ${CMAKE_CURRENT_SOURCE_DIR}/group_b_llm/test_llm.cpp
)

target_include_directories(test_llm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(test_llm PRIVATE
    rag_core
    dataset_lib
)

# Group C (VectorDB) integrated tests
add_executable(test_vectordb_integrated
    ${CMAKE_CURRENT_SOURCE_DIR}/group_c_vectordb/test_vectordb_integrated.cpp
)

target_include_directories(test_vectordb_integrated PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(test_vectordb_integrated PRIVATE
    rag_core
    dataset_lib
)

# RAG Pipeline integration tests
add_executable(test_rag_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_rag_pipeline_integration.cpp
)

target_include_directories(test_rag_integration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(test_rag_integration PRIVATE
    rag_core
    rag_pipeline_dataset
    dataset_lib
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

