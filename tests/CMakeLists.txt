# Tests CMakeLists.txt

# Set RPATH so executables can find shared libraries
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Get prebuilt directory from parent CMakeLists
if(USE_PREBUILT)
    set(PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt")
    set(TARGET_PLATFORM "linux-x86_64")
    if(ANDROID)
        set(TARGET_PLATFORM "android-aarch64")
    endif()

    # Add RPATH for prebuilt libraries
    set(CMAKE_BUILD_RPATH
        "${PREBUILT_DIR}/${TARGET_PLATFORM}/MNN"
        "${PREBUILT_DIR}/${TARGET_PLATFORM}/faiss"
        "${PREBUILT_DIR}/${TARGET_PLATFORM}/llama"
    )
else()
    set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party")
endif()

# Find required libraries for tests
if(USE_PREBUILT)
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
    find_library(GGML_LIBRARY NAMES ggml
        HINTS "${PREBUILT_DIR}/${TARGET_PLATFORM}/llama"
        NO_DEFAULT_PATH
    )
    find_library(GGML_BASE_LIBRARY NAMES ggml-base
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

target_compile_definitions(test_vectordb PRIVATE PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")

add_executable(test_command_line_args
    ${CMAKE_CURRENT_SOURCE_DIR}/test_command_line_args.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cli/CommandLineArgs.cpp
)

target_include_directories(test_command_line_args PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

add_executable(test_query_file_loader
    ${CMAKE_CURRENT_SOURCE_DIR}/test_query_file_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cli/QueryFileLoader.cpp
)

target_include_directories(test_query_file_loader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

add_executable(test_llama_logging
    ${CMAKE_CURRENT_SOURCE_DIR}/test_llama_logging.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/llm/LlamaCppLogging.cpp
)

target_include_directories(test_llama_logging PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${PREBUILT_DIR}/include/llama
)

target_link_libraries(test_llama_logging PRIVATE
    ${LLAMA_LIBRARY}
    ${GGML_LIBRARY}
    ${GGML_BASE_LIBRARY}
)

add_executable(test_prompt_utils
    ${CMAKE_CURRENT_SOURCE_DIR}/test_prompt_utils.cpp
)

target_include_directories(test_prompt_utils PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

add_executable(test_evidence_features
    ${CMAKE_CURRENT_SOURCE_DIR}/test_evidence_features.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/controller/EvidenceFeatures.cpp
)

target_include_directories(test_evidence_features PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

add_executable(test_graph_selector
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_selector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/controller/GraphSelector.cpp
)

target_include_directories(test_graph_selector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

add_executable(test_semantic_hashing
    ${CMAKE_CURRENT_SOURCE_DIR}/group_c_vectordb/test_semantic_hashing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/vector_db/SqliteVectorDB.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/retrieval/SemanticHash.cpp
)

target_include_directories(test_semantic_hashing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if (TARGET SQLite::SQLite3)
    target_link_libraries(test_semantic_hashing PRIVATE SQLite::SQLite3)
else()
    target_link_libraries(test_semantic_hashing PRIVATE ${SQLite3_LIBRARIES})
    target_include_directories(test_semantic_hashing PRIVATE ${SQLite3_INCLUDE_DIRS})
endif()

add_executable(test_rag_semantic_hash_prefilter
    ${CMAKE_CURRENT_SOURCE_DIR}/test_rag_semantic_hash_prefilter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/RAGPipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/controller/EvidenceFeatures.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/controller/GraphSelector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/vector_db/SqliteVectorDB.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/retrieval/SemanticHash.cpp
)

target_include_directories(test_rag_semantic_hash_prefilter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if (TARGET SQLite::SQLite3)
    target_link_libraries(test_rag_semantic_hash_prefilter PRIVATE SQLite::SQLite3)
else()
    target_link_libraries(test_rag_semantic_hash_prefilter PRIVATE ${SQLite3_LIBRARIES})
    target_include_directories(test_rag_semantic_hash_prefilter PRIVATE ${SQLite3_INCLUDE_DIRS})
endif()

if(USE_PREBUILT)
    set_target_properties(
        test_command_line_args
        test_llama_logging
        PROPERTIES
        BUILD_RPATH "${CMAKE_BUILD_RPATH}"
        INSTALL_RPATH "${CMAKE_BUILD_RPATH}"
    )
endif()

# Core RAG library (implementation files)
add_library(rag_core STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/controller/EvidenceFeatures.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/controller/GraphSelector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/RAGPipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/retrieval/SemanticHash.cpp
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

# Link SQLite for SqliteVectorDB used inside rag_core
find_package(SQLite3 REQUIRED)
if (TARGET SQLite::SQLite3)
    target_link_libraries(rag_core PUBLIC SQLite::SQLite3)
else()
    target_link_libraries(rag_core PUBLIC ${SQLite3_LIBRARIES})
    target_include_directories(rag_core PUBLIC ${SQLite3_INCLUDE_DIRS})
endif()

add_executable(test_sqlite_vectordb_backend
    ${CMAKE_CURRENT_SOURCE_DIR}/group_c_vectordb/test_sqlite_vectordb_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/vector_db/SqliteVectorDB.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/retrieval/SemanticHash.cpp
)

target_include_directories(test_sqlite_vectordb_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if (TARGET SQLite::SQLite3)
    target_link_libraries(test_sqlite_vectordb_backend PRIVATE SQLite::SQLite3)
else()
    target_link_libraries(test_sqlite_vectordb_backend PRIVATE ${SQLite3_LIBRARIES})
    target_include_directories(test_sqlite_vectordb_backend PRIVATE ${SQLite3_INCLUDE_DIRS})
endif()

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

# Set RPATH for test_embedding
if(USE_PREBUILT)
    set_target_properties(test_embedding PROPERTIES
        BUILD_RPATH "${CMAKE_BUILD_RPATH}"
        INSTALL_RPATH "${CMAKE_BUILD_RPATH}"
    )
endif()

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

# Set RPATH for test_llm
if(USE_PREBUILT)
    set_target_properties(test_llm PROPERTIES
        BUILD_RPATH "${CMAKE_BUILD_RPATH}"
        INSTALL_RPATH "${CMAKE_BUILD_RPATH}"
    )
endif()

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

# Set RPATH for test_vectordb_integrated
if(USE_PREBUILT)
    set_target_properties(test_vectordb_integrated PROPERTIES
        BUILD_RPATH "${CMAKE_BUILD_RPATH}"
        INSTALL_RPATH "${CMAKE_BUILD_RPATH}"
    )
endif()

target_compile_definitions(test_vectordb_integrated PRIVATE PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")

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

# Set RPATH for test_rag_integration
if(USE_PREBUILT)
    set_target_properties(test_rag_integration PROPERTIES
        BUILD_RPATH "${CMAKE_BUILD_RPATH}"
        INSTALL_RPATH "${CMAKE_BUILD_RPATH}"
    )
endif()

# Enable testing
enable_testing()
add_test(NAME CommandLineArgsTest COMMAND $<TARGET_FILE:test_command_line_args>)
add_test(NAME QueryFileLoaderTest COMMAND $<TARGET_FILE:test_query_file_loader>)
add_test(NAME LlamaLoggingTest COMMAND $<TARGET_FILE:test_llama_logging>)
add_test(NAME PromptUtilsTest COMMAND $<TARGET_FILE:test_prompt_utils>)
add_test(NAME EvidenceFeaturesTest COMMAND $<TARGET_FILE:test_evidence_features>)
add_test(NAME GraphSelectorTest COMMAND $<TARGET_FILE:test_graph_selector>)
add_test(NAME SemanticHashingTest COMMAND $<TARGET_FILE:test_semantic_hashing>)
add_test(NAME RAGSemanticHashPrefilterTest COMMAND $<TARGET_FILE:test_rag_semantic_hash_prefilter>)
add_test(NAME SqliteVectorDBBackendTest COMMAND $<TARGET_FILE:test_sqlite_vectordb_backend>)
add_test(NAME VectorDBTest COMMAND $<TARGET_FILE:test_vectordb>)
add_test(NAME EmbeddingTest COMMAND $<TARGET_FILE:test_embedding>)
add_test(NAME LLMTest COMMAND $<TARGET_FILE:test_llm>)
add_test(NAME VectorDBIntegratedTest COMMAND $<TARGET_FILE:test_vectordb_integrated>)
add_test(NAME RAGIntegrationTest COMMAND $<TARGET_FILE:test_rag_integration>)

# Set test properties
set_tests_properties(
    CommandLineArgsTest
    QueryFileLoaderTest
    LlamaLoggingTest
    PromptUtilsTest
    EvidenceFeaturesTest
    GraphSelectorTest
    SemanticHashingTest
    RAGSemanticHashPrefilterTest
    SqliteVectorDBBackendTest
    EmbeddingTest
    LLMTest
    VectorDBTest
    VectorDBIntegratedTest
    RAGIntegrationTest
    PROPERTIES
    TIMEOUT 120
    LABELS "integration"
)

if(USE_PREBUILT)
    set_tests_properties(
        LlamaLoggingTest
        PROPERTIES
        TIMEOUT 120
        LABELS "integration"
        ENVIRONMENT "LD_LIBRARY_PATH=${PREBUILT_DIR}/${TARGET_PLATFORM}/llama:$ENV{LD_LIBRARY_PATH}"
    )
endif()

message(STATUS "Tests configured:")
message(STATUS "  - test_query_file_loader (batch query file parsing)")
message(STATUS "  - test_prompt_utils (generation output cleanup)")
message(STATUS "  - test_evidence_features (evidence feature extraction)")
message(STATUS "  - test_semantic_hashing (Group C - semantic hash)")
message(STATUS "  - test_rag_semantic_hash_prefilter (query-time semantic hash prefilter)")
message(STATUS "  - test_sqlite_vectordb_backend (SQLite lexical/vector backend)")
message(STATUS "  - test_embedding (Group A)")
message(STATUS "  - test_llm (Group B)")
message(STATUS "  - test_vectordb (Group C - basic)")
message(STATUS "  - test_vectordb_integrated (Group C - integrated)")
message(STATUS "  - test_rag_integration (Full pipeline)")
