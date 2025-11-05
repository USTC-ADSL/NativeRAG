#include "dataset/VectorDataGenerator.hpp"
#include <fstream>
#include <random>
#include <iostream>
#include <filesystem>
#include <cstring>

namespace fs = std::filesystem;

namespace mobile_rag {

bool VectorDataGenerator::generate_from_dataset(
    const std::shared_ptr<IDataset>& dataset,
    const std::string& output_dir,
    int embedding_dim) {
    
    if (!dataset || dataset->size() == 0) {
        std::cerr << "Dataset is empty" << std::endl;
        return false;
    }

    // 创建输出目录
    try {
        fs::create_directories(output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create output directory: " << e.what() << std::endl;
        return false;
    }

    // 生成向量数据
    std::vector<std::vector<float>> vectors;
    std::vector<std::string> doc_ids;
    
    auto samples = dataset->get_all();
    for (const auto& sample : samples) {
        // 为每个文档生成一个向量
        for (const auto& doc : sample.documents) {
            std::vector<float> vector(embedding_dim);
            
            // 使用文档内容的哈希值作为种子生成确定性向量
            std::hash<std::string> hasher;
            size_t seed = hasher(doc);
            std::mt19937 gen(seed);
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            
            for (int i = 0; i < embedding_dim; ++i) {
                vector[i] = dis(gen);
            }
            
            // 归一化向量
            float norm = 0.0f;
            for (float v : vector) {
                norm += v * v;
            }
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (auto& v : vector) {
                    v /= norm;
                }
            }
            
            vectors.push_back(vector);
            doc_ids.push_back(sample.id + "_" + std::to_string(doc_ids.size()));
        }
    }

    // 保存向量
    std::string vectors_file = output_dir + "/vectors.bin";
    if (!save_vectors(vectors_file, vectors)) {
        return false;
    }

    // 保存元数据
    std::string metadata_file = output_dir + "/metadata.txt";
    if (!save_metadata(metadata_file, doc_ids)) {
        return false;
    }

    std::cout << "Generated " << vectors.size() << " vectors with dimension " 
              << embedding_dim << " to " << output_dir << std::endl;
    return true;
}

bool VectorDataGenerator::generate_random_vectors(
    int count,
    int dimension,
    const std::string& output_file) {
    
    std::vector<std::vector<float>> vectors;
    std::mt19937 gen(42);  // 固定种子以保证可重复性
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < count; ++i) {
        std::vector<float> vector(dimension);
        for (int j = 0; j < dimension; ++j) {
            vector[j] = dis(gen);
        }
        vectors.push_back(vector);
    }

    return save_vectors(output_file, vectors);
}

std::vector<std::vector<float>> VectorDataGenerator::load_vectors(const std::string& file_path) {
    std::vector<std::vector<float>> vectors;
    
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return vectors;
    }

    // 读取向量数量和维度
    int count, dimension;
    file.read(reinterpret_cast<char*>(&count), sizeof(int));
    file.read(reinterpret_cast<char*>(&dimension), sizeof(int));

    // 读取向量数据
    for (int i = 0; i < count; ++i) {
        std::vector<float> vector(dimension);
        file.read(reinterpret_cast<char*>(vector.data()), dimension * sizeof(float));
        vectors.push_back(vector);
    }

    file.close();
    return vectors;
}

bool VectorDataGenerator::save_vectors(
    const std::string& file_path,
    const std::vector<std::vector<float>>& vectors) {
    
    if (vectors.empty()) {
        std::cerr << "No vectors to save" << std::endl;
        return false;
    }

    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << file_path << std::endl;
        return false;
    }

    // 写入向量数量和维度
    int count = vectors.size();
    int dimension = vectors[0].size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(int));
    file.write(reinterpret_cast<const char*>(&dimension), sizeof(int));

    // 写入向量数据
    for (const auto& vector : vectors) {
        file.write(reinterpret_cast<const char*>(vector.data()), dimension * sizeof(float));
    }

    file.close();
    return true;
}

bool VectorDataGenerator::save_metadata(
    const std::string& file_path,
    const std::vector<std::string>& metadata) {
    
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << file_path << std::endl;
        return false;
    }

    for (const auto& item : metadata) {
        file << item << "\n";
    }

    file.close();
    return true;
}

std::vector<std::string> VectorDataGenerator::load_metadata(const std::string& file_path) {
    std::vector<std::string> metadata;
    
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return metadata;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            metadata.push_back(line);
        }
    }

    file.close();
    return metadata;
}

}  // namespace mobile_rag

