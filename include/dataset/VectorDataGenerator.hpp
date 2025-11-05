#pragma once

#include <string>
#include <vector>
#include <memory>
#include "IDataset.hpp"

namespace mobile_rag {

/**
 * 向量数据生成工具
 * 用于 Group C (VectorDB) 的测试
 * 
 * 功能:
 * 1. 从数据集生成向量数据
 * 2. 保存向量数据到文件
 * 3. 加载向量数据
 * 4. 支持不同的向量维度
 */
class VectorDataGenerator {
public:
    /**
     * 从数据集生成向量数据
     * @param dataset 数据集
     * @param output_dir 输出目录
     * @param embedding_dim 向量维度
     * @return 是否生成成功
     */
    static bool generate_from_dataset(
        const std::shared_ptr<IDataset>& dataset,
        const std::string& output_dir,
        int embedding_dim = 384
    );

    /**
     * 生成随机向量（用于测试）
     * @param count 向量数量
     * @param dimension 向量维度
     * @param output_file 输出文件路径
     * @return 是否生成成功
     */
    static bool generate_random_vectors(
        int count,
        int dimension,
        const std::string& output_file
    );

    /**
     * 加载向量数据
     * @param file_path 文件路径
     * @return 向量数据 (count x dimension)
     */
    static std::vector<std::vector<float>> load_vectors(const std::string& file_path);

    /**
     * 保存向量数据
     * @param file_path 文件路径
     * @param vectors 向量数据
     * @return 是否保存成功
     */
    static bool save_vectors(
        const std::string& file_path,
        const std::vector<std::vector<float>>& vectors
    );

    /**
     * 保存元数据（文档ID、查询等）
     * @param file_path 文件路径
     * @param metadata 元数据
     * @return 是否保存成功
     */
    static bool save_metadata(
        const std::string& file_path,
        const std::vector<std::string>& metadata
    );

    /**
     * 加载元数据
     * @param file_path 文件路径
     * @return 元数据
     */
    static std::vector<std::string> load_metadata(const std::string& file_path);
};

}  // namespace mobile_rag

