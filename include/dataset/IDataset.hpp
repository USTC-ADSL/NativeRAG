#pragma once

#include <string>
#include <vector>
#include <memory>

namespace mobile_rag {

/**
 * 数据集中的单个样本
 */
struct DataSample {
    std::string id;              // 样本唯一标识
    std::string query;           // 查询/问题
    std::vector<std::string> documents;  // 相关文档
    std::vector<std::string> answers;    // 答案（可选）
};

/**
 * 统一的数据集接口
 * 支持不同格式的数据集（JSON、CSV等）
 */
class IDataset {
public:
    virtual ~IDataset() = default;

    /**
     * 从文件加载数据集
     * @param file_path 数据集文件路径
     * @return 是否加载成功
     */
    virtual bool load(const std::string& file_path) = 0;

    /**
     * 获取数据集中的样本总数
     */
    virtual size_t size() const = 0;

    /**
     * 获取指定索引的样本
     * @param index 样本索引
     * @return 数据样本，如果索引越界返回空
     */
    virtual DataSample get(size_t index) const = 0;

    /**
     * 获取所有样本
     */
    virtual std::vector<DataSample> get_all() const = 0;

    /**
     * 获取数据集名称
     */
    virtual std::string get_name() const = 0;
};

}  // namespace mobile_rag

