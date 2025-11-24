# Web UI 使用说明

这是一个简单的Web界面，支持端侧推理和云端请求的异步处理。

## 功能特性

1. **双模式查询**：同时进行端侧推理和云端请求
2. **实时反馈**：端侧推理完成后立即返回结果，云端完成后也实时返回
3. **简单UI**：基于Flask的Web界面，易于使用

## 安装步骤

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 2. 编译C++程序（如果尚未编译）

```bash
cd build
cmake ..
make
cd ..
```

### 3. 启动Web服务

```bash
python web_ui.py
```

### 4. 打开浏览器

访问: http://localhost:5000

## 配置说明

在Web界面中需要配置以下参数：

- **LLM模型路径**: 本地LLM模型的路径（例如: `/path/to/model.mnn`）
- **Embedding模型路径**: Embedding模型的路径（例如: `/path/to/embedding.mnn`）
- **索引路径**: Faiss索引文件的路径（例如: `./index.faiss`）
- **SQLite数据库路径**: SQLite数据库文件的路径（例如: `./rag.db`）

## 云端API配置

默认云端API地址为 `http://localhost:2024`，可以在 `web_ui.py` 中修改：

```python
BASE_URL = "http://localhost:2024"  # 修改为你的云端API地址
GRAPH_ID = "agent"  # 修改为你的graph_id
```

## 工作流程

1. 用户输入查询问题
2. 系统同时启动两个任务：
   - **端侧推理**：调用本地C++ RAG程序进行推理
   - **云端请求**：异步转发请求到云端API
3. 端侧推理完成后立即显示结果
4. 云端请求完成后也显示结果

## 技术实现

- **后端**: Flask + Python
- **前端**: HTML + JavaScript (原生，无框架)
- **实时通信**: Server-Sent Events (SSE) 和 HTTP轮询
- **异步处理**: Python threading

## 注意事项

1. 确保C++程序已正确编译并位于 `build/mobile_rag_cli`
2. 确保所有模型文件和索引文件路径正确
3. 云端API需要按照 `test.py` 中的格式实现
4. 这是一个简化实现，生产环境建议使用Redis等存储查询结果

## 故障排除

- **找不到RAG程序**: 检查 `build/mobile_rag_cli` 是否存在，或通过环境变量指定路径
- **云端请求失败**: 检查云端API是否运行在 `http://localhost:2024`
- **端侧推理超时**: 检查模型路径和索引路径是否正确

