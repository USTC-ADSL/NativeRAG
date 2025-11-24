#!/bin/bash
# 启动Web UI的脚本

echo "启动NativeRAG Web UI..."
echo "确保已安装依赖: pip install -r requirements.txt"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 检查依赖
if ! python3 -c "import flask" 2>/dev/null; then
    echo "警告: Flask未安装，正在安装依赖..."
    pip install -r requirements.txt
fi

# 启动服务
python3 web_ui.py

