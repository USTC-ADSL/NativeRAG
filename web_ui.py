#!/usr/bin/env python3
"""
简单的Web UI，支持端侧推理和云端请求的异步处理

使用方法:
1. 安装依赖: pip install -r requirements.txt
2. 确保已编译C++程序: cd build && cmake .. && make
3. 运行: python web_ui.py
4. 在浏览器中打开: http://localhost:5000
"""
import os
import json
import time
import threading
import subprocess
import requests
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 配置
BASE_URL = "http://localhost:8000"  # 云端API地址
GRAPH_ID = "agent"  # 从test.py中获取

# 全局变量存储assistant_id和thread_id（简化实现）
assistant_id = None
thread_id = None

def get_or_create_assistant():
    """获取或创建assistant"""
    global assistant_id
    if assistant_id:
        return assistant_id
    
    try:
        response = requests.post(
            f"{BASE_URL}/assistants",
            json={"graph_id": GRAPH_ID},
            timeout=5
        )
        response.raise_for_status()
        assistant_id = response.json()["assistant_id"]
        return assistant_id
    except Exception as e:
        print(f"Error creating assistant: {e}")
        return None

def get_or_create_thread():
    """获取或创建thread"""
    global thread_id
    if thread_id:
        return thread_id
    
    assistant_id = get_or_create_assistant()
    if not assistant_id:
        return None
    
    try:
        response = requests.post(
            f"{BASE_URL}/threads",
            json={"assistant_id": assistant_id},
            timeout=5
        )
        response.raise_for_status()
        thread_id = response.json()["thread_id"]
        return thread_id
    except Exception as e:
        print(f"Error creating thread: {e}")
        return None

def run_local_rag(query, config):
    """运行本地RAG推理"""
    try:
        # 获取可执行文件路径
        rag_binary = config.get('rag_binary', './build/mobile_rag_cli')
        if not os.path.exists(rag_binary):
            # 尝试其他可能的路径
            possible_paths = [
                './build/mobile_rag_cli',
                '../build/mobile_rag_cli',
                'build/mobile_rag_cli',
                os.path.join(os.path.dirname(__file__), 'build/mobile_rag_cli')
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    rag_binary = path
                    break
            else:
                return {"success": False, "error": f"RAG binary not found. Please specify --rag-binary or ensure it's in build/"}
        
        # 构建命令
        cmd = [rag_binary, '--query', query]
        
        # 添加可选参数
        if config.get('llm_model'):
            cmd.extend(['--llm-model', config['llm_model']])
        if config.get('embedding_model'):
            cmd.extend(['--embedding-model', config['embedding_model']])
        if config.get('index_path'):
            cmd.extend(['--index-path', config['index_path']])
        if config.get('sqlite_db'):
            cmd.extend(['--sqlite-db', config['sqlite_db']])
        if config.get('top_k'):
            cmd.extend(['--top-k', str(config['top_k'])])
        
        # 添加load-index标志
        if config.get('index_path'):
            cmd.append('--load-index')
        
        # 运行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
            cwd=os.path.dirname(__file__)  # 在项目根目录运行
        )
        
        if result.returncode == 0:
            # 提取答案
            output = result.stdout.strip()
            # 尝试找到答案部分（通常在最后，跳过调试信息）
            lines = output.split('\n')
            # 查找可能的答案部分（跳过包含[INFO], [QUERY], [RETRIEVAL]等的行）
            answer_lines = []
            for line in reversed(lines):
                if line.strip() and not line.startswith('[') and not line.startswith('✓'):
                    answer_lines.insert(0, line)
                if len(answer_lines) >= 20:  # 最多取20行
                    break
            
            answer = '\n'.join(answer_lines) if answer_lines else output
            return {"success": True, "answer": answer, "output": output}
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return {"success": False, "error": error_msg}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Local RAG timeout (exceeded 5 minutes)"}
    except Exception as e:
        return {"success": False, "error": f"Error running local RAG: {str(e)}"}

def run_cloud_request(query):
    """异步运行云端请求"""
    try:
        # 获取或创建thread
        tid = get_or_create_thread()
        if not tid:
            return {"success": False, "error": "Failed to create thread"}
        
        assistant_id = get_or_create_assistant()
        if not assistant_id:
            return {"success": False, "error": "Failed to get assistant"}
        
        # 提交run请求
        run_response = requests.post(
            f"{BASE_URL}/threads/{tid}/runs",
            json={
                "assistant_id": assistant_id,
                "input": {
                    "messages": [
                        {
                            "type": "human",
                            "content": query
                        }
                    ],
                    "initial_search_query_count": 3,
                    "max_research_loops": 2,
                    "reasoning_model": "gpt-4o-mini"
                }
            },
            timeout=10
        )
        run_response.raise_for_status()
        run_data = run_response.json()
        run_id = run_data["run_id"]
        
        # 轮询获取结果
        while True:
            status_response = requests.get(
                f"{BASE_URL}/threads/{tid}/runs/{run_id}",
                timeout=5
            )
            status_response.raise_for_status()
            status = status_response.json()
            
            if status["status"] == "success":
                # 获取线程状态中的消息
                state_response = requests.get(
                    f"{BASE_URL}/threads/{tid}/state",
                    timeout=5
                )
                state_response.raise_for_status()
                state = state_response.json()
                
                # 从状态中提取消息
                messages = state.get("values", {}).get("messages", [])
                if messages:
                    last_message = messages[-1]
                    return {
                        "success": True,
                        "answer": last_message.get("content", "")
                    }
                return {"success": True, "answer": "No message found"}
            elif status["status"] == "error":
                return {"success": False, "error": status.get("error", "Unknown error")}
            
            time.sleep(2)  # 等待2秒后重试
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

# 全局结果存储（简化实现，生产环境应使用Redis等）
query_results = {}

@app.route('/api/query', methods=['POST'])
def query():
    """处理查询请求"""
    data = request.json
    query_text = data.get('query', '')
    config = data.get('config', {})
    
    if not query_text:
        return jsonify({"error": "Query is required"}), 400
    
    # 生成查询ID
    import uuid
    query_id = str(uuid.uuid4())
    
    # 初始化结果存储
    query_results[query_id] = {
        "local": None,
        "cloud": None,
        "local_done": False,
        "cloud_done": False
    }
    
    def local_task():
        """本地RAG任务"""
        result = run_local_rag(query_text, config)
        query_results[query_id]["local"] = result
        query_results[query_id]["local_done"] = True
    
    def cloud_task():
        """云端请求任务"""
        result = run_cloud_request(query_text)
        query_results[query_id]["cloud"] = result
        query_results[query_id]["cloud_done"] = True
    
    # 启动两个线程
    local_thread = threading.Thread(target=local_task)
    cloud_thread = threading.Thread(target=cloud_task)
    
    local_thread.start()
    cloud_thread.start()
    
    # 等待本地完成（立即返回）
    local_thread.join()
    
    # 返回本地结果，云端结果稍后通过SSE获取
    return jsonify({
        "local": query_results[query_id]["local"],
        "local_done": True,
        "cloud_done": False,
        "query_id": query_id
    })

@app.route('/api/status/<query_id>', methods=['GET'])
def get_status(query_id):
    """获取查询状态（用于轮询）"""
    if query_id not in query_results:
        return jsonify({"error": "Query ID not found"}), 404
    
    result = query_results[query_id]
    return jsonify({
        "local_done": result["local_done"],
        "cloud_done": result["cloud_done"],
        "local": result["local"],
        "cloud": result["cloud"]
    })

@app.route('/api/stream/<query_id>')
def stream_results(query_id):
    """使用SSE流式推送结果"""
    def generate():
        if query_id not in query_results:
            yield f"data: {json.dumps({'error': 'Query ID not found'})}\n\n"
            return
        
        # 先发送本地结果（如果已完成）
        result = query_results[query_id]
        if result["local_done"]:
            yield f"data: {json.dumps({'type': 'local', 'data': result['local']})}\n\n"
        
        # 轮询云端结果
        max_wait = 300  # 最多等待5分钟
        wait_time = 0
        while wait_time < max_wait:
            if result["cloud_done"]:
                yield f"data: {json.dumps({'type': 'cloud', 'data': result['cloud']})}\n\n"
                break
            time.sleep(1)
            wait_time += 1
        
        # 清理（可选，保留一段时间以便客户端重连）
        # del query_results[query_id]
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

