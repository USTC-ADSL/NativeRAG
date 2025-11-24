import requests
import json
import time

# API 基础URL
BASE_URL = "http://localhost:2024"  # 开发模式
# BASE_URL = "http://localhost:8123"  # 生产模式

GRAPH_ID = "agent"  # 这是 langgraph.json 中定义的 graph_id

# 0. 首先创建或获取 assistant（使用 graph_id）
print("创建 assistant...")
assistant_response = requests.post(
    f"{BASE_URL}/assistants",
    json={"graph_id": GRAPH_ID}
)
assistant_response.raise_for_status()
assistant_data = assistant_response.json()
assistant_id = assistant_data["assistant_id"]
print(f"Assistant ID: {assistant_id}")

# 1. 创建线程（注意：端点是 /threads，不是 /assistants/{id}/threads）
print("\n创建线程...")
thread_response = requests.post(
    f"{BASE_URL}/threads",
    json={"assistant_id": assistant_id}
)
thread_response.raise_for_status()
thread_data = thread_response.json()
thread_id = thread_data["thread_id"]
print(f"Thread ID: {thread_id}")

# 2. 提交研究请求（注意：端点是 /threads/{thread_id}/runs）
print("\n提交研究请求...")
run_response = requests.post(
    f"{BASE_URL}/threads/{thread_id}/runs",
    json={
        "assistant_id": assistant_id,
        "input": {
            "messages": [
                {
                    "type": "human",
                    "content": "什么是人工智能的最新发展趋势？"
                }
            ],
            "initial_search_query_count": 3,
            "max_research_loops": 2,
            "reasoning_model": "gpt-4o-mini"
        }
    }
)
run_response.raise_for_status()
run_data = run_response.json()
run_id = run_data["run_id"]
print(f"Run ID: {run_id}")

# 3. 轮询获取结果（注意：端点是 /threads/{thread_id}/runs/{run_id}）
print("\n等待结果...")
while True:
    status_response = requests.get(
        f"{BASE_URL}/threads/{thread_id}/runs/{run_id}"
    )
    status_response.raise_for_status()
    status = status_response.json()
    
    print(f"状态: {status.get('status')}")
    
    if status["status"] == "success":
        # 获取线程状态中的消息
        state_response = requests.get(
            f"{BASE_URL}/threads/{thread_id}/state"
        )
        state_response.raise_for_status()
        state = state_response.json()
        
        # 从状态中提取消息
        messages = state.get("values", {}).get("messages", [])
        if messages:
            last_message = messages[-1]
            print("\n" + "="*50)
            print("最终答案:")
            print("="*50)
            print(last_message.get("content", ""))
        break
    elif status["status"] == "error":
        print("错误:", status.get("error"))
        break
    
    time.sleep(2)  # 等待2秒后重试