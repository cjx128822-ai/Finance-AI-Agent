from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
import os

# 导入你之前写好的绝密武器
from tools import get_realtime_stock_price, search_local_news

# ==========================================
# 🌟 大厂级后端：FastAPI 接口层必备导入
# ==========================================
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 1. 建立与 vLLM 引擎的极速链接
llm = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    model_name="finance-llm",
    temperature=0.1,
    max_tokens=1024,
    stop=["Observation:", "\nObservation"]  # ⬅️ 直接这样写
)

# 2. 组装工具箱
tools = [
    Tool(
        name="get_realtime_stock_price",
        func=get_realtime_stock_price,
        description="【在线查股价】查询指定股票的最新在线价格。参数必须是标准的纯英文大写代码，如 'AAPL'、'MSFT'。"
    ),
    Tool(
        name="search_local_news",
        func=search_local_news,
        description="【查公司背景】查询本地新闻、财报和公司动向。参数必须是中文公司名或具体的事件关键词。"
    )
]

# 3. 注入精心打磨的 ReAct 灵魂模板
react_template = """你是一个专业的金融分析助手。你必须严格基于工具的返回结果来回答问题，绝不能凭空捏造！
你可以使用以下工具：
{tools}

必须严格遵循以下ReAct对话格式（注意冒号必须是半角英文冒号）：

Question: 用户的输入问题
Thought: 思考你需要做什么。
Action: [{tool_names}] 中的某一个工具名称
Action Input: 传给工具的具体搜索参数
Observation: (工具返回的结果)
... (Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我已经掌握了需要的信息，准备输出最终答案。
Final Answer: 
【数据查询结果】：在这里写具体的数字或数据。如果没有，写“无”。
【新闻与背景】：在这里写查询到的新闻细节。如果没有，写“无”。
【综合结论】：在这里写最终的简短总结。

--- 下面是一个标准的操作示例，请严格模仿这个格式 ---
Question: 帮我查一下特斯拉(TSLA)的股价，以及它最近的自动驾驶新闻。
Thought: 我需要先用 get_realtime_stock_price 查 TSLA 的股价，然后再用 search_local_news 查自动驾驶的新闻。
Action: get_realtime_stock_price
Action Input: TSLA
Observation: TSLA 当前的最新收盘价为 180.50 USD
Thought: 股价查到了，我现在需要查新闻。
Action: search_local_news
Action Input: 特斯拉 自动驾驶
Observation: 检索到的极其精准且覆盖面广的背景知识：特斯拉宣布FSD自动驾驶系统重大升级...
Thought: 股价和新闻都查到了，我现在可以输出最终回答了。
Final Answer: 
【数据查询结果】：TSLA 当前的最新收盘价为 180.50 USD。
【新闻与背景】：特斯拉宣布FSD自动驾驶系统迎来了重大升级。
【综合结论】：特斯拉(TSLA)目前股价为 180.50 USD。近期公司在自动驾驶(FSD)技术上取得了重要突破。
--- 示例结束 ---

🚨【最高警告】🚨：
1. 绝对不要在 Thought 中直接写答案！答案必须写在 Final Answer: 后面！
2. 必须且只能包含一个 Final Answer: (务必使用英文冒号)。
3. 请忘掉你脑子里预训练的旧知识，一切以 Observation 返回的结果为准！

过去的聊天记录（如果有的话，请作为上下文参考，但请务必优先回答当前的 Question）：
{chat_history}

现在，开始当前任务！
Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_template)

# 创建记忆对象 (保留原有的记忆功能)
memory = ConversationBufferMemory(memory_key="chat_history")

# 4. 组装智能体执行器
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,  # 开启终端透视眼
    handle_parsing_errors=True,
    max_iterations=5  # 最多思考 5 轮
)

# ==========================================
# 🌟 启动 FastAPI 后端服务
# ==========================================
app = FastAPI(title="金融全能智能体核心 API", version="3.0")

# 配置跨域资源共享（CORS），这步不加，后面的 HTML 前端会因为跨域报错连不上！
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义前端传过来的数据包结构
class ChatRequest(BaseModel):
    message: str


# 暴露对话接口给前端
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"\n[🚀 API 接收] 前端传来问题: {request.message}")

    try:
        # 把前端的问题喂给我们的 Agent 大脑
        result = agent_executor.invoke({"input": request.message})

        # 返回标准 JSON 给前端
        return {
            "status": "success",
            "reply": result['output']
        }
    except Exception as e:
        print(f"[💥 API 报错] {str(e)}")
        return {
            "status": "error",
            "reply": f"不好意思，我的大脑短路了：{str(e)}"
        }


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 FastAPI 核心引擎已启动！")
    print("👉 接口监听地址: http://0.0.0.0:8080/api/chat")
    print("👉 Swagger UI 测试地址: http://127.0.0.1:8080/docs")
    print("=" * 50 + "\n")

    # 替换 Gradio，使用工业级的 Uvicorn 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8080)