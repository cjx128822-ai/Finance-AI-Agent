from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1" # 大小写都加上，彻底防死
# 导入你之前写好的绝密武器
from tools import get_realtime_stock_price, search_local_news
from fastapi.responses import StreamingResponse
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
    temperature=0.4,
    max_tokens=1024,
    streaming=True, # ⬅️ 关键：开启流式输出
    stop=["Observation:", "\nObservation"]
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

必须严格遵循以下ReAct对话格式（注意冒号必须是半角英文冒号，且严格保持换行）：

Question: 用户的输入问题
Thought: 思考你需要做什么。
Action: [{tool_names}] 中的某一个工具名称
Action Input: 传给工具的具体搜索参数
Observation: (工具返回的结果)
... (Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我已经掌握了需要的信息，准备输出最终答案。
Final Answer: 你的最终回答。

--- [场景1：需要查询数据的标准操作示例] ---
Question: 帮我查一下特斯拉(TSLA)的股价。
Thought: 我需要先用 get_realtime_stock_price 查 TSLA 的股价。
Action: get_realtime_stock_price
Action Input: TSLA
Observation: TSLA 当前的最新收盘价为 180.50 USD
Thought: 股价查到了，我现在可以输出最终回答了。
Final Answer: 
【数据查询结果】：TSLA 当前的最新收盘价为 180.50 USD。
【新闻与背景】：无。
【综合结论】：特斯拉(TSLA)目前股价为 180.50 USD。

--- [场景2：遇到日常闲聊或挑衅辱骂时的标准操作示例] ---
Question: (用户打招呼或辱骂的话)
Thought: 这是一个不需要使用工具的场景。我绝对不能去查股票或新闻，我应该直接输出 Final Answer。
Final Answer: [根据用户的语气自由发挥。如果用户打招呼，请礼貌回应；如果用户辱骂，请进行强硬反击。每次务必使用完全不同的词汇，严禁当复读机！]
--- 示例结束 ---
🚨【最高警告】🚨：
1. 绝对不要在 Thought 中直接写答案！答案必须单独写在 Final Answer: 的后面！
2. 遇到闲聊或辱骂，绝对不允许使用工具！你可以参考[场景2]的思维格式，但【必须使用你自己的话】来输出 Final Answer！遇到辱骂要强硬反击，每次用词都要不一样，绝对不能当一字不差的复读机！
3. 请忘掉你脑子里预训练的旧知识，查数据必须看 Observation！

过去的聊天记录（如果有的话，请作为上下文参考，但请务必优先回答当前的 Question）：
{chat_history}

现在，开始当前任务！
Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_template)

# 创建记忆对象 (保留原有的记忆功能)
# 加上 output_key="output"，明确告诉它只存 output
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="output"  # 👈 加上这行，彻底治好它的强迫症
)

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
# 暴露对话接口给前端（流式版本）
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"\n[🚀 API 接收] 前端传来问题: {request.message}")

    # 定义一个内部生成器函数，用来“挤牙膏”
    async def generate_response():
        try:
            # 开启异步的 agent stream
            async for chunk in agent_executor.astream_events(
                    {"input": request.message},
                    version="v1"  # 必须指定 v1 版本 API
            ):
                # 只捕获最终大模型说话的内容，跳过工具调用的内部思考
                if chunk["event"] == "on_chat_model_stream":
                    content = chunk["data"]["chunk"].content
                    if content:
                        # 核心：将文本编码成字节流吐出去
                        yield content.encode("utf-8")

        except Exception as e:
            error_msg = f"\n❌ 大脑短路了: {str(e)}"
            yield error_msg.encode("utf-8")

    # 使用 FastAPI 专用的 StreamingResponse 返回这个生成器
    return StreamingResponse(generate_response(), media_type="text/plain")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 FastAPI 核心引擎已启动！")
    print("👉 接口监听地址: http://0.0.0.0:8080/api/chat")
    print("👉 Swagger UI 测试地址: http://127.0.0.1:8080/docs")
    print("=" * 50 + "\n")

    # 替换 Gradio，使用工业级的 Uvicorn 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8080)