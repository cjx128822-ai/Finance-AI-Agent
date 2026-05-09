import gradio as gr
from openai import OpenAI
from tools import search_local_news  # 直接导入你写好的检索工具

# 1. 建立与本地 vLLM 引擎的直连桥梁
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)


def predict(message, history):
    # 1. 清洗消息：如果是 Gradio 5.x 传来的对象，提取纯文本
    if isinstance(message, dict):
        user_input = message.get("text", "")
    else:
        user_input = str(message)

    # 2. 智能过滤：定义寒暄词库，防止“你好”也去翻图书馆
    greetings = ["你好", "您好", "在吗", "hello", "hi", "早上好", "中午好", "晚上好"]

    # 判定是否需要检索：如果不是寒暄，且包含关键词或长度适中
    should_search = user_input.strip() not in greetings and len(user_input.strip()) > 1

    if should_search:
        print(f"--- 触发检索：{user_input} ---")
        context = search_local_news(user_input)
    else:
        print("--- 跳过检索：检测为日常对话 ---")
        context = "无需背景知识。"

    # 3. 构造系统 Prompt
    system_content = f"你是一个专业的金融全能智能体。请结合背景回答，若无背景则直接回答。背景知识：{context}"
    messages = [{"role": "system", "content": system_content}]

    # 4. 彻底清洗历史记录，抛弃多余的 JSON 结构（解决卡顿的核心）
    # 修改：只保留最近的 2 轮（4条消息）历史记录，防止 Prompt 无限膨胀
    MAX_HISTORY_MESSAGES = 4
    recent_history = history[-MAX_HISTORY_MESSAGES:] if len(history) > MAX_HISTORY_MESSAGES else history

    for entry in recent_history:
        raw = entry["content"]
        clean_text = "".join([item.get("text", "") for item in raw if isinstance(item, dict)]) if isinstance(raw,
                                                                                                             list) else str(
            raw)
        messages.append({"role": entry["role"], "content": clean_text})

    messages.append({"role": "user", "content": user_input})

    # 5. 向 vLLM 发起调用（temperature 调低能稍微快一点点）
    # 5. 向 vLLM 发起调用
    response = client.chat.completions.create(
        model="finance-llm",
        messages=messages,
        max_tokens=1024,  # ⬅️ 必须加上这行！绝对不能让它飙到 8000+
        temperature=0.1,
        stream=True
    )

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield partial_message
# 创建聊天界面
demo = gr.ChatInterface(
    fn=predict,
    title="🚀 金融全能智能体 (RAG 增强版)",
    description="""
    🔥 引擎：Qwen2.5-7B-Instruct (vLLM 驱动) \n
    📚 知识库：已挂载本地金融新闻知识库。
    """,
    examples=[
        "星辰科技最近有什么科研突破？",
        "雷霆汽车一季度的净利润是多少？",
        "数字人民币最近在跨境结算方面有什么动向？",
        "原材料价格回升对雷霆汽车有什么影响？"
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)