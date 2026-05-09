import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import Tool

# 导入你写的工具函数
from tools import get_realtime_stock_price, search_local_news

print("正在装载金融微调大脑 (Qwen2.5 + LoRA)...")
base_model_path = "qwen/Qwen2.5-1.5B-Instruct"
lora_path = "./output/Qwen2.5_finance/checkpoint-125"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 显式清除 max_length，消除长度冲突警告
if hasattr(model.generation_config, "max_length"):
    model.generation_config.max_length = None

model = PeftModel.from_pretrained(model, lora_path)

# =====================================================================
# 【核心修复 1】：自定义 HuggingFace 停止词准则（硬截断 / 物理刹车）
# =====================================================================
class StopWordCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_words):
        self.tokenizer = tokenizer
        self.stop_words = stop_words

    def __call__(self, input_ids, scores, **kwargs):
        # 每次生成新 token 时，解码最后 20 个 token 判断是否触碰“红线”
        # 这样可以在模型刚要吐出 "Observation" 的瞬间强制终止生成
        decoded_text = self.tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True)
        for stop_word in self.stop_words:
            if stop_word in decoded_text:
                return True
        return False

# 只要模型尝试自己生成 "Observation"，立刻在底层切断它！
stopping_criteria = StoppingCriteriaList([
    StopWordCriteria(tokenizer, ["Observation:", "\nObservation", "Observation"])
])
# =====================================================================

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=512,
    temperature=0.1, 
    do_sample=True,
    return_full_text=False,
    stopping_criteria=stopping_criteria # 🚀 注入物理硬截断逻辑
)

llm = HuggingFacePipeline(pipeline=pipe)
# LangChain 的软截断依然保留作为双重保险
llm_with_stop = llm.bind(stop=["Observation:", "\nObservation"])

# 2. 将两把武器都交给 Agent，并换上最严厉的“说明书”
tools = [
    Tool(
        name="get_realtime_stock_price",
        func=get_realtime_stock_price,
        description="【仅限明确要求查最新股价，且已知英文代码时使用】查询在线股票价格。警告：如果用户只输入了中文公司名（如星辰科技），绝对不要使用此工具！"
    ),
    Tool(
        name="search_local_news",
        func=search_local_news,
        description="【默认查询工具】查询本地新闻和财报。只要用户输入的是中文公司名（如雷霆汽车、星辰科技）或模糊的提问，必须优先且仅使用此工具！"
    )
]

# =====================================================================
# 【核心修复 2】：为 Qwen2.5-Instruct 增加 ChatML 对话标签模板
# Instruct 模型非常依赖这种特定格式，加上之后能大幅减少它的胡言乱语
# =====================================================================
# 【关键修复 2】：为小模型增加极度严苛的格式约束和防崩溃指南
# =====================================================================
# 【核心修复】：为双工具模式优化的防过度思考 ChatML 模板
# =====================================================================
react_template = """<|im_start|>system
你是一个严谨的金融助手。你可以使用以下两个工具：

{tools}

你必须严格按照以下固定格式回答：

Question: 用户的输入问题
Thought: 思考我需要用哪个工具，或者是否已经知道答案
Action: 工具的名称，必须是 [{tool_names}] 中的一个
Action Input: 传给工具的具体搜索词（【警告】查股价必须填纯英文大写代码如 AAPL；查新闻填中文名）
Observation: 工具返回的结果
... (可以重复思考和调用工具)
Thought: 我现在知道最终答案了
Final Answer: 回答用户的问题

【职场天条 - 必须严格遵守】：
1. 遇到不懂的问题，必须调用工具。
2. 如果调用 get_realtime_stock_price 报错，或者 search_local_news 没找到信息，你必须在 Final Answer 中如实回答“很抱歉，查询失败或未找到信息”，【绝对不允许】编造任何数字！
3. 不要画蛇添足！如果通过 search_local_news 已经找到了用户询问的公司新闻，直接用新闻内容回答即可，【绝对不要】自作主张再去查它的实时股价！<|im_end|>
<|im_start|>user
开始任务！

Question: {input}
Thought: {agent_scratchpad}"""
prompt = PromptTemplate.from_template(react_template)

agent = create_react_agent(llm_with_stop, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True, 
    max_iterations=5
)

# 把这段替换掉原来的 run_finance_task
def run_finance_task(query: str) -> str:
    print(f"\n{'='*20} 网页端发来任务 {'='*20}")
    print(f"用户需求: {query}")
    try:
        # invoke 运行 Agent 循环
        result = agent_executor.invoke({"input": query})
        output = result['output']
        print(f"\n✅ [Agent 最终解答]:\n{output}")
        return output  # 关键：一定要 return 结果！
    except Exception as e:
        error_msg = f"❌ Agent 运行出错: {str(e)}"
        print(error_msg)
        return "不好意思，我的大脑遇到了一点短路，请稍微换个说法再试一次。"

if __name__ == "__main__":
    # 终端本地测试依然有效
    print(run_finance_task("请使用工具帮我查询苹果公司(AAPL)的股价。"))