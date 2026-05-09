import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 配置路径
base_model_path = "qwen/Qwen2.5-1.5B-Instruct"
lora_path = "./output/Qwen2.5_finance/checkpoint-125" # 选你效果最好的那个

# 2. 加载模型（金融微调后的 Qwen）
print("正在加载金融微调模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, lora_path)

# 3. 加载检索系统（RAG）
print("正在连接知识库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
# 直接读取你刚才在 rag_test.py 里存好的知识（如果没有存，就重新建一个）
with open("knowledge/finance_news.txt", "r", encoding="utf-8") as f:
    knowledge_text = f.read()
vector_db = FAISS.from_texts([knowledge_text], embeddings)

# 4. 定义回答逻辑
def ask_ai(question):
    # 第一步：检索相关知识
    docs = vector_db.similarity_search(question, k=1)
    context = docs[0].page_content if docs else "无相关背景"
    
    # 第二步：构建 Prompt（把知识和问题塞给模型）
    prompt = f"""<|im_start|>system
你是一个专业的金融助手。请结合以下背景知识回答用户的问题。
背景知识：{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    # 第三步：生成回答
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    return tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

# --- 运行测试 ---
print("\n" + "="*30)
query = "星辰科技最近有什么重大科研突破？"
print(f"问：{query}")
print(f"答：{ask_ai(query)}")
print("\n" + "="*30)
query = "雷霆汽车一季度的营收增长了多少？"
print(f"问：{query}")
print(f"答：{ask_ai(query)}")
print("\n" + "="*30)
query = "考虑到碳酸锂价格回升，你对雷霆汽车未来的盈利预期怎么看？"
print(f"问：{query}")
print(f"答：{ask_ai(query)}")
