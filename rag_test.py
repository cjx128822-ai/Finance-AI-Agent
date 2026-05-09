import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter # 换成这个
import os

# 1. 加载 Embedding 模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2. 读取并切分知识库
file_path = "knowledge/finance_news.txt"
if not os.path.exists(file_path):
    print(f"❌ 错误：找不到文件 {file_path}")
    exit()

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 检查文件是否为空
if len(raw_text) == 0:
    print("❌ 错误：finance_news.txt 是空的！请加点内容进去。")
    exit()

# 使用递归切分器，chunk_size 设小一点方便演示
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
texts = text_splitter.split_text(raw_text)

print(f"DEBUG: 成功切分为 {len(texts)} 段文本")

# 3. 创建向量数据库
vector_db = FAISS.from_texts(texts, embeddings)
print("✅ 向量数据库已建立！")

# 4. 模拟检索
query = "股价暴涨"
docs = vector_db.similarity_search(query)

if docs:
    print(f"\n检索到的相关知识：\n{docs[0].page_content}")