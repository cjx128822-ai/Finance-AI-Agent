import yfinance as yf
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import os
import jieba
from rank_bm25 import BM25Okapi


def get_realtime_stock_price(symbol: str) -> str:
    """获取指定股票代码的实时收盘价。参数 symbol 必须是标准的股票代码，如 'AAPL' 或 '600519.SS'。"""
    print(f"\n[Debug - Tool内部] 接收到原始 symbol 参数: '{symbol}'")

    clean_symbol = symbol.replace("symbol=", "").replace("symbol:", "").replace('"', '').replace("'", "")
    clean_symbol = re.sub(r'(?i)observation|thought|action|[:\n\r\s]', '', clean_symbol)
    clean_symbol = re.sub(r'[^a-zA-Z0-9\.]', '', clean_symbol).upper()

    if clean_symbol.startswith("SYMBOL"):
        clean_symbol = clean_symbol.replace("SYMBOL", "", 1)

    print(f"[Debug - Tool内部] 自动清洗后的 symbol: '{clean_symbol}'")

    if not clean_symbol:
        return "错误：未能从输入中提取到有效的股票代码。"

    try:
        ticker = yf.Ticker(clean_symbol)
        todays_data = ticker.history(period='1d')
        if todays_data.empty:
            return f"无法获取代码 {clean_symbol} 的数据，该股票可能已退市或代码不正确。"

        price = todays_data['Close'].iloc[-1]
        currency = ticker.info.get('currency', 'USD')
        return f"{clean_symbol} 当前的最新收盘价为 {price:.2f} {currency}"
    except Exception as e:
        return f"查询出错: {str(e)}"


# ==========================================
# 🌟 初始化阶段：构建 FAISS 和 BM25 索引
# ==========================================
print("[初始化] 正在构建本地 FAISS + BM25 双路混合知识库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

vector_db = None
bm25_retriever = None
chunks = []

file_path = "knowledge/finance_news.txt"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    if raw_text:
        texts = [t.strip() for t in raw_text.split('\n\n') if t.strip()]
        chunks = texts

        # FAISS 向量库
        vector_db = FAISS.from_texts(chunks, embeddings)

        # BM25 关键词索引
        tokenized_corpus = [list(jieba.cut(chunk)) for chunk in chunks]
        bm25_retriever = BM25Okapi(tokenized_corpus)

        print(f"[初始化] 双路 RAG 知识库加载完成，共 {len(texts)} 个纯净文本块。")

print("[初始化] 正在加载 Reranker 精排模型 (这需要一点点显存，请耐心等待)...")
reranker_model = CrossEncoder('BAAI/bge-reranker-base')
print("[初始化] Reranker 精排模型加载完毕！")


def search_local_news(query: str) -> str:
    """【查公司背景与政策】查询本地新闻、财报和宏观政策。"""
    print(f"\n[Debug - 高阶 RAG] 收到原始搜索词: '{query}'")

    if not vector_db or not bm25_retriever:
        return "本地知识库未找到或未初始化。"

    clean_query = re.sub(r'[^\w\u4e00-\u9fa5\s]', '', query)
    clean_query = re.sub(r'(?i)observation', '', clean_query)
    print(f"[Debug - 高阶 RAG] 清洗后的真正搜索词: '{clean_query}'")

    if not clean_query:
        return "错误：未能提取到有效的搜索词。"

    # ==========================================
    # 🌟 第一阶段：双路召回 (MMR + BM25)
    # ==========================================
    print(f"[Debug - 高阶 RAG] 正在启动 MMR 向量召回与 BM25 关键词召回...")

    # 【路 1】：恢复你的 MMR 算法！保障片段多样性
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 50}
    )
    mmr_docs = retriever.invoke(clean_query)
    faiss_results = [doc.page_content for doc in mmr_docs]

    # 【路 2】：BM25 关键词检索 (死锁专有名词)
    tokenized_query = list(jieba.cut(clean_query))
    bm25_results = bm25_retriever.get_top_n(tokenized_query, chunks, n=15)

    # 【合并去重】
    combined_results = list(set(faiss_results + bm25_results))
    print(f"[Debug - 高阶 RAG] 双路召回合并后，共捞出 {len(combined_results)} 个独立候选片段。")

    if not combined_results:
        return "知识库中未找到相关内容。"

    # ==========================================
    # 🌟 第二阶段：精排重打分 (Reranking)
    # ==========================================
    print(f"[Debug - 高阶 RAG] 交由 Reranker 进行深度交叉打分...")
    sentence_pairs = [[clean_query, doc] for doc in combined_results]
    scores = reranker_model.predict(sentence_pairs)

    scored_docs = list(zip(scores, combined_results))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    THRESHOLD = 0.5
    top_docs = []

    for score, doc in scored_docs:
        if score > THRESHOLD:
            top_docs.append(doc)
            print(f"[Debug] 命中有效文档！得分: {score:.4f}")

        if len(top_docs) >= 3:
            break

    if not top_docs:
        return f"知识库中没有找到与 '{query}' 高度相关的确切信息。"

    final_text = "\n\n".join(top_docs)
    return f"检索到的极其精准且覆盖面广的背景知识：\n{final_text}"