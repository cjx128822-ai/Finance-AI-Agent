import yfinance as yf
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import os


def get_realtime_stock_price(symbol: str) -> str:
    """获取指定股票代码的实时收盘价。参数 symbol 必须是标准的股票代码，如 'AAPL' 或 '600519.SS'。"""

    # 【新增调试信息】：看看大模型到底传进来了什么鬼东西
    print(f"\n[Debug - Tool内部] 接收到原始 symbol 参数: '{symbol}'")

    # ==========================================
    # 关键修复：清洗小模型传来的“脏数据”
    # ==========================================
    # 0. 物理超度：首先剥离大模型最爱加的变量名声明和引号
    clean_symbol = symbol.replace("symbol=", "").replace("symbol:", "").replace('"', '').replace("'", "")

    # 1. 剔除大模型容易连带输出的关键词（忽略大小写），以及换行符、空格、冒号等
    clean_symbol = re.sub(r'(?i)observation|thought|action|[:\n\r\s]', '', clean_symbol)

    # 2. 进一步移除非法字符，只保留字母、数字，以及用于区分市场的点号（如 .SS, .SZ）
    # 最后统一转为大写，确保 yfinance 能识别
    clean_symbol = re.sub(r'[^a-zA-Z0-9\.]', '', clean_symbol).upper()

    # 3. 终极兜底：如果经历了上面步骤，开头还是粘上了 SYMBOL 这个词，强行切掉
    if clean_symbol.startswith("SYMBOL"):
        clean_symbol = clean_symbol.replace("SYMBOL", "", 1)

    print(f"[Debug - Tool内部] 自动清洗后的 symbol: '{clean_symbol}'")

    # 如果清洗后变成了空字符串，说明传进来的全是废话
    if not clean_symbol:
        return "错误：未能从输入中提取到有效的股票代码。"

    try:
        # 使用清洗后的纯净代码进行查询
        ticker = yf.Ticker(clean_symbol)

        # 获取最新的一笔交易价格
        todays_data = ticker.history(period='1d')
        if todays_data.empty:
            return f"无法获取代码 {clean_symbol} 的数据，该股票可能已退市或代码不正确。"

        price = todays_data['Close'].iloc[-1]
        currency = ticker.info.get('currency', 'USD')
        return f"{clean_symbol} 当前的最新收盘价为 {price:.2f} {currency}"
    except Exception as e:
        return f"查询出错: {str(e)}"
print("[初始化] 正在构建本地 FAISS 向量知识库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_db = None
file_path = "knowledge/finance_news.txt"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    if raw_text:
        texts = [t.strip() for t in raw_text.split('\n\n') if t.strip()]    
        vector_db = FAISS.from_texts(texts, embeddings)
        print(f"[初始化] RAG 知识库加载完成，共 {len(texts)} 个极其纯净的文本块。")

# 【新增】加载 Reranker 模型
print("[初始化] 正在加载 Reranker 精排模型 (这需要一点点显存，请耐心等待)...")
reranker_model = CrossEncoder('BAAI/bge-reranker-base')
print("[初始化] Reranker 精排模型加载完毕！")
def search_local_news(query: str) -> str:
    """【查公司背景与政策】查询本地新闻、财报和宏观政策。注意：搜索词必须极其精简！优先使用官方全称（如'人民银行'而非'央行'，'存款准备金'而非'降准'），切勿输入过长的句子。"""
    print(f"\n[Debug - RAG 进阶] 收到原始搜索词: '{query}'")

    if not vector_db:
        return "本地知识库未找到或未初始化。"

    # 清洗逻辑保持不变
    clean_query = re.sub(r'[^\w\u4e00-\u9fa5\s]', '', query)
    clean_query = re.sub(r'(?i)observation', '', clean_query)

    print(f"[Debug - RAG 进阶] 清洗后的真正搜索词: '{clean_query}'")

    if not clean_query:
        return "错误：未能提取到有效的搜索词。"

    # ==========================================
    # 第一阶段：高级粗排召回 (Retriever + MMR)
    # ==========================================
    print(f"[Debug - RAG 进阶] 正在配置高级 Retriever (MMR算法)...")

    # 核心升级点：配置 as_retriever
    # fetch_k=20: 先在底层暴力找 20 个
    # k=10: 用 MMR 算法过滤去重，最终输出 10 个最多样化的片段
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 50}  # 先暴力捞50个，再挑出15个给Reranker
    )

    # 使用 invoke 方法触发检索
    initial_docs = retriever.invoke(clean_query)
    print(f"[Debug - RAG 进阶] 第一阶段 MMR 粗排完成，选出了 {len(initial_docs)} 条最具差异化的候选片段。")

    if not initial_docs:
        return "知识库中未找到相关内容。"

    # ==========================================
    # 第二阶段：精排重打分 (Reranking) - 优中选优
    # ==========================================
    # 构造交叉打分的输入格式：[(问题, 片段1), (问题, 片段2)...]
    sentence_pairs = [[clean_query, doc.page_content] for doc in initial_docs]

    # 让 Reranker 模型给这些组合进行深度逻辑打分
    scores = reranker_model.predict(sentence_pairs)

    # 将得分和文档组合在一起
    scored_docs = list(zip(scores, initial_docs))

    # 按得分从高到低排序
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # 🌟 关键修复：引入“及格线”机制（Score Threshold）
    THRESHOLD = 0.5  # 设定及格线，满分是 1.0，0.5 以下的通常是废话

    top_docs = []
    for score, doc in scored_docs:
        if score > THRESHOLD:
            top_docs.append(doc)
            print(f"[Debug] 命中有效文档！得分: {score:.4f}")

        # 最多还是只拿前 3 条（防止太多撑爆 Prompt）
        if len(top_docs) >= 3:
            break
        # 兜底：如果连一个及格的都没有

    if not top_docs:
        return f"知识库中没有找到与 '{query}' 高度相关的确切信息。"
    # ==========================================
    # 组装最终纯净的上下文
    # ==========================================
    final_text = "\n\n".join([doc.page_content for doc in top_docs])
    return f"检索到的极其精准且覆盖面广的背景知识：\n{final_text}"