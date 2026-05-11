# 金融全能智能体 (Finance LLM Agent)

基于 **Qwen2.5-1.5B-Instruct** 的金融领域垂直大模型项目，完整覆盖从数据准备、QLoRA 微调、模型合并，到 RAG 增强检索与 Agent 工具调用的全链路。

---

## 项目架构

```
用户输入
   │
   ▼
LangChain ReAct Agent
   ├── 工具1：get_realtime_stock_price（yfinance 在线查股价）
   └── 工具2：search_local_news（本地 RAG 知识库检索）
                    │
                    ├── FAISS 向量召回 (MMR, BGE Embedding)
                    ├── BM25 关键词召回 (jieba 分词)
                    └── CrossEncoder Reranker 精排 (bge-reranker-base)
   │
   ▼
Qwen2.5-1.5B（QLoRA 金融微调）
   │
   ├── 本地推理模式：agent_engine.py（PEFT + HuggingFacePipeline）
   └── vLLM 加速模式：web_agent.py（FastAPI + vLLM OpenAI 接口）
```

---

## 目录结构

```
llm_pro_one/
├── prepare_data.py       # 从 HuggingFace 下载金融 SFT 数据集
├── train.py              # QLoRA 微调脚本（4bit 量化，适配 6GB 显存）
├── merge_model.py        # 将 LoRA 权重融合进基座模型
├── inference.py          # 轻量推理封装，快速验证微调效果
├── rag_test.py           # 构建 FAISS 向量库并测试检索
├── tools.py              # Agent 工具函数 + 双路 RAG 引擎初始化
├── agent_engine.py       # 本地模型 ReAct Agent（LangChain）
├── web_demo.py           # Gradio 演示界面（接 vLLM）
├── web_agent.py          # FastAPI 后端（流式输出，接 vLLM）
├── data/
│   └── finance_data.jsonl      # 训练用金融 SFT 数据
├── knowledge/
│   └── finance_news.txt        # 本地金融知识库（人工维护）
└── Qwen2.5_finance_merged/     # merge_model.py 输出的完整模型
```

---

## 核心模块说明

### 数据与训练

| 文件 | 说明 |
|---|---|
| `prepare_data.py` | 从 `gbharti/finance-alpaca` 下载 1000 条金融指令数据，保存为 JSONL |
| `train.py` | QLoRA（r=8, α=32）+ bitsandbytes 4bit 量化，SwanLab 监控训练曲线 |
| `merge_model.py` | `merge_and_unload()` 将 LoRA 补丁物理融合进基座，输出可独立部署的完整模型 |

训练配置（针对 6GB 显存优化）：
- `per_device_train_batch_size=1` + `gradient_accumulation_steps=8`
- `gradient_checkpointing=True` + `fp16=True`
- LoRA target: `q_proj / k_proj / v_proj / o_proj`

### RAG 检索引擎（`tools.py`）

采用**三阶段混合检索**策略：

1. **双路召回**：FAISS MMR 向量检索（`fetch_k=50, k=15`）+ BM25 关键词检索（Top 15）
2. **合并去重**：两路结果取并集
3. **Reranker 精排**：`BAAI/bge-reranker-base` 交叉编码打分，过滤阈值 0.5，取 Top 3

Embedding 模型：`BAAI/bge-small-zh-v1.5`

### Agent 引擎

提供两个工具给 ReAct Agent：

- `get_realtime_stock_price`：调用 `yfinance` 查询实时收盘价，需要英文股票代码（如 `AAPL`）
- `search_local_news`：查询本地 `knowledge/finance_news.txt` 知识库，适合中文公司名或模糊问题

**两种部署形态：**

- `agent_engine.py`：直接加载 LoRA 权重 + `HuggingFacePipeline`，适合本地调试；包含自定义 `StopWordCriteria` 防止模型幻觉式地自生成 `Observation`
- `web_agent.py`：FastAPI 后端，连接 vLLM OpenAI 接口（`localhost:8000`），支持流式输出（SSE），适合生产部署

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers peft bitsandbytes datasets
pip install langchain langchain-huggingface langchain-community langchain-openai
pip install faiss-cpu sentence-transformers rank-bm25 jieba yfinance
pip install gradio fastapi uvicorn swanlab
```

### 2. 准备知识库与训练数据

```bash
# 下载金融 SFT 数据集（需要访问 HuggingFace）
python prepare_data.py

# 将本地金融新闻 / 财报文本放入知识库，段落之间用空行分隔
# knowledge/finance_news.txt

# 验证 RAG 检索是否正常
python rag_test.py
```

### 3. 微调模型

```bash
python train.py
# 模型 checkpoint 保存到 ./output/Qwen2.5_finance/
```

训练过程可在 [SwanLab](https://swanlab.cn) 查看实时曲线（项目名：`Qwen2.5-Finance-SFT`）。

### 4. 验证微调效果

```bash
python inference.py
```

### 5. 融合 LoRA 权重（可选，用于 vLLM 部署）

```bash
python merge_model.py
# 输出到 ./Qwen2.5_finance_merged/
```

### 6. 启动服务

**方案 A：本地 Agent（直接加载模型）**

```bash
python agent_engine.py
```

**方案 B：vLLM + Gradio 演示（推荐）**

```bash
# 先用 vLLM 启动推理引擎
vllm serve ./Qwen2.5_finance_merged --served-model-name finance-llm --port 8000

# 再启动 Gradio 界面
python web_demo.py
# 访问 http://localhost:7860
```

**方案 C：vLLM + FastAPI 后端（生产）**

```bash
vllm serve ./Qwen2.5_finance_merged --served-model-name finance-llm --port 8000

python web_agent.py
# API 地址：http://0.0.0.0:8080/api/chat
# Swagger UI：http://127.0.0.1:8080/docs
```

---

## 硬件要求

| 场景 | 显存需求 |
|---|---|
| QLoRA 微调（`train.py`） | ≥ 6 GB |
| 本地推理（`inference.py` / `agent_engine.py`） | ≥ 6 GB（fp16） |
| vLLM 部署（合并后完整模型） | ≥ 4 GB |
| Reranker + Embedding 模型 | 额外约 1-2 GB |

- 代码默认 `device_map="auto"`，自动分配 GPU。
- `get_realtime_stock_price` 需要能访问 Yahoo Finance 的网络环境。

---

## 技术栈

- **基座模型**：Qwen2.5-1.5B-Instruct
- **微调框架**：PEFT (LoRA) + bitsandbytes (QLoRA 4bit)
- **训练监控**：SwanLab
- **RAG 框架**：LangChain + FAISS + BM25 + BGE Reranker
- **推理加速**：vLLM
- **前端界面**：Gradio
- **后端接口**：FastAPI + Uvicorn
