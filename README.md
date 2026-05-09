# llm_pro_one

轻量说明：本项目以 Qwen2.5 为基础，集成 LoRA 微调、RAG 本地检索与一个基于 LangChain 的 Agent 引擎，目标是构建金融领域问答与检索服务原型。

## 目录结构（重要文件）
- `agent_engine.py`：Agent 核心实现，集成工具调用流程、停止准则与 React Agent 模板。
- `final_app.py`：一个示例流程，展示如何把微调模型与本地 FAISS 向量库结合用于问答。
- `inference.py`：轻量的推理封装函数 `chat()`，加载 LoRA 权重并提供推理接口。
- `prepare_data.py`：从 Hugging Face 下载并生成训练用的 JSONL 数据样例（`data/finance_data.jsonl`）。
- `train.py`：微调脚本，使用 PEFT/LoRA 与 bitsandbytes 量化方案，适配低显存训练。
- `rag_test.py`：构建并测试本地 FAISS 向量库（基于 `knowledge/finance_news.txt`）。
- `tools.py`：对外暴露的工具函数（如 `get_realtime_stock_price`、`search_local_news`）以及 RAG 向量库初始化。
- `web_demo.py`：基于 Gradio 的简单演示界面，调用 `agent_engine.run_finance_task`。

## 快速开始

1. 创建并激活 Python 环境（推荐 conda）并安装依赖：

```bash
pip install -r requirements.txt
```

2. 准备知识库与训练数据：

```bash
# 将金融知识文本放到 knowledge/finance_news.txt
python prepare_data.py
python rag_test.py   # 构建本地向量库并测试检索
```

3. 本地微调（示例）：

```bash
python train.py
```

4. 运行推理或演示：

```bash
python inference.py   # 交互式测试 chat()
python web_demo.py    # 启动 gradio 演示
```

## 环境与注意事项
- 代码默认使用 GPU（`device_map="auto"` 与 `.to("cuda")`）。确保 CUDA 与显卡驱动兼容。
- 训练与推理使用半精度/量化（`fp16`, `bitsandbytes`），在显存有限的机器上可运行，但需对应环境支持。
- `tools.get_realtime_stock_price` 使用 `yfinance`，需要网络访问外部数据源；`search_local_news` 依赖本地 `knowledge/finance_news.txt`。

## 开发建议与扩展点
- 将模型与 LoRA 权重路径配置化（环境变量或 config 文件），避免硬编码路径。
- 把 Vector DB 初始化移动到单独模块，并添加持久化/加载机制，避免每次启动都重建。
- 为 Agent 添加更完善的输入校验与异常隔离（例如工具调用失败时的降级策略）。

## 我做了什么
- 已快速扫描并理解项目主要模块。基于代码，生成本文件作为项目 README 草案。

如需，我可以继续：
- 将 `README.md` 调整为英文版本；
- 把路径硬编码替换为配置文件（`.env` 或 `config.py`）；
- 为 `web_demo.py` 增加启动参数说明或 Dockerfile。

---
生成时间：2026-04-28
