# ===== Stage 1: builder =====
# 单独的构建阶段：只在这里装编译工具，最终镜像不带这些垃圾，体积更小
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# 依赖单独一层：requirements.txt 没变时这一层会被缓存，加速构建
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ===== Stage 2: runtime =====
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/appuser/.local/bin:$PATH \
    APP_VERSION=1.0.0 \
    MODEL_NAME=mock-llm

# 安全：用非 root 用户跑服务，容器逃逸时不能直接拿到宿主机 root
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 -m -d /home/appuser appuser

# 健康检查需要 curl
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 拷贝已安装的依赖
COPY --from=builder /root/.local /home/appuser/.local

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Docker 层的健康检查；docker-compose / k8s 都能识别
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# 用 exec 形式：PID 1 是 uvicorn 本体，能正确接收 SIGTERM 优雅退出
# --workers 1 配合外部 Nginx/K8s 做水平扩展；多 worker 会让内存翻倍
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
