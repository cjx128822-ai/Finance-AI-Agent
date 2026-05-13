import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | request_id=%(request_id)s | %(message)s',
)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


logger = logging.getLogger("ai_infra")
logger.addFilter(RequestIdFilter())

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MODEL_NAME = os.getenv("MODEL_NAME", "mock-llm")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "service startup | version=%s | model=%s", APP_VERSION, MODEL_NAME,
        extra={"request_id": "boot"},
    )
    # 真实场景在此处加载模型、连接数据库、预热缓存
    app.state.ready = True
    app.state.started_at = time.time()
    yield
    app.state.ready = False
    logger.info("service shutdown", extra={"request_id": "boot"})


app = FastAPI(
    title="AI Infra Demo",
    description="生产级 FastAPI 模板：健康探针 / 请求追踪 / 结构化日志 / Nginx 友好",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    # 优先取 Nginx 透传的 X-Request-ID，方便全链路追踪
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception(
            "unhandled exception | path=%s", request.url.path,
            extra={"request_id": request_id},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "request_id": request_id},
        )

    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{latency_ms:.2f}"

    logger.info(
        "%s %s -> %d (%.2f ms)",
        request.method, request.url.path, response.status_code, latency_ms,
        extra={"request_id": request_id},
    )
    return response


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户输入")
    user_id: str | None = Field(default=None, description="可选用户标识，用于日志/限流")


class ChatResponse(BaseModel):
    reply: str
    model: str
    latency_ms: float
    request_id: str


@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "ai-infra-demo",
        "version": APP_VERSION,
        "model": MODEL_NAME,
        "uptime_sec": round(time.time() - app.state.started_at, 2),
    }


# Liveness：进程是否还活着。失败 → 重启容器
@app.get("/health", tags=["probe"])
async def health():
    return {"status": "alive"}


# Readiness：是否可以接收流量。失败 → 摘流量但不重启
@app.get("/ready", tags=["probe"])
async def ready():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="service not ready")
    return {"status": "ready"}


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["llm"])
async def chat(req: ChatRequest, request: Request):
    request_id = request.state.request_id
    start = time.perf_counter()

    logger.info(
        "chat | user=%s | msg_len=%d",
        req.user_id or "anon", len(req.message),
        extra={"request_id": request_id},
    )

    # 占位：真实场景对接 vLLM / 自研 Agent / 外部 LLM API
    reply = f"[{MODEL_NAME}] 你说的是：{req.message}"

    latency_ms = (time.perf_counter() - start) * 1000
    return ChatResponse(
        reply=reply,
        model=MODEL_NAME,
        latency_ms=round(latency_ms, 2),
        request_id=request_id,
    )
