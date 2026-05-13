import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from celery.result import AsyncResult
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from celery_app import celery_app
from tasks import run_inference, _stream_key

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

APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen2.5-7B-Instruct")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# SSE 调优
SSE_BLOCK_MS = 15000        # XREAD 单次阻塞时长
SSE_IDLE_TIMEOUT_S = 600    # 整个连接最长空闲时间，超时主动关闭
SSE_HEARTBEAT_S = 15        # 心跳间隔，防止 Nginx/网关 idle 断开


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        await app.state.redis.ping()
        app.state.ready = True
    except Exception:
        logger.exception("redis ping failed", extra={"request_id": "boot"})
        app.state.ready = False
    app.state.started_at = time.time()
    logger.info(
        "service startup | version=%s | model=%s | redis=%s",
        APP_VERSION, MODEL_NAME, REDIS_URL,
        extra={"request_id": "boot"},
    )
    yield
    app.state.ready = False
    await app.state.redis.aclose()
    logger.info("service shutdown", extra={"request_id": "boot"})


app = FastAPI(
    title="AI Infra Demo (Async)",
    description="FastAPI + Celery + Redis 异步 LLM 服务：投递任务 + SSE 流式输出",
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
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
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


# ===== Schemas =====
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="用户输入")
    user_id: str | None = Field(default=None, description="可选用户标识，用于日志/限流")


class ChatSubmitResponse(BaseModel):
    task_id: str
    stream_url: str
    status_url: str
    request_id: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    ready: bool
    result: dict | None = None
    error: str | None = None


# ===== Meta / Probe =====
@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "ai-infra-demo",
        "version": APP_VERSION,
        "model": MODEL_NAME,
        "uptime_sec": round(time.time() - app.state.started_at, 2),
    }


@app.get("/health", tags=["probe"])
async def health():
    return {"status": "alive"}


@app.get("/ready", tags=["probe"])
async def ready():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="service not ready")
    try:
        await app.state.redis.ping()
    except Exception:
        raise HTTPException(status_code=503, detail="redis unreachable")
    return {"status": "ready"}


# ===== Chat: 投递 =====
@app.post("/api/v1/chat", response_model=ChatSubmitResponse, status_code=202, tags=["llm"])
async def submit_chat(req: ChatRequest, request: Request):
    request_id = request.state.request_id
    async_result: AsyncResult = run_inference.apply_async(
        kwargs={
            "message": req.message,
            "user_id": req.user_id,
            "request_id": request_id,
        },
        # 通过 Celery headers 透传 request_id，方便 worker 端日志关联
        headers={"X-Request-ID": request_id},
    )
    task_id = async_result.id

    logger.info(
        "chat submit | task_id=%s | user=%s | msg_len=%d",
        task_id, req.user_id or "anon", len(req.message),
        extra={"request_id": request_id},
    )

    return ChatSubmitResponse(
        task_id=task_id,
        stream_url=f"/api/v1/chat/stream/{task_id}",
        status_url=f"/api/v1/tasks/{task_id}",
        request_id=request_id,
    )


# ===== Chat: SSE 流式输出 =====
async def _sse_event_stream(redis_client: aioredis.Redis, task_id: str, request_id: str):
    """从 Redis Stream 读取 worker 产生的事件，转为 SSE。"""
    stream = _stream_key(task_id)
    last_id = "0-0"  # 从头读，保证客户端晚于 task 启动也能补到所有事件
    deadline = time.monotonic() + SSE_IDLE_TIMEOUT_S
    last_event_at = time.monotonic()

    while True:
        if time.monotonic() > deadline:
            yield {"event": "error", "data": json.dumps({"detail": "sse idle timeout"})}
            return

        try:
            entries = await redis_client.xread({stream: last_id}, block=SSE_BLOCK_MS, count=64)
        except asyncio.CancelledError:
            logger.info("sse client disconnect | task_id=%s", task_id, extra={"request_id": request_id})
            raise
        except Exception:
            logger.exception("sse xread failed | task_id=%s", task_id, extra={"request_id": request_id})
            yield {"event": "error", "data": json.dumps({"detail": "stream backend error"})}
            return

        if not entries:
            # 空轮询期间发心跳，防止 Nginx idle 断开
            if time.monotonic() - last_event_at > SSE_HEARTBEAT_S:
                yield {"event": "ping", "data": "{}"}
                last_event_at = time.monotonic()
            continue

        for _key, items in entries:
            for entry_id, fields in items:
                last_id = entry_id
                last_event_at = time.monotonic()
                event_name = fields.get("event", "message")
                yield {"event": event_name, "data": fields.get("data", "{}")}
                if event_name in ("done", "error"):
                    return


@app.get("/api/v1/chat/stream/{task_id}", tags=["llm"])
async def stream_chat(task_id: str, request: Request):
    request_id = request.state.request_id
    redis_client: aioredis.Redis = app.state.redis
    logger.info("sse subscribe | task_id=%s", task_id, extra={"request_id": request_id})

    return EventSourceResponse(
        _sse_event_stream(redis_client, task_id, request_id),
        ping=SSE_HEARTBEAT_S,
        headers={"X-Accel-Buffering": "no"},  # 双保险，提示 Nginx 不要缓冲
    )


# ===== 任务状态查询（断线重连 / 兜底）=====
@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse, tags=["llm"])
async def task_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    if res.failed():
        return TaskStatusResponse(
            task_id=task_id,
            status=res.status,
            ready=True,
            error=str(res.result),
        )
    if res.successful():
        return TaskStatusResponse(
            task_id=task_id,
            status=res.status,
            ready=True,
            result=res.result if isinstance(res.result, dict) else {"value": res.result},
        )
    return TaskStatusResponse(
        task_id=task_id,
        status=res.status,
        ready=False,
    )
