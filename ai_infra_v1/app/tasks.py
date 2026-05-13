import json
import logging
import os
import time

import redis
from celery.exceptions import SoftTimeLimitExceeded
from openai import OpenAI, APIError, APIConnectionError

from celery_app import celery_app

logger = logging.getLogger(__name__)

# vLLM OpenAI 兼容服务地址：worker 容器通过 host-gateway 访问宿主机
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://host.docker.internal:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "/home/jinxiangchen/models/Qwen2.5-7B-Instruct")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
REDIS_STREAM_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
STREAM_MAXLEN = 1024  # 单个 task stream 最多保留多少 token 事件

# 模块级单例：复用底层 TCP 连接池，避免每个任务都重建
_openai_client: OpenAI | None = None
_redis_client: redis.Redis | None = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=300.0,
        )
    return _openai_client


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_STREAM_URL, decode_responses=True)
    return _redis_client


def _stream_key(task_id: str) -> str:
    return f"chat:stream:{task_id}"


def _publish(r: redis.Redis, task_id: str, event: str, payload: dict) -> None:
    """向 Redis Stream 追加一条事件，FastAPI 端 XREAD 消费"""
    r.xadd(
        _stream_key(task_id),
        {"event": event, "data": json.dumps(payload, ensure_ascii=False)},
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )


@celery_app.task(
    name="tasks.run_inference",
    bind=True,
    autoretry_for=(APIConnectionError,),
    retry_backoff=True,
    retry_backoff_max=30,
    retry_jitter=True,
    max_retries=2,
)
def run_inference(self, message: str, user_id: str | None = None, request_id: str | None = None) -> dict:
    """
    调用 vLLM 流式生成，token 实时写入 Redis Stream 供 SSE 转发，
    最终汇总文本通过 Celery result backend 返回。
    """
    task_id = self.request.id
    r = _get_redis()
    client = _get_openai()

    started = time.perf_counter()
    _publish(r, task_id, "start", {
        "task_id": task_id,
        "model": VLLM_MODEL,
        "request_id": request_id,
    })

    full_text_parts: list[str] = []
    try:
        stream = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": message}],
            stream=True,
            max_tokens=1024,
            temperature=0.7,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue
            full_text_parts.append(delta)
            _publish(r, task_id, "token", {"delta": delta})

    except SoftTimeLimitExceeded:
        _publish(r, task_id, "error", {"detail": "soft time limit exceeded"})
        raise
    except APIError as e:
        # vLLM 返回的业务错误，无需 retry，直接失败
        _publish(r, task_id, "error", {"detail": f"upstream api error: {e}"})
        raise
    except Exception as e:
        logger.exception("inference failed | task_id=%s | request_id=%s", task_id, request_id)
        _publish(r, task_id, "error", {"detail": str(e)})
        raise

    full_text = "".join(full_text_parts)
    latency_ms = (time.perf_counter() - started) * 1000
    _publish(r, task_id, "done", {
        "content": full_text,
        "latency_ms": round(latency_ms, 2),
    })

    # Stream key 过期时间和 result backend 对齐，避免无人消费时残留
    r.expire(_stream_key(task_id), 3600)

    return {
        "content": full_text,
        "latency_ms": round(latency_ms, 2),
        "model": VLLM_MODEL,
        "request_id": request_id,
        "user_id": user_id,
    }
