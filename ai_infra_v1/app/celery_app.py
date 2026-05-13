import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")

celery_app = Celery(
    "ai_infra",
    broker=REDIS_URL,
    backend=RESULT_BACKEND,
    include=["tasks"],
)

celery_app.conf.update(
    # 序列化：json 更安全，跨语言可读，禁止 pickle
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=False,

    # 关键：LLM 是长任务，必须 acks_late + prefetch=1
    # 否则 worker 崩了任务直接丢，或者一个 worker 把队列全吞了
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,

    # 单任务硬超时 / 软超时：软超时抛 SoftTimeLimitExceeded 让任务有机会清理
    task_time_limit=600,
    task_soft_time_limit=540,

    # 结果保留 1 小时，避免 Redis 撑爆
    result_expires=3600,

    # worker 跑完 200 个任务自动重启，防内存泄漏堆积
    worker_max_tasks_per_child=200,

    # 任务路由（目前只有一个队列，预留扩展）
    task_default_queue="inference",
    task_routes={
        "tasks.run_inference": {"queue": "inference"},
    },
)
