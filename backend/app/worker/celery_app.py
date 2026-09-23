from celery import Celery

from app.config import settings

celery_app = Celery(
    "payment_operations_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend_url,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
celery_app.autodiscover_tasks(["app.worker"])
