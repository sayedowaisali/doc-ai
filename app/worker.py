"""Celery application used for asynchronous document processing.

Task state lives in Postgres (document status), so no Celery result backend is configured.
"""

from celery import Celery

from app.config import get_settings

celery_app = Celery("docu_ai", broker=get_settings().redis_url, include=["app.tasks"])
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    # Re-deliver a document if the worker dies mid-processing instead of silently losing it.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Document processing is long-running; don't let one worker hoard queued jobs.
    worker_prefetch_multiplier=1,
)
