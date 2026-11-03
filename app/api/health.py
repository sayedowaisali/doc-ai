"""Liveness/readiness endpoint covering the service's hard dependencies."""

import redis
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.config import get_settings
from app.db import engine

router = APIRouter(tags=["monitoring"])


def _database_ok() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _redis_ok() -> bool:
    try:
        return bool(redis.Redis.from_url(get_settings().redis_url, socket_timeout=2).ping())
    except Exception:
        return False


@router.get("/health")
def health(response: Response) -> dict:
    checks = {"database": _database_ok(), "redis": _redis_ok()}
    healthy = all(checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if healthy else "degraded", "checks": checks}
