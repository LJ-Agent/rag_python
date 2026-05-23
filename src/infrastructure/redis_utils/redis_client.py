"""Redis cache client — lightweight caching, aligned with Java RedisConfig."""
from typing import Any

import redis
from redis import ConnectionPool

from common.config_loader import get_config
from common.util.logger import get_logger
from common.util.utils import json_dumps, json_loads

logger = get_logger()


class RedisCache:
    """Redis client for caching vector params, QA results, and system config."""

    def __init__(self):
        cfg = get_config()["redis"]
        self._pool = ConnectionPool(
            host=cfg["host"],
            port=cfg["port"],
            password=cfg["password"] or None,
            db=cfg["database"],
            max_connections=cfg.get("pool_max_active", 20),
            socket_timeout=cfg.get("timeout_ms", 5000) / 1000,
        )
        self._client = redis.Redis(connection_pool=self._pool)
        self._default_ttl = cfg.get("cache_ttl_seconds", 3600)

    def get(self, key: str) -> str | None:
        try:
            val = self._client.get(key)
            return val.decode() if val else None
        except redis.RedisError as e:
            logger.warning(f"Redis get failed: {key} — {e}")
            return None

    def set(self, key: str, value: str, ttl: int | None = None):
        try:
            self._client.setex(key, ttl or self._default_ttl, value)
        except redis.RedisError as e:
            logger.warning(f"Redis set failed: {key} — {e}")

    def get_json(self, key: str) -> Any | None:
        raw = self.get(key)
        return json_loads(raw) if raw else None

    def set_json(self, key: str, value: Any, ttl: int | None = None):
        self.set(key, json_dumps(value), ttl)

    def delete(self, key: str):
        try:
            self._client.delete(key)
        except redis.RedisError as e:
            logger.warning(f"Redis delete failed: {key} — {e}")

    def exists(self, key: str) -> bool:
        try:
            return bool(self._client.exists(key))
        except redis.RedisError:
            return False

    def close(self):
        self._pool.disconnect()

    @property
    def client(self) -> redis.Redis:
        return self._client


# Singleton
_redis_cache: RedisCache | None = None


def get_redis_cache() -> RedisCache:
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache
