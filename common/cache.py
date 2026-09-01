"""
Thin Redis wrapper for caching aggregated search results at the
gateway. Cache key is a hash of the normalized query + params so
identical repeated queries (a realistic pattern for search traffic -
head queries dominate) skip re-querying every shard.
"""
import hashlib
import json
import os
from typing import Optional

import redis


class SearchCache:
    def __init__(self, host: Optional[str] = None, port: int = 6379, ttl_seconds: int = 60):
        host = host or os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", port))
        self.ttl = ttl_seconds
        self._client = redis.Redis(host=host, port=port, decode_responses=True, socket_connect_timeout=2)

    @staticmethod
    def make_key(query: str, top_k: int, method: str) -> str:
        raw = f"{query.strip().lower()}|{top_k}|{method}"
        return "search:" + hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        try:
            val = self._client.get(key)
        except redis.RedisError:
            return None
        return json.loads(val) if val else None

    def set(self, key: str, value: dict) -> None:
        try:
            self._client.set(key, json.dumps(value), ex=self.ttl)
        except redis.RedisError:
            pass  # cache is best-effort; a Redis outage must not fail search

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False
