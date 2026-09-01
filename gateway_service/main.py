"""
Gateway service: the single entry point clients hit. On each search
request it:
  1. checks Redis for a cached response (keyed on query+top_k+method)
  2. on a miss, fans the query out to every shard service concurrently
     (asyncio.gather over httpx async calls)
  3. merges each shard's top-k into a single global top-k by score
  4. caches the merged response and returns it

This is the piece that turns N independent shard indexes into one
logical search API, and the piece a benchmark script measures for
end-to-end latency/throughput.
"""
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import List

import httpx
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.cache import SearchCache  # noqa: E402
from common.logging_config import get_logger  # noqa: E402
from common.schemas import HealthResponse, ResultItem, SearchRequest, SearchResponse  # noqa: E402

logger = get_logger("gateway")

SHARD_URLS: List[str] = [
    u.strip() for u in os.environ.get(
        "SHARD_URLS",
        "http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004",
    ).split(",") if u.strip()
]
SHARD_TIMEOUT_S = float(os.environ.get("SHARD_TIMEOUT_S", "3.0"))

cache = SearchCache(ttl_seconds=int(os.environ.get("CACHE_TTL_SECONDS", "60")))

app = FastAPI(title="Search Gateway")


async def _query_shard(client: httpx.AsyncClient, url: str, q: str, top_k: int, method: str):
    try:
        resp = await client.get(
            f"{url}/search",
            params={"q": q, "top_k": top_k, "method": method},
            timeout=SHARD_TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        # A shard being down degrades results, it must not fail the
        # whole query - the gateway just logs it and merges what it got.
        logger.warning("shard query failed", extra={"shard_url": url, "error": str(exc)})
        return None


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.get("/search", response_model=SearchResponse)
async def search(q: str, top_k: int = 10, method: str = "bm25"):
    if not q.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not SHARD_URLS:
        raise HTTPException(status_code=503, detail="no shards configured")

    t0 = time.perf_counter()
    cache_key = cache.make_key(q, top_k, method)
    cached = cache.get(cache_key)
    if cached:
        cached["cache_hit"] = True
        cached["took_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("cache hit", extra={"query": q})
        return SearchResponse(**cached)

    async with httpx.AsyncClient() as client:
        shard_results = await asyncio.gather(
            *[_query_shard(client, url, q, top_k, method) for url in SHARD_URLS]
        )

    all_results: List[ResultItem] = []
    shards_ok = 0
    for shard_resp in shard_results:
        if shard_resp is None:
            continue
        shards_ok += 1
        for r in shard_resp["results"]:
            all_results.append(ResultItem(**r))

    all_results.sort(key=lambda r: r.score, reverse=True)
    top_results = all_results[:top_k]

    took_ms = round((time.perf_counter() - t0) * 1000, 2)
    response = SearchResponse(
        query=q,
        method=method,
        top_k=top_k,
        total_candidates=len(all_results),
        took_ms=took_ms,
        cache_hit=False,
        shards_queried=shards_ok,
        results=top_results,
    )

    cache.set(cache_key, response.model_dump())
    logger.info(
        "gateway search",
        extra={
            "query": q,
            "shards_ok": shards_ok,
            "shards_total": len(SHARD_URLS),
            "candidates": len(all_results),
            "took_ms": took_ms,
        },
    )
    return response
