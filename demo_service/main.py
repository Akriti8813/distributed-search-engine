"""
Standalone demo service: the SAME ranking/indexing/aggregation code as
the distributed system (common/ranking.py, common/inverted_index.py,
common/query_parser.py), but running as one process instead of one
gateway + 4 networked shard services.

Why this exists: the distributed architecture (see shard_service/ +
gateway_service/ + docker-compose.yml) is what's actually built and
tested - this file exists only because hosting 5 always-on containers
costs money, and a free single-service host is enough to give a
resume link something real to click. The search logic below is
identical; only the "network hop between shards" is replaced with an
in-process loop over the same shard indexes.
"""
import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.cache import SearchCache  # noqa: E402
from common.inverted_index import ShardIndex  # noqa: E402
from common.logging_config import get_logger  # noqa: E402
from common.query_parser import parse_query  # noqa: E402
from common.ranking import score_documents  # noqa: E402
from common.schemas import HealthResponse, ResultItem, SearchResponse  # noqa: E402

logger = get_logger("demo")

SHARD_DATA_DIR = os.environ.get("SHARD_DATA_DIR", "data/shards")
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "4"))

cache = SearchCache(ttl_seconds=int(os.environ.get("CACHE_TTL_SECONDS", "60")))

state: dict = {"shards": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    shards = []
    for i in range(NUM_SHARDS):
        path = Path(SHARD_DATA_DIR) / f"shard_{i}.pkl"
        if path.exists():
            shards.append(ShardIndex.load(path))
    state["shards"] = shards
    total_docs = sum(s.total_docs for s in shards)
    logger.info("demo service ready", extra={"shards_loaded": len(shards), "total_docs": total_docs})
    yield


app = FastAPI(title="Distributed Search Engine - Live Demo", lifespan=lifespan)


def _search_one_shard(idx: ShardIndex, parsed, top_k: int, method: str) -> List[ResultItem]:
    scored = score_documents(
        query_terms=parsed.terms,
        postings=idx.postings,
        doc_lengths=idx.doc_lengths,
        avg_doc_len=idx.avg_doc_len,
        total_docs=max(idx.total_docs, 1),
        method=method,
    )
    if parsed.phrases:
        for i, (doc_id, score) in enumerate(scored):
            title = idx.documents[doc_id].get("title", "").lower()
            if any(p.lower() in title for p in parsed.phrases):
                scored[i] = (doc_id, score * 1.25)

    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for doc_id, score in scored[:top_k]:
        doc = idx.documents[doc_id]
        results.append(
            ResultItem(
                doc_id=doc_id,
                score=round(score, 4),
                title=doc.get("title", ""),
                snippet=doc.get("body", "")[:140],
                shard_id=idx.shard_id,
            )
        )
    return results


@app.get("/health", response_model=HealthResponse)
def health():
    total_docs = sum(s.total_docs for s in state["shards"])
    return HealthResponse(status="ok", docs_indexed=total_docs)


@app.get("/search", response_model=SearchResponse)
async def search(q: str, top_k: int = 10, method: str = "bm25"):
    if not q.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    t0 = time.perf_counter()
    cache_key = cache.make_key(q, top_k, method)
    cached = cache.get(cache_key)
    if cached:
        cached["cache_hit"] = True
        cached["took_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return SearchResponse(**cached)

    parsed = parse_query(q)

    # Same "fan out to every shard, then merge" pattern as the gateway
    # service - just via asyncio.to_thread instead of an HTTP call,
    # since all shards live in this one process.
    per_shard = await asyncio.gather(
        *[asyncio.to_thread(_search_one_shard, idx, parsed, top_k, method) for idx in state["shards"]]
    )

    all_results: List[ResultItem] = [r for shard_results in per_shard for r in shard_results]
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
        shards_queried=len(state["shards"]),
        results=top_results,
    )
    cache.set(cache_key, response.model_dump())
    logger.info("demo search", extra={"query": q, "took_ms": took_ms})
    return response
