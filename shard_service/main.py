"""
Shard service: owns one partition of the inverted index and answers
search requests against only its own documents. Multiple instances
of this same service (SHARD_ID=0..N-1) are what makes the system
"distributed" - the gateway fans a single query out to all of them
in parallel and merges the results.
"""
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.inverted_index import ShardIndex  # noqa: E402
from common.logging_config import get_logger  # noqa: E402
from common.query_parser import parse_query  # noqa: E402
from common.ranking import score_documents  # noqa: E402
from common.schemas import HealthResponse, ResultItem, ShardSearchResponse  # noqa: E402

SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
SHARD_DATA_DIR = os.environ.get("SHARD_DATA_DIR", "data/shards")

logger = get_logger(f"shard-{SHARD_ID}")

state: dict = {"index": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(SHARD_DATA_DIR) / f"shard_{SHARD_ID}.pkl"
    if path.exists():
        state["index"] = ShardIndex.load(path)
        logger.info(
            "shard index loaded",
            extra={"shard_id": SHARD_ID, "docs": state["index"].total_docs},
        )
    else:
        state["index"] = ShardIndex(shard_id=SHARD_ID)
        logger.warning("no index file found, starting empty", extra={"path": str(path)})
    yield


app = FastAPI(title=f"Shard Service {SHARD_ID}", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    idx: ShardIndex = state["index"]
    return HealthResponse(status="ok", shard_id=SHARD_ID, docs_indexed=idx.total_docs)


@app.get("/search", response_model=ShardSearchResponse)
def search(q: str, top_k: int = 10, method: str = "bm25"):
    if not q.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    idx: ShardIndex = state["index"]
    t0 = time.perf_counter()

    parsed = parse_query(q)
    scored = score_documents(
        query_terms=parsed.terms,
        postings=idx.postings,
        doc_lengths=idx.doc_lengths,
        avg_doc_len=idx.avg_doc_len,
        total_docs=max(idx.total_docs, 1),
        method=method,
    )

    # phrase boost: if a quoted phrase appears verbatim in the title, bump the score
    if parsed.phrases:
        for doc_id, score in list(scored):
            title = idx.documents[doc_id].get("title", "").lower()
            if any(p.lower() in title for p in parsed.phrases):
                scored[[d for d, _ in scored].index(doc_id)] = (doc_id, score * 1.25)

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    results = []
    for doc_id, score in top:
        doc = idx.documents[doc_id]
        results.append(
            ResultItem(
                doc_id=doc_id,
                score=round(score, 4),
                title=doc.get("title", ""),
                snippet=doc.get("body", "")[:140],
                shard_id=SHARD_ID,
            )
        )

    took_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "shard search",
        extra={"query": q, "candidates": len(scored), "took_ms": round(took_ms, 2)},
    )
    return ShardSearchResponse(shard_id=SHARD_ID, took_ms=round(took_ms, 2), results=results)
