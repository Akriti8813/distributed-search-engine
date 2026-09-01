from typing import List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Raw query string")
    top_k: int = Field(10, ge=1, le=100)
    method: str = Field("bm25", pattern="^(bm25|tfidf)$")


class ResultItem(BaseModel):
    doc_id: int
    score: float
    title: str
    snippet: str
    shard_id: int


class ShardSearchResponse(BaseModel):
    shard_id: int
    took_ms: float
    results: List[ResultItem]


class SearchResponse(BaseModel):
    query: str
    method: str
    top_k: int
    total_candidates: int
    took_ms: float
    cache_hit: bool
    shards_queried: int
    results: List[ResultItem]


class HealthResponse(BaseModel):
    status: str
    shard_id: Optional[int] = None
    docs_indexed: Optional[int] = None
