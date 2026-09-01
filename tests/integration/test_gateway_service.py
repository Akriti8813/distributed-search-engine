import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def gateway_client(monkeypatch):
    monkeypatch.setenv("SHARD_URLS", "http://shard-a,http://shard-b")
    # Point at a redis port nothing is listening on so the cache
    # degrades to "always miss" - keeps this test independent of a
    # running Redis instance while still exercising the real code path.
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_PORT", "6390")

    sys.modules.pop("gateway_service.main", None)
    module = importlib.import_module("gateway_service.main")

    async def fake_query_shard(client, url, q, top_k, method):
        if url == "http://shard-a":
            return {
                "shard_id": 0,
                "took_ms": 1.0,
                "results": [
                    {"doc_id": 1, "score": 9.5, "title": "A1", "snippet": "...", "shard_id": 0},
                    {"doc_id": 2, "score": 3.0, "title": "A2", "snippet": "...", "shard_id": 0},
                ],
            }
        if url == "http://shard-b":
            return {
                "shard_id": 1,
                "took_ms": 1.2,
                "results": [
                    {"doc_id": 3, "score": 8.0, "title": "B1", "snippet": "...", "shard_id": 1},
                ],
            }
        return None

    monkeypatch.setattr(module, "_query_shard", fake_query_shard)
    with TestClient(module.app) as client:
        yield client, module


def test_search_merges_and_sorts_results_across_shards(gateway_client):
    client, _ = gateway_client
    resp = client.get("/search", params={"q": "gradient descent", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["shards_queried"] == 2
    assert body["total_candidates"] == 3
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert body["results"][0]["doc_id"] == 1  # highest score (9.5) wins overall


def test_search_respects_top_k(gateway_client):
    client, _ = gateway_client
    resp = client.get("/search", params={"q": "gradient descent", "top_k": 1})
    assert len(resp.json()["results"]) == 1


def test_one_shard_down_still_returns_partial_results(gateway_client, monkeypatch):
    client, module = gateway_client

    async def fake_query_shard_one_down(client_, url, q, top_k, method):
        if url == "http://shard-a":
            return {
                "shard_id": 0,
                "took_ms": 1.0,
                "results": [{"doc_id": 1, "score": 5.0, "title": "A1", "snippet": "...", "shard_id": 0}],
            }
        return None  # shard-b unreachable

    monkeypatch.setattr(module, "_query_shard", fake_query_shard_one_down)
    resp = client.get("/search", params={"q": "gradient", "top_k": 5})
    body = resp.json()
    assert body["shards_queried"] == 1
    assert len(body["results"]) == 1


def test_empty_query_returns_400(gateway_client):
    client, _ = gateway_client
    resp = client.get("/search", params={"q": ""})
    assert resp.status_code == 400
