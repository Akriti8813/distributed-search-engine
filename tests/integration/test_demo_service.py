import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from common.inverted_index import build_shard_index

SAMPLE_DOCS_SHARD_0 = [
    {"doc_id": 1, "title": "Gradient Descent Basics", "body": "gradient descent optimizes loss functions"},
]
SAMPLE_DOCS_SHARD_1 = [
    {"doc_id": 2, "title": "Raft Consensus", "body": "raft is a consensus protocol for distributed systems"},
]


@pytest.fixture()
def demo_client(tmp_path, monkeypatch):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    build_shard_index(0, SAMPLE_DOCS_SHARD_0).save(shard_dir / "shard_0.pkl")
    build_shard_index(1, SAMPLE_DOCS_SHARD_1).save(shard_dir / "shard_1.pkl")

    monkeypatch.setenv("SHARD_DATA_DIR", str(shard_dir))
    monkeypatch.setenv("NUM_SHARDS", "2")
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_PORT", "6391")  # nothing listening -> cache degrades to miss

    sys.modules.pop("demo_service.main", None)
    module = importlib.import_module("demo_service.main")
    with TestClient(module.app) as client:
        yield client


def test_health_reports_docs_from_all_shards(demo_client):
    resp = demo_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["docs_indexed"] == 2


def test_search_merges_results_across_in_process_shards(demo_client):
    resp = demo_client.get("/search", params={"q": "consensus protocol", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["shards_queried"] == 2
    assert len(body["results"]) == 1
    assert body["results"][0]["doc_id"] == 2


def test_empty_query_returns_400(demo_client):
    resp = demo_client.get("/search", params={"q": ""})
    assert resp.status_code == 400
