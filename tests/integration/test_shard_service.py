import importlib
import os
import sys

import pytest
from fastapi.testclient import TestClient

from common.inverted_index import ShardIndex, build_shard_index

SAMPLE_DOCS = [
    {"doc_id": 1, "title": "Gradient Descent Basics", "body": "gradient descent optimizes loss functions"},
    {"doc_id": 2, "title": "Raft Consensus", "body": "raft is a consensus protocol for distributed systems"},
    {"doc_id": 3, "title": "Query Optimization", "body": "gradient boosting is unrelated to gradient descent"},
]


@pytest.fixture()
def shard_client(tmp_path, monkeypatch):
    # Build a tiny index and point a fresh shard_service import at it via env vars.
    idx = build_shard_index(0, SAMPLE_DOCS)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    idx.save(shard_dir / "shard_0.pkl")

    monkeypatch.setenv("SHARD_ID", "0")
    monkeypatch.setenv("SHARD_DATA_DIR", str(shard_dir))

    sys.modules.pop("shard_service.main", None)
    module = importlib.import_module("shard_service.main")
    with TestClient(module.app) as client:
        yield client


def test_health_reports_docs_indexed(shard_client):
    resp = shard_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["docs_indexed"] == 3


def test_search_returns_relevant_doc_first(shard_client):
    resp = shard_client.get("/search", params={"q": "gradient descent", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["shard_id"] == 0
    assert len(body["results"]) > 0
    assert body["results"][0]["doc_id"] == 1  # exact phrase match ranks highest


def test_search_empty_query_returns_400(shard_client):
    resp = shard_client.get("/search", params={"q": "   "})
    assert resp.status_code == 400


def test_search_unknown_term_returns_no_results(shard_client):
    resp = shard_client.get("/search", params={"q": "nonexistentzzz"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []
