from pathlib import Path

from common.inverted_index import ShardIndex, assign_shard, build_shard_index


SAMPLE_DOCS = [
    {"doc_id": 1, "title": "Gradient Descent", "body": "gradient descent is used in training"},
    {"doc_id": 2, "title": "Consensus Protocol", "body": "raft is a consensus protocol"},
    {"doc_id": 3, "title": "Query Optimization", "body": "gradient boosting improves accuracy"},
]


def test_build_shard_index_creates_postings_for_every_term():
    idx = build_shard_index(0, SAMPLE_DOCS)
    assert idx.total_docs == 3
    assert "gradient" in idx.postings
    assert set(idx.postings["gradient"].keys()) == {1, 3}


def test_doc_lengths_recorded_per_document():
    idx = build_shard_index(0, SAMPLE_DOCS)
    assert idx.doc_lengths[1] > 0
    assert all(doc_id in idx.doc_lengths for doc_id in (1, 2, 3))


def test_avg_doc_len_is_mean_of_lengths():
    idx = build_shard_index(0, SAMPLE_DOCS)
    expected = sum(idx.doc_lengths.values()) / len(idx.doc_lengths)
    assert idx.avg_doc_len == expected


def test_save_and_load_roundtrip(tmp_path: Path):
    idx = build_shard_index(2, SAMPLE_DOCS)
    path = tmp_path / "shard_2.pkl"
    idx.save(path)

    loaded = ShardIndex.load(path)
    assert loaded.shard_id == 2
    assert loaded.total_docs == idx.total_docs
    assert loaded.postings == idx.postings


def test_assign_shard_is_deterministic_and_bounded():
    for doc_id in range(100):
        shard = assign_shard(doc_id, num_shards=4)
        assert 0 <= shard < 4
        assert assign_shard(doc_id, 4) == shard  # deterministic
