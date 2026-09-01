"""
In-memory inverted index for a single shard, with pickle
serialization so shard services can load a prebuilt index at startup
instead of rebuilding it on every container start.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from common.tokenizer import tokenize


@dataclass
class ShardIndex:
    shard_id: int
    # term -> {doc_id: term_frequency}
    postings: Dict[str, Dict[int, int]] = field(default_factory=dict)
    # doc_id -> token count (needed for BM25 length normalization)
    doc_lengths: Dict[int, int] = field(default_factory=dict)
    # doc_id -> original document text (kept so the shard can return
    # a snippet without a separate document store)
    documents: Dict[int, dict] = field(default_factory=dict)

    @property
    def total_docs(self) -> int:
        return len(self.doc_lengths)

    @property
    def avg_doc_len(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def add_document(self, doc_id: int, doc: dict) -> None:
        text = f"{doc.get('title', '')} {doc.get('body', '')}"
        tokens = tokenize(text)
        self.doc_lengths[doc_id] = len(tokens) or 1
        self.documents[doc_id] = doc
        term_counts: Dict[str, int] = {}
        for tok in tokens:
            term_counts[tok] = term_counts.get(tok, 0) + 1
        for term, tf in term_counts.items():
            self.postings.setdefault(term, {})[doc_id] = tf

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "ShardIndex":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_shard_index(shard_id: int, docs: List[dict]) -> ShardIndex:
    """docs: list of {"doc_id": int, "title": str, "body": str, "topic": str}"""
    idx = ShardIndex(shard_id=shard_id)
    for doc in docs:
        idx.add_document(doc["doc_id"], doc)
    return idx


def assign_shard(doc_id: int, num_shards: int) -> int:
    """Consistent hashing-lite: simple modulo partitioning. Documented
    tradeoff in the README (real deployments would use consistent
    hashing to avoid a full reshuffle when num_shards changes)."""
    return doc_id % num_shards
