"""
Ranking functions: BM25 (default) and TF-IDF, computed against an
InvertedIndex's postings lists. Pure functions over plain dicts so
they're trivial to unit test in isolation from FastAPI/Redis.
"""
import math
from typing import Dict, List, Tuple

# BM25 hyperparameters (standard defaults from Robertson & Zaragoza).
K1 = 1.5
B = 0.75


def bm25_score(
    term_freq: int,
    doc_len: int,
    avg_doc_len: float,
    doc_freq: int,
    total_docs: int,
    k1: float = K1,
    b: float = B,
) -> float:
    """Okapi BM25 score contribution for a single query term in a
    single document."""
    idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
    denom = term_freq + k1 * (1 - b + b * doc_len / avg_doc_len)
    return idf * (term_freq * (k1 + 1)) / denom


def tfidf_score(term_freq: int, doc_freq: int, total_docs: int) -> float:
    """Simple log-weighted TF-IDF, offered as an alternate ranking
    strategy (the resume bullet calls out both TF-IDF and BM25)."""
    tf = 1 + math.log(term_freq)
    idf = math.log(total_docs / doc_freq)
    return tf * idf


def score_documents(
    query_terms: List[str],
    postings: Dict[str, Dict[int, int]],
    doc_lengths: Dict[int, int],
    avg_doc_len: float,
    total_docs: int,
    method: str = "bm25",
) -> List[Tuple[int, float]]:
    """
    Score every candidate document that contains at least one query
    term against the full query, for one shard.

    postings: term -> {doc_id: term_freq}   (this shard's inverted index)
    doc_lengths: doc_id -> token count       (this shard's doc lengths)
    Returns [(doc_id, score), ...] unsorted.
    """
    scores: Dict[int, float] = {}
    for term in query_terms:
        term_postings = postings.get(term)
        if not term_postings:
            continue
        doc_freq = len(term_postings)
        for doc_id, tf in term_postings.items():
            if method == "bm25":
                s = bm25_score(
                    term_freq=tf,
                    doc_len=doc_lengths[doc_id],
                    avg_doc_len=avg_doc_len,
                    doc_freq=doc_freq,
                    total_docs=total_docs,
                )
            else:
                s = tfidf_score(tf, doc_freq, total_docs)
            scores[doc_id] = scores.get(doc_id, 0.0) + s
    return list(scores.items())
