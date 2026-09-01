import math

import pytest

from common.ranking import bm25_score, score_documents, tfidf_score


def test_bm25_score_increases_with_term_frequency():
    low_tf = bm25_score(term_freq=1, doc_len=100, avg_doc_len=100, doc_freq=5, total_docs=1000)
    high_tf = bm25_score(term_freq=10, doc_len=100, avg_doc_len=100, doc_freq=5, total_docs=1000)
    assert high_tf > low_tf


def test_bm25_score_decreases_for_longer_documents_at_same_tf():
    short_doc = bm25_score(term_freq=3, doc_len=50, avg_doc_len=100, doc_freq=5, total_docs=1000)
    long_doc = bm25_score(term_freq=3, doc_len=400, avg_doc_len=100, doc_freq=5, total_docs=1000)
    assert short_doc > long_doc


def test_bm25_rewards_rarer_terms_more_idf():
    rare_term = bm25_score(term_freq=2, doc_len=100, avg_doc_len=100, doc_freq=2, total_docs=1000)
    common_term = bm25_score(term_freq=2, doc_len=100, avg_doc_len=100, doc_freq=500, total_docs=1000)
    assert rare_term > common_term


def test_tfidf_score_positive_for_valid_inputs():
    score = tfidf_score(term_freq=3, doc_freq=10, total_docs=1000)
    assert score > 0


def test_score_documents_ranks_matching_docs_above_nonmatching():
    postings = {
        "gradient": {1: 5, 2: 1},
        "descent": {1: 3},
    }
    doc_lengths = {1: 50, 2: 50}
    scored = dict(
        score_documents(
            query_terms=["gradient", "descent"],
            postings=postings,
            doc_lengths=doc_lengths,
            avg_doc_len=50,
            total_docs=2,
        )
    )
    # doc 1 matches both query terms with higher tf -> must outrank doc 2
    assert scored[1] > scored[2]


def test_score_documents_returns_empty_for_unknown_terms():
    scored = score_documents(
        query_terms=["nonexistent"],
        postings={"gradient": {1: 5}},
        doc_lengths={1: 50},
        avg_doc_len=50,
        total_docs=1,
    )
    assert scored == []


def test_bm25_matches_hand_computed_value():
    # k1=1.5, b=0.75 defaults; doc_len == avg_doc_len removes length norm term
    tf, doc_len, avg_doc_len, doc_freq, total_docs = 2, 100, 100, 10, 100
    idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
    expected = idf * (tf * 2.5) / (tf + 1.5)
    assert bm25_score(tf, doc_len, avg_doc_len, doc_freq, total_docs) == pytest.approx(expected)
