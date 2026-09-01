from common.tokenizer import tokenize


def test_lowercases_and_splits_on_punctuation():
    assert tokenize("Hello, World! Search-Engine.") == ["hello", "world", "search", "engine"]


def test_removes_stopwords_by_default():
    tokens = tokenize("the cat is on the mat")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "cat" in tokens
    assert "mat" in tokens


def test_can_keep_stopwords():
    tokens = tokenize("the cat is here", remove_stopwords=False)
    assert "the" in tokens
    assert "is" in tokens


def test_empty_string_returns_empty_list():
    assert tokenize("") == []


def test_numbers_are_kept_as_tokens():
    assert tokenize("bm25 top10 results") == ["bm25", "top10", "results"]
