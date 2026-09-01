from common.query_parser import parse_query


def test_parses_plain_terms():
    parsed = parse_query("gradient descent training")
    assert parsed.terms == ["gradient", "descent", "training"]
    assert parsed.phrases == []


def test_extracts_quoted_phrase_separately():
    parsed = parse_query('"gradient descent" optimizer')
    assert parsed.phrases == ["gradient descent"]
    assert "optimizer" in parsed.terms
    assert "gradient" in parsed.terms  # phrase terms are also searchable individually


def test_dedupes_terms_preserving_order():
    parsed = parse_query("gradient gradient descent")
    assert parsed.terms == ["gradient", "descent"]


def test_empty_query_returns_no_terms():
    parsed = parse_query("")
    assert parsed.terms == []
    assert parsed.phrases == []
