"""
Minimal query parser: tokenizes free-text terms and pulls out any
"quoted phrases" separately so shard search can apply a small
relevance boost to documents whose title contains the exact phrase.
Kept intentionally simple (no boolean operators) - documented as a
"next step" in the README rather than overbuilt for a portfolio demo.
"""
import re
from dataclasses import dataclass, field
from typing import List

from common.tokenizer import tokenize

_PHRASE_RE = re.compile(r'"([^"]+)"')


@dataclass
class ParsedQuery:
    raw: str
    terms: List[str] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)


def parse_query(raw_query: str) -> ParsedQuery:
    phrases = _PHRASE_RE.findall(raw_query)
    remainder = _PHRASE_RE.sub(" ", raw_query)
    terms = tokenize(remainder)
    for phrase in phrases:
        terms.extend(tokenize(phrase))
    # de-dupe while preserving order
    seen = set()
    unique_terms = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique_terms.append(t)
    return ParsedQuery(raw=raw_query, terms=unique_terms, phrases=phrases)
