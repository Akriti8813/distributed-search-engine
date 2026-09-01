"""
Lightweight tokenizer + stopword filter shared by the indexer and the
query parser, so that documents and queries are always tokenized the
same way.
"""
import re
from typing import List

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small, fixed stopword list (kept in-process so we don't pull in nltk
# just for this).
STOPWORDS = frozenset(
    """
    a an the and or but if while is are was were be been being
    to of in on at for with by from as it its this that these those
    i you he she we they them his her their our your my
    not no do does did doing have has had having
    """.split()
)


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Lowercase + alnum tokenize. Deterministic and dependency-free so
    the same function can run inside every shard container without
    extra downloads at build time."""
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
