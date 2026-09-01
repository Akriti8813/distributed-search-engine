"""
Generates a synthetic document corpus for benchmarking the search
engine. Documents are built from topic-specific vocabularies so that
ranking quality (precision@k) can be measured against known ground
truth: for a query built from topic X's keywords, the "relevant"
documents are exactly the documents generated with primary topic X.

Usage:
    python scripts/generate_corpus.py --num-docs 50000 --out data/corpus/corpus.jsonl
"""
import argparse
import json
import random
from pathlib import Path

TOPICS = {
    "distributed_systems": [
        "consensus", "replication", "sharding", "latency", "throughput",
        "partition", "raft", "paxos", "cluster", "node", "leader",
        "election", "quorum", "failover", "gossip", "vector", "clock",
    ],
    "machine_learning": [
        "gradient", "neural", "network", "training", "inference",
        "embedding", "regression", "classifier", "overfitting",
        "dataset", "feature", "epoch", "optimizer", "accuracy", "loss",
    ],
    "databases": [
        "index", "query", "transaction", "isolation", "schema",
        "join", "commit", "rollback", "btree", "cache", "storage",
        "durability", "replica", "shard", "vacuum",
    ],
    "web_development": [
        "api", "endpoint", "frontend", "backend", "javascript",
        "component", "render", "request", "response", "session",
        "authentication", "middleware", "route", "template", "cache",
    ],
    "finance": [
        "portfolio", "equity", "dividend", "interest", "inflation",
        "bond", "asset", "liability", "valuation", "market", "risk",
        "hedge", "yield", "capital", "leverage",
    ],
    "sports": [
        "match", "tournament", "score", "player", "coach", "league",
        "season", "championship", "stadium", "referee", "goal",
        "training", "team", "victory", "draft",
    ],
    "cooking": [
        "recipe", "ingredient", "oven", "simmer", "roast", "spice",
        "flavor", "garnish", "kitchen", "bake", "grill", "sauce",
        "dough", "marinade", "kettle",
    ],
    "astronomy": [
        "galaxy", "orbit", "telescope", "planet", "nebula", "comet",
        "gravity", "star", "asteroid", "cosmic", "satellite",
        "eclipse", "spectrum", "meteor", "constellation",
    ],
}

FILLER = [
    "the", "system", "process", "result", "study", "report", "team",
    "approach", "method", "analysis", "example", "case", "review",
    "update", "project", "design", "model", "value", "level", "group",
]


def make_document(doc_id: int, rng: random.Random) -> dict:
    topic = rng.choice(list(TOPICS.keys()))
    vocab = TOPICS[topic]
    title_words = rng.sample(vocab, k=min(4, len(vocab)))
    title = " ".join(w.capitalize() for w in title_words)

    body_len = rng.randint(40, 120)
    body_words = []
    for _ in range(body_len):
        # 70% topic vocabulary, 30% generic filler -> realistic noise
        # while keeping topic terms dense enough to rank well.
        pool = vocab if rng.random() < 0.7 else FILLER
        body_words.append(rng.choice(pool))
    body = " ".join(body_words)

    return {"doc_id": doc_id, "topic": topic, "title": title, "body": body}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-docs", type=int, default=50_000)
    ap.add_argument("--out", type=str, default="data/corpus/corpus.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for doc_id in range(args.num_docs):
            doc = make_document(doc_id, rng)
            f.write(json.dumps(doc) + "\n")

    print(f"Wrote {args.num_docs} documents to {out_path}")


if __name__ == "__main__":
    main()
