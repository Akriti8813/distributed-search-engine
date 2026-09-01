"""
Partitions the corpus across N shards (doc_id % num_shards) and
builds + serializes an inverted index for each shard.

Usage:
    python scripts/build_shards.py --corpus data/corpus/corpus.jsonl \
        --num-shards 4 --out-dir data/shards
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.inverted_index import ShardIndex, assign_shard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="data/corpus/corpus.jsonl")
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default="data/shards")
    args = ap.parse_args()

    docs_by_shard = defaultdict(list)
    with open(args.corpus) as f:
        for line in f:
            doc = json.loads(line)
            shard_id = assign_shard(doc["doc_id"], args.num_shards)
            docs_by_shard[shard_id].append(doc)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for shard_id in range(args.num_shards):
        docs = docs_by_shard[shard_id]
        idx = ShardIndex(shard_id=shard_id)
        for doc in docs:
            idx.add_document(doc["doc_id"], doc)
        path = out_dir / f"shard_{shard_id}.pkl"
        idx.save(path)
        print(
            f"shard {shard_id}: {len(docs)} docs, "
            f"{len(idx.postings)} unique terms -> {path}"
        )
    print(f"Built {args.num_shards} shards in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
