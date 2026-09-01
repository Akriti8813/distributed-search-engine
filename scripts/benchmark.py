"""
Benchmarks the deployed gateway for:
  - latency (p50/p95/p99) and throughput (QPS) under concurrent load
  - ranking quality (Precision@k, NDCG@k) using synthetic ground truth
    derived from the corpus's known topic labels

Usage:
    python scripts/benchmark.py --gateway http://localhost:8000 \
        --corpus data/corpus/corpus.jsonl --requests 500 --concurrency 50
"""
import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import httpx

from generate_corpus import TOPICS  # noqa: E402


def load_topic_map(corpus_path: str) -> dict:
    doc_topic = {}
    with open(corpus_path) as f:
        for line in f:
            d = json.loads(line)
            doc_topic[d["doc_id"]] = d["topic"]
    return doc_topic


async def timed_request(client: httpx.AsyncClient, base_url: str, query: str, top_k: int):
    t0 = time.perf_counter()
    try:
        resp = await client.get(f"{base_url}/search", params={"q": query, "top_k": top_k}, timeout=10)
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return elapsed_ms, resp.json()
    except httpx.HTTPError:
        return None, None


async def run_latency_throughput(base_url: str, total_requests: int, concurrency: int):
    # 80% of traffic hits a small "head query" set (cache-friendly, like
    # real search traffic), 20% hits unique long-tail queries.
    head_queries = [" ".join(v[:3]) for v in TOPICS.values()]
    tail_queries = [" ".join(v[i:i + 3]) for v in TOPICS.values() for i in range(3, len(v) - 3, 3)]

    import random
    rng = random.Random(7)
    queries = []
    for _ in range(total_requests):
        if rng.random() < 0.8:
            queries.append(rng.choice(head_queries))
        else:
            queries.append(rng.choice(tail_queries) if tail_queries else rng.choice(head_queries))

    latencies = []
    cache_hits = 0
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def bound_request(q):
            async with sem:
                return await timed_request(client, base_url, q, 10)

        t_start = time.perf_counter()
        results = await asyncio.gather(*[bound_request(q) for q in queries])
        wall_time_s = time.perf_counter() - t_start

    for elapsed_ms, payload in results:
        if elapsed_ms is not None:
            latencies.append(elapsed_ms)
            if payload and payload.get("cache_hit"):
                cache_hits += 1

    latencies.sort()
    n = len(latencies)
    def pct(p):
        if not n:
            return None
        idx = min(int(p * n), n - 1)
        return round(latencies[idx], 2)

    return {
        "total_requests": total_requests,
        "concurrency": concurrency,
        "successful": n,
        "wall_time_s": round(wall_time_s, 3),
        "throughput_qps": round(n / wall_time_s, 2) if wall_time_s > 0 else None,
        "latency_ms": {
            "min": round(min(latencies), 2) if latencies else None,
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": round(max(latencies), 2) if latencies else None,
            "mean": round(statistics.mean(latencies), 2) if latencies else None,
        },
        "cache_hit_rate": round(cache_hits / n, 3) if n else None,
    }


async def run_ranking_quality(base_url: str, doc_topic: dict, top_k: int = 10):
    per_topic = {}
    async with httpx.AsyncClient() as client:
        for topic, vocab in TOPICS.items():
            query = " ".join(vocab[:4])
            elapsed_ms, payload = await timed_request(client, base_url, query, top_k)
            if not payload:
                continue
            results = payload["results"]
            relevances = [1 if doc_topic.get(r["doc_id"]) == topic else 0 for r in results]

            precision_at_k = sum(relevances) / len(relevances) if relevances else 0.0

            def dcg(rels):
                return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

            ideal = sorted(relevances, reverse=True)
            idcg = dcg(ideal)
            ndcg = dcg(relevances) / idcg if idcg > 0 else 0.0

            per_topic[topic] = {
                "query": query,
                "precision_at_k": round(precision_at_k, 3),
                "ndcg_at_k": round(ndcg, 3),
            }

    avg_precision = round(statistics.mean(v["precision_at_k"] for v in per_topic.values()), 3)
    avg_ndcg = round(statistics.mean(v["ndcg_at_k"] for v in per_topic.values()), 3)
    return {"per_topic": per_topic, "avg_precision_at_k": avg_precision, "avg_ndcg_at_k": avg_ndcg}


async def main_async(args):
    doc_topic = load_topic_map(args.corpus)

    print(f"Running ranking-quality eval (top_{args.top_k}) against {len(TOPICS)} topics...")
    quality = await run_ranking_quality(args.gateway, doc_topic, args.top_k)

    print(f"Running load test: {args.requests} requests @ concurrency {args.concurrency}...")
    perf = await run_latency_throughput(args.gateway, args.requests, args.concurrency)

    summary = {"performance": perf, "ranking_quality": quality, "corpus_size": len(doc_topic)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Latency / Throughput ===")
    print(json.dumps(perf, indent=2))
    print("\n=== Ranking Quality ===")
    print(f"avg Precision@{args.top_k}: {quality['avg_precision_at_k']}")
    print(f"avg NDCG@{args.top_k}: {quality['avg_ndcg_at_k']}")
    print(f"\nFull results written to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", type=str, default="http://localhost:8000")
    ap.add_argument("--corpus", type=str, default="data/corpus/corpus.jsonl")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=str, default="benchmark_results.json")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
