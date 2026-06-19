#!/usr/bin/env python3
"""Pre-warm + validate every reranker in the search space (offline, hf-mirror for flashrank).

Constructing a Reranker triggers model download/load, so this both fills caches and
confirms each of the 8 choices loads on server3 before the multi-hour run.
"""
import sys
import time
sys.path.insert(0, "/data1/yujia/Yingli/HEART")
from Rerank.RerankFactory import get_reranker

CHOICES = [
    "flashrank::ms-marco-TinyBERT-L-2-v2",
    "flashrank::ms-marco-MiniLM-L-12-v2",
    "qwen_reranker::qwen3-reranker-0.6b",
    "transformer_ranker::mxbai-rerank-base",
    "transformer_ranker::bge-reranker-v2-m3",
    "transformer_ranker::jina-reranker-base-multilingual",
    "transformer_ranker::gte-multilingual-reranker-base",
    "upr::t5-base",
]

ok, bad = [], []
for ch in CHOICES:
    t0 = time.time()
    try:
        get_reranker({"reranker_choice": ch, "reranker_top_k": 5})
        dt = time.time() - t0
        print(f"[OK]   {ch}   ({dt:.1f}s)", flush=True)
        ok.append(ch)
    except Exception as e:
        print(f"[FAIL] {ch}\n        {type(e).__name__}: {str(e)[:300]}", flush=True)
        bad.append(ch)

print(f"\n=== warmup summary: {len(ok)}/{len(CHOICES)} OK ===", flush=True)
if bad:
    print("FAILED:", bad, flush=True)
    sys.exit(1)
print("ALL RERANKERS LOAD OFFLINE — safe to launch full run.", flush=True)
