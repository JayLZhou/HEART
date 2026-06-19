# Query Clustering for HEART per-cluster RAG tuning

HEART's LGBO tuner groups queries into `k` clusters and tunes one RAG config per cluster.
This directory holds the winning clustering method and the KMeans baseline it beats.

**Key finding:** cluster on *what RAG config a query needs* (a small set of config-aligned
dimensions estimated by an LLM), **not** on semantic similarity. Config-coherent clusters
make per-cluster tuning generalize; semantic (embedding) clusters look alike but mix optimal
configs, so the tuned config overfits.

## Results (HotpotQA 1000q, k=5, 9params LGBO, 1200 budget / 10 rounds, same machine)

| method | features | before → after | gain |
|---|---|---|---|
| **profile-only (WINNER)** | 8 config-aligned LLM dims | 38.5% → **45.0%** | **+6.5** |
| KMeans (baseline) | pure Qwen3 embedding | 38.5% → 42.2% | +3.7 |

Every cluster improved under profile-only, including the largest (442 queries, +6.1) — because
the clusters are config-coherent. Imbalance (442 vs 50) stopped mattering once that was true.
(Variants that mixed the embedding with hand NER features only matched KMeans, so they were dropped.)

> Compare clustering only **on the same machine** — absolute accuracy shifts a few points
> across machines (reranker/vLLM differences), which can swamp the clustering effect.

## The winning method: `profiler_llm.py` → `cluster_profile.py`

1. **`profiler_llm.py`** — one LLM call per query (local Qwen2.5-7B) estimates 12 config-aligned
   profile dimensions, each mapping to a HEART search-space knob. Query-side only (no answer
   needed → deployable). Inspired by METIS (Ray et al., 2025) query profiling.

   | dim | range | → knob |
   |---|---|---|
   | `n_facts` | 1–10 | `top_k` |
   | `n_hops` | 1–5 | `num_queries` |
   | `comparison` | 0/1 | `query_decomposition` |
   | `exact_term_need` | 0–2 | `method` / `bm25_weight` |
   | `numeric_temporal` | 0/1 | sparse/BM25 |
   | `paraphrase_gap` | 0–2 | dense ↔ sparse |
   | `distractor_risk` | 0/1 | `fusion_mode` |
   | `answer_length` | 1–5 | `reranker_top_k` |
   | `compositional`, `entity_density`, `reasoning_depth`, `ambiguity` | — | dropped: near-constant on HotpotQA (no signal) |

2. **`cluster_profile.py`** — standardize the profile dims, auto-drop near-constant ones
   (a single value covers >95%), and run plain KMeans(k=5). Writes a dataset variant with the
   new `cluster_id`.

Clustering is injected as a precomputed `cluster_id` in a dataset variant — **no core code
change**. Run the variant with a config that has `cluster_kmeans_enabled: false`.
The KMeans baseline uses the built-in `cluster_kmeans_enabled: true` path (embedding KMeans).

## Files

```
profiler_llm.py        LLM query profiler (12 config-aligned dims)         [WINNER step 1]
cluster_profile.py     KMeans on profile dims only -> dataset variant       [WINNER step 2]
warm_rerankers.py      util: pre-warm / validate the 8 rerankers load offline
data/
  profile_llm.json     precomputed 12-dim profiles for the 1000 queries (reproducibility)
```

Run configs live with the rest in `Option/`:
- `Option/LGBO_9params_profile.yaml` — profile-only run (`cluster_kmeans_enabled: false`)
- `Option/LGBO_9params_kmeans_local.yaml` — KMeans baseline (embedding KMeans)

## Reproduce the winner

```bash
# 1. profile every query (LLM via :8001)
python clustering/profiler_llm.py datasets/hotpotqa_1000_c5_real/Question.json clustering/data/profile_llm.json

# 2. cluster on the profile dims -> writes datasets/hotpotqa_1000_c5_profile/
PROF_FILE=clustering/data/profile_llm.json python clustering/cluster_profile.py

# 3. run the 9params LGBO pipeline on the variant (HF_HUB_OFFLINE=1 if rerankers are cached)
python main.py -opt Option/LGBO_9params_profile.yaml -dataset_name hotpotqa_1000_c5_profile
```

`profiler_llm.py` / `cluster_profile.py` read endpoints and paths from env vars
(`LLM_URL`, `EMB_URL`, `SRC_DS`, `OUT_DS`, `K`, …) — see the top of each file.
