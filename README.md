# HEART: Hyperparameter-Efficient Adaptive RAG Tuning

HEART is a framework for automatically tuning RAG (Retrieval-Augmented Generation) pipeline hyperparameters using Bayesian Optimization with LLM guidance.

## Overview

RAG pipelines have many interdependent hyperparameters — chunking strategy, retrieval method, top-k, reranker choice, etc. HEART treats pipeline configuration as a black-box optimization problem and searches for high-accuracy configurations using a small evaluation budget.

### Methods

| Method | Description |
|--------|-------------|
| **LGBO** (ours) | LLM-Guided Bayesian Optimization. Clusters queries by difficulty, allocates budget per cluster, and uses an LLM to guide surrogate model construction and propose candidates. Within each cluster, training queries are selected by importance sampling (Horvitz–Thompson estimator) rather than a round-robin window. |
| **LLAMBO** | LLM-based Bayesian Optimization baseline, inspired by the LLAMBO paper. Uses an LLM to directly propose next hyperparameter configs given past observations. |
| **GPBO** | Gaussian Process Bayesian Optimization via BoTorch. Standard GP surrogate with EI acquisition. |
| **TPE** | Tree-structured Parzen Estimator via Optuna. Non-LLM statistical baseline. |

## Results

Evaluated on **HotpotQA** (1000 queries, 5 KMeans clusters):

| Method | Pre-eval | Post-eval | Delta |
|--------|----------|-----------|-------|
| TPE    | 38.0%    | 38.4%     | +0.4% |
| GPBO   | 38.2%    | 38.5%     | +0.3% |
| LLAMBO | 38.4%    | 42.8%     | +4.4% |
| **LGBO (ours)** | 38.0% | **43.1%** | **+5.1%** |

LGBO achieves the highest post-evaluation accuracy, showing that LLM guidance improves over both statistical BO and direct LLM proposal baselines.

### Improved configuration & clustering ablation

With the full pipeline — query **decomposition enabled** (HotpotQA is multi-hop), **within-cluster importance sampling**, and a **leaned synthesis space** (see Search Space) — LGBO improves further. Runs use local Qwen2.5-7B + Qwen3-Embedding-0.6B, 10 rounds × budget 120 (≈50 trials), full pre/post eval over all 1000 queries:

| Clustering | Pre-eval | Post-eval | Delta |
|------------|----------|-----------|-------|
| **K-means (embedding)** | 37.9% | **44.9%** | **+7.0%** |
| Search-space-aligned features | 37.9% | 43.7% | +5.8% |
| Bipartite (token + embedding) | 38.4% | 43.9% | +5.5% |

Key levers, each isolated by ablation:
- **Query decomposition is essential on multi-hop QA** — enabling it lifts K-means+LGBO from −0.7% to +3.1% (and to +7.0% with the lean synthesis space).
- **Search-space hygiene > more dimensions under a small budget.** Response synthesis is fixed to `direct`: across all trials, `direct` averaged 43.3% vs `refine` 40.2% vs `map_reduce` 31.9% (map_reduce/refine summarize away the bridge entities multi-hop QA needs). Dropping `synthesis_mode` + `intermediate_length` concentrates the ~10-trial-per-cluster budget on retrieval/reranker/decomposition and lifts every clustering by +2 to +4 points.
- **Clustering choice is second-order once the pipeline is right** — under the full pipeline the three clustering schemes land within ~1 point of each other.

> Note: baseline rows (TPE/GPBO/LLAMBO) above were not re-run under the improved pipeline; the table here isolates LGBO configuration choices.

## Setup

### Prerequisites

- Python 3.10+
- vLLM server running an LLM (default: qwen2.5-7b on port 8001)
- vLLM server running an embedding model (default: Qwen3-Embedding-0.6B on port 8017)
- GPU with sufficient VRAM for your chosen rerankers (or CPU-only mode)

### Installation



### Environment Variables



### Configuration

Edit  for base settings (LLM endpoints, embedding, data paths).

Method-specific configs are in :
-  — LGBO with cluster-aware budget allocation
-  — LLAMBO flat (5 rounds x 200 budget each)
-  — GPBO flat
-  — TPE flat

Copy  to  and fill in your API keys (file is gitignored).

## Running



Output is written to :
-  — per-query answers and scores
-  — pre/post accuracy by cluster
-  — cluster assignments
-  — accuracy curves per round

## Code Structure



## Search Space

HEART jointly optimizes over:
- RAG method: dense, sparse, hybrid
- Top-k retrieval (2-32)
- Hybrid BM25 weight (0.2-0.8)
- Query decomposition (enabled/disabled, LLM choice, num queries)
- FAISS index params (HNSW M, efSearch, efConstruction, metric)
- Reranker choice (flashrank variants, transformer rankers, Qwen3-reranker, UPR-T5)
- Reranker top-k (2-100)

Response synthesis is **fixed to `direct`** and is no longer tuned: `map_reduce`/`refine` (and their `intermediate_length`) consistently underperformed on multi-hop QA and wasted the small per-cluster budget. The flow falls back to `direct` automatically when these params are absent.
