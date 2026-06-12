# HEART: Hyperparameter-Efficient Adaptive RAG Tuning

HEART is a framework for automatically tuning RAG (Retrieval-Augmented Generation) pipeline hyperparameters using Bayesian Optimization with LLM guidance.

## Overview

RAG pipelines have many interdependent hyperparameters — chunking strategy, retrieval method, top-k, reranker choice, etc. HEART treats pipeline configuration as a black-box optimization problem and searches for high-accuracy configurations using a small evaluation budget.

### Methods

| Method | Description |
|--------|-------------|
| **LGBO** (ours) | LLM-Guided Bayesian Optimization. Clusters queries by difficulty, allocates budget per cluster, and uses an LLM to guide surrogate model construction and propose candidates. |
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
