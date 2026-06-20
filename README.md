<div align="center">

# 🫀 HEART
### Hyperparameter-Efficient Adaptive RAG Tuning

*Tune RAG pipelines with LLM-guided Bayesian Optimization — and cluster queries by the config they **need**, not by what they look like.*

**LGBO +5.1** &nbsp;•&nbsp; config-aware clustering 🥇 **45.0%** beats same-machine K-means **42.2%**

</div>

---

## 🔭 Overview

RAG pipelines have many interdependent hyperparameters — chunking, retrieval method, top-k, reranker, decomposition, fusion. HEART treats pipeline configuration as a **black-box optimization** problem and searches for high-accuracy configs under a small evaluation budget, then **specializes a config per query cluster**.

### 🧠 Methods

| Method | Description |
|--------|-------------|
| 🥇 **LGBO** (ours) | **LLM-Guided Bayesian Optimization.** Clusters queries, allocates budget per cluster, and uses an LLM to guide surrogate construction + propose candidates. Within each cluster, training queries are picked by **importance sampling** (Horvitz–Thompson estimator) instead of a round-robin window. |
| **LLAMBO** | LLM-based BO baseline — an LLM directly proposes the next config from past observations. |
| **GPBO** | Gaussian-Process BO via BoTorch (GP surrogate + EI acquisition). |
| **TPE** | Tree-structured Parzen Estimator via Optuna — non-LLM statistical baseline. |

---

## 📊 Results

Evaluated on **HotpotQA** (1000 queries, 5 clusters):

| Method | Pre-eval | Post-eval | Δ |
|--------|:--------:|:---------:|:--:|
| TPE    | 38.0% | 38.4% | +0.4% |
| GPBO   | 38.2% | 38.5% | +0.3% |
| LLAMBO | 38.4% | 42.8% | +4.4% |
| 🥇 **LGBO (ours)** | 38.0% | **43.1%** | **+5.1%** |

LGBO reaches the highest post-eval accuracy — LLM guidance beats both statistical BO and direct LLM proposal.

### 🔬 Improved configuration & clustering ablation

With the full pipeline — query **decomposition enabled** (HotpotQA is multi-hop), **within-cluster importance sampling**, and a **lean synthesis space** — LGBO improves further. Local Qwen2.5-7B + Qwen3-Embedding-0.6B, 10 rounds × budget 120 (≈50 trials), full pre/post eval over all 1000 queries:

| Clustering | Pre-eval | Post-eval | Δ |
|------------|:--------:|:---------:|:--:|
| **K-means (embedding)** | 37.9% | **44.9%** | **+7.0%** |
| Search-space-aligned features | 37.9% | 43.7% | +5.8% |
| Bipartite (token + embedding) | 38.4% | 43.9% | +5.5% |

Key levers, each isolated by ablation:
- 🪓 **Query decomposition is essential on multi-hop QA** — enabling it lifts K-means+LGBO from −0.7% to +3.1% (and to +7.0% with the lean synthesis space).
- 🧹 **Search-space hygiene > more dimensions under a small budget.** Synthesis is fixed to `direct` (averaged 43.3% vs `refine` 40.2% vs `map_reduce` 31.9% — the latter summarize away the bridge entities multi-hop QA needs). Dropping `synthesis_mode` + `intermediate_length` concentrates the ~10-trial-per-cluster budget on retrieval/reranker/decomposition (+2 to +4 points everywhere).
- ⚖️ **Among similarity-based schemes, clustering is second-order** — embedding/token/feature schemes land within ~1 point. **But clustering on _config-need_ (below) breaks this.**

> Baseline rows (TPE/GPBO/LLAMBO) were not re-run under the improved pipeline; this table isolates LGBO configuration choices.

---

## 🧭 Config-aware clustering — cluster by *need*, not by *looks* 🥇

HEART tunes **one config per cluster**, so the clustering's only job is to group queries that want the **same config**. Embeddings cluster queries that *read* alike — but those still want different `top_k` / retrieval method / decomposition. Instead, an **LLM rates each query on config-aligned dimensions** ("how many facts? how many hops? exact-match?") and we cluster on *those* → **config-coherent** clusters where per-cluster tuning stops overfitting.

### Same-machine comparison

| Clustering | features | Pre-eval | Post-eval | Δ |
|------------|----------|:--------:|:---------:|:--:|
| 🥇 **Config-aligned LLM profile** | 8 LLM dims | 38.5% | **45.0%** | 🟢 **+6.5%** |
| K-means (baseline) | pure embedding | 38.5% | 42.2% | +3.7% |

**Profile-only beats same-machine K-means by +2.8 points.** Every cluster improves — including the 442-query giant (+6.1%), where an embedding cluster of that size mixes config needs and barely moves.

| cluster | n | what it is | pre → post | Δ |
|:--:|:--:|:--|:--:|:--:|
| `c0` | 187 | 🟢 simple lookups | 29.4 → 33.2 | +3.8 |
| `c1` | 442 | 🧗 heavy multi-hop | 31.9 → 38.0 | **+6.1** |
| `c2` | 256 | 🔗 2-hop + exact terms | 49.6 → 56.6 | +7.0 |
| `c3` | 50 | ❓ yes/no comparison | 50.0 → 66.0 | **+16.0** |
| `c4` | 65 | 🔢 numeric / date | 56.9 → 64.6 | +7.7 |

> ⚠️ Compare clustering **only on the same machine** — absolute accuracy drifts a few points across boxes (reranker/vLLM differences), enough to swamp the clustering effect. (Jovyan K-means hit 44.9%; the *same* K-means on this box hits 42.2%.)

### The profile — `clustering/profiler_llm.py`

One LLM call per query (local Qwen2.5-7B) → 12 dimensions, each mapped to a search-space knob. Query-side only ⇒ **no answer needed ⇒ deployable**. Inspired by [METIS (Ray et al., 2025)](https://arxiv.org/abs/2412.10543).

| 🎚️ dimension | range | → knob |
|:--|:--:|:--|
| `n_facts` | 1–10 | `top_k` 📚 |
| `n_hops` | 1–5 | `num_queries` 🔗 |
| `comparison` | 0/1 | `query_decomposition` 🪓 |
| `exact_term_need` | 0–2 | `method` / `bm25_weight` 🔍 |
| `numeric_temporal` | 0/1 | sparse / BM25 🔢 |
| `paraphrase_gap` | 0–2 | dense ↔ sparse 🌐 |
| `distractor_risk` | 0/1 | `fusion_mode` 🎛️ |
| `answer_length` | 1–5 | `reranker_top_k` 📏 |
| ~~`compositional` · `entity_density` · `reasoning_depth` · `ambiguity`~~ | — | 🗑️ auto-dropped — near-constant on HotpotQA (no signal) |

`clustering/cluster_profile.py` standardizes the dims, **auto-drops near-constant ones** (one value > 95%), runs `KMeans(k=5)`, and writes a dataset variant carrying the new `cluster_id`. ✨ **Zero core-code change** — run it with a config that sets `cluster_kmeans_enabled: false`.

### 🚀 Reproduce

```bash
# 1️⃣  profile every query (LLM via :8001)
python clustering/profiler_llm.py datasets/hotpotqa_1000_c5_real/Question.json clustering/data/profile_llm.json
# 2️⃣  cluster on the profile dims → datasets/hotpotqa_1000_c5_profile/
PROF_FILE=clustering/data/profile_llm.json python clustering/cluster_profile.py
# 3️⃣  run the LGBO pipeline on the variant  (HF_HUB_OFFLINE=1 if rerankers are cached)
python main.py -opt Option/LGBO_9params_profile.yaml -dataset_name hotpotqa_1000_c5_profile
```

---

## ⚙️ Setup

**Prerequisites**
- Python 3.10+
- vLLM serving an LLM (default `qwen2.5-7b` on `:8001`)
- vLLM serving an embedding model (default `Qwen3-Embedding-0.6B` on `:8017`)
- GPU with VRAM for the rerankers (or CPU-only)

**Configuration** — edit `Option/Config2.yaml` for base settings (LLM endpoints, embedding, data paths). Method-specific configs live in `Option/`:
- `LGBO_9params_profile.yaml` — config-aware profile clustering (winner)
- `LGBO_9params_kmeans_local.yaml` — K-means baseline
- `LLAMBO_*.yaml` / `GPBO_*.yaml` / `TPE_*.yaml` — baselines

## ▶️ Running

```bash
python main.py -opt Option/<config>.yaml -dataset_name <dataset>
```

Output lands in `runs/<dataset>/<exp_name>/Results/`:
- per-query answers and scores
- `full_eval_before_after_by_cluster.json` — pre/post accuracy by cluster
- cluster assignments + per-round accuracy curves

## 🗂️ Code structure

```
main.py            entry point (build index → cluster → LGBO tune → full eval)
Tuner/             LGBO / BO tuners, budget allocation, §6 importance sampling
Rerank/            reranker zoo (flashrank, transformer, Qwen3, UPR-T5, …)
Index/             FAISS + BM25 indexing & retrieval
Config/            search space + config schema
clustering/        config-aware query clustering (profiler + scripts + data)
Option/            run configs
datasets/ · runs/  inputs and outputs
```

## 🧪 Search space

HEART jointly optimizes:
- **RAG method** — dense / sparse / hybrid
- **Top-k** retrieval (2–32)
- **Hybrid BM25 weight** (0.2–0.8)
- **Query decomposition** — on/off, LLM choice, num sub-queries (2–5)
- **FAISS `efSearch`** (8–256) — the only tuned index param (HNSW `M=32`, `efConstruction=40`, metric `L2` are fixed build-time params; rebuilding the index per trial is expensive, `efSearch` is query-time)
- **Reranker** — flashrank variants, transformer rankers, Qwen3-reranker, UPR-T5
- **Reranker top-k** (2–32, capped to retrieval top-k)

Response synthesis is **fixed to `direct`** — `map_reduce`/`refine` (and `intermediate_length`) underperform on multi-hop QA and waste the small per-cluster budget; the flow falls back to `direct` when these params are absent.
