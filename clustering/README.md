<div align="center">

# 🧭 Config-Aware Query Clustering for HEART

**Cluster queries by the RAG config they _need_ — not by what they _look like_.**

`profile-only clustering` 🥇 **45.0%** &nbsp;•&nbsp; KMeans baseline **42.2%** &nbsp;•&nbsp; **+2.8 pts**, same machine

</div>

---

## 💡 TL;DR

HEART's LGBO tuner splits queries into `k` clusters and tunes **one RAG config per cluster**.
So the clustering's _only_ job is to group together queries that want the **same config**.

> 🔑 **Key insight:** an embedding clusters queries that *read* alike — but those still want
> different `top_k` / retrieval method / decomposition. We instead ask an **LLM** to rate each
> query on a handful of **config-aligned dimensions** ("how many facts? how many hops? exact-match?")
> and cluster on *those*. The clusters become **config-coherent**, so per-cluster tuning stops
> overfitting and actually generalizes. 🎯

---

## 📊 Results

`HotpotQA` · 1000 queries · `k=5` · 9-param LGBO · 1200 budget / 10 rounds · **same machine**

| 🏷️ method | features | before → after | gain |
|:--|:--|:--:|:--:|
| 🥇 **profile-only** | 8 config-aligned **LLM** dims | 38.5% → **45.0%** | 🟢 **+6.5** |
| ⚪ KMeans (baseline) | pure Qwen3 embedding | 38.5% → 42.2% | +3.7 |

### Per-cluster breakdown (profile-only) 🔬

| cluster | n | what it is | before → after | Δ |
|:--:|:--:|:--|:--:|:--:|
| `c0` | 187 | 🟢 simple lookups | 29.4 → 33.2 | +3.8 |
| `c1` | 442 | 🧗 heavy multi-hop | 31.9 → 38.0 | **+6.1** |
| `c2` | 256 | 🔗 2-hop + exact terms | 49.6 → 56.6 | +7.0 |
| `c3` | 50 | ❓ yes/no comparison | 50.0 → 66.0 | **+16.0** |
| `c4` | 65 | 🔢 numeric / date | 56.9 → 64.6 | +7.7 |

**Every** cluster improved — even the 442-query giant (+6.1). With semantic clustering the
big cluster mixes config needs and barely moves; here it's config-coherent, so one tuned
config fits all 442. Imbalance (442 vs 50) **stopped mattering** once clusters were coherent. ⚖️

> ⚠️ **Compare on the same machine only.** Absolute accuracy drifts a few points across
> machines (reranker / vLLM differences) — enough to swamp the clustering effect. Our jovyan
> KMeans hit 44.9% but the *same* KMeans on this box hits 42.2%.

---

## 🧬 The method

### 1️⃣ `profiler_llm.py` — LLM query profiler

One LLM call per query (local **Qwen2.5-7B**) → 12 profile dimensions, each mapped to a HEART
search-space knob. Query-side only → **no answer needed → deployable**. Inspired by
[METIS (Ray et al., 2025)](https://arxiv.org/abs/2412.10543) query profiling.

| 🎚️ dimension | range | → HEART knob |
|:--|:--:|:--|
| `n_facts` | 1–10 | `top_k` 📚 |
| `n_hops` | 1–5 | `num_queries` 🔗 |
| `comparison` | 0/1 | `query_decomposition` 🪓 |
| `exact_term_need` | 0–2 | `method` / `bm25_weight` 🔍 |
| `numeric_temporal` | 0/1 | sparse / BM25 🔢 |
| `paraphrase_gap` | 0–2 | dense ↔ sparse 🌐 |
| `distractor_risk` | 0/1 | `fusion_mode` 🎛️ |
| `answer_length` | 1–5 | `reranker_top_k` 📏 |
| ~~`compositional`~~ ~~`entity_density`~~ ~~`reasoning_depth`~~ ~~`ambiguity`~~ | — | 🗑️ dropped — near-constant on HotpotQA (no signal) |

### 2️⃣ `cluster_profile.py` — cluster on those dims

Standardize → **auto-drop near-constant dims** (one value covers > 95 %) → plain `KMeans(k=5)`
→ write a dataset variant carrying the new `cluster_id`.

✨ **Zero core-code change.** Clustering is injected as a precomputed `cluster_id` in a dataset
variant; run it with a config that sets `cluster_kmeans_enabled: false`. The KMeans baseline
just flips that to `true` (built-in embedding KMeans).

---

## 🗂️ Layout

```
clustering/
├── 📄 README.md
├── 🐍 profiler_llm.py        LLM query profiler (12 config-aligned dims)   [step 1]
├── 🐍 cluster_profile.py     KMeans on profile dims → dataset variant      [step 2]
├── 🐍 warm_rerankers.py      util · pre-warm / validate the 8 rerankers offline
└── 📦 data/
    └── profile_llm.json      precomputed 12-dim profiles (1000 queries · reproducibility)

Option/
├── ⚙️ LGBO_9params_profile.yaml        profile-only run  (cluster_kmeans_enabled: false)
└── ⚙️ LGBO_9params_kmeans_local.yaml   KMeans baseline   (embedding KMeans)
```

---

## 🚀 Reproduce the winner

```bash
# 1️⃣  profile every query (LLM via :8001)
python clustering/profiler_llm.py \
    datasets/hotpotqa_1000_c5_real/Question.json \
    clustering/data/profile_llm.json

# 2️⃣  cluster on the profile dims → writes datasets/hotpotqa_1000_c5_profile/
PROF_FILE=clustering/data/profile_llm.json python clustering/cluster_profile.py

# 3️⃣  run the 9-param LGBO pipeline on the variant
#     (HF_HUB_OFFLINE=1 if the rerankers are already cached)
python main.py -opt Option/LGBO_9params_profile.yaml -dataset_name hotpotqa_1000_c5_profile
```

🔧 **Knobs** (env vars, see the top of each script): `LLM_URL`, `EMB_URL`, `SRC_DS`, `OUT_DS`,
`K`, `PROF_FILE`, `WORKERS`.

---

## 🧪 Caveats

- 📈 **Single run.** The +6.5 vs +3.7 gap is large, but confirm a noise band before publishing
  (re-run KMeans & profile 1–2× each).
- 🖥️ **Same-machine only** for any clustering comparison (see warning above).
- 🤖 Profile dims are **LLM estimates** — coarse integers, so clusters are lumpy (one modal
  "multi-hop" type dominates). That coarseness is fine *because* the dims track config need.

<div align="center">

**Cluster on need, not on looks.** 🎯

</div>
