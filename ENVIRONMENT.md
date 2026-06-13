# HEART Environment Guide (3haoji server)

## Code

| Item | Path |
|------|------|
| Main codebase | `/data1/yujia/HEART/HEART/` |
| Git remote (fork) | `git@github.com:YouranSun/HEART.git` |
| Git remote (upstream) | `git@github.com:JayLZhou/HEART.git` |
| Active branch | `tuning-cleaned` (local: `rebuild-tuning`) |
| Python environment | `/data1/yujia/envs/graphrag/bin/python` |

## Datasets

| Dataset | Path | Size |
|---------|------|------|
| `hotpotqa_1000_c5_real` | `/data1/yujia/HEART/HEART/datasets/hotpotqa_1000_c5_real/` | 1000 queries, 5 KMeans clusters |
| `hotpotqa_1000_c5` | `/data1/yujia/HEART/HEART/datasets/hotpotqa_1000_c5/` | 1000 queries (alternate clustering) |
| `hotpotqa_5_smoke` | `/data1/yujia/HEART/HEART/datasets/hotpotqa_5_smoke/` | 5 queries (smoke test) |

Each dataset folder contains `Corpus.json` and `Question.json`.

## LLM & Embedding Services

| Service | Model | Local path | Port |
|---------|-------|------------|------|
| LLM (load balancer) | `qwen2.5-7b` | — | **8001** (proxy → 8002/8003) |
| vLLM instance A | `Qwen2.5-7B-Instruct` | `/data1/yujia/models/Qwen2.5-7B-Instruct` | 8002 (GPU1) |
| vLLM instance B | `Qwen2.5-7B-Instruct` | `/data1/yujia/models/Qwen2.5-7B-Instruct` | 8003 (GPU2) |
| Embedding server | `Qwen3-Embedding-0.6B` | `/data1/yujia/models/Qwen3-Embedding-0.6B` | **8017** (GPU2) |
| LB proxy script | — | `/data1/yujia/lb_proxy.py` | — |

HEART config always points to port **8001** (LLM) and **8017** (embedding). Do not change these.

## Reranker Models

| Reranker choice (in YAML) | Model path on disk |
|--------------------------|-------------------|
| `flashrank::ms-marco-TinyBERT-L-2-v2` | `/data1/yujia/HEART/HEART/cache/models/ms-marco-TinyBERT-L-2-v2/` |
| `flashrank::ms-marco-MiniLM-L-12-v2` | `/data1/yujia/HEART/HEART/cache/models/ms-marco-MiniLM-L-12-v2/` |
| `qwen_reranker::qwen3-reranker-0.6b` | `~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/` |
| `transformer_ranker::mxbai-rerank-base` | `~/.cache/huggingface/hub/models--mixedbread-ai--mxbai-rerank-base-v1/` |
| `transformer_ranker::bge-reranker-v2-m3` | `~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/` |
| `transformer_ranker::jina-reranker-base-multilingual` | `~/.cache/huggingface/hub/models--jinaai--jina-reranker-v2-base-multilingual/` |
| `transformer_ranker::gte-multilingual-reranker-base` | `~/.cache/huggingface/hub/models--Alibaba-NLP--gte-multilingual-reranker-base/` |
| `upr::t5-base` | `~/.cache/huggingface/hub/models--google--t5-base-lm-adapt/` |

All models are cached locally. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before running.

## GPU Status

| GPU | Status | Used / Total | Occupied by |
|-----|--------|-------------|-------------|
| GPU0 | ❌ Hardware fault | — | — |
| GPU1 | 🟡 ~92% full | 37.5 / 40 GB | vLLM (port 8002) |
| GPU2 | 🔴 ~99% full | 40.3 / 40 GB | vLLM (port 8003) + embedding (port 8017) |

**Rerankers must run on CPU.** Use `CUDA_VISIBLE_DEVICES=""` to prevent HEART from grabbing GPU memory.

## Running an Experiment

```bash
cd /data1/yujia/HEART/HEART

# Required env vars
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""

# Run (replace YAML and dataset as needed)
nohup /data1/yujia/envs/graphrag/bin/python main.py \
    -opt Option/LGBO_9params_0613.yaml \
    -dataset_name hotpotqa_1000_c5_real \
    > agent_workspace/runs/lgbo_9params_0613.log 2>&1 &

echo "PID: $!"
```

## Experiment Output

All run outputs go to `agent_workspace/runs/<exp_name>/`:

| File | Contents |
|------|---------|
| `Results/results.json` | Per-query answers and accuracy scores |
| `Results/full_eval_before_after_by_cluster.json` | Pre/post accuracy by cluster |
| `Results/kmeans_clustered_questions.jsonl` | Cluster assignments |
| `Metrics/` | Accuracy curves per round |
| `../lgbo_9params_0613.log` | Full stdout/stderr log |

## Experiment Configs

| Config file | Method | Description |
|-------------|--------|-------------|
| `Option/LGBO_9params_0613.yaml` | LGBO | 9 params, 8 rerankers, 10 rounds, budget=1200 |
| `Option/LGBO_1000q_real_5c_1200budget_0609.yaml` | LGBO | Original run that achieved 43.1% |
| `Option/LLAMBO_1000q_flat_5rounds_0612_v4.yaml` | LLAMBO | Baseline, achieved 42.8% |
| `Option/GPBO_1000q_flat_5rounds_0611.yaml` | GPBO | Baseline, achieved 38.5% |
| `Option/TPE_1000q_flat_5rounds_0610.yaml` | TPE | Baseline, achieved 38.4% |

## SSH Access

```bash
# From local machine (WSL) — VPN required on Windows side
ssh.exe 3haoji

# GitHub key on server
~/.ssh/id_rsa_cmy1   # authorized for YouranSun and JayLZhou/HEART
```
